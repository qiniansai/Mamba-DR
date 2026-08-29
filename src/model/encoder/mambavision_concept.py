"""MambaVision Concept Model with Ordinal DR Head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ModelOutput
import os
import math
import logging
import numpy as np


class SVMIntervention:
    """
    SVM-guided Positive Centroid Interpolation for test-time intervention.

    Core idea: for FN tokens (model missed a lesion), replace them with
    the average token of "has lesion" samples (positive centroid), keeping
    the original token norm. This directly tells the disease head "this
    lesion is present" using a token from the real training distribution.

    Method:
    1. Collect concept tokens from test data
    2. For each lesion, train SVM to find positive/negative centroids
    3. For FN tokens: interpolate toward positive centroid
       intervened = (1-alpha)*token + alpha*positive_centroid_normalized*||token||
       alpha = strength (0.0=no change, 1.0=full replacement)
    """

    def __init__(self, num_lesions, embed_dim, intervention_strength=1.0,
                 use_sv_centroid=True,
                 max_cascade_rounds=3, cascade_strength_multiplier=1.5):
        self.num_lesions = num_lesions
        self.embed_dim = embed_dim
        self.intervention_strength = intervention_strength
        self.use_sv_centroid = use_sv_centroid
        self.max_cascade_rounds = max_cascade_rounds
        self.cascade_strength_multiplier = cascade_strength_multiplier

        self.positive_centroids = None
        self.svm_models = None
        self.is_fitted = False

        self._concept_tokens_list = []
        self._concept_labels_list = []

    def collect(self, concept_tokens, concept_labels):
        self._concept_tokens_list.append(concept_tokens.detach().cpu())
        self._concept_labels_list.append(concept_labels.detach().cpu())

    def fit(self):
        from sklearn.svm import SVC

        all_tokens = torch.cat(self._concept_tokens_list, dim=0)
        all_labels = torch.cat(self._concept_labels_list, dim=0)

        positive_centroids = torch.zeros(self.num_lesions, self.embed_dim)
        svm_models = []

        for i in range(self.num_lesions):
            tokens_i = all_tokens[:, i, :]
            labels_i = all_labels[:, i].numpy()

            unique_labels = np.unique(labels_i)
            if len(unique_labels) < 2:
                logging.warning(
                    f"SVMIntervention: lesion {i} has only one class "
                    f"({unique_labels}), skipping"
                )
                svm_models.append(None)
                continue

            svm = SVC(kernel='linear', class_weight='balanced')
            svm.fit(tokens_i.numpy(), labels_i)
            svm_models.append(svm)

            if self.use_sv_centroid and hasattr(svm, 'n_support_') and svm.n_support_.sum() >= 2:
                n_neg_sv = svm.n_support_[0]
                pos_sv = svm.support_vectors_[n_neg_sv:]
                centroid = torch.from_numpy(pos_sv.mean(axis=0).copy()).float()
                centroid_type = "SV_pos_centroid"
            else:
                pos_mask = labels_i == 1
                centroid = tokens_i[pos_mask].mean(dim=0)
                centroid_type = "all_pos_centroid"

            positive_centroids[i] = centroid

            train_acc = svm.score(tokens_i.numpy(), labels_i)
            n_pos = int((labels_i == 1).sum())
            n_neg = int((labels_i == 0).sum())
            n_sv = int(svm.n_support_.sum()) if hasattr(svm, 'n_support_') else 0
            logging.info(
                f"SVMIntervention: lesion {i} - "
                f"acc={train_acc:.3f}, n_pos={n_pos}, n_neg={n_neg}, "
                f"n_sv={n_sv}, centroid={centroid_type}, "
                f"centroid_norm={centroid.norm():.2f}"
            )

        self.positive_centroids = positive_centroids
        self.svm_models = svm_models
        self.is_fitted = True

        self._concept_tokens_list = []
        self._concept_labels_list = []

        logging.info(
            f"SVMIntervention: fitted {self.num_lesions} classifiers, "
            f"centroid_type={'SV' if self.use_sv_centroid else 'all'}, "
            f"strength={self.intervention_strength}"
        )

        return positive_centroids

    @torch.no_grad()
    def intervene(self, concept_tokens, lesion_preds, lesion_labels,
                  strength_override=None):
        """Interpolate FN tokens toward positive centroid.

        intervened = (1-alpha)*token + alpha*centroid_normalized*||token||
        alpha = strength (0.0=no change, 1.0=full replacement with centroid)

        Args:
            concept_tokens: [B, num_lesions, embed_dim]
            lesion_preds: [B, num_lesions] - model predictions (0 or 1)
            lesion_labels: [B, num_lesions] - ground truth labels (0 or 1)
            strength_override: override base strength for cascade rounds

        Returns:
            intervened_tokens: [B, num_lesions, embed_dim]
            diagnostics: dict with intervention statistics
        """
        if not self.is_fitted:
            return concept_tokens, {}

        false_negatives = (lesion_preds == 0) & (lesion_labels == 1)

        if not false_negatives.any():
            return concept_tokens, {}

        alpha = strength_override if strength_override is not None else self.intervention_strength

        pos_centroids = self.positive_centroids.to(concept_tokens.device)
        centroid_norms = pos_centroids.norm(dim=-1, keepdim=True)
        centroid_norms = torch.clamp(centroid_norms, min=1e-8)
        normalized_centroids = pos_centroids / centroid_norms

        original_norms = concept_tokens.norm(dim=-1, keepdim=True)
        target_tokens = normalized_centroids.unsqueeze(0) * original_norms

        mask = false_negatives.float().unsqueeze(-1)

        intervened_tokens = concept_tokens + alpha * mask * (target_tokens - concept_tokens)

        num_fn = int(false_negatives.sum().item())
        fn_tokens_orig = concept_tokens[false_negatives]
        fn_tokens_intr = intervened_tokens[false_negatives]
        cos_sim = F.cosine_similarity(fn_tokens_orig, fn_tokens_intr, dim=-1)
        angular_change = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)) * 180.0 / math.pi

        diag = {
            'num_intervened': num_fn,
            'avg_angular_change': float(angular_change.mean()),
            'max_angular_change': float(angular_change.max()),
            'strength': alpha,
        }

        logging.info(
            f"SVMIntervention: {num_fn} FN intervened, "
            f"avg_angle={angular_change.mean():.1f}°, "
            f"max_angle={angular_change.max():.1f}°, "
            f"strength={alpha:.2f}"
        )

        return intervened_tokens, diag

    def reset(self):
        self._concept_tokens_list = []
        self._concept_labels_list = []
        self.is_fitted = False
        self.positive_centroids = None
        self.svm_models = None


from .mambavision_core.mamba_vision import (
    mamba_vision_T,
    mamba_vision_T2,
    mamba_vision_S,
    mamba_vision_B,
    mamba_vision_B_21k,
    mamba_vision_L,
    mamba_vision_L_21k,
    mamba_vision_L2,
    mamba_vision_L2_512_21k,
    mamba_vision_L3_256_21k,
    mamba_vision_L3_512_21k,
)

__all__ = [
    "mambavision_tiny_concept",
    "mambavision_tiny2_concept",
    "mambavision_small_concept",
    "mambavision_base_concept",
    "mambavision_base_21k_concept",
    "mambavision_large_concept",
    "mambavision_large_21k_concept",
    "mambavision_large2_concept",
    "mambavision_large2_512_21k_concept",
    "mambavision_large3_256_21k_concept",
    "mambavision_large3_512_21k_concept",
    "MambaVisionConcept",
    "BACKBONE_CONFIGS",
    "OrdinalDRHead",
    "LevelSpecificConceptAttention",
    "SVMIntervention",
]

CACHE_DIR = "./weights"

BACKBONE_CONFIGS = {
    "T": {
        "embed_dim": 640,
        "backbone_func": mamba_vision_T,
        "weight_file": "mambavision_tiny_1k.pth.tar",
        "display_name": "MambaVision",
    },
    "T2": {
        "embed_dim": 640,
        "backbone_func": mamba_vision_T2,
        "weight_file": "mambavision_tiny2_1k.pth.tar",
        "display_name": "MambaVision-T2",
    },
    "S": {
        "embed_dim": 768,
        "backbone_func": mamba_vision_S,
        "weight_file": "mambavision_small_1k.pth.tar",
        "display_name": "MambaVision-S",
    },
    "B": {
        "embed_dim": 1024,
        "backbone_func": mamba_vision_B,
        "weight_file": "mambavision_base_1k.pth.tar",
        "display_name": "MambaVision-B",
    },
    "B_21k": {
        "embed_dim": 1024,
        "backbone_func": mamba_vision_B_21k,
        "weight_file": "mambavision_base_21k.pth.tar",
        "display_name": "MambaVision-B-21k",
    },
    "L": {
        "embed_dim": 1568,
        "backbone_func": mamba_vision_L,
        "weight_file": "mambavision_large_1k.pth.tar",
        "display_name": "MambaVision-L",
    },
    "L_21k": {
        "embed_dim": 1568,
        "backbone_func": mamba_vision_L_21k,
        "weight_file": "mambavision_large_21k.pth.tar",
        "display_name": "MambaVision-L-21k",
    },
    "L2": {
        "embed_dim": 1568,
        "backbone_func": mamba_vision_L2,
        "weight_file": "mambavision_large2_1k.pth.tar",
        "display_name": "MambaVision-L2",
    },
    "L2_512_21k": {
        "embed_dim": 1568,
        "backbone_func": mamba_vision_L2_512_21k,
        "weight_file": "mambavision_L2_21k_240m_512.pth.tar",
        "display_name": "MambaVision-L2-512-21k",
    },
    "L3_256_21k": {
        "embed_dim": 2048,
        "backbone_func": mamba_vision_L3_256_21k,
        "weight_file": "mambavision_L3_21k_740m_256.pth.tar",
        "display_name": "MambaVision-L3-256-21k",
    },
    "L3_512_21k": {
        "embed_dim": 2048,
        "backbone_func": mamba_vision_L3_512_21k,
        "weight_file": "mambavision_L3_21k_740m_512.pth.tar",
        "display_name": "MambaVision-L3-512-21k",
    },
}


class OrdinalDRHead(nn.Module):
    """
    序数DR分类头，使用分层病变关注机制。
    """
    def __init__(self, embed_dim, num_classes=5, num_concepts=4, num_heads=8,
                 use_ordinal_constraint=True, use_level_specific_attention=True,
                 lesion_names=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.num_heads = num_heads
        self.use_ordinal_constraint = use_ordinal_constraint
        self.use_level_specific_attention = use_level_specific_attention

        if use_level_specific_attention:
            from .level_specific_attention import LevelSpecificConceptAttention
            self.level_attention = LevelSpecificConceptAttention(
                num_classes=num_classes,
                num_concepts=num_concepts,
                embed_dim=embed_dim,
                num_heads=num_heads,
                lesion_names=lesion_names,
                use_learnable_mask=True,
                mask_temperature=0.1,
            )
        else:
            self.dr_level_queries = nn.Parameter(torch.randn(num_classes - 1, embed_dim))
            nn.init.normal_(self.dr_level_queries, std=0.02)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )

            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.1)
            )

            self.concept_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
            )

        self.ordinal_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim // 2, 1)
            ) for _ in range(num_classes - 1)
        ])

    def forward(self, concept_tokens, return_ordinal_logits=False):
        if self.use_level_specific_attention:
            level_features, attn_weights, level_masks = self.level_attention(
                concept_tokens=concept_tokens,
            )
        else:
            B = concept_tokens.shape[0]

            gate_weights = self.concept_gate(concept_tokens)
            gate_weights = torch.sigmoid(gate_weights)

            gated_concepts = concept_tokens * gate_weights

            queries = self.dr_level_queries.unsqueeze(0).expand(B, -1, -1)

            attn_output, attn_weights = self.cross_attn(
                query=queries,
                key=gated_concepts,
                value=gated_concepts
            )

            attn_output = self.norm1(queries + attn_output)

            ffn_output = self.ffn(attn_output)
            level_features = self.norm2(attn_output + ffn_output)

        # Each classifier estimates a *conditional* transition probability:
        # P(Y >= k | Y >= k - 1), k = 1, ..., K - 1.  Multiplying the
        # transitions produces ordered cumulative probabilities by construction.
        conditional_logits = []
        for i, classifier in enumerate(self.ordinal_classifiers):
            logit = classifier(level_features[:, i, :])
            conditional_logits.append(logit)
        conditional_logits = torch.cat(conditional_logits, dim=1)

        disease_logits, _ = self.conditional_logits_to_class_logits(conditional_logits)

        if return_ordinal_logits:
            return disease_logits, attn_weights, conditional_logits
        return disease_logits, attn_weights

    @staticmethod
    def conditional_logits_to_class_logits(conditional_logits):
        """Convert conditional ordinal transitions to valid class log-probabilities.

        The resulting cumulative probabilities are monotone even when individual
        transition logits are unconstrained, avoiding the negative probability
        repair used by the former independent-threshold implementation.
        """
        transition_probs = torch.sigmoid(conditional_logits)
        cumulative_probs = torch.cumprod(transition_probs, dim=1)

        first_class = 1.0 - cumulative_probs[:, :1]
        middle_classes = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
        last_class = cumulative_probs[:, -1:]
        class_probs = torch.cat([first_class, middle_classes, last_class], dim=1)

        return torch.log(class_probs.clamp_min(1e-7)), cumulative_probs

    def _ordinal_to_multiclass(self, ordinal_logits):
        # Kept as a compatibility alias for older call sites.
        class_logits, _ = self.conditional_logits_to_class_logits(ordinal_logits)
        return class_logits

    @staticmethod
    def conditional_ordinal_loss(conditional_logits, targets):
        """Conditional BCE for a CORN-style ordinal chain.

        Threshold k is supervised only for samples that reached grade k.  This
        makes each classifier learn P(Y >= k+1 | Y >= k), rather than four
        unrelated binary tasks.
        """
        thresholds = torch.arange(
            conditional_logits.shape[1], device=targets.device
        ).unsqueeze(0)
        eligible = targets.unsqueeze(1) >= thresholds
        conditional_targets = (targets.unsqueeze(1) >= (thresholds + 1)).to(
            conditional_logits.dtype
        )

        per_entry = F.binary_cross_entropy_with_logits(
            conditional_logits, conditional_targets, reduction='none'
        )
        return (per_entry * eligible.to(per_entry.dtype)).sum() / eligible.sum().clamp_min(1)

    def ordinal_regression_loss(self, ordinal_logits, targets):
        return self.conditional_ordinal_loss(ordinal_logits, targets)


class GatedDiseaseHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_concepts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, num_classes),
        )
        self.num_concepts = num_concepts

    def forward(self, concept_tokens):
        gating_weights = self.gate(concept_tokens)
        gating_weights = F.softmax(gating_weights, dim=1)

        weighted_token = (concept_tokens * gating_weights).sum(dim=1)
        return self.classifier(weighted_token), gating_weights


class MambaVisionConcept(nn.Module):
    def __init__(
        self,
        num_lesions,
        num_classes=5,
        img_size=384,
        backbone_name="T",
        use_ordinal_head=True,
        ordinal_num_heads=8,
        use_level_specific_attention=True,
        lesion_names=None,
        intervention_strength=1.0,
        intervention_temperature=2.0,
        intervention_epsilon=0.01,
        **kwargs,
    ):
        super().__init__()
        if backbone_name not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE_CONFIGS.keys())}")

        config = BACKBONE_CONFIGS[backbone_name]

        self.num_lesions = num_lesions
        self.num_concepts = num_lesions
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone_name = backbone_name
        self.embed_dim = config["embed_dim"]
        self.use_ordinal_head = use_ordinal_head

        self.backbone = config["backbone_func"](pretrained=False, num_classes=1000, resolution=img_size)

        if use_ordinal_head:
            self.disease_head = OrdinalDRHead(
                self.embed_dim, self.num_classes, self.num_concepts,
                num_heads=ordinal_num_heads,
                use_level_specific_attention=use_level_specific_attention,
                lesion_names=lesion_names,
            )
        else:
            self.disease_head = GatedDiseaseHead(
                self.embed_dim, self.num_classes, self.num_concepts,
            )
        self.disease_head.apply(self._init_weights)

        self.lesion_head = nn.Conv2d(self.embed_dim, self.num_concepts, kernel_size=[1, 1])
        self.lesion_head.apply(self._init_weights)

        self.concept_cls_token = nn.Parameter(
            torch.zeros(1, self.num_concepts, self.embed_dim)
        )
        self.pos_embed_concept_cls = nn.Parameter(
            torch.zeros(1, self.num_concepts, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed_concept_cls, std=0.02)
        nn.init.normal_(self.concept_cls_token, std=0.02)

        self.intervention_strength = intervention_strength
        self.intervention_temperature = intervention_temperature
        self.intervention_epsilon = intervention_epsilon
        self.intervention_directions = nn.Parameter(
            torch.randn(num_lesions, self.embed_dim) * 0.02
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.backbone.patch_embed(x)

        for level in self.backbone.levels:
            x = level(x)

        x = self.backbone.norm(x)

        B, C, H, W = x.shape
        patch_tokens = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        cls_tokens = self.concept_cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_concept_cls

        patch_mean = patch_tokens.mean(dim=1, keepdim=True).repeat(1, self.num_concepts, 1)
        processed_concept_tokens = cls_tokens + patch_mean

        return (
            processed_concept_tokens,
            patch_tokens,
        )

    def apply_concept_corrections(self, concept_tokens, lesion_logits,
                                  correction_mask, corrected_probs):
        """Apply explicit clinician lesion corrections without using labels.

        ``correction_mask`` selects reviewed base lesions and ``corrected_probs``
        contains the clinician-confirmed probabilities in [0, 1].  Only selected
        concept tokens move along their learned lesion directions; a subsequent
        disease-head pass converts those corrections into a revised DR grade.
        """
        if correction_mask.shape != corrected_probs.shape:
            raise ValueError("correction_mask and corrected_probs must have identical shapes")
        if correction_mask.shape[1] != self.num_lesions:
            raise ValueError("corrections must contain one entry per base lesion")

        current_logits = lesion_logits[:, :self.num_lesions]
        target_logits = torch.logit(
            corrected_probs.to(current_logits.dtype).clamp(
                self.intervention_epsilon, 1.0 - self.intervention_epsilon
            )
        )
        delta = target_logits - current_logits
        scales = self.intervention_strength * torch.tanh(
            delta / self.intervention_temperature
        )
        directions = F.normalize(self.intervention_directions, dim=-1)
        updated_tokens = concept_tokens.clone()
        selected = correction_mask.to(concept_tokens.dtype).unsqueeze(-1)
        updated_tokens[:, :self.num_lesions] = (
            updated_tokens[:, :self.num_lesions]
            + selected * scales.unsqueeze(-1) * directions.unsqueeze(0)
        )
        reported_logits = lesion_logits.clone()
        reported_logits[:, :self.num_lesions] = torch.where(
            correction_mask.bool(), target_logits, current_logits
        )
        return updated_tokens, reported_logits

    def forward_heads_from_concept_tokens(self, concept_tokens, concept_patch_logits,
                                          reported_lesion_logits=None):
        lesion_logits = (concept_tokens.mean(-1) + concept_patch_logits) / 2
        if reported_lesion_logits is not None:
            lesion_logits = reported_lesion_logits
        disease_output = self.disease_head(
            concept_tokens, return_ordinal_logits=True
        ) if self.use_ordinal_head else self.disease_head(concept_tokens)
        return disease_output[0], lesion_logits

    def forward(
        self,
        x,
        return_attn=False,
        intervened_concept_tokens=None,
    ):
        (
            concept_tokens,
            patch_tokens,
        ) = self.forward_features(x)

        n, p, c = patch_tokens.shape
        w0 = h0 = int(math.sqrt(p))

        patch_tokens = torch.reshape(patch_tokens, [n, w0, h0, c])
        patch_tokens = patch_tokens.permute([0, 3, 1, 2])
        patch_tokens = patch_tokens.contiguous()

        concept_patch = self.lesion_head(patch_tokens)

        concept_patch_pooled = F.adaptive_max_pool2d(
            concept_patch, (1, 1)
        )
        concept_patch_logits = torch.flatten(concept_patch_pooled, 1)

        self._last_concept_patch_logits = concept_patch_logits

        concept_logits = concept_tokens.mean(-1)
        concept_logits = (concept_logits + concept_patch_logits) / 2

        base_lesion_logits = concept_logits[:, :self.num_lesions]

        lesion_logits = concept_logits

        concept_tokens_for_disease = concept_tokens

        if intervened_concept_tokens is not None:
            concept_tokens_for_disease = concept_tokens.clone()
            concept_tokens_for_disease[:, :self.num_lesions, :] = intervened_concept_tokens

        if self.use_ordinal_head:
            disease_output = self.disease_head(
                concept_tokens_for_disease,
                return_ordinal_logits=True,
            )
            disease_logits = disease_output[0]
            attn_weights = disease_output[1]
            ordinal_logits = disease_output[2]
        else:
            disease_output = self.disease_head(
                concept_tokens_for_disease,
            )
            disease_logits = disease_output[0]
            ordinal_logits = None

        if return_attn:
            cams = concept_patch.detach().clone()
            cams = F.relu(cams)

            return ModelOutput(
                disease_logits,
                lesion_logits,
                concept_tokens,
                cams,
                ordinal_logits,
            ), None

        return ModelOutput(
            disease_logits, lesion_logits, concept_tokens, None, ordinal_logits
        ), None

    def learn_concept_embed(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.lesion_head.parameters():
            param.requires_grad = True
        self.concept_cls_token.requires_grad = True

    def learn_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_lesion_token(self):
        self.concept_cls_token.requires_grad = False


def _load_pretrained_weights(model, backbone_name):
    config = BACKBONE_CONFIGS[backbone_name]
    pretrained_path = os.path.join(CACHE_DIR, config["weight_file"])

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained {config['display_name']} weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k[7:]
            else:
                new_k = k
            new_state_dict[new_k] = v

        model_dict = model.backbone.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_dict and 'head' not in k}
        model_dict.update(pretrained_dict)
        model.backbone.load_state_dict(model_dict, strict=False)
        print(f"Successfully loaded {len(pretrained_dict)} pretrained layers.")
    else:
        print(f"Warning: Pretrained weights file not found at {pretrained_path}")
    return model


def mambavision_tiny_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="T", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "T")
    return model


def mambavision_tiny2_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="T2", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "T2")
    return model


def mambavision_small_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="S", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "S")
    return model


def mambavision_base_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="B", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "B")
    return model


def mambavision_base_21k_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="B_21k", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "B_21k")
    return model


def mambavision_large_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L")
    return model


def mambavision_large_21k_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L_21k", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L_21k")
    return model


def mambavision_large2_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L2", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L2")
    return model


def mambavision_large2_512_21k_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L2_512_21k", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L2_512_21k")
    return model


def mambavision_large3_256_21k_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L3_256_21k", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L3_256_21k")
    return model


def mambavision_large3_512_21k_concept(pretrained=False, **kwargs):
    model = MambaVisionConcept(backbone_name="L3_512_21k", **kwargs)
    if pretrained:
        model = _load_pretrained_weights(model, "L3_512_21k")
    return model

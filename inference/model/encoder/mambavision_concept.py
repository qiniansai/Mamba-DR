"""MambaVision Concept Model — 推理精简版"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ModelOutput


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
from .mambavision_core.mamba_vision import mamba_vision_T

BACKBONE_CONFIGS = {
    "T": {"embed_dim": 640, "backbone_func": mamba_vision_T},
}

# ---------------------------------------------------------------------------
# Level-Specific Concept Attention
# ---------------------------------------------------------------------------
from .level_specific_attention import LevelSpecificConceptAttention

# ---------------------------------------------------------------------------
# Ordinal DR Head
# ---------------------------------------------------------------------------
class OrdinalDRHead(nn.Module):
    """序数DR分类头，使用分层病变关注机制。"""

    def __init__(self, embed_dim, num_classes=5, num_concepts=5, num_heads=8,
                 use_level_specific_attention=True, lesion_names=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.use_level_specific_attention = use_level_specific_attention

        if use_level_specific_attention:
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
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(0.1),
            )
            self.concept_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4), nn.Tanh(),
                nn.Linear(embed_dim // 4, 1),
            )

        self.ordinal_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(embed_dim // 2, 1),
            ) for _ in range(num_classes - 1)
        ])

    def forward(self, concept_tokens, return_ordinal_logits=False):
        if self.use_level_specific_attention:
            level_features, attn_weights, _ = self.level_attention(concept_tokens)
        else:
            B = concept_tokens.shape[0]
            gate_weights = torch.sigmoid(self.concept_gate(concept_tokens))
            gated_concepts = concept_tokens * gate_weights
            queries = self.dr_level_queries.unsqueeze(0).expand(B, -1, -1)
            attn_output, attn_weights = self.cross_attn(query=queries, key=gated_concepts, value=gated_concepts)
            attn_output = self.norm1(queries + attn_output)
            level_features = self.norm2(attn_output + self.ffn(attn_output))

        cond_logits = torch.cat([clf(level_features[:, i, :]) for i, clf in enumerate(self.ordinal_classifiers)], dim=1)
        disease_logits, _ = self._cond_to_class(cond_logits)
        if return_ordinal_logits:
            return disease_logits, attn_weights, cond_logits
        return disease_logits, attn_weights

    @staticmethod
    def _cond_to_class(cond_logits):
        tp = torch.sigmoid(cond_logits)
        cp = torch.cumprod(tp, dim=1)
        probs = torch.cat([1.0 - cp[:, :1], cp[:, :-1] - cp[:, 1:], cp[:, -1:]], dim=1)
        return torch.log(probs.clamp_min(1e-7)), cp


# ---------------------------------------------------------------------------
# Gated Disease Head (fallback)
# ---------------------------------------------------------------------------
class GatedDiseaseHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_concepts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.Tanh(), nn.Linear(embed_dim // 4, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, num_classes))

    def forward(self, concept_tokens):
        gw = F.softmax(self.gate(concept_tokens), dim=1)
        return self.classifier((concept_tokens * gw).sum(dim=1)), gw


# ---------------------------------------------------------------------------
# MambaVisionConcept (主模型)
# ---------------------------------------------------------------------------
class MambaVisionConcept(nn.Module):
    def __init__(self, num_lesions, num_classes=5, img_size=384,
                 backbone_name="T", use_ordinal_head=True, ordinal_num_heads=8,
                 use_level_specific_attention=True, lesion_names=None, **kwargs):
        super().__init__()
        cfg = BACKBONE_CONFIGS[backbone_name]
        self.num_lesions = num_lesions
        self.num_concepts = num_lesions
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = cfg["embed_dim"]
        self.use_ordinal_head = use_ordinal_head

        self.backbone = cfg["backbone_func"](pretrained=False, num_classes=1000, resolution=img_size)

        if use_ordinal_head:
            self.disease_head = OrdinalDRHead(
                self.embed_dim, self.num_classes, self.num_concepts,
                num_heads=ordinal_num_heads,
                use_level_specific_attention=use_level_specific_attention,
                lesion_names=lesion_names,
            )
        else:
            self.disease_head = GatedDiseaseHead(self.embed_dim, self.num_classes, self.num_concepts)
        self.disease_head.apply(self._init_weights)

        self.lesion_head = nn.Conv2d(self.embed_dim, self.num_concepts, kernel_size=1)
        self.lesion_head.apply(self._init_weights)

        self.concept_cls_token = nn.Parameter(torch.zeros(1, self.num_concepts, self.embed_dim))
        self.pos_embed_concept_cls = nn.Parameter(torch.zeros(1, self.num_concepts, self.embed_dim))
        nn.init.normal_(self.pos_embed_concept_cls, std=0.02)
        nn.init.normal_(self.concept_cls_token, std=0.02)

    @staticmethod
    def _init_weights(m):
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
        patch_tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cls_tokens = self.concept_cls_token.expand(B, -1, -1) + self.pos_embed_concept_cls
        patch_mean = patch_tokens.mean(dim=1, keepdim=True).repeat(1, self.num_concepts, 1)
        return cls_tokens + patch_mean, patch_tokens

    def forward(self, x, return_attn=False):
        concept_tokens, patch_tokens = self.forward_features(x)

        n, p, c = patch_tokens.shape
        w0 = h0 = int(math.sqrt(p))
        pt = patch_tokens.reshape(n, w0, h0, c).permute(0, 3, 1, 2).contiguous()
        concept_patch = self.lesion_head(pt)
        concept_patch_logits = torch.flatten(F.adaptive_max_pool2d(concept_patch, (1, 1)), 1)

        concept_logits = (concept_tokens.mean(-1) + concept_patch_logits) / 2
        lesion_logits = concept_logits

        if self.use_ordinal_head:
            disease_logits, attn, ordinal = self.disease_head(concept_tokens, return_ordinal_logits=True)
        else:
            disease_logits, attn = self.disease_head(concept_tokens)
            ordinal = None

        if return_attn:
            cams = F.relu(concept_patch.detach().clone())
            return ModelOutput(disease_logits, lesion_logits, concept_tokens, cams, ordinal), None
        return ModelOutput(disease_logits, lesion_logits, concept_tokens, None, ordinal), None

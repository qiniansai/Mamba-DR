import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from pathlib import Path
import numpy as np

from model.data import DataItem
from model.encoder import load_encoder
from model.utils import ModelOutput
from model.losses.KnowledgeGuideLoss import KnowledgeGuideLoss
from model.encoder.text_encoder import TextEncoderWithPrompts, definitions, lesion_abbreviations



class Model(LightningModule):
    def __init__(
        self,
        disease_names: list[str],
        lesion_names: list[str],
        img_size: int = 384,
        arch_name: str = "mambavision_tiny_concept",
        pretrained: bool = True,
        disease_loss_weight: float = 1.0,
        disease_label_smoothing: float = 0.1,
        lesion_loss_weight: float = 1.0,
        KG_loss_weight: float = 0.1,
        with_EK: bool = False,
        dropout_rate: float = 0.4,
        weight_decay: float = 1e-4,
        kg_loss_base_weight: float = 1.0,
        use_ordinal_head: bool = True,
        ordinal_num_heads: int = 8,
        intervention_strength: float = 1.0,
        intervention_temperature: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_disease = len(disease_names)
        self.num_lesions = len(lesion_names)
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.img_size = img_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kg_loss_base_weight = kg_loss_base_weight
        self.use_ordinal_head = use_ordinal_head
        self.disease_label_smoothing = disease_label_smoothing
        self.intervention_strength = intervention_strength
        self.intervention_temperature = intervention_temperature

        self.model = load_encoder(
            arch_name,
            pretrained=pretrained,
            num_classes=self.num_disease,
            num_lesions=self.num_lesions,
            img_size=img_size,
            use_ordinal_head=use_ordinal_head,
            ordinal_num_heads=ordinal_num_heads,
            use_level_specific_attention=True,
            lesion_names=lesion_names,
            intervention_strength=intervention_strength,
            intervention_temperature=intervention_temperature,
        )
        self.disease_loss_weight = disease_loss_weight
        self.lesion_loss_weight = lesion_loss_weight
        self.KG_loss_weight = KG_loss_weight

        self.text_encoder = TextEncoderWithPrompts(
            bert_type='emilyalsentzer/Bio_ClinicalBERT',
            proj_dim=self.model.embed_dim,
            caption="A fundus photograph of [CLS]",
        )

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = self.text_encoder.to(device)

        lesion_full_names = [lesion_abbreviations.get(name, name) for name in lesion_names]

        presence_embeds, absence_embeds = self.text_encoder.compute_presence_absence_embeddings(
            categories=lesion_full_names,
            domain_knowledge=definitions,
            device=device
        )

        self.token2concept = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.embed_dim, presence_embeds.shape[1]),
        )



        self.loss_lesion = nn.MultiLabelSoftMarginLoss()

        self.loss_knowledge_guide = KnowledgeGuideLoss(
            presence_embeds=presence_embeds,
            absence_embeds=absence_embeds,
            lesion_names=lesion_names,
            base_weight=kg_loss_base_weight,
        )


    def forward(
        self,
        batch: DataItem,
        return_attn: bool = False,
    ):
        imgs = batch.image
        disease_lbls = batch.disease_lbls
        lesion_lbls = batch.lesion_lbls

        output: ModelOutput = self.model(
            imgs,
            return_attn=return_attn,
        )

        if isinstance(output, tuple):
            output, _ = output

        disease_logits = output.disease_logits
        lesion_logits = output.lesion_logits
        cams = output.cams

        disease_targets = disease_lbls
        if disease_lbls.dim() > 1 and disease_lbls.shape[1] > 1:
            disease_targets = torch.argmax(disease_lbls, dim=1)
        disease_loss = F.cross_entropy(
            disease_logits, disease_targets, label_smoothing=self.disease_label_smoothing
        )

        base_lesion_lbls = lesion_lbls[:, :self.num_lesions]
        base_lesion_logits = lesion_logits[:, :self.num_lesions]

        lesion_pos_counts = base_lesion_lbls.sum(dim=0)
        lesion_neg_counts = len(base_lesion_lbls) - lesion_pos_counts
        lesion_weights = lesion_neg_counts / (lesion_pos_counts + 1e-6)
        lesion_weights = torch.clamp(lesion_weights, 0.5, 5.0)

        base_lesion_loss = F.binary_cross_entropy_with_logits(
            base_lesion_logits,
            base_lesion_lbls,
            weight=lesion_weights.unsqueeze(0).expand_as(base_lesion_lbls),
            reduction='mean'
        )

        lesion_loss = base_lesion_loss

        if self.KG_loss_weight > 0:
            concept_tokens = self.token2concept(output.lesion_tokens)
            kg_loss = self.loss_knowledge_guide(
                inputs=concept_tokens,
                targets=lesion_lbls,
            )
        else:
            kg_loss = 0

        total_loss = (
            self.disease_loss_weight * disease_loss
            + self.lesion_loss_weight * lesion_loss
            + self.KG_loss_weight * kg_loss
        )

        return {
            "loss": total_loss,
            "disease_logits": disease_logits,
            "lesion_logits": lesion_logits,
            "disease_loss": disease_loss,
            "lesion_loss": lesion_loss,
            "kg_loss": kg_loss,
            "cams": cams,
        }

    def shared_step(self, batch: DataItem, batch_idx: int, stage: str):
        output = self(batch, return_attn=False)
        self.log(
            f"{stage}/loss",
            output["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.image.shape[0],
        )
        self.log(
            f"{stage}/disease_loss",
            output["disease_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.image.shape[0],
        )
        self.log(
            f"{stage}/lesion_loss",
            output["lesion_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.image.shape[0],
        )
        self.log(
            f"{stage}/kg_loss",
            output["kg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.image.shape[0],
        )

        return output

    def training_step(self, batch: DataItem, batch_idx: int):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: DataItem, batch_idx: int):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch: DataItem, batch_idx: int):
        output = self.forward(batch, return_attn=True)
        # Standard test metrics are label-blind model predictions.  Clinician
        # corrections are evaluated only through the explicit intervene API.
        return {
            "disease_logits": output["disease_logits"],
            "lesion_logits": output["lesion_logits"],
            "cams": output["cams"],
        }

    @torch.no_grad()
    def intervene(self, batch: DataItem, correction_mask: torch.Tensor,
                  corrected_probs: torch.Tensor):
        """Re-grade a case after externally supplied clinician corrections."""
        encoder_output = self.model(batch.image, return_attn=True)
        if isinstance(encoder_output, tuple):
            encoder_output, _ = encoder_output
        corrected_tokens, reported_logits = self.model.apply_concept_corrections(
            encoder_output.lesion_tokens, encoder_output.lesion_logits,
            correction_mask, corrected_probs,
        )
        disease_logits, lesion_logits = self.model.forward_heads_from_concept_tokens(
            corrected_tokens, self.model._last_concept_patch_logits, reported_logits,
        )
        return {
            "disease_logits": disease_logits,
            "lesion_logits": lesion_logits,
            "cams": encoder_output.cams,
        }

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=1e-4, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint):
        import logging

        state_dict = checkpoint.get('state_dict', {})
        model_state_dict = self.state_dict()

        matched_keys = []
        unexpected_keys = []
        missing_keys = []

        for key in list(state_dict.keys()):
            if key in model_state_dict:
                if state_dict[key].shape == model_state_dict[key].shape:
                    matched_keys.append(key)
                else:
                    unexpected_keys.append(f"{key} (shape mismatch: {state_dict[key].shape} vs {model_state_dict[key].shape})")
                    del state_dict[key]
            else:
                unexpected_keys.append(key)
                del state_dict[key]

        for key in model_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)

        logging.info("\n" + "=" * 80)
        logging.info("Checkpoint Loading Summary")
        logging.info("=" * 80)
        logging.info(f"✅ Matched keys: {len(matched_keys)}")

        if unexpected_keys:
            logging.info(f"⚠️  Filtered keys (unexpected): {len(unexpected_keys)}")
            for key in unexpected_keys[:10]:
                logging.info(f"   - {key}")
            if len(unexpected_keys) > 10:
                logging.info(f"   ... and {len(unexpected_keys) - 10} more")

        if missing_keys:
            logging.info(f"⚠️  Missing keys (will be randomly initialized): {len(missing_keys)}")
            for key in missing_keys[:10]:
                logging.info(f"   - {key}")
            if len(missing_keys) > 10:
                logging.info(f"   ... and {len(missing_keys) - 10} more")

        logging.info("=" * 80)

        checkpoint['state_dict'] = state_dict

        logging.info("Concept-intervention parameters are initialized from the current config.")
        
        logging.info(f"\n✅ Concept-intervention parameters restored:")
        logging.info(f"   - intervention_strength: {self.intervention_strength}")
        logging.info(f"   - intervention_temperature: {self.intervention_temperature}")

        if 'optimizer_states' in checkpoint:
            del checkpoint['optimizer_states']
            logging.info("📝 Optimizer states removed from checkpoint")

        if 'lr_schedulers' in checkpoint:
            del checkpoint['lr_schedulers']
            logging.info("📝 LR scheduler states removed from checkpoint")

        logging.info("✅ Only model weights will be loaded (training state reset)")
        logging.info("=" * 80)

        self._weights_only = True

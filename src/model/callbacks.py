import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from model.data import DataItem
from model.model import Model
from model.utils import CLSandCPTMetrics, fit_rs2table, test_rs2table


class FlatModelCheckpoint(ModelCheckpoint):
    """
    自定义ModelCheckpoint，将checkpoint直接保存在指定目录下，不创建子文件夹
    """
    def __init__(self, dirpath=None, filename=None, monitor=None, mode="min",
                 save_top_k=1, save_weights_only=False, **kwargs):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            **kwargs
        )

    def format_checkpoint_name(self, metrics, filename=None, ver=None):
        filename = filename or self.filename

        if filename:
            for k, v in metrics.items():
                filename = filename.replace(f"{{{k}}}", str(v))
                if ":" in k:
                    key_parts = k.split(":")
                    if len(key_parts) == 2:
                        key_name = key_parts[0]
                        if key_name in metrics:
                            filename = filename.replace(f"{{{k}}}", str(metrics[key_name]))

        if filename:
            return f"{filename}.ckpt"
        else:
            return f"checkpoint.ckpt"

    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None):
        filepath = self.format_checkpoint_name(monitor_candidates)

        if self.dirpath:
            filepath = os.path.join(self.dirpath, filepath)
        else:
            filepath = os.path.join(trainer.default_root_dir, "checkpoints", filepath)

        return filepath


class MetricsCaculator(Callback):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: Model) -> None:
        self.best_metrics = None
        self.best_epoch = None
        assert isinstance(pl_module, Model), (
            f"`pl_module` must be Model Class, but got {type(pl_module)}"
        )
        self.train_metrics = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)
        self.val_metrics = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)

    def on_test_start(self, trainer: Trainer, pl_module: Model) -> None:
        assert isinstance(pl_module, Model), (
            f"`pl_module` must be Model Class, but got {type(pl_module)}"
        )
        self.test_metrics_normal = CLSandCPTMetrics(
            pl_module.disease_names, pl_module.lesion_names
        ).to(pl_module.device)

    def _get_base_lesion_logits(self, pl_module: Model, lesion_logits: torch.Tensor) -> torch.Tensor:
        return lesion_logits

    def _get_base_lesion_labels(self, pl_module: Model, lesion_lbls: torch.Tensor) -> torch.Tensor:
        return lesion_lbls

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
    ) -> None:
        base_lesion_logits = self._get_base_lesion_logits(pl_module, outputs["lesion_logits"])
        base_lesion_lbls = self._get_base_lesion_labels(pl_module, batch.lesion_lbls)

        self.train_metrics.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            base_lesion_logits,
            base_lesion_lbls,
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        base_lesion_logits = self._get_base_lesion_logits(pl_module, outputs["lesion_logits"])
        base_lesion_lbls = self._get_base_lesion_labels(pl_module, batch.lesion_lbls)

        self.val_metrics.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            base_lesion_logits,
            base_lesion_lbls,
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        base_lesion_lbls = self._get_base_lesion_labels(pl_module, batch.lesion_lbls)

        base_lesion_logits = self._get_base_lesion_logits(pl_module, outputs["lesion_logits"])
        self.test_metrics_normal.update(
            outputs["disease_logits"],
            batch.disease_lbls,
            base_lesion_logits,
            base_lesion_lbls,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        train_updated = False
        if hasattr(self.train_metrics, 'cls_metrics') and hasattr(self.train_metrics.cls_metrics, 'kappa'):
            train_updated = self.train_metrics.cls_metrics.kappa.update_called

        if train_updated:
            train_rs = self.train_metrics.compute()
        else:
            train_rs = {}

        val_updated = False
        if hasattr(self.val_metrics, 'cls_metrics') and hasattr(self.val_metrics.cls_metrics, 'kappa'):
            val_updated = self.val_metrics.cls_metrics.kappa.update_called

        if val_updated:
            val_rs = self.val_metrics.compute()
        else:
            val_rs = {}

        self.train_metrics.reset()
        self.val_metrics.reset()
        self.current_val_rs = val_rs

        table = fit_rs2table(
            trainer.current_epoch, train_rs, val_rs, self.best_metrics, self.best_epoch
        )
        logging.info(f"\n{table}") if self.verbose else None

        for name, value in train_rs.items():
            pl_module.log(
                f"train/{name}",
                value,
                prog_bar=True if name in ["kappa"] else False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        for name, value in val_rs.items():
            pl_module.log(
                f"val/{name}",
                value,
                prog_bar=True if name in ["kappa"] else False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logging.info("\n" + "=" * 80)
        logging.info("TEST RESULTS - STANDARD (LABEL-BLIND) INFERENCE")
        logging.info("=" * 80)
        table_normal = test_rs2table(self.test_metrics_normal.compute())
        logging.info(f"\n{table_normal}") if self.verbose else None

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint,
    ) -> None:
        self.best_metrics = self.current_val_rs
        self.best_epoch = trainer.current_epoch


class GenHeatmap(Callback):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

    def on_test_start(self, trainer: Trainer, pl_module: Model) -> None:
        assert isinstance(pl_module, Model), (
            f"`pl_module` must be Model Class, but got {type(pl_module)}"
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Model,
        outputs: dict,
        batch: DataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.enabled:
            return
        img_size = pl_module.img_size
        logger = pl_module.logger.experiment.add_image

        if "normal" in outputs and "intervened" in outputs:
            output_to_use = outputs["normal"]
        else:
            output_to_use = outputs

        cams = output_to_use.get("cams")
        disease_logits = output_to_use["disease_logits"]
        disease_lbls = batch.disease_lbls
        img_ids = batch.id
        img_paths = batch.img_path

        if cams is None:
            return

        cams = F.interpolate(
            cams,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        cam_list = []
        lesion_confidence = torch.sigmoid(output_to_use["lesion_logits"])
        cam_list.append(cams)
        cls_attentions = torch.sum(torch.stack(cam_list), dim=0)

        num_total_concepts = cams.shape[1]

        concept_names = pl_module.lesion_names.copy()

        for b in range(batch.image.shape[0]):
            for concept_ind in range(num_total_concepts):
                concept_cls_score = format(
                    lesion_confidence[b, concept_ind].cpu().numpy(), ".4f"
                )

                concept_cls_attention = cls_attentions[b, concept_ind]

                attention_min = concept_cls_attention.min()
                attention_max = concept_cls_attention.max()

                if torch.isnan(attention_min) or torch.isnan(attention_max):
                    concept_cls_attention = torch.zeros_like(concept_cls_attention)
                elif attention_max - attention_min < 1e-8:
                    concept_cls_attention = torch.full_like(concept_cls_attention, 0.5)
                else:
                    concept_cls_attention = (
                        concept_cls_attention - attention_min
                    ) / (attention_max - attention_min)

                concept_cls_attention = torch.clamp(concept_cls_attention, 0.0, 1.0)

                concept_cls_attention = concept_cls_attention.detach().cpu().numpy()

                concept_cls_attention = np.nan_to_num(concept_cls_attention, nan=0.0, posinf=1.0, neginf=0.0)
                concept_cls_attention = np.clip(concept_cls_attention, 0.0, 1.0)

                img_path = img_paths[b]
                raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                raw_img = np.array(cv2.resize(raw_img, (img_size, img_size)))
                heatmap = cv2.applyColorMap(
                    (255 * concept_cls_attention).astype(np.uint8), cv2.COLORMAP_JET
                )
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                cam = 0.5 * raw_img.astype(np.float32) / 255.0 + 0.5 * heatmap / 255.0
                cam = cam.transpose((2, 0, 1))

                line = np.ones((3, img_size, 2))
                vis = np.concatenate([cam, line], axis=2)
                ds = disease_logits[b].argmax().item()

                concept_name = concept_names[concept_ind] if concept_ind < len(concept_names) else f"concept_{concept_ind}"
                logger(
                    f"{concept_name}/ls_{concept_cls_score}/ds_{ds}/lbl_{disease_lbls[b]}/id_{img_ids[b]}",
                    vis,
                    global_step=trainer.global_step,
                )

# fixed bugs in China by Yutao Zhao 2026.1.28
import logging
import os
from collections import  namedtuple
from typing import Literal, Optional
import pandas as pd
import prettytable as pt
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    F1Score,
    Metric,
    MetricCollection,
    Recall,
    Specificity,
)

ModelOutput = namedtuple(
    "ModelOutput",
    [
        "disease_logits",
        "lesion_logits",
        "lesion_tokens",
        "cams",
        "ordinal_logits",  # 新增：序数logits用于计算序数回归损失
    ],
    defaults=(None,),  # ordinal_logits默认为None，保持向后兼容
)


# region logging，日志功能
def logging_config(log_dir: Optional[str] = None, rank: Optional[int] = None):
    """Configure logging

    Args:
        log_dir (str, optional): The directory to save the log file. Defaults to None.
    """
    if log_dir:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = f"records_r{rank}" if rank is not None else "records"
        handler = logging.FileHandler(f"{log_dir}/{log_name}.log")
        formatter = logging.Formatter(
            "%(asctime)s%(message)s", datefmt="[%Y/%m/%d %H:%M:%S]"
        )
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().handlers = [handler]

# endregion


# region data处理函数
def kfold_split(
    kfold: int, fold_num: int, disease_labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    train_idx, val_idx = list(
        skf.split(disease_labels, y=disease_labels.iloc[:, 1:].values.argmax(axis=1))
    )[fold_num]
    train_disease_labels = disease_labels.iloc[train_idx, :]
    val_disease_labels = disease_labels.iloc[val_idx, :]
    test_disease_labels = val_disease_labels
    return train_disease_labels, val_disease_labels, test_disease_labels  # type: ignore


def handout_split(
    val_size: Optional[float], test_size: float, disease_labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size == 1:
        return None, None, disease_labels  # type: ignore

    train_disease_labels, test_disease_labels = train_test_split(
        disease_labels,
        test_size=test_size,
        stratify=disease_labels.iloc[:, 1:],
        random_state=42,
    )

    if val_size:
        train_disease_labels, val_disease_labels = train_test_split(
            train_disease_labels,
            test_size=val_size,
            stratify=train_disease_labels.iloc[:, 1:],
            random_state=42,
        )
    else:
        val_disease_labels = test_disease_labels

    return train_disease_labels, val_disease_labels, test_disease_labels


# endregion


# region metrics
# 计算评价指标
def configure_metrics(
    num_disease: int,
    num_lesion: int,
    disease_avg: Literal["micro", "macro", "weighted"] = "micro",
    lesion_avg: Literal["micro", "macro", "weighted"] = "micro",
    device: str = "cpu",
) -> tuple[MetricCollection, MetricCollection]:
    """Configure metrics for classification and concept detection tasks

    Args:
        n_cls (int): Integer specifying the number of classes. Defaults to None.
        n_cpt (int): Integer specifying the number of concepts. Defaults to None.
        average (Literal["micro", "macro"], optional): Defines the reduction that is applied over labels. Defaults to "micro".

    Returns:
        Tuple[MetricCollection, MetricCollection]: 1st element is classification metrics, 2rd element is concept detection metrics
    """
    if num_disease is None:
        raise ValueError("`num_disease` must be specified")
    if num_lesion is None:
        raise ValueError("`num_lesion` must be specified")
    task = "multiclass" if num_disease > 1 else "binary"
    cls_metrics = MetricCollection(
        {
            "kappa": CohenKappa(
                task=task, num_classes=num_disease, weights="quadratic"
            ),
            "sensitivity": (
                Recall(task=task, num_classes=num_disease, average=disease_avg)
                if task == "multiclass"
                else Recall(task=task, num_labels=num_disease, average=disease_avg)
            ),
            "specificity": (
                Specificity(task=task, num_classes=num_disease, average=disease_avg)
                if task == "multiclass"
                else Specificity(task=task, num_labels=num_disease, average=disease_avg)
            ),
            "auc": (
                AUROC(task=task, num_classes=num_disease, average="macro")
                if task == "multiclass"
                else AUROC(task=task, num_labels=num_disease, average="macro")
            ),
        }
    )
    if disease_avg != "micro":
        cls_metrics.add_metrics(
            {
                "acc": (
                    Accuracy(task=task, num_classes=num_disease, average=disease_avg)
                    if task == "multiclass"
                    else Accuracy(
                        task=task, num_labels=num_disease, average=disease_avg
                    )
                ),
                "f1": (
                    F1Score(task=task, num_classes=num_disease, average=disease_avg)
                    if task == "multiclass"
                    else F1Score(task=task, num_labels=num_disease, average=disease_avg)
                ),
            }
        )

    cpt_metrics = MetricCollection(
        {
            "f1": F1Score(task="multilabel", num_labels=num_lesion, average=lesion_avg),
            "acc": Accuracy(
                task="multilabel", num_labels=num_lesion, average=lesion_avg
            ),
            "auc": AUROC(task="multilabel", num_labels=num_lesion, average="macro"),
        },
        prefix="cpt_",
    )

    return cls_metrics.to(device), cpt_metrics.to(device)


class CLSandCPTMetrics(Metric):
    def __init__(
        self,
        cls_names: list[str],
        cpt_names: list[str],
        cls_avg: Literal["micro", "macro", "weighted"] = "micro",
        cpt_avg: Literal["micro", "macro", "weighted"] = "micro",
        **kwargs,
    ):
        """
        Metrics for classification and concept detection tasks,
        inherited from torchmetrics.Metric

        Args:
            cls_names (List[str]): The list of class names
            cpt_names (List[str]): The list of concept names
            cls_avg (Literal["micro", "macro"], optional): The average method
                for classification metrics. Defaults to "micro".
            cpt_avg (Literal["micro", "macro"], optional): The average method
                for concept detection metrics. Defaults to "micro".
        """
        super().__init__(**kwargs)
        self.n_cls = len(cls_names)
        self.n_cpt = len(cpt_names)
        self.cls_names = cls_names
        self.cpt_names = cpt_names
        self.cls_metrics, self.cpt_metrics = configure_metrics(
            self.n_cls, self.n_cpt, cls_avg, cpt_avg
        )

    def reset(self) -> None:
        self.cls_metrics.reset()
        self.cpt_metrics.reset()

    def update(self, cls_logits, cls_lbls, cpt_logits, cpt_lbls) -> None:
        # Convert labels to long type as required by torchmetrics
        cls_lbls_long = cls_lbls.long() if cls_lbls.dtype != torch.long else cls_lbls
        cpt_lbls_long = cpt_lbls.long() if cpt_lbls.dtype != torch.long else cpt_lbls
        
        # 修复维度问题：确保预测和标签维度匹配
        if self.n_cls > 1:
            # 多分类：预测需要是概率分布 [B, num_classes]
            cls_preds = cls_logits.detach().softmax(1)
            # 标签需要是类别索引 [B]（如果是one-hot需要转换）
            if cls_lbls_long.dim() > 1 and cls_lbls_long.shape[1] > 1:
                # 如果是one-hot编码，转换为类别索引
                cls_lbls_indices = torch.argmax(cls_lbls_long, dim=1)
            else:
                # 确保标签是1D的 [B]
                cls_lbls_indices = cls_lbls_long.view(-1)
        else:
            # 二分类：预测和标签都是[B]
            cls_preds = cls_logits.detach().sigmoid().view(-1)
            cls_lbls_indices = cls_lbls_long.view(-1)
            
        self.cls_metrics.update(cls_preds, cls_lbls_indices)
        self.cpt_metrics.update(cpt_logits.detach().sigmoid(), cpt_lbls_long)

    def compute(self) -> dict:
        cls_metrics = self.cls_metrics.compute()
        cpt_metrics = self.cpt_metrics.compute()
        self.result = {**cls_metrics, **cpt_metrics}
        return self.result


# endregion


# region visualization 下面两个日志函数
def fit_rs2table(
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    best_metrics: Optional[dict] = None,
    best_epoch: Optional[int] = None,
) -> pt.PrettyTable:
    table = pt.PrettyTable()
    
    # 如果train_metrics为空，使用val_metrics的keys作为列名
    if train_metrics:
        keys = list(train_metrics.keys())
    elif val_metrics:
        keys = list(val_metrics.keys())
    else:
        keys = []
    
    table.field_names = [f"Epoch {epoch}", *keys]
    
    if train_metrics:
        table.add_row(["Train"] + [f"{v:.2%}" for v in train_metrics.values()])
    else:
        table.add_row(["Train"] + ["-" for _ in keys])
    
    if val_metrics:
        table.add_row(["Val"] + [f"{v:.2%}" for v in val_metrics.values()])
    else:
        table.add_row(["Val"] + ["-" for _ in keys])
    
    if best_metrics and best_epoch:
        table.add_row(
            [f"Best ep{best_epoch}"] + [f"{v:.2%}" for v in best_metrics.values()]
        )
    return table


def test_rs2table(rs: dict) -> pt.PrettyTable:
    table = pt.PrettyTable()
    table.field_names = [""] + list(rs.keys())
    row_data = [f"{v * 100:.2f}" for v in rs.values()]
    table.add_row(["Test"] + row_data)
    return table


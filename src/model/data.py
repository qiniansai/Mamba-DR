import re
from collections import namedtuple
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from pandas import DataFrame
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
import torch

from model.utils import handout_split, kfold_split

DataItem = namedtuple(
    "DataItem", ["image", "disease_lbls", "lesion_lbls", "id", "img_path"]
)


def custom_collate_fn(batch):
    """自定义collate函数，处理包含字符串的DataItem"""
    images = torch.stack([item.image for item in batch])
    disease_lbls = torch.stack([torch.as_tensor(item.disease_lbls) for item in batch])
    lesion_lbls_list = [item.lesion_lbls for item in batch]
    lesion_arrays = [np.array(lbls, dtype=np.float32) for lbls in lesion_lbls_list]
    lesion_lbls = torch.from_numpy(np.stack(lesion_arrays))
    ids = [item.id for item in batch]
    img_paths = [item.img_path for item in batch]
    
    return DataItem(
        image=images,
        disease_lbls=disease_lbls,
        lesion_lbls=lesion_lbls,
        id=ids,
        img_path=img_paths,
    )


class FundusDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        disease_names: list[str],
        lesion_names: list[str],
        val_size: Optional[float] = None,
        test_size: float = 0.2,
        kfold: int = 0,
        fold_num: int = -1,
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 384,
        root_dir: str = "./data",
        enhanced_aug: bool = True,  # 增强数据增强开关
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.disease_names = disease_names
        self.lesion_names = lesion_names
        self.val_size = val_size
        self.test_size = test_size
        self.kfold = kfold
        self.fold_num = fold_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.root_dir = Path(root_dir)
        self.enhanced_aug = enhanced_aug
        # print configs
        print("Batch size: {}, Image size: {}".format(self.batch_size, self.img_size))
        # 构建数据增强管道
        self.train_transforms = self._build_transforms()
        self.eval_transforms = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        )
    
    def _build_transforms(self):
        """构建数据增强管道，支持增强模式"""
        if self.enhanced_aug:
            # 强化版数据增强策略 - 针对过拟合优化
            print("Dataset: {}, Using STRONG data augmentation for overfitting reduction.".format(self.dataset_name))
            transforms = [
                A.Resize(self.img_size, self.img_size),
                # 几何变换 - 增强
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.4),
                A.RandomRotate90(p=0.4),
                A.Affine(
                    translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
                    scale=(0.75, 1.25),  # 更大的缩放范围
                    rotate=(-30, 30),    # 更大的旋转范围
                    shear=(-15, 15),
                    p=0.7,
                ),
                # 弹性变形（模拟不同设备成像差异）
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.05, p=0.5),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    A.ElasticTransform(alpha=150, sigma=150 * 0.05, p=0.5),
                ], p=0.4),
                # 噪声与模糊（模拟图像质量下降）
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                    A.ISONoise(intensity=(0.1, 0.5), p=0.3),
                    A.Blur(blur_limit=5, p=0.3),
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.MedianBlur(blur_limit=5, p=0.2),
                ], p=0.4),
                # 颜色与对比度（模拟不同光照条件）
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
                    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.4),
                    A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.3),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ], p=0.6),
                # 针对眼底的CoarseDropout（模拟病变区域遮挡）
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15),
                    fill_value=0,
                    p=0.5
                ),
                # 添加CoarseDropout作为Cutout的替代（albumentations没有Cutout）
                A.CoarseDropout(
                    num_holes_range=(2, 4),
                    hole_height_range=(0.1, 0.2),
                    hole_width_range=(0.1, 0.2),
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
        else:
            # 标准数据增强策略
            print("Dataset: {}, Using standard data augmentation.".format(self.dataset_name))
            transforms = [
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    scale=(0.85, 1.15),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.ElasticTransform(alpha=50, sigma=5, p=0.1),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.3),
                    A.CLAHE(p=0.3),
                ], p=0.5),
                A.CoarseDropout(
                    p=0.3
                ),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ]
            
        return A.Compose(transforms)

    def setup(self, stage: str) -> None:
        method_name = f"setup_{self.dataset_name}_dataset"
        setup_method = getattr(self, method_name, None)
        assert setup_method is not None, (
            f"Dataset {self.dataset_name} not supported, please choose from {self.dataset_support_list}."
        )

        trainset, valset, testset = setup_method()
        if stage == "fit" or stage is None:
            self.trainset = trainset
            self.valset = valset
        elif stage == "test" or stage == "predict":
            self.testset = testset

        return super().setup(stage)

    def setup_DDR_dataset(self):
        disease_annotation_file = "data/annotation_DDR_disease.csv"
        lesion_annotation_file = "data/annotation_DDR_lesion.csv"
        root_dir = "data/DDR/fundus_384"  # Modify this to your own path

        disease_df = pd.read_csv(disease_annotation_file)
        lesion_df = pd.read_csv(lesion_annotation_file)
        
        # 统一列名为image_id（如果需要）
        if 'ID' in disease_df.columns and 'image_id' not in disease_df.columns:
            disease_df = disease_df.rename(columns={'ID': 'image_id'})
        if 'ID' in lesion_df.columns and 'image_id' not in lesion_df.columns:
            lesion_df = lesion_df.rename(columns={'ID': 'image_id'})
        
        # split data
        train_disease_annotation, val_disease_annotation, test_disease_annotation = (
            kfold_split(self.kfold, self.fold_num, disease_df)
            if self.kfold > 1
            else handout_split(self.val_size, self.test_size, disease_df)
        )
        train_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(train_disease_annotation["image_id"])
        ]
        val_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(val_disease_annotation["image_id"])
        ]
        test_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(test_disease_annotation["image_id"])
        ]

        # create dataset
        trainset = FundusDataset(
            disease_df=train_disease_annotation,
            lesion_df=train_lesion_annotation,
            root_dir=root_dir,
            transforms=self.train_transforms,
            lesion_names=self.lesion_names,
        )
        valset = FundusDataset(
            disease_df=val_disease_annotation,
            lesion_df=val_lesion_annotation,
            root_dir=root_dir,
            transforms=self.eval_transforms,
            lesion_names=self.lesion_names,
        )
        testset = FundusDataset(
            disease_df=test_disease_annotation,
            lesion_df=test_lesion_annotation,
            root_dir=root_dir,
            transforms=self.eval_transforms,
            lesion_names=self.lesion_names,
        )
        return trainset, valset, testset

    def setup_FGADR_dataset(self):
        disease_annotation_file = "data/annotation_FGADR_disease.csv"
        lesion_annotation_file = "data/annotation_FGADR_lesion.csv"

        root_dir = "data/FGADR/fundus"  # Modify this to your own path

        disease_df = pd.read_csv(disease_annotation_file)
        lesion_df = pd.read_csv(lesion_annotation_file)
        # 统一列名为image_id（如果需要）
        if 'ID' in disease_df.columns and 'image_id' not in disease_df.columns:
            disease_df = disease_df.rename(columns={'ID': 'image_id'})
        if 'ID' in lesion_df.columns and 'image_id' not in lesion_df.columns:
            lesion_df = lesion_df.rename(columns={'ID': 'image_id'})
        # split data
        train_disease_annotation, val_disease_annotation, test_disease_annotation = (
            kfold_split(self.kfold, self.fold_num, disease_df)
            if self.kfold > 1
            else handout_split(self.val_size, self.test_size, disease_df)
        )
        train_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(train_disease_annotation["image_id"])
        ]
        val_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(val_disease_annotation["image_id"])
        ]
        test_lesion_annotation = lesion_df[
            lesion_df["image_id"].isin(test_disease_annotation["image_id"])
        ]

        # create dataset
        trainset = FundusDataset(
            disease_df=train_disease_annotation,
            lesion_df=train_lesion_annotation,
            root_dir=root_dir,
            transforms=self.train_transforms,
            lesion_names=self.lesion_names,
            picture_extension=".png",
        )
        valset = FundusDataset(
            disease_df=val_disease_annotation,
            lesion_df=val_lesion_annotation,
            root_dir=root_dir,
            transforms=self.eval_transforms,
            lesion_names=self.lesion_names,
            picture_extension=".png",
        )
        testset = FundusDataset(
            disease_df=test_disease_annotation,
            lesion_df=test_lesion_annotation,
            root_dir=root_dir,
            transforms=self.eval_transforms,
            lesion_names=self.lesion_names,
            picture_extension=".png",
        )
        return trainset, valset, testset

    def setup_FGADDR_dataset(self):
        from torch.utils.data import ConcatDataset

        DDR_train, DDR_val, DDR_test = self.setup_DDR_dataset()
        FGADR_train, FGADR_val, FGADR_test = self.setup_FGADR_dataset()
        trainset = ConcatDataset([DDR_train, FGADR_train])
        valset = ConcatDataset([DDR_val, FGADR_val])
        testset = ConcatDataset([DDR_test, FGADR_test])
        return trainset, valset, testset

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
        )

    @classmethod
    def dataset_support_list(cls):
        return [
            re.match("setup_(.*)_dataset", name).group(1)  # type: ignore
            for name in dir(cls)
            if re.match("setup_(.*)_dataset", name)
        ]


class FundusDataset(Dataset):
    def __init__(
        self,
        disease_df: DataFrame,
        lesion_df: DataFrame,
        root_dir: str,
        transforms: A.Compose,
        lesion_names: list[str],
        picture_extension: str = ".jpg",
    ):
        self.disease_df = disease_df
        self.lesion_df = lesion_df
        self.root_dir = root_dir
        self.transforms = transforms
        self.lesion_names = lesion_names
        self.picture_extension = picture_extension

    def __len__(self):
        return len(self.disease_df)

    def __getitem__(self, index):
        row = self.disease_df.iloc[index]
        image_id = row["image_id"]
        img_path = f"{self.root_dir}/{image_id}{self.picture_extension}"
        image = np.array(Image.open(img_path).convert("RGB"))

        disease_onehot = row[self.disease_df.columns[1:]].values.astype(np.float32)
        disease_label = np.argmax(disease_onehot)
        lesion_row = self.lesion_df[self.lesion_df["image_id"] == image_id]
        lesion_labels = lesion_row[self.lesion_names].values[0].astype(np.float32)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return DataItem(
            image=image,
            disease_lbls=torch.tensor(disease_label, dtype=torch.long),
            lesion_lbls=torch.from_numpy(lesion_labels),
            id=image_id,
            img_path=img_path,
        )

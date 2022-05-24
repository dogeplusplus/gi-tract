import cv2
import random
import numpy as np
import pandas as pd
import typing as t
import albumentations as A
import monai.transforms as transforms

from pathlib import Path
from itertools import chain
from multiprocessing import Pool
from einops import rearrange, repeat
from dataclasses import dataclass
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import StratifiedGroupKFold


class GITract3D(Dataset):
    def __init__(self, image_stacks: t.List[Path], label_stacks: t.List[Path], transforms=None):
        self.image_stacks = image_stacks
        self.label_stacks = label_stacks
        self.transforms = transforms

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        image_stack = np.load(self.image_stacks[idx])
        label_stack = np.load(self.label_stacks[idx])

        image_stack = np.asarray(image_stack, dtype=np.float32)
        image_stack /= image_stack.max()

        return image_stack, label_stack


class GITract(Dataset):
    def __init__(self, images: t.List[Path], labels: t.List[Path], transforms=None):
        assert len(images) == len(
            labels), f"Images and Labels unequal length, Images: {len(images)}, Labels: {len(labels)}"
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.transforms = transforms

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        img = np.load(img_path)
        label = np.load(label_path)

        img = np.asarray(img, dtype=np.float32)
        img /= img.max()
        img = repeat(img, "h w -> c h w", c=3)
        label = np.asarray(label, dtype=np.float32)
        label = rearrange(label, "h w c -> c h w")

        if self.transforms:
            data = self.transforms({"image": img, "label": label})
            img = data["image"]
            label = data["label"]

        return img, label


@dataclass
class DataPaths:
    images: t.List[Path]
    labels: t.List[Path]


def split_cases(input_dir: Path, val_ratio: float) -> t.Tuple[DataPaths, DataPaths]:
    image_dir = input_dir / "images"

    cases = np.array([x.name for x in image_dir.iterdir()])
    val_len = int(val_ratio * len(cases))
    train_len = len(cases) - val_len

    train_idx, val_idx = random_split(np.arange(len(cases)), [train_len, val_len])
    train_cases = cases[train_idx]
    val_cases = cases[val_idx]

    train_images = list(chain.from_iterable((image_dir / case).rglob("**/*.npy") for case in train_cases))
    val_images = list(chain.from_iterable((image_dir / case).rglob("**/*.npy") for case in val_cases))

    train_labels = [Path(str(p).replace("images", "labels")) for p in train_images]
    val_labels = [Path(str(p).replace("images", "labels")) for p in val_images]

    return DataPaths(train_images, train_labels), DataPaths(val_images, val_labels)


def split_images(input_dir: Path, val_ratio: float) -> t.Tuple[DataPaths, DataPaths]:
    image_dir = input_dir / "images"

    all_images = list(image_dir.rglob("**/*.npy"))
    random.shuffle(all_images)
    val_len = int(val_ratio * len(all_images))

    train_images, val_images = all_images[val_len:], all_images[:val_len]
    train_labels = [Path(str(p).replace("images", "labels")) for p in train_images]
    val_labels = [Path(str(p).replace("images", "labels")) for p in val_images]

    return DataPaths(train_images, train_labels), DataPaths(val_images, val_labels)


def augmentations(image_size: t.Tuple[int, int]) -> A.Compose:
    transforms = A.Compose([
        A.Resize(*image_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.25),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size[0] // 20,
            max_width=image_size[1] // 20,
            fill_value=0,
            mask_fill_value=0,
            p=0.5,
        ),
    ], p=1.0)

    return transforms


def monai_augmentations(image_size: t.Tuple[int, int]) -> transforms.Compose:
    augmentation = transforms.Compose([
        transforms.Resized(keys=["image", "label"], spatial_size=image_size),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandRotated(keys=["image", "label"], prob=0.2, range_x=[0.3, 0.3]),
        transforms.RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.2, 0.3), prob=0.2),
        transforms.OneOf([
            transforms.RandGridDistortiond(keys=["image", "label"], num_cells=5, prob=0.2),
            transforms.Rand2DElasticd(
                keys=["image", "label"],
                spacing=(20, 20),
                magnitude_range=(1, 2),
                prob=0.2,
            ),
        ], weights=[0.5, 0.5]),
        transforms.RandCoarseDropoutd(
            keys=["image", "label"],
            holes=8,
            spatial_size=(image_size[0] // 20, image_size[1] // 20),
            fill_value=0,
            prob=0.3,
        ),
        transforms.RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 4.5)),
    ])

    return augmentation


def is_empty(path: Path) -> bool:
    mask = np.load(path)
    return np.max(mask) == 0


def kfold_split(label_dir: Path, folds: int = 5, seed: int = 42) -> t.Tuple[t.List[DataPaths], t.List[DataPaths]]:
    labels = list(label_dir.rglob("*.npy"))
    case_names = [p.parent.name for p in labels]

    with Pool() as pool:
        empty = pool.map(is_empty, labels)

    df = pd.DataFrame({
        "labels": labels,
        "cases": case_names,
        "empty": empty,
    })

    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = skf.split(df, df["empty"], groups=df["cases"])

    train_datasets = []
    val_datasets = []
    for val_indices, train_indices in splits:
        train_fold = df.iloc[train_indices]
        train_label_paths = train_fold["labels"].tolist()
        train_image_paths = [Path(str(p).replace("labels", "images")) for p in train_label_paths]
        train_datasets.append(DataPaths(train_image_paths, train_label_paths))

        val_fold = df.iloc[val_indices]
        val_label_paths = val_fold["labels"].tolist()
        val_image_paths = [Path(str(p).replace("labels", "images")) for p in val_label_paths]
        val_datasets.append(DataPaths(val_image_paths, val_label_paths))

    return train_datasets, val_datasets

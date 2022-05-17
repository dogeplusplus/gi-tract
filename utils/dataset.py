import random
import numpy as np
import typing as t

from pathlib import Path
from itertools import chain
from einops import rearrange, repeat
from dataclasses import dataclass
from torch.utils.data import Dataset, random_split


class GITract(Dataset):
    def __init__(self, images: t.List[Path], labels: t.List[Path], transforms=None):
        assert len(images) == len(
            labels), f"Images and Labels unequal length, Images: {len(images)}, Labels: {len(labels)}"
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        img = np.load(img_path)
        label = np.load(label_path)

        img = np.asarray(img, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)

        if self.transforms:
            data = self.transforms(image=rearrange(img, "h w -> h w 1"), mask=label)
            img = data["image"]
            label = data["mask"]

        img = repeat(img, "h w 1 -> h w c", c=3)
        img /= img.max()
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

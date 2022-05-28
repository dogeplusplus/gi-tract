import tqdm
import torch
import mlflow
import numpy as np
import typing as t
import pandas as pd
import torchmetrics
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupShuffleSplit
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    RandBiasFieldd,
    RandCoarseDropoutd,
    RandFlipd,
    RandGridDistortiond,
    Resized,
    RandRotated,
    OneOf,
    Rand3DElasticd,
    RandAdjustContrastd,
)

from utils.losses import criterion
from utils.dataset import GITract3D
from utils.metrics import iou_coef, dice_coef


def augmentation_3d(image_size: t.List[int]) -> Compose:
    augmentation = Compose(
        [
            Resized(keys=["image", "label"], spatial_size=image_size, mode="nearest"),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandRotated(keys=["image", "label"], prob=0.2, range_x=[0.3, 0.3]),
            RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.2, 0.3), prob=0.2),
            OneOf(
                [
                    RandGridDistortiond(keys=["image", "label"], num_cells=5, prob=0.2),
                    Rand3DElasticd(
                        keys=["image", "label"],
                        magnitude_range=(1, 2),
                        sigma_range=(5, 7),
                        prob=0.2,
                    ),
                ],
                weights=[0.5, 0.5],
            ),
            RandCoarseDropoutd(
                keys=["image", "label"],
                holes=8,
                spatial_size=(image_size[0] // 20, image_size[1] // 20, image_size[2] // 20),
                fill_value=0,
                prob=0.3,
            ),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 4.5)),
        ]
    )

    return augmentation


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    display_every: int,
    loss_fn: t.Callable,
    optimizer: Optimizer,
    accumulator: float,
    lr_schedule: object = None,
) -> t.Dict[str, float]:
    metrics = {
        "loss": torchmetrics.MeanMetric(),
        "dice": torchmetrics.MeanMetric(),
        "iou": torchmetrics.MeanMetric(),
        "gpu_mem": torchmetrics.MeanMetric(),
    }
    for metric in metrics.values():
        metric.to(device)
    train_bar = tqdm.tqdm(data_loader, total=len(data_loader), desc=f"Train Epoch: {epoch}")

    scaler = GradScaler()
    for i, (x, y) in enumerate(train_bar):
        x = x.to(device)
        y = y.to(device)
        with autocast():
            pred = model(x)
            loss = loss_fn(pred, y)
            loss = loss / accumulator

        scaler.scale(loss).backward()

        if (i+1) % accumulator == 0:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            if lr_schedule is not None:
                lr_schedule.step()

        loss = loss.detach()

        pred = nn.Sigmoid()(pred)
        dice = dice_coef(y, pred).detach()
        iou = iou_coef(y, pred).detach()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

        metrics["loss"].update(loss)
        metrics["dice"].update(dice)
        metrics["iou"].update(iou)
        metrics["gpu_mem"].update(mem)

        if i % display_every == 0:
            display_metrics = {k: f"{float(v.compute().cpu().numpy()):.4f}" for k, v in metrics.items()}
            train_bar.set_postfix(**display_metrics)

    metrics = {
        k: float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }

    return metrics


@torch.no_grad()
def valid_epoch(model, data_loader, device, epoch, display_every, loss_fn):
    metrics = {
        "loss": torchmetrics.MeanMetric(),
        "dice": torchmetrics.MeanMetric(),
        "iou": torchmetrics.MeanMetric(),
        "gpu_mem": torchmetrics.MeanMetric(),
    }
    for metric in metrics.values():
        metric.to(device)

    val_bar = tqdm.tqdm(data_loader, total=len(data_loader), desc=f"Valid Epoch: {epoch}")
    for i, (x, y) in enumerate(val_bar):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        pred = nn.Sigmoid()(pred)
        dice = dice_coef(y, pred).detach()
        iou = iou_coef(y, pred).detach()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

        metrics["loss"].update(loss)
        metrics["dice"].update(dice)
        metrics["iou"].update(iou)
        metrics["gpu_mem"].update(mem)

        if i % display_every == 0:
            display_metrics = {k: f"{float(v.compute().cpu().numpy()):.4f}" for k, v in metrics.items()}
            val_bar.set_postfix(**display_metrics)

    metrics = {
        k: float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }

    return metrics


def main():
    # Note some stacks only have 80 z-slices but most seem to be 144
    image_size = (144, 224, 224)
    transforms = augmentation_3d(image_size)
    dataset_dir = Path("datasets", "3d")

    images = list((dataset_dir / "images").iterdir())
    labels = list((dataset_dir / "labels").iterdir())
    df = pd.DataFrame({"images": images, "labels": labels})

    skf = GroupShuffleSplit(1, test_size=0.2)

    df["case"] = df["images"].apply(lambda x: x.stem.split("_")[0])
    # Dummy column to allow for grouped splitting
    df["dummy"] = 1
    splits = list(skf.split(df["images"], df["dummy"], df["case"]))[0]
    train_idx, val_idx = splits
    train_sub = df.iloc[train_idx]
    val_sub = df.iloc[val_idx]

    train_set = train_sub[["images", "labels"]].to_dict("list")
    val_set = val_sub[["images", "labels"]].to_dict("list")
    train_ds = GITract3D(train_set["images"], train_set["labels"], transforms)
    val_ds = GITract3D(val_set["images"], val_set["labels"], transforms)

    batch_size = 4
    num_workers = 4
    prefetch_factor = 4
    epochs = 100
    show_every = 10
    accumulator = max(1, 32 // batch_size)
    loss_fn = criterion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        channels=(4, 8, 16),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    net.to(device)

    optimizer = AdamW(net.parameters())

    train_ds = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        prefetch_factor=prefetch_factor,
    )

    val_ds = DataLoader(
        val_ds,
        2*batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        prefetch_factor=prefetch_factor,
    )

    best_loss = np.inf
    mlflow.set_experiment("3d_unet")

    for epoch in range(epochs):
        train_metrics = train_epoch(net, train_ds, device, epoch, show_every, loss_fn, optimizer, accumulator)
        val_metrics = valid_epoch(net, val_ds, device, epoch, show_every, loss_fn)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
        mlflow.log_metrics({f"valid_{k}": v for k, v in val_metrics.items()}, step=epoch)

        val_loss = val_metrics["loss"]
        if val_loss < best_loss:
            best_loss = val_loss
            mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    main()

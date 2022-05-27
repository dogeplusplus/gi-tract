import torch
import typing as t
import numpy as np
import torchmetrics
import torch.nn as nn
import segmentation_models_pytorch as smp

from ray import tune
from pathlib import Path
from einops import rearrange
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from monai.metrics import compute_meandice
from torch.cuda.amp import autocast, GradScaler

from utils.dataset import GITract, split_cases, monai_augmentations


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    loss_fn: t.Callable,
    optimizer: Optimizer,
) -> t.Dict[str, float]:
    metrics = {
        "loss": torchmetrics.MeanMetric(),
        "dice": torchmetrics.MeanMetric(),
        "gpu_mem": torchmetrics.MeanMetric(),
    }
    for metric in metrics.values():
        metric.to(device)

    scaler = GradScaler()
    for (x, y) in data_loader:
        optimizer.zero_grad()
        x = rearrange(x, "b h w c -> b c h w")
        y = rearrange(y, "b h w c -> b c h w")
        x = x.to(device)
        y = y.to(device)
        with autocast():
            pred = model(x)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss = loss.detach()

        pred = nn.Sigmoid()(pred)
        dice = torch.nan_to_num(compute_meandice(pred, y)).mean().detach()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

        metrics["loss"].update(loss)
        metrics["dice"].update(dice)
        metrics["gpu_mem"].update(mem)

    metrics_numpy = {
        k: float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }

    return metrics_numpy


@torch.no_grad()
def valid_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    loss_fn: t.Callable
) -> t.Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {
        "loss": torchmetrics.MeanMetric(),
        "dice": torchmetrics.MeanMetric(),
        "gpu_mem": torchmetrics.MeanMetric(),
    }
    for metric in metrics.values():
        metric.to(device)

    for (x, y) in data_loader:
        x = rearrange(x, "b h w c -> b c h w")
        y = rearrange(y, "b h w c -> b c h w")
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        pred = nn.Sigmoid()(pred)
        dice = torch.nan_to_num(compute_meandice(pred, y)).mean()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

        metrics["loss"].update(loss)
        metrics["dice"].update(dice)
        metrics["gpu_mem"].update(mem)

    metrics_numpy = {
        k: float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }

    return metrics_numpy


def train(config):
    num_workers = 2
    prefetch_factor = 8
    image_size = (224, 224)

    model = smp.Unet(
        encoder_name=config["encoder"]["name"],
        encoder_weights=config["encoder"]["weights"],
        in_channels=config["in_dim"],
        classes=config["out_dim"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transforms = monai_augmentations(image_size)

    val_ratio = 0.2
    train_set, val_set = split_cases(config["dataset_dir"], val_ratio)

    train_ds = GITract(train_set.images, train_set.labels, transforms)
    val_ds = GITract(val_set.images, val_set.labels, transforms)

    train_ds = DataLoader(
        train_ds,
        config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        prefetch_factor=prefetch_factor,
    )
    val_ds = DataLoader(
        val_ds,
        config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        prefetch_factor=prefetch_factor,
    )

    weight_decay = 1e-6
    optimizer = AdamW(model.parameters(), config["learning_rate"], weight_decay=weight_decay)

    loss_fn = config["loss_fn"]

    # Remove these tags as they are automatically populated
    best_val_loss = np.inf
    best_train_loss = np.inf

    for e in range(config["epochs"]):
        train_metrics = train_epoch(model, train_ds, device, loss_fn, optimizer)
        val_metrics = valid_epoch(model, val_ds, device, loss_fn)

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        if train_loss < best_train_loss:
            best_train_loss = train_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        tune.report(
            iterations=e,
            train_loss=train_loss,
            val_loss=val_loss,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
        )


def criterion(y_pred, y_true):
    bce = smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true)
    tve = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)(y_pred, y_true)
    return (bce + tve) / 2


def main():
    # For some reason need absolute paths to use with ray tune
    dataset_dir = Path("dataset").resolve()

    configs = {
        "epochs": 3,
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
        "batch_size": 4,
        "loss_fn": tune.choice([
            smp.losses.SoftBCEWithLogitsLoss(),
            smp.losses.DiceLoss(mode="multilabel"),
            smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
            criterion,
        ]),
        "encoder": tune.choice([
            {"name": "resnext50_32x4d", "weights": "imagenet"},
            {"name": "efficientnet-b1", "weights": "imagenet"},
            {"name": "densenet121", "weights": "imagenet"},
        ]),
        "in_dim": 3,
        "out_dim": 3,
        "dataset_dir": dataset_dir,
    }

    resources_per_trial = {
        "cpu": 12,
        "gpu": 0.5,
    }

    tune.run(
        train,
        config=configs,
        num_samples=10,
        resources_per_trial=resources_per_trial,
    )


if __name__ == "__main__":
    main()

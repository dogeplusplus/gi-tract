import tqdm
import torch
import mlflow
import logging
import typing as t
import numpy as np
import torchmetrics
import torch.nn as nn
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, lr_scheduler, Optimizer

from utils.losses import criterion
from utils.metrics import iou_coef, dice_coef
from utils.dataset import GITract, split_cases, augmentation_2d, kfold_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    dataset_dir = Path("datasets", "2d")

    val_ratio = 0.2
    batch_size = 128
    num_workers = 4
    prefetch_factor = 8
    image_size = (224, 224)
    accumulator = max(1, 32 // batch_size)
    epochs = 30
    lr = 2e-3
    weight_decay = 1e-6

    in_dim = 3
    out_dim = 3
    t_max = int(30000 / batch_size * epochs) + 50

    train_set, val_set = split_cases(dataset_dir, val_ratio)
    transforms = augmentation_2d(image_size)
    train_ds = GITract(train_set.images, train_set.labels, transforms)
    val_ds = GITract(val_set.images, val_set.labels, transforms)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=in_dim,
        classes=out_dim,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr, weight_decay=weight_decay)
    lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

    losses = dict(
        bce_loss=smp.losses.SoftBCEWithLogitsLoss(),
        dice_loss=smp.losses.DiceLoss(mode="multilabel"),
        tve_loss=smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
        mixed_loss=criterion
    )

    loss_name = "mixed_loss"
    loss_fn = losses[loss_name]

    mlflow.log_param("loss_type", loss_name)

    case_split = {
        "train": list(train_set.cases),
        "valid": list(val_set.cases),
    }
    mlflow.log_dict(case_split, "cases.json")

    display_every = 50
    best_loss = np.inf

    for e in range(epochs):
        train_metrics = train_epoch(
            model,
            train_ds,
            device,
            e,
            display_every,
            loss_fn,
            optimizer,
            accumulator,
            lr_schedule
        )
        val_metrics = valid_epoch(model, val_ds, device, e, display_every, loss_fn)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=e)
        mlflow.log_metrics({f"valid_{k}": v for k, v in val_metrics.items()}, step=e)

        val_loss = val_metrics["loss"]
        if val_loss < best_loss:
            best_loss = val_loss
            mlflow.pytorch.log_model(model, "model")


def train_ensemble():
    dataset_dir = Path("datasets", "2d")

    batch_size = 128
    num_workers = 4
    prefetch_factor = 8
    image_size = (224, 224)
    accumulator = max(1, 32 // batch_size)
    epochs = 15
    lr = 2e-3
    weight_decay = 1e-6

    in_dim = 3
    out_dim = 3
    t_max = int(30000 / batch_size * epochs) + 50

    train_folds, val_folds = kfold_split(dataset_dir / "labels")
    transforms = augmentation_2d(image_size)

    train_datasets = [GITract(fold.images, fold.labels, transforms) for fold in train_folds]
    val_datasets = [GITract(fold.images, fold.labels, transforms) for fold in val_folds]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_experiment("ensemble")
    for i, (train_ds, val_ds) in enumerate(zip(train_datasets, val_datasets)):
        with mlflow.start_run(run_name=f"fold_{i}"):
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

            model = smp.Unet(
                encoder_name="efficientnet-b1",
                encoder_weights="imagenet",
                in_channels=in_dim,
                classes=out_dim,
            )
            model.to(device)

            optimizer = AdamW(model.parameters(), lr, weight_decay=weight_decay)
            lr_schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)

            losses = dict(
                bce_loss=smp.losses.SoftBCEWithLogitsLoss(),
                dice_loss=smp.losses.DiceLoss(mode="multilabel"),
                tve_loss=smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
                mixed_loss=criterion,
            )

            loss_name = "mixed_loss"
            loss_fn = losses[loss_name]

            mlflow.log_param("loss_type", loss_name)
            display_every = 50

            best_loss = np.inf

            for e in range(epochs):
                train_metrics = train_epoch(
                    model,
                    train_ds,
                    device,
                    e,
                    display_every,
                    loss_fn,
                    optimizer,
                    accumulator,
                    lr_schedule
                )
                val_metrics = valid_epoch(model, val_ds, device, e, display_every, loss_fn)

                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=e)
                mlflow.log_metrics({f"valid_{k}": v for k, v in val_metrics.items()}, step=e)

                val_loss = val_metrics["loss"]
                if val_loss < best_loss:
                    logger.info("Validation improved. Logging model...")
                    best_loss = val_loss
                    mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()

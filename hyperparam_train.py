import cv2
import torch
import mlflow
import numpy as np
import torchmetrics
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp

from ray import tune
from pathlib import Path
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from monai.metrics import compute_meandice
from torch.cuda.amp import autocast, GradScaler
from ray.tune.integration.mlflow import MLflowLoggerCallback

from utils.dataset import GITract, split_train_test_cases


def train_epoch(model, data_loader, device, epoch, loss_fn, optimizer):
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

    mlflow.log_metrics({
        f"train_{k}": float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }, step=epoch)


@torch.no_grad()
def valid_epoch(model, data_loader, device, epoch, loss_fn):
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

    mlflow.log_metrics({
        f"val_{k}": float(v.compute().cpu().numpy()) for k,
        v in metrics.items()
    }, step=epoch)

    return metrics["loss"].compute().cpu().numpy()


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

    transforms = A.Compose([
        A.Normalize((0.5), (0.5)),
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
        ),
    ], p=1.0)

    train_ds = GITract(config["train_set"].images, config["train_set"].labels, transforms)
    val_ds = GITract(config["val_set"].images, config["val_set"].labels, transforms)

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

    weight_decay = 1e-2
    optimizer = AdamW(model.parameters(), config["learning_rate"], weight_decay=weight_decay)

    loss_fn = config["loss_fn"]
    mlflow.log_param("loss_type", str(loss_fn))

    # Remove these tags as they are automatically populated
    best_loss = np.inf

    for e in range(config["epochs"]):
        train_epoch(model, train_ds, device, e, loss_fn, optimizer)
        val_loss = valid_epoch(model, val_ds, device, e, loss_fn)
        if val_loss < best_loss:
            best_loss = val_loss
            mlflow.pytorch.log_model(model, "model")


def main():
    # For some reason need absolute paths to use with ray tune
    dataset_dir = Path("dataset").resolve()
    val_ratio = 0.2
    train_set, val_set = split_train_test_cases(dataset_dir, val_ratio)

    configs = {
        "epochs": 15,
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5]),
        "batch_size": 4,
        "loss_fn": tune.choice([
            smp.losses.SoftBCEWithLogitsLoss(),
            smp.losses.DiceLoss(mode="multilabel"),
            smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
        ]),
        "encoder": tune.choice([
            {"name": "resnext50_32x4d", "weights": "imagenet"},
            {"name": "efficientnet-b1", "weights": "imagenet"},
            {"name": "densenet121", "weights": "imagenet"},
        ]),
        "in_dim": 1,
        "out_dim": 3,
        "train_set": train_set,
        "val_set": val_set,
    }

    callbacks = [MLflowLoggerCallback(experiment_name="ray_tune", save_artifact=True)]
    resources_per_trial = {
        "cpu": 12,
        "gpu": 0.5,
    }

    tune.run(
        train,
        config=configs,
        callbacks=callbacks,
        resources_per_trial=resources_per_trial,
    )


if __name__ == "__main__":
    main()

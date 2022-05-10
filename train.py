import tqdm
import torch
import mlflow
import numpy as np
import torchmetrics

from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_meandice

from models.unet import UNet
from utils.dataset import GITract, collate_fn, split_train_test_cases


def main():
    dataset_dir = Path("dataset")

    val_ratio = 0.2
    batch_size = 128
    num_workers = 4

    train_set, val_set = split_train_test_cases(dataset_dir, val_ratio)

    train_ds = GITract(train_set.images, train_set.labels)
    val_ds = GITract(val_set.images, val_set.labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    val_ds = DataLoader(
        val_ds,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    filters = [16, 32, 64, 64]
    in_dim = 1
    out_dim = 3
    kernel_size = (3, 3)
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-2

    model = UNet(filters, in_dim, out_dim, kernel_size)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler()
    loss_fn = DiceLoss()

    desc = "Train Epoch: {}"
    val_desc = "Valid Epoch: {}"
    display_every = 50

    best_loss = -np.inf

    for e in range(epochs):
        train_metrics = {
            "loss": torchmetrics.MeanMetric(),
            "dice": torchmetrics.MeanMetric(),
            "hausdorff": torchmetrics.MeanMetric(),
        }
        for metric in train_metrics.values():
            metric.to(device)
        train_bar = tqdm.tqdm(train_ds, total=len(train_ds), desc=desc.format(e))

        for i, (x, y) in enumerate(train_bar):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            with autocast():
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.detach()
            hausdorff = torch.nan_to_num(compute_hausdorff_distance(pred, y)).mean().detach()
            dice = torch.nan_to_num(compute_meandice(pred, y)).mean().detach()
            train_metrics["loss"].update(loss)
            train_metrics["hausdorff"].update(hausdorff)
            train_metrics["dice"].update(dice)

            if i % display_every == 0:
                display_metrics = {k: f"{float(v.compute().cpu().numpy()):.4f}" for k, v in train_metrics.items()}
                train_bar.set_postfix(**display_metrics)

        mlflow.log_metrics({
            f"train_{k}": float(v.compute().cpu().numpy()) for k,
            v in train_metrics.items()
        }, step=e)

        val_metrics = {
            "loss": torchmetrics.MeanMetric(),
            "dice": torchmetrics.MeanMetric(),
            "hausdorff": torchmetrics.MeanMetric(),
        }
        for metric in val_metrics.values():
            metric.to(device)

        val_bar = tqdm.tqdm(val_ds, total=len(val_ds), desc=val_desc.format(e))

        with torch.no_grad():
            for i, (x, y) in enumerate(val_bar):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)
                hausdorff = torch.nan_to_num(compute_hausdorff_distance(pred, y)).mean()
                dice = torch.nan_to_num(compute_meandice(pred, y)).mean()

                val_metrics["loss"].update(loss)
                val_metrics["hausdorff"].update(hausdorff)
                val_metrics["dice"].update(dice)

                if i % display_every == 0:
                    display_metrics = {k: f"{float(v.compute().cpu().numpy()):.4f}" for k, v in val_metrics.items()}
                    val_bar.set_postfix(**display_metrics)

        mlflow.log_metrics({
            f"val_{k}": float(v.compute().cpu().numpy()) for k,
            v in val_metrics.items()
        }, step=e)

        val_loss = val_metrics["loss"].compute().cpu().numpy()
        if val_loss < best_loss:
            best_loss = val_loss
            mlflow.pytorch.log_model(model, "stomech")


if __name__ == "__main__":
    main()

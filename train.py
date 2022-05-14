import cv2
import tqdm
import torch
import mlflow
import numpy as np
import torchmetrics
import albumentations as A
import segmentation_models_pytorch as smp

from pathlib import Path
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from monai.metrics import compute_meandice
from torch.cuda.amp import autocast, GradScaler

# from models.unet import UNet
from utils.dataset import GITract, split_train_test_cases


def main():
    dataset_dir = Path("dataset")

    val_ratio = 0.2
    batch_size = 64
    num_workers = 0
    image_size = (320, 320)

    train_set, val_set = split_train_test_cases(dataset_dir, val_ratio)

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
        ),
    ], p=1.0)

    train_ds = GITract(train_set.images, train_set.labels, transforms)
    val_ds = GITract(val_set.images, val_set.labels, transforms)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )
    val_ds = DataLoader(
        val_ds,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    epochs = 100
    lr = 1e-3
    weight_decay = 1e-2

    in_dim = 1
    out_dim = 4

    # filters = [16, 32, 64, 64]
    # kernel_size = (3, 3)
    # activation = nn.LeakyReLU
    # final_activation = nn.Sigmoid()

    # model = UNet(filters, in_dim, out_dim, kernel_size, activation, final_activation)
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=in_dim,
        classes=out_dim,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr, weight_decay=weight_decay)
    scaler = GradScaler()

    losses = dict(
        bce_loss=smp.losses.SoftBCEWithLogitsLoss(),
        dice_loss=smp.losses.DiceLoss(mode="multilabel"),
        tve_loss=smp.losses.TverskyLoss(mode="multilabel", log_loss=False)
    )

    loss_name = "bce_loss"
    loss_fn = losses[loss_name]

    mlflow.log_param("loss_type", loss_name)
    desc = "Train Epoch: {}"
    val_desc = "Valid Epoch: {}"
    display_every = 50

    best_loss = np.inf

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
            dice = torch.nan_to_num(compute_meandice(pred, y)).mean().detach()
            train_metrics["loss"].update(loss)
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
                dice = torch.nan_to_num(compute_meandice(pred, y)).mean()

                val_metrics["loss"].update(loss)
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
            mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()

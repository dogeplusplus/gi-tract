import torch
import logging
import numpy as np
import torch.nn as nn
import monai.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from utils.dataset import GITract
from argparse import ArgumentParser
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader

from utils.metrics import dice_coef, iou_coef, hausdorff_dist


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser("Evaluate model performance")
    parser.add_argument("--version", type=str, help="Model version number.")
    parser.add_argument("--dataset", type=str, help="Path to dataset.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [
        torch.load(model_path, map_location=device)
        for model_path in Path("model_instances", str(args.version)).iterdir()
    ]
    for model in models:
        model.eval()

    images_dir = Path(args.dataset) / "images"
    images = list(images_dir.rglob("*.npy"))
    labels = [Path(str(p).replace("images", "labels")) for p in images]

    preprocessing = transforms.Compose([
        transforms.Resized(keys=["image", "label"], spatial_size=(224, 224), mode="nearest"),
    ])
    dataset = GITract(images, labels, preprocessing)
    batch_size = 32

    loader = DataLoader(dataset, batch_size)

    thr = 0.5

    mean_dice = MeanMetric().to(device)
    miou = MeanMetric().to(device)
    hausdorff = MeanMetric().to(device)
    max_dist = np.sqrt(224 ** 2 + 224 ** 2)

    for xs, ys in tqdm(loader, total=len(loader)):
        xs = xs.to(device)
        ys = ys.to(device)

        combined = torch.zeros_like(ys)
        for model in models:
            pred = model(xs)
            pred = nn.Sigmoid()(pred)
            combined += pred

        combined = combined / len(models)
        prediction = combined > thr

        dice = dice_coef(ys, prediction)
        iou = iou_coef(ys, prediction)
        # Hausdorff computation seems to run on CPU
        distance = hausdorff_dist(prediction, ys, max_dist)

        mean_dice.update(dice)
        miou.update(iou)
        hausdorff.update(distance)

    mean_dice = mean_dice.compute().cpu().numpy()
    mean_iou = miou.compute().cpu().numpy()
    mean_hausdorff = hausdorff.compute().cpu().numpy()

    logger.info(f"Mean Dice: {mean_dice}")
    logger.info(f"Mean IoU: {mean_iou}")
    logger.info(f"Hausdorff Distance: {mean_hausdorff}")
    logger.info(f"Estimated Kaggle Score 0.4 * dice + 0.6 * hausdorff: {0.4 * mean_dice + 0.6 * mean_hausdorff}")


if __name__ == "__main__":
    main()

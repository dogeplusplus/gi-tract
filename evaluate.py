import torch
import logging
import torch.nn as nn
import monai.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from utils.dataset import GITract
from argparse import ArgumentParser
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader

from train import dice_coef, iou_coef


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
        transforms.Resized(keys=["image", "label"], spatial_size=(224, 224)),
    ])
    dataset = GITract(images, labels, preprocessing)
    batch_size = 32

    loader = DataLoader(dataset, batch_size)

    thr = 0.5

    mean_dice = MeanMetric().to(device)
    miou = MeanMetric().to(device)

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

        mean_dice.update(dice)
        miou.update(iou)

    logger.info(f"Mean Dice: {mean_dice.compute().cpu().numpy()}")
    logger.info(f"Mean IoU: {miou.compute().cpu().numpy()}")


if __name__ == "__main__":
    main()

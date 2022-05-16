import cv2
import sys
import torch
import random
import mlflow
import numpy as np
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from einops import rearrange

sys.path.append(".")


def predict(model: nn.Module, image: torch.Tensor, device: str) -> torch.Tensor:
    shape = image.shape
    preprocessing = A.Compose([
        A.Normalize((0.5), (0.5)),
        A.Resize(320, 320, interpolation=cv2.INTER_NEAREST),
    ])

    image = rearrange(image, "h w -> h w 1")
    data = preprocessing(image=image)
    image = data["image"]
    image = rearrange(image, "h w 1 -> 1 1 h w")
    image = torch.from_numpy(image).to(device)
    pred = model(image)
    pred = nn.Sigmoid()(pred[0])
    pred = torch.argmax(pred, dim=0)
    pred = pred.cpu().detach().numpy()

    pred_resized = A.Resize(*shape, interpolation=cv2.INTER_NEAREST)(image=pred)["image"]

    return pred_resized


def display_predictions(model: nn.Module, num_images: int, images_path: Path, device: str):
    images = list(images_path.rglob("*.npy"))
    image_paths = random.sample(images, num_images)
    mask_paths = [Path(str(path).replace("images", "labels")) for path in image_paths]

    images = [np.load(p).astype(np.float32) for p in image_paths]
    masks = [np.argmax(np.load(p), axis=-1) for p in mask_paths]

    cols = 3
    fig, ax = plt.subplots(num_images, cols)
    predictions = [predict(model, image, device) for image in images]

    ax[0, 0].set_title("Image")
    ax[0, 1].set_title("Ground Truth")
    ax[0, 2].set_title("Prediction")

    for i in range(num_images):
        ax[i, 0].imshow(images[i])
        ax[i, 1].imshow(masks[i])
        ax[i, 2].imshow(predictions[i])

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
        ax[i, 2].set_axis_off()

    plt.tight_layout()
    plt.show()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--run", "-r", help="Run ID", type=str)
    parser.add_argument("--images", "-i", help="Number of images to display", type=int, default=5)

    args = parser.parse_args()
    return args


def main(args):
    images_path = Path("dataset", "images")
    logged_model = f"runs:/{args.run}/model"
    model = mlflow.pytorch.load_model(logged_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    display_predictions(model, args.images, images_path, device)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

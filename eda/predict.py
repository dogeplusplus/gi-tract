import sys
import torch
import random
import mlflow
import numpy as np
import torch.nn as nn
import monai.transforms as transforms
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from einops import rearrange, repeat

sys.path.append(".")


def predict_single_image(model: nn.Module, image: torch.Tensor, device: str) -> torch.Tensor:
    shape = image.shape
    preprocessing = transforms.Compose([
        transforms.Resize(spatial_size=(224, 224), mode="nearest"),
    ])

    image = repeat(image, "h w -> 3 h w")
    image = preprocessing(image)
    image = rearrange(image, "c h w -> 1 c h w")

    image = torch.from_numpy(image).to(device)
    pred = model(image)
    pred = nn.Sigmoid()(pred[0])
    pred = (pred > 0.5).to(torch.float32)
    pred = pred.cpu().detach().numpy()
    # Take the first non-background class
    pred = transforms.Resize(spatial_size=shape, mode="nearest")(pred)
    pred = np.argmax(pred, axis=0)

    return pred


def display_predictions(model: nn.Module, num_images: int, images_path: Path, device: str):
    images = list(images_path.rglob("*.npy"))
    image_paths = random.sample(images, num_images)
    mask_paths = [Path(str(path).replace("images", "labels")) for path in image_paths]

    images = [np.load(p).astype(np.float32) for p in image_paths]
    images = [img / img.max() for img in images]
    masks = [np.argmax(np.load(p), axis=-1) for p in mask_paths]

    cols = 3
    fig, ax = plt.subplots(num_images, cols)
    predictions = [predict_single_image(model, image, device) for image in images]

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
    images_path = Path("datasets", "2d", "images")
    logged_model = f"runs:/{args.run}/model"
    model = mlflow.pytorch.load_model(logged_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    display_predictions(model, args.images, images_path, device)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

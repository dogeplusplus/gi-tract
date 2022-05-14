import sys
import torch
import random
import mlflow
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from einops import rearrange
from torchvision import transforms

sys.path.append(".")


def predict(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    shape = image.shape
    preprocessing = torch.nn.Sequential(
        transforms.Resize((320, 320)),
        transforms.Normalize((0.456), (0.225))
    )

    image = rearrange(image, "h w -> 1 1 h w")
    image = preprocessing(image)
    pred = model(image)
    pred = torch.argmax(pred, dim=1)

    pred_resized = transforms.Resize(shape)(pred)
    return pred_resized


def display_predictions(model: nn.Module, num_images: int, images_path: Path, device: str):
    images = list(images_path.rglob("*.npy"))
    image_paths = random.sample(images, num_images)
    mask_paths = [Path(str(path).replace("images", "labels")) for path in image_paths]

    images = [np.load(p).astype(np.float32) for p in image_paths]
    masks = [np.argmax(np.load(p), axis=-1) for p in mask_paths]

    cols = 3
    fig, ax = plt.subplots(num_images, cols)
    predictions = [predict(model, torch.from_numpy(image).to(device)) for image in images]
    predictions = [p.cpu().detach().numpy()[0] for p in predictions]

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


if __name__ == "__main__":
    num_images = 5
    images_path = Path("dataset", "images")

    logged_model = 'runs:/54db63cc351242399e8fc208d55e0ed7/model'
    model = mlflow.pytorch.load_model(logged_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    display_predictions(model, num_images, images_path, device)

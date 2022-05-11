import cv2
import random
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots


def create_color_mask(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    classes = {
        0: (255, 0, 0, 128),
        1: (0, 255, 0, 128),
        2: (0, 0, 255, 128),
    }

    image = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(overlay)

    for cls, color in classes.items():
        mask = label[..., cls] * 255
        mask = Image.fromarray(mask, mode="L")
        drawing.bitmap((0, 0), mask, color)

    combined_image = Image.alpha_composite(image, overlay)
    return np.array(combined_image)


def visualise_random_images(images_path: Path, num_images: int):
    all_images = list(images_path.rglob("*.png"))

    image_paths = random.sample(all_images, num_images)
    label_paths = [Path(str(img).replace("images", "labels")) for img in image_paths]
    color_masks = [
        create_color_mask(
            cv2.imread(str(ip), cv2.IMREAD_UNCHANGED),
            cv2.imread(str(lp), cv2.IMREAD_UNCHANGED),
        ) for (ip, lp) in zip(image_paths, label_paths)
    ]
    cols = 4
    rows = num_images // cols
    fig = make_subplots(rows, cols)
    fig.update_layout(title_text="Image Examples")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    for i in range(rows):
        for j in range(cols):
            fig.add_trace(go.Image(z=color_masks[i * cols + j]), row=i+1, col=j+1)

    fig.show()

import random
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots


def image_histogram_equalization(image: np.ndarray) -> np.ndarray:
    max_u16 = 65536
    hist, _ = np.histogram(image.flatten(), max_u16, [0, max_u16])
    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * max_u16 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint16)

    image = cdf[image]
    image = (image / 256).astype(np.uint8)
    return image


def create_color_mask(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    classes = {
        1: (255, 0, 0, 128),
        2: (0, 255, 0, 128),
        3: (0, 0, 255, 128),
    }

    image = image_histogram_equalization(image)
    image = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(overlay)
    for cls, color in classes.items():
        mask = np.array(label[..., cls] * 255, np.uint8)
        mask = Image.fromarray(mask, mode="L")
        drawing.bitmap((0, 0), mask, color)

    combined_image = Image.alpha_composite(image, overlay)
    return np.array(combined_image)


def visualise_random_images(images_path: Path, num_images: int):
    all_images = list(images_path.rglob("*.npy"))

    image_paths = random.sample(all_images, num_images)
    label_paths = [Path(str(img).replace("images", "labels")) for img in image_paths]
    color_masks = [
        create_color_mask(
            np.load(ip),
            np.load(lp),
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


def main():
    images_path = Path("dataset", "images")
    num_images = 16
    visualise_random_images(images_path, num_images)


if __name__ == "__main__":
    main()

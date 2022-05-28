import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from functools import reduce
from collections import Counter
from multiprocessing import Pool


def image_class_distribution():
    df = pd.read_csv(Path("datasets", "train.csv"))
    df["values"] = np.where(df["segmentation"].isna(), 0, 1)
    summary = df.pivot(index=["id"], columns=["class"], values=["values"])
    num_images = len(summary)
    class_counts = summary.sum(axis=0) / num_images

    return class_counts


def count_pixels(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    flat = image.flatten()
    return Counter(flat)


def pixel_distributions(dataset_dir: Path):
    image_paths = list(dataset_dir.rglob("**/*.png"))
    with Pool(os.cpu_count()) as pool:
        results = pool.map(count_pixels, image_paths)

    pixel_counts = reduce(lambda x, y: x + y, results, Counter())
    freq = [pixel_counts[i] for i in range(max(pixel_counts) + 1)]

    # Trick to rebin the histogram using pre-computed bins
    plt.hist(np.arange(0, max(pixel_counts)+1), weights=freq)
    plt.show()


def display_image_mask(image_path: Path, mask_path: Path):
    fig, ax = plt.subplots(1, 2)

    image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)

    plt.title(image_path)
    ax[0].imshow(image, cmap="gray")
    ax[1].matshow(mask, cmap="gray")

    plt.show()


if __name__ == "__main__":
    image_list = list(Path("dataset/images").rglob("**/*.png"))
    image_path = random.choice(image_list)
    mask_path = Path(str(image_path).replace("images", "labels"))
    display_image_mask(image_path, mask_path)

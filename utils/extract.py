import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import repeat
from multiprocessing import Pool


def parse_segmentation(
    rle_segmentation: str,
    image_size: np.ndarray,
) -> np.ndarray:
    flat_mask = np.zeros(np.product(image_size), dtype=np.int8)
    runs = [int(x) for x in rle_segmentation.split()]

    for idx in range(0, len(runs), 2):
        start = runs[idx]
        length = runs[idx+1]
        flat_mask[start:start+length] = 1

    mask = np.reshape(flat_mask, image_size)

    return mask


def generate_mask(segments: pd.DataFrame, image_size: np.ndarray) -> np.ndarray:
    segments = segments.sort_values(by="class")
    masks = []

    for _, seg in segments.iterrows():
        segment_mask = parse_segmentation(seg["segmentation"], image_size)
        masks.append(segment_mask)

    onehot_mask = np.stack(masks, axis=-1)
    return onehot_mask


def process_case(case_dir: Path, df: pd.DataFrame, image_dir: Path, label_dir: Path):
    case_image_dir = image_dir / case_dir.name
    case_label_dir = label_dir / case_dir.name

    case_image_dir.mkdir(exist_ok=True, parents=True)
    case_label_dir.mkdir(exist_ok=True, parents=True)

    for case_day in case_dir.iterdir():
        case = case_day.name

        for scan in (case_day / "scans").iterdir():
            parts = scan.stem.split("_")
            _, slice_num, height, width, pixel_height, pixel_width = parts
            height = int(height)
            width = int(width)
            pixel_height = float(pixel_height)
            pixel_width = float(pixel_width)
            slice_id = f"{case}_slice_{slice_num}"
            rows = df.loc[df["id"] == slice_id]
            mask = generate_mask(rows, (width, height))
            image_path = case_image_dir / f"{slice_id}.npy"
            label_path = case_label_dir / f"{slice_id}.npy"

            image = cv2.imread(str(scan), cv2.IMREAD_UNCHANGED)
            np.save(str(image_path), image)
            np.save(str(label_path), mask)


def preprocess_dataset(df: pd.DataFrame, input_dir: Path, dataset_dir: Path):
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = list(input_dir.iterdir())

    with Pool() as pool:
        pool.starmap(process_case, zip(case_dirs, repeat(df), repeat(image_dir), repeat(label_dir)))


def main():
    input_dir = Path("datasets", "raw_dataset")
    dataset_dir = Path("datasets", "2d")
    df = pd.read_csv(Path("datasets", "train.csv"))
    df["segmentation"].fillna(value="", inplace=True)

    preprocess_dataset(df, input_dir, dataset_dir)


if __name__ == "__main__":
    main()

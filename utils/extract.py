import cv2
import shutil
import numpy as np
import pandas as pd

from pathlib import Path


CLASS_MAPPING = {
    "background": 0,
    "large_bowel": 1,
    "small_bowel": 2,
    "stomach": 3,
}


def parse_segmentation(
    rle_segmentation: str,
    label: str,
    image_size: np.ndarray,
) -> np.ndarray:
    flat_mask = np.zeros(np.product(image_size), dtype=np.int8)
    runs = [int(x) for x in rle_segmentation.split()]

    for idx in range(0, len(runs), 2):
        start = runs[idx]
        length = runs[idx+1]
        flat_mask[start:start+length] = CLASS_MAPPING[label]

    mask = np.reshape(flat_mask, image_size)
    return mask


def running_length(arr: np.ndarray) -> str:
    runs = []
    length = 0
    start = 0

    for i in range(len(arr)):
        if arr[i] == 0 and length > 0:
            runs += [str(start), str(length)]
            length = 0
            start = -1
        elif arr[i] == 1:
            if length == 0:
                start = i
            length += 1

    if length > 0:
        runs += [str(start), str(length)]

    return " ".join(runs)


def generate_mask(segments: pd.DataFrame, image_size: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.int8)

    for _, seg in segments.iterrows():
        if pd.isna(seg["segmentation"]):
            continue
        segment_mask = parse_segmentation(seg["segmentation"], seg["class"], image_size)
        # Avoid cases where the RLE mask overlap
        mask = np.maximum(mask, segment_mask)

    return mask


def preprocess_dataset(df: pd.DataFrame, input_dir: Path, dataset_dir: Path):
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in input_dir.iterdir():
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
                image_path = case_image_dir / f"{slice_id}.png"
                label_path = case_label_dir / f"{slice_id}.png"
                shutil.copy(scan, image_path)
                cv2.imwrite(str(label_path), mask)

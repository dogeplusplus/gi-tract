from utils.extract import generate_mask
import cv2
import sys
import einops
import numpy as np
import typing as t
import pandas as pd

from pathlib import Path
from itertools import repeat
from multiprocessing import Pool

sys.path.append(".")


def parse_filepath(path: Path) -> t.Dict[str, t.Any]:
    case_day = path.parent.parent.name
    case, day = case_day.split("_")
    case = int(case.replace("case", ""))
    day = int(day.replace("day", ""))

    _, number, height, width, pix_height, pix_width = path.stem.split("_")
    number = int(number)
    height = int(height)
    width = int(width)
    pix_height = float(pix_height)
    pix_width = float(pix_width)

    row = {
        "case": case,
        "day": day,
        "slice": number,
        "width": width,
        "height": height,
        "pix_height": pix_height,
        "pix_width": pix_width,
        "path": path,
    }

    return row


def stack_images(df: pd.DataFrame, case_day: str, images_dir: Path, labels_dir: Path):
    df.sort_values("slice", inplace=True)
    unique_paths = df["path"].unique()
    images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in unique_paths]
    ids = df["id"].unique()

    shape = images[0].shape
    image_stack = np.stack(images)
    image_stack = einops.repeat(image_stack, "d h w -> c d h w", c=3)
    masks = [
        generate_mask(df[df["id"] == idx], shape) for idx in ids
    ]
    mask_stack = np.stack(masks)
    mask_stack = einops.rearrange(mask_stack, "d h w c -> c d h w")

    np.save(images_dir / f"{case_day}.npy", image_stack)
    np.save(labels_dir / f"{case_day}.npy", mask_stack)


def preprocess_dataset_3d(df: pd.DataFrame, input_dir: Path, dataset_dir: Path):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    df["case_day"] = df.apply(lambda x: str(x.case) + "_" + str(x.day), axis=1)

    all_images = list(input_dir.rglob("*.png"))

    with Pool() as pool:
        rows = pool.map(parse_filepath, all_images)

    image_df = pd.DataFrame(rows)
    df = pd.merge(df, image_df, on=["case", "day", "slice"])

    case_days = df["case_day"].unique()

    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    dataset_dir.mkdir()
    image_dir.mkdir()
    label_dir.mkdir()

    groupings = df.groupby("case_day")
    sub_dfs = [groupings.get_group(x) for x in groupings.groups]

    with Pool() as pool:
        pool.starmap(stack_images, zip(sub_dfs, case_days, repeat(image_dir), repeat(label_dir)))


def main():
    input_dir = Path("datasets", "raw_dataset")
    dataset_dir = Path("datasets", "3d")
    df = pd.read_csv(Path("datasets", "train.csv"))
    df["segmentation"].fillna(value="", inplace=True)

    preprocess_dataset_3d(df, input_dir, dataset_dir)


if __name__ == "__main__":
    main()

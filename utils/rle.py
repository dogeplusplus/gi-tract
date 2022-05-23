import numpy as np
import pandas as pd


def running_length(arr: np.ndarray) -> str:
    msk = np.array(arr)
    pixels = msk.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def convert_to_rle(mask: np.ndarray, id: str) -> pd.DataFrame:
    rows = []
    channel_index = {
        "large_bowel": 0,
        "small_bowel": 1,
        "stomach": 2,
    }

    for name, index in channel_index.items():
        submask = mask[..., index]
        # ordered from top to bottom, left to right
        flat_mask = submask.flatten(order="F")
        predicted = running_length(flat_mask)
        entry = {
            "id": id,
            "class": name,
            "predicted": predicted or None,
        }
        rows.append(entry)

    df = pd.DataFrame(rows)
    return df

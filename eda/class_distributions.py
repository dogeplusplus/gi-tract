import pandas as pd
import numpy as np


def image_class_distribution():
    df = pd.read_csv("train.csv")
    df["values"] = np.where(df["segmentation"].isna(), 0, 1)
    summary = df.pivot(index=["id"], columns=["class"], values=["values"])
    num_images = len(summary)
    class_counts = summary.sum(axis=0) / num_images

    return class_counts


if __name__ == "__main__":
    counts = image_class_distribution()
    print(counts)

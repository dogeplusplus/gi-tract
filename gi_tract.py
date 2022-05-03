from pathlib import Path
from torch.utils.data import DataLoader

from utils.dataset import GITract, collate_fn, split_train_test_cases


def main():
    dataset_dir = Path("tract_gi_dataset")

    val_ratio = 0.2
    batch_size = 16
    train_set, val_set = split_train_test_cases(dataset_dir, val_ratio)

    train_ds = GITract(train_set.images, train_set.labels)
    val_ds = GITract(val_set.images, val_set.labels)

    train_ds = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_ds = DataLoader(val_ds, batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == "__main__":
    main()

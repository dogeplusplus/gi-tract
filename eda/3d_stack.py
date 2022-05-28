import numpy as np


from pathlib import Path


def case_shapes(images_dir: Path):
    for case_dir in images_dir.iterdir():
        case = case_dir.name
        images = list(case_dir.iterdir())
        num_slices = len(images)
        img = np.load(images[0])
        combined_shape = (num_slices,) + img.shape
        print(f"Case {case}: Shape {combined_shape}")


def main():
    image_dir = Path("dataset", "images")
    case_shapes(image_dir)


if __name__ == "__main__":
    main()

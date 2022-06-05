import random
import argparse
import numpy as np
import nibabel as nib

from pathlib import Path
from einops import rearrange


def convert_image_nifti(path: Path, destination: Path, mask: bool = False):
    img = np.load(path)
    # Transpose to make orientation the same in itk-snap
    img = np.flip(img, axis=-1)
    img = np.flip(img, axis=-2)

    if mask:
        zeros = np.where(np.max(img, axis=0) == 0, 1, 0)[np.newaxis, ...]
        img = np.concatenate([zeros, img])
        img = np.argmax(img, axis=0)
        img = img.astype(np.uint8)
    else:
        img = img[0]

    affine = np.eye(4, 4)

    img = rearrange(img, "z y x -> x y z")
    nifti = nib.Nifti1Image(img, affine)

    nib.save(nifti, destination)


def parse_arguments():
    images_3d = list(Path("datasets", "3d", "images").rglob("*.npy"))
    random_image = random.choice(images_3d)

    parser = argparse.ArgumentParser("Converter for 3D images to nifti")
    parser.add_argument("--image", type=str, help="Path to 3D image", default=random_image)
    parser.add_argument("--output", type=str, help="Path to output file", default="mri")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    img_path = Path(args.image)
    mask_path = Path(str(args.image).replace("images", "labels"))

    # Pixel dimensions for xyzt just for visualisation
    img_dest = Path(args.output + "_image.nii.gz")
    mask_dest = Path(args.output + "_mask.nii.gz")

    convert_image_nifti(img_path, img_dest)
    convert_image_nifti(mask_path, mask_dest, mask=True)


if __name__ == "__main__":
    main()

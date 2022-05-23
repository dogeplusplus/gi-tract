import shutil
import logging
import subprocess

from pathlib import Path
from argparse import ArgumentParser


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def commit_model(model_run: str, mlruns_folder: Path, dataset_dir: Path) -> int:
    model_path = mlruns_folder / model_run / "artifacts/model/data/model.pth"
    last_version = [int(x.stem[1:]) for x in dataset_dir.rglob("*.pth")]
    version = max(last_version) + 1
    shutil.copyfile(model_path, dataset_dir / f"v{version}.pth")
    command = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        "model_instances",
        "-m",
        '"New model"',
        "-r",
        "zip",
    ]
    logger.info("Starting to upload new dataset version...")
    proc = subprocess.run(command)
    logger.info("Finished attempt to upload")
    return proc.returncode


def parse_arguments():
    parser = ArgumentParser("Model dataset management utility.")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to model dataset directory",
        default="model_instances",
    )
    parser.add_argument("--run", type=str, help="Id of the model run")

    args = parser.parse_args()
    return args


def main(args):
    mlruns_folder = Path("mlruns/0")
    return_code = commit_model(args.run, Path(mlruns_folder), Path(args.directory))
    if return_code == 0:
        logger.info("Successfully created new dataset on kaggle.")
    else:
        logger.error("Ran into some error when uploading the dataset. Check the kaggle API.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

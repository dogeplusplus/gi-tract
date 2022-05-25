import shutil
import logging
import subprocess
import typing as t

from pathlib import Path
from argparse import ArgumentParser


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def commit_model(model_runs: t.List[str], mlruns_folder: Path, dataset_dir: Path) -> int:
    last_version = [int(x.stem) for x in dataset_dir.rglob("*/**")]
    version = max(last_version) + 1

    model_dir = dataset_dir / str(version)
    model_dir.mkdir()

    for i, run in enumerate(model_runs):
        model_path = mlruns_folder / run / "artifacts/model/data/model.pth"
        shutil.copyfile(model_path, model_dir / f"model_{i}.pth")

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
    parser.add_argument("--runs", type=str, nargs="+", help="Id of the model run")
    parser.add_argument("--experiment", type=str, help="Experiment ID", default=0)

    args = parser.parse_args()
    return args


def main(args):
    mlruns_folder = Path("mlruns") / str(args.experiment)
    return_code = commit_model(args.runs, Path(mlruns_folder), Path(args.directory))
    if return_code == 0:
        logger.info("Successfully created new dataset on kaggle.")
    else:
        logger.error("Ran into some error when uploading the dataset. Check the kaggle API.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

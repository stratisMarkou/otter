import argparse
import logging
import os
import shutil
import sys

import xarray as xr

sys.path.append(".")
from .utils import (  # noqa: E402
    ALL_GRID_VARIABLE_LEVELS,
    ALL_GRID_VARIABLE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_ERA5_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h"

DATASETS = {
    "era5_64x32": (f"{BASE_ERA5_PATH}-64x32_equiangular_conservative.zarr"),
    "era5_240x121": (
        f"{BASE_ERA5_PATH}-240x121_equiangular_with_poles_conservative.zarr"
    ),
}

NUM_TIMES_TEST = 12


def download_zarr_and_store(
    dataset: str,
    output_dir: str,
    overwrite_existing: bool,
    test: bool,
) -> None:
    path = DATASETS[dataset]
    output_path = f"{output_dir}/{dataset}.zarr"
    path_exists = os.path.exists(output_path)

    if path_exists and not overwrite_existing:
        logging.info(
            "Path %s exists and overwrite not specified. Skipping...",
            output_path,
        )
        return

    elif path_exists:
        logging.info("Path %s exists. Deleting old path...", output_path)
        shutil.rmtree(output_path)

    logging.info("Downloading %s from %s to %s...", dataset, path, output_path)

    # Open remote zarr and save it locally.
    zarr = xr.open_zarr(path).sel(level=ALL_GRID_VARIABLE_LEVELS)[
        ALL_GRID_VARIABLE_NAMES
    ]
    for var in zarr:
        del zarr[var].encoding["chunks"]
    zarr = zarr.chunk(dict(time=1))

    # If testing, only download a subset of the data.
    if test:
        zarr = zarr.isel(time=slice(0, NUM_TIMES_TEST, 1))

    zarr.to_zarr(f"{output_dir}/{dataset}.zarr", mode="w")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[])
    parser.add_argument("--output_dir", type=str, default="_data")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:
        download_zarr_and_store(
            dataset,
            args.output_dir,
            args.overwrite_existing,
            args.test,
        )


if __name__ == "__main__":
    main()

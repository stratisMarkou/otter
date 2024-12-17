import argparse
import logging
import shutil
import os

import xarray as xr

from utils import ALL_GRID_VARIABLE_NAMES, ALL_GRID_VARIABLE_LEVELS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

DATASETS = {
    "era5_64x32": (
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
    ),
    "era5_240x121": (
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    )
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', default=[])
    parser.add_argument("--output_dir", type=str, default="_data")
    parser.add_argument("--overwrite_existing", action="store_true")
    args = parser.parse_args()

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:

        path = DATASETS[dataset]
        output_path = f"{args.output_dir}/{dataset}.zarr"
        path_exists = os.path.exists(output_path)

        if path_exists and not args.overwrite_existing:
            logging.info("Path %s exists and overwrite not specified. Skipping...", output_path)
            continue
        
        elif path_exists:
            logging.info("Path %s exists. Deleting old path...", output_path)
            shutil.rmtree(output_path)

        logging.info("Downloading %s from %s to %s...", dataset, path, output_path)

        # Open remote zarr and save it locally.
        zarr = xr.open_zarr(path).sel(level=ALL_GRID_VARIABLE_LEVELS)[ALL_GRID_VARIABLE_NAMES]
        zarr.to_zarr(f"{args.output_dir}/{dataset}.zarr", mode="w")

if __name__ == "__main__":
    main()

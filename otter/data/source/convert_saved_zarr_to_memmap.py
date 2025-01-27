import argparse
import logging
import os

import xarray as xr

from otter.data.source.utils import DATASETS
from otter.data.utils import save_xarray_dataset_as_memmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[])
    parser.add_argument("--output_dir", type=str, default="_data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:
        zarr_path = os.path.join(args.output_dir, f"{dataset}.zarr")
        save_path = os.path.join(args.output_dir, f"{dataset}_single_array")

        save_xarray_dataset_as_memmap(
            dataset=xr.open_zarr(zarr_path).sel(
                time=slice("1979-01-01 00:00:00", None)
            ),
            path=save_path,
            iteration_dimension="time",
            sorted_dims=[
                "time",
                "latitude",
                "longitude",
                "variable_and_level",
            ],
        )


if __name__ == "__main__":
    main()

import argparse
import logging
import os

import xarray as xr

from otter.data.source.utils import DATASETS
from otter.data.utils import (
    save_xarray_data_array_as_memmap,
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)


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

        # Split the dataset into different variables along the
        # variable_and_level dimension.
        split_dataset = split_into_different_variables_along_dim(
            xr.open_zarr(zarr_path),
            dim="level",
        )
        stacked_array = stack_dataset_variable_and_levels(split_dataset)

        save_xarray_data_array_as_memmap(
            data=stacked_array,
            path=save_path,
            iteration_dimension="time",
            sorted_dims=["time", "lat", "lon", "variable_and_level"],
        )


if __name__ == "__main__":
    main()

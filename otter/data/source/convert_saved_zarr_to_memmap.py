import argparse
import logging
import os

import xarray as xr

from otter.data.source.utils import DATASETS
from otter.data.utils import (
    join_xarray_dataset_across_levels_and_save_as_single_memmap,
    save_xarray_dataset_in_memmap_format,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[])
    parser.add_argument("--zarr_dir", type=str, default="_data")
    parser.add_argument("--output_dir", type=str, default="_data")
    parser.add_argument("--single_memmap", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:
        zarr_path = os.path.join(args.zarr_dir, f"{dataset}.zarr")
        xarray_dataset = xr.open_zarr(zarr_path)

        save_path = os.path.join(args.output_dir, dataset)

        if args.single_memmap:
            join_xarray_dataset_across_levels_and_save_as_single_memmap(
                dataset=xarray_dataset,
                path=f"{save_path}_single_array",
                iteration_dimension="time",
            )

        else:
            save_xarray_dataset_in_memmap_format(
                dataset=xarray_dataset,
                path=save_path,
                iteration_dimension="time",
            )


if __name__ == "__main__":
    main()

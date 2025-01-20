import argparse
import logging
import os

import xarray as xr

from otter.data.source.utils import DATASETS
from otter.data.utils import save_xarray_dataset_in_memmap_format


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
        xarray_dataset = xr.open_zarr(zarr_path)

        save_path = os.path.join(args.output_dir, dataset)
        save_xarray_dataset_in_memmap_format(
            dataset=xarray_dataset,
            path=save_path,
            iteration_dimension="time",
        )


if __name__ == "__main__":
    main()

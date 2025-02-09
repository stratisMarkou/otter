"""Checks that determinstic config initializes as expected."""

import os
from tempfile import TemporaryDirectory

import pytest
import xarray as xr

from otter.data.source.utils import download_zarr_and_store
from otter.data.utils import save_xarray_dataset_as_memmap
from otter.experiment.utils import initialize_experiment

TEST_DATASET = "era5_240x121"


@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
def test_deterministic_config() -> None:

    with TemporaryDirectory() as temp_dir:
        download_zarr_and_store(
            dataset=TEST_DATASET,
            output_dir=temp_dir,
            overwrite_existing=False,
            test=True,
        )
        zarr_path = os.path.join(temp_dir, f"{TEST_DATASET}.zarr")
        dataset = xr.open_zarr(zarr_path)

        save_xarray_dataset_as_memmap(
            dataset=dataset,
            path=f"{temp_dir}/{TEST_DATASET}_single_array",
            iteration_dimension="time",
            sorted_dims=[
                "time",
                "latitude",
                "longitude",
                "variable_and_level",
            ],
        )

        path = os.path.join(temp_dir, f"{TEST_DATASET}_single_array")
        initialize_experiment(
            args_list=[
                "--debug",
                "--config",
                "otter/experiment/config/deterministic.yaml",
                f"constants.data.path_to_labelled_array={path}",
            ],
        )

import os
from tempfile import TemporaryDirectory

import pytest
import xarray as xr

from otter.data.datasets import make_gridded_weather_task
from otter.data.source.utils import (
    download_zarr_and_store,
)
from otter.data.utils import save_xarray_dataset_as_memmap


TEST_DATASET = "era5_240x121"


# Tests that th length of the dataset is correct
@pytest.mark.parametrize(
    (
        "start_date",
        "end_date",
        "temporal_resolution_hours",
        "num_context_frames",
        "num_target_frames",
        "task_sub_sampling_factor",
    ),
    [
        ("2020-01-01 00:00:00", "2020-01-05 18:00:00", 6, 1, 2, 1),
        ("2020-04-01 00:00:00", "2020-04-06 18:00:00", 6, 2, 3, 2),
        ("2020-06-30 00:00:00", "2020-07-03 17:00:00", 6, 2, 4, 1),
        ("2020-01-01 00:00:00", "2020-01-03 18:00:00", 12, 1, 3, 2),
        ("2020-01-01 00:00:00", "2020-01-10 21:00:00", 12, 1, 2, 5),
        ("2020-07-01 00:00:00", "2020-09-03 15:00:00", 24, 1, 3, 10),
    ],
)
@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
def test_len_dataset(
    start_date: str,
    end_date: str,
    num_context_frames: int,
    num_target_frames: int,
    temporal_resolution_hours: int,
    task_sub_sampling_factor: int,
) -> None:
    # First download the dataset
    with TemporaryDirectory() as temp_dir:
        download_zarr_and_store(
            dataset=TEST_DATASET,
            output_dir=temp_dir,
            overwrite_existing=False,
            test=False,
            start_date=start_date,
            end_date=end_date,
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

        # Construct dataset
        dataset = make_gridded_weather_task(
            path_to_labelled_array=path,
            start_date=start_date,
            end_date=end_date,
            num_context_frames=num_context_frames,
            num_target_frames=num_target_frames,
            temporal_resolution_hours=temporal_resolution_hours,
            ctx_variables_to_exclude=[],
            trg_variables_to_exclude=[],
            task_sub_sampling_factor=task_sub_sampling_factor,
        )

    # Assert that this gives a target time after end date
    with pytest.raises(AssertionError):
        dataset.__getitem__(len(dataset))

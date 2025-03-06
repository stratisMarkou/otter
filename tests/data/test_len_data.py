import os

import pytest

from otter.data.datasets import make_gridded_weather_task


# Tests that th length of the dataset is correct
@pytest.mark.parametrize(
    (
        "start_date",
        "end_date",
        "temporal_resolution_hours",
        "num_context_frames",
        "num_target_frames",
        "correct_len",
    ),
    [
        ("2020-01-01 00:00:00", "2020-01-05 18:00:00", 6, 1, 2, 18),
        ("2020-04-01 00:00:00", "2020-04-06 18:00:00", 6, 2, 3, 20),
        ("2020-01-01 00:00:00", "2020-01-03 15:00:00", 6, 1, 3, 8),
        ("2020-06-30 00:00:00", "2020-07-03 17:00:00", 6, 2, 4, 10),
        ("2020-01-01 00:00:00", "2020-01-03 18:00:00", 12, 1, 3, 3),
        ("2020-01-01 00:00:00", "2020-01-04 21:00:00", 12, 1, 2, 6),
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
    correct_len: int,
) -> None:
    # Construct dataset
    dataset = make_gridded_weather_task(
        path_to_labelled_array="/scratches/richardson_2/otter_data/era5_240x121_single_array_float32",
        start_date=start_date,
        end_date=end_date,
        num_context_frames=num_context_frames,
        num_target_frames=num_target_frames,
        temporal_resolution_hours=temporal_resolution_hours,
        ctx_variables_to_exclude=[],
        trg_variables_to_exclude=[],
        task_sub_sampling_factor=1,
    )

    # Assert the length is correct
    assert len(dataset) == correct_len

    # Extract last index
    last_datapoint = dataset.__getitem__(len(dataset) - 1)
    zero_time = last_datapoint.zero_time
    target_timedeltas = dataset.target_timedeltas.astype("timedelta64[h]")
    target_times = zero_time + target_timedeltas
    # Compute the realisable end date
    target_end_date = (
        dataset.end_date
        - (dataset.end_date - dataset.start_date) % dataset.temporal_resolution
    )
    # Assert last date is the realisable end date
    assert target_times[-1] == target_end_date

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zarr

from otter.data.datasets import ForecastingSample
from otter.experiment.save_samples import initialise_zarr, save_forecasts


@pytest.fixture
def experiment_setup(tmp_path):
    experiment_path = tmp_path / "experiment"
    experiment_path.mkdir()

    # Mock GriddedWeatherTask
    ds = MagicMock()
    ds.__len__.return_value = 2
    ds.first_zero_time = np.datetime64("2023-01-01T00:00:00")
    ds.temporal_resolution = np.timedelta64(6, "h")
    ds.task_sub_sampling_factor = 1
    ds.trg_variables_and_levels = ["t2m", "u10"]

    params = {
        "num_lats": 4,
        "num_lons": 8,
        "num_forecast_times": 3,
        "num_trajectories": 2,
        "vars_and_levels": 2,
        "experiment_path": experiment_path,
        "ds": ds,
    }
    return params


def test_initialise_zarr(experiment_setup):
    p = experiment_setup
    root = initialise_zarr(
        p["ds"],
        p["num_lats"],
        p["num_lons"],
        p["num_forecast_times"],
        p["num_trajectories"],
        p["vars_and_levels"],
        p["experiment_path"],
    )

    assert isinstance(root, zarr.Group)
    assert "forecasts" in root
    assert root["forecasts"].shape == (2, 4, 8, 3, 2, 2, 2)
    assert root["zero_time"].shape == (2,)
    assert root["lat"].shape == (4,)
    assert root["lon"].shape == (8,)
    assert root["variable_and_level"].shape == (2,)
    assert root["forecasts"].attrs["_ARRAY_DIMENSIONS"] == [
        "zero_time",
        "lat",
        "lon",
        "forecast_time_idx",
        "trajectory_idx",
        "statistic",
        "variable_and_level",
    ]


def test_save_forecasts(experiment_setup):
    p = experiment_setup
    root = initialise_zarr(
        p["ds"],
        p["num_lats"],
        p["num_lons"],
        p["num_forecast_times"],
        p["num_trajectories"],
        p["vars_and_levels"],
        p["experiment_path"],
    )

    # Mock model
    model = MagicMock()
    mock_dist = MagicMock()
    batch_size = 1
    mock_dist.mean = torch.randn(
        batch_size,
        p["num_lats"],
        p["num_lons"],
        p["num_forecast_times"],
        p["vars_and_levels"],
    )
    mock_dist.stddev = torch.randn(
        batch_size,
        p["num_lats"],
        p["num_lons"],
        p["num_forecast_times"],
        p["vars_and_levels"],
    )
    model.return_value = (mock_dist, None)

    # Mock dataloader
    zero_time = np.datetime64("2023-01-01T00:00:00")
    batch = MagicMock(spec=ForecastingSample)
    batch.zero_time = [zero_time]
    batch.to.return_value = batch
    dataloader = [batch]

    # Mock preprocess_batch
    with patch(
        "otter.experiment.save_samples.preprocess_batch", return_value=batch
    ):
        save_forecasts(
            root,
            model,
            dataloader,
            data_scale=1,
            rollout_scales=torch.tensor([1]),
            num_trajectories=p["num_trajectories"],
        )

    # Check if data was saved
    forecasts = root["forecasts"]
    # Mean is at statistic index 0, Std is at statistic index 1
    np.testing.assert_array_almost_equal(
        forecasts[0, :, :, :, 0, 0, :], mock_dist.mean[0].numpy()
    )
    np.testing.assert_array_almost_equal(
        forecasts[0, :, :, :, 0, 1, :], mock_dist.stddev[0].numpy()
    )

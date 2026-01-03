# %%
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import zarr
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import ForecastingSample, GriddedWeatherTask
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import load_experiment_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.multiscale_model import MultiscaleModel


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Use a trained model to generate forceasting samples."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="The experiment name from which to load the trained model.",
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to collect (default: 10).",
    )
    parser.add_argument(
        "--val-subsampling-factor",
        type=int,
        default=59,
        help="Factor for subsampling validation data (default: 59).",
    )

    args = parser.parse_args()
    return args


def initialise_zarr(
    ds: GriddedWeatherTask,
    num_lats: int,
    num_lons: int,
    num_forecast_times: int,
    num_trajectories: int,
    vars_and_levels: int,
    experiment_path: Path,
) -> zarr.Group:
    out_shape = (
        len(ds),  # Number of zero times
        num_lats,
        num_lons,
        num_forecast_times,  # e.g. [6, 12, 18, 24] hrs after zero time
        num_trajectories,  # Number of trajectories that are sampled
        2,  # Mean and std
        vars_and_levels,
    )
    coord_names = [
        "zero_time",
        "lat",
        "lon",
        "forecast_time_idx",
        "trajectory_idx",
        "statistic",
        "variable_and_level",
    ]
    logging.info(f"Zarr output shape: {out_shape}")
    logging.info(f"Zarr coord names: {coord_names}")

    file_size = np.prod(out_shape) * np.dtype(np.float32).itemsize
    file_size_gb = file_size / (1024**3)
    logging.info(f"Estimated uncompressed file size: {file_size_gb:.2f} GB")

    if file_size_gb > 100:
        raise ValueError("File size is larger than 100 GB.")

    zero_times = [
        ds.first_zero_time
        + i * ds.temporal_resolution * ds.task_sub_sampling_factor
        for i in range(len(ds))
    ]
    lats = np.linspace(-90, 90, num_lats, dtype=np.float32)
    lons = np.linspace(-180, 180, num_lons, dtype=np.float32)
    forecast_time_indices = np.arange(num_forecast_times, dtype=np.int32)
    trajectory_indices = np.arange(num_trajectories, dtype=np.int32)
    statistics = ["mean", "std"]

    forecasts_path = experiment_path / "forecasts.zarr"
    logging.info(f"Allocating zarr at {forecasts_path}")

    root = zarr.open_group(str(forecasts_path), mode="w")

    # Create the main data array
    # We use chunks to allow for efficient writing and reading.
    # Chunking by zero_time (1) and trajectory (1) seems reasonable for the current save loop.
    root.create_array(
        "forecasts",
        shape=out_shape,
        chunks=(
            1,
            num_lats,
            num_lons,
            num_forecast_times,
            1,
            2,
            vars_and_levels,
        ),
        dtype=np.float32,
    )

    # Save coordinates as separate datasets or attributes
    root.create_array("zero_time", data=np.array(zero_times).astype(str))
    root.create_array("lat", data=lats)
    root.create_array("lon", data=lons)
    root.create_array("forecast_time_idx", data=forecast_time_indices)
    root.create_array("trajectory_idx", data=trajectory_indices)
    root.create_array("statistic", data=np.array(statistics).astype(str))
    root.create_array(
        "variable_and_level",
        data=np.array(ds.trg_variables_and_levels).astype(str),
    )

    # Add metadata about dimensions
    root["forecasts"].attrs["_ARRAY_DIMENSIONS"] = coord_names

    return root


def save_forecasts(
    zarr_root: zarr.Group,
    model: MultiscaleModel,
    dataloader: torch.utils.data.DataLoader[ForecastingSample],
    data_scale: int,
    rollout_scales: torch.Tensor,
    num_trajectories: int,
) -> None:
    forecasts_dataset: zarr.Array = zarr_root["forecasts"]  # type: ignore
    zero_times_coord: zarr.Array = zarr_root["zero_time"]  # type: ignore
    zero_times_list = zero_times_coord[:]

    # Create a mapping from zero_time string to index
    zero_time_to_idx = {str(zt): i for i, zt in enumerate(zero_times_list)}  # type: ignore

    for batch_idx, batch in enumerate(dataloader):
        logging.info(f"Batch {batch_idx + 1}/{len(dataloader)}")
        zero_times = batch.zero_time
        batch = batch.to("cuda")
        model_input = preprocess_batch(batch, data_scale)
        with torch.no_grad():
            for traj_idx in tqdm(range(num_trajectories)):
                prd_dist, _ = model(batch=model_input, scales=rollout_scales)
                mean_np: npt.NDArray[Any] = prd_dist.mean.cpu().numpy()
                std_np = prd_dist.stddev.cpu().numpy()

                # Iterate across the batch dimension and save the data one-by-one.
                for i, zero_time in enumerate(zero_times):
                    zt_idx = zero_time_to_idx[str(zero_time)]

                    # forecasts shape: (zero_time, lat, lon, forecast_time, trajectory, statistic, var)
                    # mean_np shape: (batch, lat, lon, forecast_time, var)
                    # We need to insert into (zt_idx, :, :, :, traj_idx, 0, :)
                    forecasts_dataset[zt_idx, :, :, :, traj_idx, 0, :] = (
                        mean_np[i, ...]
                    )
                    forecasts_dataset[zt_idx, :, :, :, traj_idx, 1, :] = (
                        std_np[i, ...]
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    config = load_experiment_config(
        args.experiment_name,
        override_args=[
            f"constants.data.val_task_subsampling_factor={args.val_subsampling_factor}",
            "data_source.path=_data/era5_240x121_single_array_float32",
        ],
    )
    config.model.device = "cuda"

    experiment = instantiate(config)
    experiment_path = Path(experiment.info.experiment_path)

    scale_scheduler: ScaleScheduler = experiment.inference_scale_scheduler
    data_scale = scale_scheduler.sample_data_scale()
    rollout_scales = scale_scheduler.get_rollout_scales()

    all_scales = [data_scale] + list(rollout_scales)
    all_scales = [int(scale) for scale in all_scales]
    logging.info(f"All scales: {all_scales}")

    dataloader = experiment.valid_dataloader
    ds: GriddedWeatherTask = dataloader.dataset

    model: MultiscaleModel = experiment.model.to("cuda")

    checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
    run_state = RunState.load(checkpoint_path)
    model.load_state_dict(run_state.model_state)

    model.eval()

    num_lats, num_lons, num_forecast_times, vars_and_levels = ds[0].trg.shape

    zarr_root = initialise_zarr(
        ds,
        num_lats,
        num_lons,
        num_forecast_times,
        args.num_trajectories,
        vars_and_levels,
        experiment_path,
    )

    save_forecasts(
        zarr_root,
        model,
        dataloader,
        data_scale,
        rollout_scales,
        args.num_trajectories,
    )

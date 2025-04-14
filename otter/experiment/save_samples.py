# %%
import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import ForecastingSample, GriddedWeatherTask
from otter.data.utils import LabelledArray, new_memmap_labelled_array
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
        "experiment_name",
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
    parser.add_argument(
        "--rollout-scales",
        type=int,
        nargs="+",
        default=[8, 4, 2],
        help="List of integer rollout scales (space-separated, default: 8 4 2).",
    )

    args = parser.parse_args()
    return args


def initialise_memmap(
    num_zero_times: int,
    num_lats: int,
    num_lons: int,
    num_forecast_times: int,
    num_trajectories: int,
    vars_and_levels: int,
    experiment_path: Path,
) -> LabelledArray:
    out_shape = (
        num_zero_times,
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
    logging.info(f"Memmap output shape: {out_shape}")
    logging.info(f"Memmap coord names: {coord_names}")

    file_size = np.prod(out_shape) * np.dtype(np.float32).itemsize
    file_size_gb = file_size / (1024**3)
    logging.info(f"File size: {file_size_gb:.2f} GB")

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

    coord_values: list[Sequence[Any] | npt.NDArray[Any]] = [
        zero_times,
        lats,
        lons,
        forecast_time_indices,
        trajectory_indices,
        statistics,
        ds.trg_variables_and_levels,
    ]

    forecasts_path = experiment_path / "forecasts"
    forecasts_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Allocating memmap file at {forecasts_path}")
    return new_memmap_labelled_array(
        forecasts_path,
        out_shape,
        coord_names=coord_names,
        coord_values=coord_values,
        dtype=np.float32,
    )


def save_forecasts(
    la: LabelledArray,
    model: MultiscaleModel,
    dataloader: torch.utils.data.DataLoader[ForecastingSample],
    data_scale: int,
    rollout_scales: torch.Tensor,
    num_trajectories: int,
) -> None:
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

                # Select the latest scale only
                mean_np = mean_np.take(indices=-1, axis=-2)
                std_np = std_np.take(indices=-1, axis=-2)

                # Iterate across the batch dimension and save the data one-by-one.
                for i, zero_time in enumerate(zero_times):
                    la.set_at(
                        mean_np[i, ...],
                        zero_time=zero_time,
                        statistic="mean",
                        trajectory_idx=traj_idx,
                    )
                    la.set_at(
                        std_np[i, ...],
                        zero_time=zero_time,
                        statistic="std",
                        trajectory_idx=traj_idx,
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    config = load_experiment_config(
        args.experiment_name,
        override_args=[
            f"constants.data.val_task_subsampling_factor={args.val_subsampling_factor}",
            f"inference_scale_scheduler.rollout_scales={args.rollout_scales}",
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

    la = initialise_memmap(
        len(ds),
        num_lats,
        num_lons,
        num_forecast_times,
        args.num_trajectories,
        vars_and_levels,
        experiment_path,
    )

    save_forecasts(
        la,
        model,
        dataloader,
        data_scale,
        rollout_scales,
        args.num_trajectories,
    )

# %%
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import GriddedWeatherTask
from otter.data.labelled_array import forecast_to_xarray
from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import load_experiment_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.multiscale_model import MultiscaleModel

# %%
logging.basicConfig(level=logging.INFO)


def load_dependencies(
    experiment_name: str, overrides: list[str] | None = None
):
    override_args = [
        "data_source.path=_data/era5_240x121_single_array_float32",
        "constants.data.batch_size=1",
    ]
    if overrides:
        override_args.extend(overrides)

    config = load_experiment_config(
        experiment_name,
        override_args=override_args,
    )
    config.model.device = "cuda"

    experiment = instantiate(config)
    experiment_path = Path(experiment.info.experiment_path)

    scale_scheduler: ScaleScheduler = experiment.inference_scale_scheduler
    data_scale = scale_scheduler.sample_data_scale()
    rollout_scales = scale_scheduler.get_rollout_scales()

    dataloader = experiment.test_dataloader
    ds: GriddedWeatherTask = dataloader.dataset

    model: MultiscaleModel = experiment.model.to("cuda")

    checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
    run_state = RunState.load(checkpoint_path)
    model.load_state_dict(run_state.model_state)
    model.eval()

    zarr_path = experiment_path / "forecasts.zarr"

    return (
        config,
        model,
        dataloader,
        ds,
        data_scale,
        rollout_scales,
        zarr_path,
    )


def generate_forecasts(
    experiment_name: str,
    model: MultiscaleModel,
    dataloader: torch.utils.data.DataLoader,
    ds: GriddedWeatherTask,
    data_scale: int,
    rollout_scales: list[int],
    zarr_path: Path,
):
    lats = ds.dataset.get_coord_by_name("latitude").values
    lons = ds.dataset.get_coord_by_name("longitude").values

    variables_and_levels = ds.trg_variables_and_levels
    temporal_resolution_hours = int(
        ds.temporal_resolution / np.timedelta64(1, "h")
    )

    # Loop through the dataloader and save to Zarr
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        zero_times = batch.zero_time
        batch = batch.to("cuda")
        model_input = preprocess_batch(batch, data_scale)
        with torch.no_grad():
            prd_dist, _ = model(batch=model_input, scales=rollout_scales)
            mean_np: npt.NDArray[Any] = prd_dist.mean.cpu().numpy()

        forecast = mean_np[..., 0, :]
        ds_batch = forecast_to_xarray(
            forecast=forecast,
            zero_times=zero_times,
            temporal_resolution_hours=temporal_resolution_hours,
            variables_and_levels=variables_and_levels,
            lats=lats,
            lons=lons,
        )

        for variable, scaling_factor in VARIABLES_WITH_SCALING_FACTORS.items():
            if variable in ds_batch:
                # Undo the scaling for some variables to adjust units.
                # This scaling doesn't apply to the zarr dataset directly, so we
                # need to reverse it here.
                ds_batch[variable] = ds_batch[variable] / scaling_factor

        if batch_idx == 0:
            # Initialize the zarr store on disk
            encoding = {
                "time": {
                    "units": "hours since 2020-01-01T00:00:00",
                    "dtype": "int64",
                }
            }
            ds_batch.to_zarr(
                zarr_path, mode="w", encoding=encoding, zarr_format=2
            )
        else:
            # Append to the existing zarr store along the 'time' dimension
            ds_batch.to_zarr(zarr_path, append_dim="time", zarr_format=2)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="Hydra config overrides",
        default=None,
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name

    (
        config,
        model,
        dataloader,
        ds,
        data_scale,
        rollout_scales,
        zarr_path,
    ) = load_dependencies(experiment_name, overrides=args.overrides)

    data_scale = int(data_scale)
    rollout_scales = [int(scale) for scale in rollout_scales]

    if not zarr_path.exists():
        logging.info(f"Generating forecasts for {experiment_name}...")
        generate_forecasts(
            experiment_name=experiment_name,
            model=model,
            dataloader=dataloader,
            ds=ds,
            data_scale=data_scale,
            rollout_scales=rollout_scales,
            zarr_path=zarr_path,
        )
    else:
        logging.info(
            f"Forecast already exists at {zarr_path}, skipping generation."
        )

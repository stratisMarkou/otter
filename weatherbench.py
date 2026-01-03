# %%
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from hydra.utils import instantiate

from otter.data.datasets import GriddedWeatherTask
from otter.data.labelled_array import forecast_to_xarray
from otter.experiment.run_state import RunState
from otter.experiment.train import preprocess_batch
from otter.experiment.utils import load_experiment_config
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.multiscale_model import MultiscaleModel

# %%
logging.basicConfig(level=logging.INFO)

experiment_name = "2025-11-06_19-08-bold-wolf"

config = load_experiment_config(
    experiment_name,
    override_args=[
        f"constants.data.val_task_subsampling_factor={365}",
        "data_source.path=_data/era5_240x121_single_array_float32",
        "constants.data.batch_size=1",
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

# %%
# Initialize Zarr store
try:
    root_forecasts = Path(config.info.root_forecasts)
except AttributeError:
    root_forecasts = Path("_forecasts")

root_forecasts.mkdir(parents=True, exist_ok=True)
zarr_path = root_forecasts / f"{experiment_name}.zarr"

lats = ds.dataset.get_coord_by_name("latitude").values
lons = ds.dataset.get_coord_by_name("longitude").values

variables_and_levels = ds.trg_variables_and_levels
temporal_resolution_hours = int(
    ds.temporal_resolution / np.timedelta64(1, "h")
)

# Loop through the dataloader and save to Zarr
for batch_idx, batch in enumerate(dataloader):
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

    if batch_idx == 0:
        # Initialize the zarr store on disk
        encoding = {
            "time": {
                "units": "hours since 2020-01-01T00:00:00",
                "dtype": "int64",
            }
        }
        ds_batch.to_zarr(zarr_path, mode="w", encoding=encoding)
    else:
        # Append to the existing zarr store along the 'time' dimension
        ds_batch.to_zarr(zarr_path, append_dim="time")

    if batch_idx % 10 == 0:
        logging.info(f"Processed zero time: {zero_times.values}")

# %%
# Verify the saved zarr
ds_saved = xr.open_zarr(zarr_path)
print(ds_saved)

# %%

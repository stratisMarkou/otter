# %%
from time import perf_counter

import hydra
import numpy as np
import xarray as xr
from hydra import compose

from otter.data.source.utils import (
    ALL_GRID_VARIABLE_LEVELS,
    ALL_GRID_VARIABLE_NAMES,
)

WEATHERBENCH_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
MEMMAP_PATH = "/home/js2731/rds/rds-vlm-weather-dHudQI6NhiE/otter/data/em626_era5_240x121_single_array_float32/era5_240x121_single_array_float32"
OFFLINE_ZARR_PATH = (
    "/home/js2731/rds/rds-vlm-weather-dHudQI6NhiE/otter/data/era5_240x121.zarr"
)

BATCH_SIZE = 4
TIMES_PER_BATCH = 3 * BATCH_SIZE
# %%
with hydra.initialize_config_module(
    config_module="otter.experiment.config", version_base="1.3"
):
    config_yaml = compose(config_name="deterministic.yaml")
    config_yaml.backbone_network.feedforward_network.hidden_features = 256
    config_yaml.constants.data.num_workers = 0
    config_yaml.constants.data.prefetch_factor = None
    config_yaml.constants.data.path_to_labelled_array = MEMMAP_PATH


trainloader = hydra.utils.instantiate(config_yaml.train_dataloader)
train_iter = iter(trainloader)


# %%
def time_run(n_batches, runner, name):
    print(f"Timing {name}...")
    start = perf_counter()
    runner.run(n_batches)
    end = perf_counter()
    time_ms = (end - start) * 1000 / n_batches
    print(f"Time per batch: {time_ms:.2f} ms")


class Runner:
    def run(self, n_batches):
        for _ in range(n_batches):
            self.run_once()

    def run_once(self):
        pass


class Memmap(Runner):
    def __init__(self, trainloader):
        self.trainloader = trainloader
        self.train_iter = iter(trainloader)

    def run_once(self):
        next(self.train_iter)


class Zarr(Runner):
    def __init__(self, url):
        self.zarr = xr.open_zarr(url).sel(level=ALL_GRID_VARIABLE_LEVELS)[
            ALL_GRID_VARIABLE_NAMES
        ]
        self.indices = range(len(self.zarr.time.values))

    def run_once(self):
        i = np.random.choice(self.indices)
        self.zarr.isel(time=slice(i, i + TIMES_PER_BATCH)).load()

    def run(self, n_batches):
        idxes = np.random.choice(
            self.indices, n_batches * TIMES_PER_BATCH, replace=False
        )
        self.zarr.isel(time=idxes).load()


zarr_weatherbench = Zarr(WEATHERBENCH_URL)
zarr_offline = Zarr(OFFLINE_ZARR_PATH)
memmap = Memmap(trainloader)

N_BATCHES = 5
time_run(N_BATCHES, zarr_weatherbench, "zarr_weatherbench")
time_run(N_BATCHES, zarr_offline, "zarr_offline")
time_run(N_BATCHES, memmap, "memmap")

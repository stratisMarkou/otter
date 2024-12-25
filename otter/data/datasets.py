import os
from dataclasses import dataclass
from typing import List

import numpy as np
import xarray as xr
from torch.utils.data import Dataset

GriddedWeatherData: type[xr.Dataset] = xr.Dataset


@dataclass
class GriddedWeatherTaskData:
    ctx: xr.Dataset
    trg: xr.Dataset


def collate_batch_wise(batch: List[xr.Dataset]) -> xr.Dataset:
    return xr.concat(batch, dim="batch")


class GriddedWeatherDataset(Dataset[xr.Dataset]):

    def __init__(
        self,
        zarr_name: str,
        start_date: str,
        end_date: str,
        base_path: str = "_data",
    ):
        super().__init__()
        zarr_path = os.path.join(base_path, zarr_name)
        self.zarr = xr.open_zarr(zarr_path)
        self.zarr = self.zarr.sel(time=slice(start_date, end_date))

    def __len__(self) -> int:
        return len(self.zarr.time)

    def __getitem__(self, idx: int) -> xr.Dataset:
        item = self.zarr.isel(time=idx).compute()
        # Making sure that mypy knows that the return type is a Dataset.
        assert isinstance(item, xr.Dataset)
        return item


class GriddedWeatherTask(Dataset[GriddedWeatherTaskData]):

    def __init__(
        self,
        zarr_name: str,
        start_date: str,
        end_date: str,
        num_context_frames: int,
        temporal_resolution_hours: int,
        num_target_timesteps: int,
        base_path: str = "_data",
    ):
        super().__init__()
        self.gridded_weather_dataset = GriddedWeatherDataset(
            zarr_name=zarr_name,
            start_date=start_date,
            end_date=end_date,
            base_path=base_path,
        )

        self.start_time = np.datetime64(start_date)
        self.end_time = np.datetime64(end_date)
        self.temporal_resolution = np.timedelta64(
            temporal_resolution_hours, "h"
        )
        self.num_target_timesteps = num_target_timesteps
        self.single_task_time_window = np.timedelta64(
            temporal_resolution_hours
            * (num_context_frames + num_target_timesteps - 1),
            "h",
        )

    def __len__(self) -> int:
        return len(self.gridded_weather_dataset) - 1

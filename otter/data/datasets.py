import os
from typing import List

import torch
import xarray as xr


def collate_batch_wise(batch: List[xr.Dataset]) -> xr.Dataset:
    return xr.concat(batch, dim="batch")


class GriddedWeatherDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        zarr_name: str,
        start_date: str,
        end_date: str,
        base_path: str = "_data",
    ):
        zarr_path = os.path.join(base_path, zarr_name)
        self.zarr = xr.open_zarr(zarr_path)
        self.zarr = self.zarr.sel(time=slice(start_date, end_date))

    def __len__(self):
        return len(self.zarr.time)

    def __getitem__(self, idx: int):
        return self.zarr.isel(time=idx).compute()

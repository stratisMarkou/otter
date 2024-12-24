import os
from typing import List

import xarray as xr
from torch.utils.data import Dataset


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

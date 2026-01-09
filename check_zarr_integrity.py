import numpy as np
import xarray as xr

# 1. Load the dataset lazily
# Using chunks={} tells xarray to use the chunks already on disk
ds = xr.open_zarr("~/otter-data/era5_240x121.zarr", chunks={})
# ds = xr.open_zarr(
#    "/home/jonas/Documents/code/otter/_data/era5_240x121.zarr", chunks={}
# )


def verify_dataset(ds):
    print("--- Dataset Overview ---")
    print(ds)

    # 2. Check Coordinates & Times
    print("\n--- Checking Coordinates ---")
    for coord in ds.coords:
        size = ds[coord].size
        # Check if any coordinate value is NaN
        nan_count = ds[coord].isnull().sum().values
        print(f"Coord '{coord}': size={size}, NaNs={nan_count}")

        # Check for gaps in time (assuming hourly)
        if coord == "time":
            diffs = np.diff(ds.time.values)
            unique_diffs = np.unique(diffs)
            print(f"  Time spacing found: {unique_diffs}")

    # 3. Check for NaNs in Data Variables
    # WARNING: This triggers a read of the data.
    # Because it's lazy, it will stream through the 1TB.
    print("\n--- Scanning for NaNs in Data (This takes time...) ---")
    for var in ds.data_vars:
        print(f"Scanning {var}...")
        # .isnull().any() is efficient because it stops at the first NaN it finds
        has_nan = ds[var].isnull().any().compute()

        if has_nan:
            total_nans = ds[var].isnull().sum().compute()
            print(f"  !! ALERT: {var} contains {total_nans} NaNs.")
        else:
            print(f"  Clean: {var} has no NaNs.")


if __name__ == "__main__":
    verify_dataset(ds)

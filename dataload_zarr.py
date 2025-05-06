import time

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class ZarrDataset(Dataset):
    def __init__(self, zarr_path, time_dim="time", slice_size=4):
        print(f"Opening Zarr dataset from {zarr_path}")
        self.zarr_path = zarr_path
        self.time_dim = time_dim
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.slice_size = slice_size

    def __len__(self):
        return self.ds.dims[self.time_dim] - self.slice_size

    def __getitem__(self, idx):
        # Create slice for time dimension
        time_slice = slice(idx, idx + self.slice_size)

        selection = {self.time_dim: time_slice}
        data = self.ds.isel(selection).load()
        return data

    def close(self):
        print("Closing Zarr dataset")
        self.ds.close()
        self.ds = None


def benchmark_random_time_access(
    zarr_path, num_samples=100, sample_size=1, num_workers=0
):
    ds = ZarrDataset(zarr_path, time_dim="time", slice_size=sample_size)

    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )

    print(
        f"Starting benchmark: Reading {num_samples} random time slices of size {sample_size}"
    )

    access_times = []
    data_iter = iter(dl)
    for i in range(num_samples):
        start_time = time.time()
        _ = next(data_iter)
        end_time = time.time()
        access_time_ms = (end_time - start_time) * 1000
        access_times.append(access_time_ms)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_samples} samples")

    # Calculate statistics
    access_times = np.array(access_times)

    # Print results
    print("\nBenchmark Results:")
    print(f"Mean access time: {np.mean(access_times):.2f} ms")
    print(f"Median access time: {np.median(access_times):.2f} ms")
    print(f"Min access time: {np.min(access_times):.2f} ms")
    print(f"Max access time: {np.max(access_times):.2f} ms")
    print(f"Std dev: {np.std(access_times):.2f} ms")
    print(f"95th percentile: {np.percentile(access_times, 95):.2f} ms")

    ds.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python script.py /path/to/dataset.zarr [num_samples] [sample_size]"
        )
        sys.exit(1)

    zarr_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    benchmark_random_time_access(zarr_path, num_samples, sample_size)

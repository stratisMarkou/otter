import argparse
import time

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class ZarrDataset(Dataset):
    def __init__(self, zarr_path, time_dim="time", slice_size=4):
        print(f"Opening Zarr dataset from {zarr_path}")
        self.zarr_path = zarr_path
        self.time_dim = time_dim
        self.ds = xr.open_zarr(zarr_path, consolidated=True, chunks=None)
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
    zarr_path, num_batches, sample_size, num_workers, batch_size
):
    ds = ZarrDataset(zarr_path, time_dim="time", slice_size=sample_size)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )

    print(
        f"Starting benchmark: Reading {num_batches} batches of size {sample_size} x {batch_size}"
    )

    access_times = []
    data_iter = iter(dl)
    for i in range(num_batches):
        start_time = time.time()
        _ = next(data_iter)
        end_time = time.time()
        access_time_ms = (end_time - start_time) * 1000
        access_times.append(access_time_ms)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_batches} batches")

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

    parser = argparse.ArgumentParser(
        description="Benchmark random time access in a Zarr dataset"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr dataset")
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of batches to access",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=4,
        help="Size of each sample (number of time steps)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader",
    )
    args = parser.parse_args()
    zarr_path = args.zarr_path
    num_batches = args.num_batches
    sample_size = args.sample_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    benchmark_random_time_access(
        zarr_path, num_batches, sample_size, num_workers, batch_size
    )

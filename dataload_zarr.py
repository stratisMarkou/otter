import random
import time

import numpy as np
import xarray as xr


def benchmark_random_time_access(zarr_path, num_samples=100, sample_size=1):
    """
    Benchmark random time slice access from a Zarr dataset and print results.

    Parameters:
    -----------
    zarr_path : str
        Path to the Zarr dataset
    num_samples : int
        Number of random samples to read
    sample_size : int
        Number of consecutive time steps to read in each sample
    """
    # Open the dataset with xarray
    print(f"Opening Zarr dataset from {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=True)

    # Get information about the dataset
    time_dim = "time"  # Default name, adjust if needed
    if time_dim not in ds.dims:
        time_dims = [dim for dim in ds.dims if "time" in dim.lower()]
        if time_dims:
            time_dim = time_dims[0]
        else:
            raise ValueError(
                f"No time dimension found. Available dimensions: {list(ds.dims.keys())}"
            )

    num_times = ds.dims[time_dim]
    print(f"Dataset contains {num_times} time steps")

    # Generate random time indices
    max_start_idx = num_times - sample_size
    random_indices = [
        random.randint(0, max_start_idx) for _ in range(num_samples)
    ]

    # Prepare results storage
    access_times = []

    # Perform benchmark
    print(
        f"Starting benchmark: Reading {num_samples} random time slices of size {sample_size}"
    )

    for i, idx in enumerate(random_indices):
        # Create slice for time dimension
        time_slice = slice(idx, idx + sample_size)

        # Build selection dictionary
        selection = {time_dim: time_slice}

        # Time the data access
        start_time = time.time()
        _ = ds.isel(selection).load()  # Force data load
        end_time = time.time()

        access_time_ms = (
            end_time - start_time
        ) * 1000  # Convert to milliseconds
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

    benchmark_random_time_access(zarr_path, num_samples, sample_size)

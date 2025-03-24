import os
import time
from datetime import datetime


def write_test(file_path, size_mb=1024, chunk_size_mb=8):
    """Test write performance by creating a file of specified size."""
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    chunks = size_mb // chunk_size_mb

    data = b"\0" * chunk_size  # Create a chunk of zeros

    print(f"Writing {size_mb}MB file in {chunk_size_mb}MB chunks...")
    start_time = time.time()

    with open(file_path, "wb") as f:
        for _ in range(chunks):
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

    end_time = time.time()
    duration = end_time - start_time
    throughput = size_mb / duration

    print(f"Write test completed in {duration:.2f} seconds")
    print(f"Write throughput: {throughput:.2f} MB/s")

    return throughput


def read_test(file_path, chunk_size_mb=8):
    """Test read performance from an existing file."""
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    print(f"Reading {file_size_mb:.0f}MB file in {chunk_size_mb}MB chunks...")
    start_time = time.time()

    with open(file_path, "rb") as f:
        while f.read(chunk_size):
            pass

    end_time = time.time()
    duration = end_time - start_time
    throughput = file_size_mb / duration

    print(f"Read test completed in {duration:.2f} seconds")
    print(f"Read throughput: {throughput:.2f} MB/s")

    return throughput


def run_all_tests(test_dir="./fs_test", file_size_mb=1024):
    """Run all filesystem performance tests and return results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    results = {}

    print("\n--- FILESYSTEM PERFORMANCE TEST ---")
    print(f"Test directory: {os.path.abspath(test_dir)}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"File size: {file_size_mb} MB")

    # Write test
    print("\n--- Write Test ---")
    file_path = f"{test_dir}/testfile_{timestamp}"
    write_speed = write_test(file_path, size_mb=file_size_mb)
    results["write_speed_mb_s"] = write_speed

    # Read test
    print("\n--- Read Test ---")
    read_speed = read_test(file_path)
    results["read_speed_mb_s"] = read_speed

    # Clean up test file
    os.remove(file_path)

    print("\n--- Summary ---")
    print(f"Write Speed: {results['write_speed_mb_s']:.2f} MB/s")
    print(f"Read Speed: {results['read_speed_mb_s']:.2f} MB/s")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test filesystem performance")
    parser.add_argument(
        "--dir",
        type=str,
        default="./fs_test",
        help="Directory to run tests in (default: ./fs_test)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="File size in MB for IO tests (default: 1024)",
    )

    args = parser.parse_args()

    run_all_tests(test_dir=args.dir, file_size_mb=args.size)

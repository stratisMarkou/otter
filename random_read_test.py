import argparse
import os
import random
import time


def create_test_file(file_path, file_size):
    """Creates a binary file filled with zeros if it doesn't already exist."""
    if not os.path.exists(file_path):
        print(f"Creating file '{file_path}' with size {file_size} bytes ...")
        with open(file_path, "wb") as f:
            # Write zeros in chunks to avoid using too much memory at once.
            chunk_size = 1024 * 1024  # 1MB at a time
            total_written = 0
            while total_written < file_size:
                write_size = min(chunk_size, file_size - total_written)
                f.write(b"\0" * write_size)
                total_written += write_size
        print("File created.")


def test_random_reads(file_path, file_size, num_reads, read_size):
    """Performs random reads from the file and measures the total time."""
    with open(file_path, "rb") as f:
        start_time = time.perf_counter()
        for _ in range(num_reads):
            offset = random.randint(0, file_size - read_size)
            f.seek(offset)
            data = f.read(read_size)
            # Dummy processing: calculate the sum of the byte values
            _ = sum(data)
        end_time = time.perf_counter()

    print(
        f"{num_reads} random reads (read size: {read_size} bytes) took {end_time - start_time:.3f} seconds"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test small random access reads on a plain binary file."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="test_file.bin",
        help="Path to the test file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100 * 1024 * 1024,
        help="Size of the test file in bytes (default: 100MB).",
    )
    parser.add_argument(
        "--num_reads",
        type=int,
        default=1000,
        help="Number of random reads to perform.",
    )
    parser.add_argument(
        "--read_size",
        type=int,
        default=1024,
        help="Number of bytes to read for each random access (default: 1024 bytes).",
    )

    args = parser.parse_args()

    create_test_file(args.file, args.size)
    test_random_reads(args.file, args.size, args.num_reads, args.read_size)


if __name__ == "__main__":
    main()

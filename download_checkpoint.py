"""Download a checkpoint from Azure storage.

Usage:
    python download_checkpoint.py <run_name>

This downloads <checkpoint_folder>/<run_name> from Azure into _results/<run_name>.
"""

import argparse
import os

from azureml.fsspec import AzureMachineLearningFileSystem

# Azure storage configuration (matching submit_azure.py)
SUBSCRIPTION_ID = "f5524f9a-9942-4ba3-bec7-cce5496c1374"
RESOURCE_GROUP = "eng-rs-ai4goodweather-otter-prod-plc-001"
WORKSPACE_NAME = "eng-mlw-otter-prod-001"
DATASTORE_NAME = "otter_blob"
CHECKPOINT_PATH = "otter_checkpoints"


def download_checkpoint(run_name: str, output_dir: str = "_results") -> None:
    """Download a checkpoint folder from Azure storage.

    Args:
        run_name: Name of the run/experiment to download.
        output_dir: Local directory to download into. Defaults to "_results".
    """
    # Construct the datastore URI
    datastore_uri = (
        f"azureml://subscriptions/{SUBSCRIPTION_ID}"
        f"/resourcegroups/{RESOURCE_GROUP}"
        f"/workspaces/{WORKSPACE_NAME}"
        f"/datastores/{DATASTORE_NAME}"
        f"/paths/{CHECKPOINT_PATH}/{run_name}"
    )

    print(f"Connecting to: {datastore_uri}")

    # Create Azure ML filesystem
    fs = AzureMachineLearningFileSystem(datastore_uri)

    # Create local output directory
    local_run_dir = os.path.join(output_dir, run_name)
    os.makedirs(local_run_dir, exist_ok=True)

    # Only download these specific files
    files_to_download = [
        "config.yaml",
        "checkpoints/latest.pt",
    ]

    print(f"Downloading {len(files_to_download)} files to: {local_run_dir}")

    for relative_path in files_to_download:
        # Construct full Azure path
        azure_path = f"{CHECKPOINT_PATH}/{run_name}/{relative_path}"

        # Check if file exists
        try:
            if not fs.exists(azure_path):
                print(f"Warning: {relative_path} not found, skipping...")
                continue
        except Exception as e:
            print(f"Warning: Could not check {relative_path}: {e}")
            continue

        local_path = os.path.join(local_run_dir, relative_path)
        local_dir = os.path.dirname(local_path)

        # Create parent directories
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        print(f"Downloading: {relative_path}")
        try:
            # Azure fs.get() downloads to a directory, creating a subdirectory with the file name
            # So we download to the parent and then move the file
            temp_download_dir = local_dir if local_dir else local_run_dir
            fs.get(azure_path, temp_download_dir)

            # The file gets downloaded as temp_download_dir/filename/filename
            # We need to move it to temp_download_dir/filename
            filename = os.path.basename(relative_path)
            downloaded_file = os.path.join(temp_download_dir, filename, filename)

            if os.path.exists(downloaded_file):
                # Move the file to the correct location
                os.rename(downloaded_file, local_path)
                # Remove the empty directory
                os.rmdir(os.path.join(temp_download_dir, filename))
        except Exception as e:
            print(f"Error downloading {relative_path}: {e}")
            continue

    print(f"\nDownload complete! Files saved to: {local_run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a checkpoint from Azure storage."
    )
    parser.add_argument("run_name", type=str, help="Name of the run to download")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_results",
        help="Local output directory (default: _results)",
    )

    args = parser.parse_args()
    download_checkpoint(args.run_name, args.output_dir)

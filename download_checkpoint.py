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

    # List all files in the checkpoint folder
    files = fs.glob("**/*")

    if not files:
        print(f"No files found for run '{run_name}'")
        return

    print(f"Found {len(files)} items to download...")
    print(f"Downloading to: {local_run_dir}")

    for file_path in files:
        # Skip directories
        try:
            if fs.isdir(file_path):
                continue
        except Exception:
            pass

        # Get relative path (remove the datastore prefix)
        # file_path is like: 'otter_checkpoints/run_name/checkpoints/file.pt'
        # We need to strip 'otter_checkpoints/run_name/' prefix
        prefix = f"{CHECKPOINT_PATH}/{run_name}/"
        if file_path.startswith(prefix):
            relative_path = file_path[len(prefix):]
        else:
            relative_path = file_path

        if not relative_path:
            continue

        local_path = os.path.join(local_run_dir, relative_path)
        local_dir = os.path.dirname(local_path)

        # Create parent directories
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        print(f"Downloading: {relative_path}")
        # Download to the parent directory - Azure will create the file with correct name
        fs.download(file_path, local_dir if local_dir else local_run_dir)

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

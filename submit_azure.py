import os
import sys

from azure.ai.ml import Input, MLClient, Output, command
from azure.identity import DefaultAzureCredential

from otter.experiment.namegen import _generate_experiment_name

wandb_api_key = os.environ.get("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY environment variable not set.")

# 1. Workspace Configuration
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="f5524f9a-9942-4ba3-bec7-cce5496c1374",
    resource_group_name="eng-rs-ai4goodweather-otter-prod-plc-001",
    workspace_name="eng-mlw-otter-prod-001",
)

era5_data = ml_client.data.get("era5-1959-2023-6h-240x121", version="1")

run_args = " ".join(sys.argv[1:])

env = "azureml:otter-env:3"
run_name = _generate_experiment_name()

experiment_path = f"${{{{outputs.checkpoint_folder}}}}/{run_name}"

job = command(
    name=run_name,
    display_name=run_name,
    # Point directly to the otter directory for the code snapshot
    code=".",
    command=(
        # Print mount locations:
        "echo 'Inputs:' && realpath ${{inputs.era5_data}} && "
        "echo 'Outputs:' && realpath ${{outputs.checkpoint_folder}} && "
        "ls -la ${{outputs.checkpoint_folder}} && "
        # Add uv to PATH for this shell session
        "export PATH=$HOME/.local/bin:$PATH && "
        # Create venv
        "uv venv .venv && "
        ". .venv/bin/activate && "
        "uv sync && "
        "uv pip install -e . && "
        f"python {run_args} data_source.path=${{inputs.era5_data}} info.experiment_path={experiment_path} --resume_experiment {run_name} --root_results ${{outputs.checkpoint_folder}}"
    ),
    inputs={
        "era5_data": Input(
            type="uri_folder",
            path=era5_data.id,
            mode="ro_mount",
        )
    },
    outputs={
        "checkpoint_folder": Output(
            type="uri_folder",
            path="azureml://datastores/otter_blob/paths/otter_checkpoints",
            mode="rw_mount",
        )
    },
    environment=env,
    compute="otter-gpu-pool",
    environment_variables={
        "WANDB_API_KEY": wandb_api_key,
        "WANDB_MODE": "online",
        "DATASET_MOUNT_CACHE_SIZE": "-40 GB",  # Leave 40 GB free for checkpoints (2 copies during atomic writes)
    },
    # Increase shared memory size for torch dataloaders
    shm_size="16g",
)

# 4. Submit
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.studio_url}")

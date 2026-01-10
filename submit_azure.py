import os
import sys

from azure.ai.ml import Input, MLClient, command
from azure.identity import DefaultAzureCredential

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

job = command(
    display_name=f"otter-{run_args[:20]}" if run_args else "otter-default",
    # Point directly to the otter directory for the code snapshot
    code=".",
    command=(
        # Add uv to PATH for this shell session
        "export PATH=$HOME/.local/bin:$PATH && "
        # Create venv
        "uv venv .venv && "
        ". .venv/bin/activate && "
        "uv sync && "
        "uv pip install -e . && "
        f"python {run_args} data_source.path=${{inputs.era5_data}}"
    ),
    inputs={
        "era5_data": Input(
            type="uri_folder",
            path=era5_data.id,
            mode="ro_mount",
        )
    },
    environment=env,
    compute="otter-gpu-pool",
    environment_variables={
        "WANDB_API_KEY": wandb_api_key,
        "WANDB_MODE": "online",
    },
)

job.set_resources(
    shm_size="16g"  # Increase shared memory size for torch dataloaders
)

# 4. Submit
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.studio_url}")

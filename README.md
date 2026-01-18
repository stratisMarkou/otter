# Otter

## Getting started
The project uses uv to manage dependencies.
If you haven't installed uv you can do so by
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then clone the repo, go to its root, create a virtual environment and sync dependencies.
```bash
git clone git@github.com:jonas-scholz123/otter.git
cd otter
uv venv
source .venv/bin/activate
uv sync
```
Last, if you want to commit code you probably want to install pre-commit hooks, to run pre-commit checks before each commit.
```bash
pre-commit install
```
To download and pre-process the ERA5 data:
```bash
# This will download the 64x32 and 240x121 grid ERA5 data from WeatherBench2.
# The larger of the two datasets, the 240x121 grid, requires about 660GB of storage.
# By default, the data will be stored under ./_data in the repo root.
python otter/data/source/download_weatherbench_data.py

# This will post-process convert it to pure-numpy format for faster loading.
python otter/data/source/convert_saved_zarr_to_memmap.py

# Normalisation factors for the ERA5 data are already computed and version controlled
# in the package, but if you want to re-generate the factors you can use the command
# python otter/data/normalisation/compute_stats.py
```

To train a deterministic model, from the repo root, run:
```bash
# This will train a model with logging to wandb. It will also check your local branch
# is clean, i.e. that it has no changes relative to the remote branch.
wandb online; python otter/experiment/train.py --config_name base

# You can disable this check by adding the flag --debug at the end of the command, as in
wandb online; python otter/experiment/train.py --config_name base --debug
```

To train a model on Azure, make sure that you first set the `WANDB_API_KEY` environment variable. This will be transmitted to the Azure VM.
You will also need to install the azure CLI:
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```
and authenticate:
```bash
az login
```

Then, run (including your overrides):
```bash
python ./submit_azure.py otter/experiment/train.py --config_name base host=azure
```
This will launch a run using the code as-is on your local machine (same git branch etc).


### RFT
You can rollout finetune a pretrained model using:
```bash
python otter/experiment/train.py --config_name rollout_finetuning constants.load_weights_from_experiment=/some/path/to/2026-01-18_13-50-fierce-eagle/checkpoints/latest.pt
```
Note that your configs need to align, i.e. the model architecture defined by the config needs to match that found in the checkpoint. We are looking to make this more ergonomic.

After a model has been trained, it can be used to generate cached forecasts for later evaluation. To do so for a given `experiment_name`, run:
```bash
python otter/evaluate/save_forecasts.py <experiment_name>
```
The experiment name is the same as in wandb or in the _results directory, e.g. `2025-11-06_19-08-bold-wolf`.

For weatherbench model evaluation (to save evaluation metrics), run:
```bash
python otter/evaluate/weatherbench.py \
--experiment_name=<experiment_name> \
--runner=DirectRunner
```
Finally, these results can be plotted and saved to the _plots folder using
```
python otter/evaluate/plot_weatherbench_output.py <experiment_name>
```

To launch a training run on HPC, from the repo root, run:
```bash
# submit.sh requests one gpu for 36 hours, the account is OTTER-SL2-GPU.
sbatch submit.sh otter/experiment/train.py --config_name base host=hpc
```

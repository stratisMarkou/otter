# Otter

## Getting started
The project uses conda to manage dependencies.
If you haven't installed conda you can do so by
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
Then clone the repo, go to its root and create a conda environment.
```bash
git clone git@github.com:stratisMarkou/otter.git
cd otter
conda env create -n otter -f env.yaml
```
The environment is built each time from scratch to run tests, so the environment yaml is always up to date.
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

To train a (deterministic) single-scale model, from the repo root, run:
```bash
# This will train a model with logging to wandb. It will also check your local branch
# is clean, i.e. that it has no changes relative to the remote branch.
wandb online; python otter/experiment/train.py --config_name base

# You can disable this check by adding the flag --debug at the end of the command, as in
wandb online; python otter/experiment/train.py --config_name base --debug
```

After a model has been trained, it can be used to generate cached forecasts for later evaluation. To do so for a given `experiment_name`, run:
```bash
python otter/experiment/save_samples.py <experiment_name>
```
The experiment name is the same as in wandb or in the _results directory.

For model evaluation (to save evaluation metrics), run:
```bash
python otter/experiment/eval.py --experiment_name <experiment_name> --results_path <results_path> --checkpoint_name <checkpoint_name>
constants.data.valid_num_target_frames=<num_target_frames>
```
The experiment name is the same as in wandb or in the <results_path> directory.

Number of target frames determines the forecasting horizon.

Checkpoint name (e.g. latest, best) determines which model weights to load.

When evaluation is done, to create a plot for power spectrum density MSE per variable, run:
```bash
python otter/plots/plot_power_spectrum_mse.py --experiment_name <experiment_name> --rollout_steps <num_target_frames> --checkpoint_name <checkpoint_name>
```

It will create a PNG image in the `experiment_name/power_spectrum_mse` folder.

To create a score card to compare two models, run:
```bash
python otter/plots/plot_score_card.py --experiment_name <experiment_name> --base_experiment_name <base_experiment_name> --rollout_steps <num_target_frames> --base_rollout_steps <base_num_target_frames> --checkpoint_name <checkpoint_name> --base_checkpoint_name <base_checkpoint_name>
```

To compare to Arches Weather M model, use `_baselines/arches_weather/validation/metrics_latest_rollout_40.pkl`.

To launch a training run on HPC, from the repo root, run:
```bash
# ENV_PATH indicates the path to otter venv, default is set to otter.
# submit.sh requests one gpu for 36 hours, the account is OTTER-SL2-GPU.
ENV_PATH="otter" sbatch submit.sh otter/experiment/train.py --config_name base host=hpc
```



__Priorities:__
- [ ] Add spatial Fourier embeddings to model.
- [ ] Train a larger model to estimate performance gains from increased size.
- [ ] Add mixture of experts functionality to the model.

__Sub goals:__
- [ ] Train an MSE model to match Stormer / ArchesWeather.
- [ ] Train a multi-scale model to match the probabilistic ArchesWeather model performance.

__Tests to implement:__
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).

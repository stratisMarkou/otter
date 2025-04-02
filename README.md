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

To launch a training run on HPC, from the repo root, run:
```bash
# ENV_PATH indicates the path to otter venv, default is set to otter.
# submit.sh requests one gpu for 36 hours, the account is OTTER-SL2-GPU.
ENV_PATH="otter" sbatch submit.sh otter/experiment/train.py --config_name base
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

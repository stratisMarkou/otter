# Otter

## Getting started
The project uses conda to manage dependencies.
If you haven't installed conda you can do so by
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
Then clone the repo, go to its root and create a conda environment.
```
git clone git@github.com:stratisMarkou/otter.git
cd otter
conda env create -f env.yaml
```
The environment is built each time from scratch to run tests, so the environment yaml is always up to date.
Last, if you want to commit code you probably want to install pre-commit hooks, to run pre-commit checks before each commit.
```
pre-commit install
```


__Priorities:__
- [x] Finalise faster training optimisations.
- [ ] Add spatial Fourier embeddings to model.
- [ ] Add model checkpointing.

__Sub goals:__
- [x] Dry-run an MSE model on the gridded weather data task in a standalone script.
- [ ] Train an MSE model to match Stormer.


__Tests to implement:__
- [x] Test that forward pass through small transformer architecture works.
- [ ] Test for gridded weather data task.
    - [ ] Decide if we need to profile this within tests (i.e. test within CI that loading is not too slow).
- [x] Test for data download.

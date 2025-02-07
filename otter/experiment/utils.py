import argparse
import importlib
import logging
import os
import re
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import git
import torch
import xarray as xr
import yaml
from hydra.utils import instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
)

import wandb

Experiment = DictConfig


def get_model_parameters(
    model: torch.nn.Module,
) -> Iterator[torch.nn.Parameter]:
    return model.parameters()


def get_num_training_steps(
    num_epochs: int, dataloader: torch.utils.data.DataLoader[Any]
) -> int:
    return num_epochs * len(dataloader)


def get_num_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def create_module_summary(experiment: Experiment) -> str:
    module_summary = get_module_summary(experiment.model)
    num_params = get_num_parameters(experiment.model)
    return module_summary + f"\nTotal # parameters: {num_params / 1e6:.1f}M."


def get_module_attr(string: str) -> Any:
    module_name, class_name = string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def initialize_experiment(
    args_list: Optional[Sequence[str]] = None,
) -> Experiment:
    """Initialize experiment by parsing the config file, checking that the
    repo is clean, creating a path for the experiment, and creating a
    writer for tensorboard.

    Arguments:
        args_list: list of arguments to parse.

    Returns:
        experiment: experiment config object.
        path: path to experiment.
        writer: tensorboard writer.
    """

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args, extra_args = parser.parse_known_args(args=args_list)

    # Create a repo object and check if local repo is clean
    repo = git.Repo(search_parent_directories=True)

    # Check that the repo is clean, and up to date with remote branch,
    # if not in debug mode.
    if not args.debug and repo.is_dirty():
        raise AssertionError("Repo is dirty, please commit changes.")

    if not args.debug and has_commits_ahead(repo):
        raise AssertionError("Repo has commits ahead, please push changes.")

    # Initialize experiment, make path and writer
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("get_module_attr", get_module_attr)
    config = OmegaConf.load(args.config)
    config_changes = OmegaConf.from_cli(extra_args)

    config = OmegaConf.merge(config, config_changes)
    assert type(config) is DictConfig  # Check this to silence mypy.
    path = make_experiment_path(config)
    config.info.path_results = path
    experiment: DictConfig = instantiate(config)

    # Write config to file together with commit hash
    with open(f"{path}/config.yml", "w") as file:
        hash = get_current_commit_hash(repo) if not args.debug else None
        config_to_save: DictConfig = OmegaConf.to_container(
            config
        )  # type: ignore
        config_to_save.update({"commit": hash})
        yaml.dump(config_to_save, file, indent=4, sort_keys=False)

    return experiment


def has_commits_ahead(repo: git.Repo) -> bool:
    """Check if there are commits ahead in the local current branch.

    Arguments:
        repo: git repo object.

    Returns:
        has_commits_ahead: True if there are commits ahead, False otherwise.
    """
    if repo.head.is_detached:
        assert not repo.is_dirty(), "Repo is dirty, please commit changes."
        return False

    else:
        current_branch = repo.active_branch.name
        commits = list(
            repo.iter_commits(f"origin/{current_branch}..{current_branch}")
        )
        return len(commits) > 0


def get_current_commit_hash(repo: git.Repo) -> str:
    """Get the current commit hash of the local repo.

    Arguments:
        repo: git repo object.

    Returns:
        commit_hash: current commit hash.
    """
    if repo.head.is_detached:
        return repo.commit(repo.head.object).hexsha

    else:
        return repo.head.commit.hexsha


def make_experiment_path(experiment: DictConfig) -> str:
    """Parse initialized experiment config and make up a path
    for the experiment, and create it if it doesn't exist,
    otherwise raise an error. Finally return the path.

    Arguments:
        config: config object.

    Returns:
        experiment_path: path to the experiment.
    """

    experiment_name = getattr(
        experiment.info,
        "experiment_name",
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )
    path = os.path.join(
        experiment.info.root_results,
        experiment_name,
    )

    if not os.path.exists(path):
        logging.info(f"Making path for experiment results: {path}.")
        os.makedirs(path)

    else:
        logging.info(
            f"Path for experiment results:\n{path}\nalready exists! Exiting."
        )
        quit()

    return path


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_module_summary(module: torch.nn.Module, indent: int = 0) -> str:
    summary = ""
    total_params = count_parameters(module)
    summary += (
        f"{' ' * indent}Module: {module.__class__.__name__} "
        f"| Parameters: {total_params / 1e6:.1f}M\n"
    )
    for name, sub_module in module.named_children():
        summary += f"{' ' * (indent + 2)}Submodule: {name}\n"
        summary += get_module_summary(sub_module, indent + 2)
    return summary


def run_with_timing(
    f: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[float, Any]:
    start = datetime.now()
    result = f(*args, **kwargs)
    end = datetime.now()
    return (end - start).total_seconds(), result


def run_and_log_time_to_wandb(
    log_name: str,
    f: Callable[..., Any],
    *f_args: Any,
    **f_kwargs: Any,
) -> Any:
    time, result = run_with_timing(f, *f_args, **f_kwargs)
    wandb.log({f"time/{log_name}": time}, commit=False)
    return result


def wandb_log_code_exclude_fn(path: str) -> bool:

    # Create regex patterns that:
    patterns = [
        # matches all paths starting with ".".
        re.compile(r"\..*"),
        # matches all paths starting with "_".
        re.compile(r"_.*"),
        # matches all paths containing "/wandb"
        re.compile(r"wandb.*"),
        # matches all paths containing "otter.egg-info"
        re.compile(r"otter.egg-info.*"),
    ]

    return any(bool(pattern.match(path)) for pattern in patterns)


def log_code_to_wandb() -> None:

    # Create code artifact to log to wandb.
    code_artifact = wandb.Artifact(name="source", type="code")
    for path in os.listdir("."):
        if not wandb_log_code_exclude_fn(path):
            if os.path.isdir(path):
                code_artifact.add_dir(path)
            else:
                code_artifact.add_file(path)

    # Log code artifact to wandb.
    wandb.log_artifact(code_artifact)


def load_std_dataset() -> xr.DataArray:

    from otter.data.normalisation.utils import load_statistic
    from otter.data.utils import (
        split_into_different_variables_along_dim,
        stack_dataset_variable_and_levels,
    )

    std_dataset = load_statistic("std")
    mean_dataset_split = split_into_different_variables_along_dim(
        std_dataset, "level"
    )
    return stack_dataset_variable_and_levels(mean_dataset_split)


def log_loss_info_to_wandb(
    loss_info: Dict[str, torch.Tensor],
    prefix: str,
) -> None:
    for k, v in loss_info.items():
        wandb.log({f"{prefix}/{k}": v}, commit=False)


def log_per_variable_per_lead_time_losses_to_wandb(
    mean_loss: Dict[str, torch.Tensor],
    mean_loss_info: Dict[str, torch.Tensor],
    std_dataset: xr.DataArray,
) -> None:

    # Log the mean loss for each time index.
    for i, loss in enumerate(mean_loss):
        # Log and add leading 0s to index i
        wandb.log({f"val/loss_time_{i:03}": loss}, commit=False)

    # Log the lwrmse loss info for each time index.
    for key, info in mean_loss_info.items():
        variable_and_level = key.replace("lwrmse/", "")
        std = std_dataset.sel(variable_and_level=variable_and_level).values

        # Loop over times and log the lwrmse loss info.
        for i in range(info.shape[0]):
            wandb.log(
                {f"val/{key}_time_{i:03}": info[i] * std},
                commit=False,
            )


def create_optimizer_and_lr_scheduler(
    module: torch.nn.Module,
    optimizer_kwargs: Dict[str, Any],
    lr_scheduler_kwargs: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:

    optimizer = torch.optim.AdamW(
        params=module.parameters(), **optimizer_kwargs
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, **lr_scheduler_kwargs
    )

    return optimizer, lr_scheduler

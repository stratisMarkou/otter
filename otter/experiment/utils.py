import argparse
import importlib
import logging
import os
import random
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
import hydra
import torch
import xarray as xr
import yaml
from distributed_shampoo import (  # type: ignore[import-untyped]
    AdaptiveAmortizedComputationFrequencyConfig,
    DistributedShampoo,
    EigenvalueCorrectedShampooPreconditionerConfig,
)
from hydra import compose
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
    device: str,
    args_list: Optional[Sequence[str]] = None,
) -> Experiment:
    """Initialize experiment by parsing the config file, checking that the
    repo is clean, creating a path for the experiment, and creating a
    writer for tensorboard.

    Arguments:
        device: device to run the experiment on.
        args_list: list of arguments to parse.

    Returns:
        experiment: experiment config object.
        path: path to experiment.
        writer: tensorboard writer.
    """

    args, extra_args = _parse_args(args_list)
    config = _get_config(args.config_module, args.config, extra_args)

    experiment_name = (
        args.resume_experiment
        if args.resume_experiment
        else _generate_experiment_name()
    )

    _initalize_wandb(experiment_name)

    repo = git.Repo(search_parent_directories=True)
    # Check that the repo is clean, and up to date with remote branch,
    # if not in debug mode.
    if not args.debug:
        _assert_repo_is_clean(repo)

    experiment_path = _create_experiment_path(config, experiment_name)

    config.info.experiment_name = experiment_name
    config.info.experiment_path = experiment_path
    config.info.start_from_checkpoint = bool(args.resume_experiment)

    config.model.device = device

    _write_config(config, experiment_path, repo)

    logging.info("Instantiating dependencies.")

    experiment: DictConfig = instantiate(config)
    return experiment


def _parse_args(
    args_list: Optional[Sequence[str]],
) -> tuple[argparse.Namespace, list[str]]:
    """
    Parse CLI arguments.

    Arguments:
        args_list: Optional list of arguments to parse.

    Returns:
        args: Known parsed arguments.
        extra_args: Additional arguments to be logged/stored.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--resume_experiment",
        type=str,
        required=False,
        help="Resume the experiment with the provided given name",
    )
    parser.add_argument(
        "--config_module", type=str, default="otter.experiment.config"
    )
    parser.add_argument("--config_name", type=str)

    return parser.parse_known_args(args=args_list)


def _generate_experiment_name() -> str:
    adjectives = [
        "swift",
        "fierce",
        "gentle",
        "clever",
        "brave",
        "cunning",
        "loyal",
        "playful",
        "quiet",
        "bold",
        "majestic",
        "nimble",
        "mighty",
        "elegant",
        "sly",
        "energetic",
        "daring",
        "graceful",
        "wild",
        "fearless",
    ]

    animals = [
        "tiger",
        "eagle",
        "wolf",
        "panther",
        "falcon",
        "fox",
        "bear",
        "lion",
        "cheetah",
        "hawk",
        "otter",
        "deer",
        "jaguar",
        "leopard",
        "owl",
        "raven",
        "buffalo",
        "cougar",
        "lynx",
        "badger",
    ]

    adjective = random.choice(adjectives)
    animal = random.choice(animals)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    return f"{adjective}-{animal}-{timestamp}"


def _initalize_wandb(experiment_name: str) -> None:
    wandb.init(
        project="otter-deterministic",
        entity="otter-weather",
        name=experiment_name,
        id=experiment_name,
    )


def _assert_repo_is_clean(repo: git.Repo) -> None:
    if repo.is_dirty():
        raise AssertionError("Repo is dirty, please commit changes.")

    if _has_commits_ahead(repo):
        raise AssertionError("Repo has commits ahead, please push changes.")


def _has_commits_ahead(repo: git.Repo) -> bool:
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


def _get_config(
    config_module: str, config_name: str, extra_args: list[str]
) -> DictConfig:
    """
    Reads the config file, evaluates expressions, and merges it with overrides.
    """
    # Register custom resolvers if not already registered.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    if not OmegaConf.has_resolver("get_module_attr"):
        OmegaConf.register_new_resolver("get_module_attr", get_module_attr)

    # Initialize experiment.
    with hydra.initialize_config_module(
        config_module=config_module, version_base="1.3"
    ):
        config_yaml = compose(config_name=config_name)

    cli_overrides = OmegaConf.from_cli(extra_args)

    config = OmegaConf.merge(config_yaml, cli_overrides)
    assert type(config) is DictConfig  # Check this to silence mypy.

    OmegaConf.set_struct(config, False)

    return config


def _get_current_commit_hash(repo: git.Repo) -> str:
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


def _create_experiment_path(config: DictConfig, experiment_name: str) -> str:
    """Creates a path for the experiment, and create it if it doesn't exist.

    Arguments:
        config: config object.
        experiment_name: name of the experiment.
    Returns:
        experiment_path: path to the experiment.
    """

    path = os.path.join(
        config.info.root_results,
        experiment_name,
    )

    if not os.path.exists(path):
        logging.info(f"Making path for experiment results: {path}.")
        os.makedirs(path)

    return path


def _write_config(config: DictConfig, path: str, repo: git.Repo) -> None:
    # Write config to file together with commit hash
    path = f"{path}/config.yml"
    logging.info(f"Writing reified config to {path}.")
    with open(path, "w") as file:
        hash = _get_current_commit_hash(repo)
        config_to_save: DictConfig = OmegaConf.to_container(config)  # type: ignore
        config_to_save.update({"commit": hash})
        yaml.dump(config_to_save, file, indent=4, sort_keys=False)


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
        # matches the "tests" directory
        re.compile(r"tests.*"),
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


def get_adamw_optimizer(
    module: torch.nn.Module, optimizer_kwargs: Dict[str, Any]
) -> torch.optim.AdamW:
    return torch.optim.AdamW(params=module.parameters(), **optimizer_kwargs)


def get_shampoo_optimizer(
    module: torch.nn.Module, optimizer_kwargs: Dict[str, Any]
) -> DistributedShampoo:
    """Instantiate Shampoo optimizer with eigenvalue-corrected preconditioner.

    See the docstring of the DistributedShampoo class for more information:
    https://github.com/facebookresearch/optimizers/blob/adaptive-update-frequency/distributed_shampoo/distributed_shampoo.py#L101-L294

    Mmemory overhead:
    For a parameter W with shape (m, n), Adam uses an additional buffer of size m*n.
    On top of this, eigenvalue-corrected Shampoo uses buffers of sizes 2 * m**2  and 2 * n**2, which increases its memory requirements.
    However, if max(m, n) > max_preconditioner_dim, the preconditioner will be blocked such that the largest dimension is max_preconditioner_dim.

    Computational overhead:
    The most significant overhead is caused by the eigendecompositions of the Kronecker factor matrices.
    Instead of computing them at a fixed frequency, we use an adaptive strategy that computes them only when the previously computed eigenbasis is no longer accurate enough.
    This strategy is controlled by the tolerance parameter of the AdaptiveAmortizedComputationFrequencyConfig class, i.e. the smaller the tolerance, the more frequently the eigendecompositions are computed.

    Arguments:
        module: Module to optimize.
        optimizer_kwargs: Optimizer arguments.

    Returns:
        DistributedShampoo optimizer instance.
    """
    # See the following docstring for more explanation on the eigenvalue correction:
    # https://github.com/facebookresearch/optimizers/blob/adaptive-update-frequency/distributed_shampoo/distributed_shampoo.py#L238-L246
    preconditioner_config = EigenvalueCorrectedShampooPreconditionerConfig(
        amortized_computation_frequency_config=AdaptiveAmortizedComputationFrequencyConfig(
            # Determines how frequently eigendecompositions are computed and hence also
            # the average step time. See the following docstring for an explanation:
            # https://github.com/facebookresearch/optimizers/blob/adaptive-update-frequency/distributed_shampoo/shampoo_types.py#L105-L112
            tolerance=optimizer_kwargs.pop(
                "adaptive_amortized_computation_frequency_tolerance"
            ),
        )
    )
    return DistributedShampoo(
        params=module.parameters(),
        preconditioner_config=preconditioner_config,
        **optimizer_kwargs,
    )


def get_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer, lr_scheduler_kwargs: Dict[str, Any]
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, **lr_scheduler_kwargs
    )

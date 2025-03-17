import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
)

import pickle
import torch
from hydra.utils import instantiate
from omegaconf import (
    DictConfig,
    OmegaConf,
)

from otter.experiment.train import (
    assert_single_device,
    run_validation_epoch,
)
from otter.experiment.utils import get_module_attr
from otter.models.losses.loss_fn import LossInfo


def initialize_evaluation(
    device: str,
    experiment_name: str,
    results_path: str = "./_results",
    lead_time: int = 4,
    args_list: Optional[list[str]] = None,
) -> DictConfig:
    """
    Initialize evaluation by loading an existing experiment's config file.

    Arguments:
        device: Device to run the evaluation on (e.g., 'cuda', 'cpu').
        results_path: Name of the directory with results.
        experiment_name: Name of the existing experiment.
        lead_time: Number of time steps to use in the validation rollout.
        args_list: Optional list of extra arguments to override config values.

    Returns:
        evaluation_config: The loaded configuration for evaluation.
    """
    experiment_dir = Path(results_path) / experiment_name
    config_path = experiment_dir / "config.yml"

    if not experiment_dir.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Experiment '{experiment_name}' not found at {experiment_dir}. "
            "Ensure the experiment exists and contains 'config.yml'."
        )

    # Load the existing config file.
    config = OmegaConf.load(str(config_path))

    # Compute num_target_frames based on lead_time and temporal resolution.
    num_target_frames = (
        lead_time * 24 // config.constants.data.temporal_resolution_hours
    )
    # Modify lead time and subsampling factor.
    OmegaConf.update(
        config, "valid_dataset.num_target_frames", num_target_frames
    )
    OmegaConf.update(config, "valid_dataset.task_sub_sampling_factor", 1)

    # Register OmegaConf resolvers.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "get_module_attr", get_module_attr, replace=True
    )

    # Merge with additional command-line arguments.
    if args_list:
        config_changes = OmegaConf.from_cli(args_list)
        config = OmegaConf.merge(config, config_changes)

    # Update necessary fields for evaluation.
    config.info.experiment_name = experiment_name
    config.info.path_results = str(experiment_dir)
    config.model.device = device

    # Instantiate the evaluation config.
    evaluation_config: DictConfig = instantiate(config)

    return evaluation_config


def get_last_ckpt_name(
    experiment_name: str,
    results_path: str,
) -> str:
    """
    Returns last checkpoint name for the specified experiment.

    Arguments:
        experiment_name: Name of the existing experiment.
        results_path: Name of the directory with results.

    Returns:
        checkpoint_name: Name of the last checkpoint.
    """
    experiment_path = os.path.join(results_path, experiment_name)
    assert os.path.exists(experiment_path)
    checkpoints_path = os.path.join(experiment_path, "checkpoints")

    checkpoint_name = sorted(os.listdir(checkpoints_path))[-1]
    return checkpoint_name


def get_mean_metrics(
    validation_metrics: Dict[str, List[LossInfo]],
) -> Dict[str, float]:
    """
    Obtain mean over batch for each loss.

    Arguments:
        validation_metrics: Dictionary with validation metrics.

    Returns:
        validation_metrics: Dictionary with mean validation metrics.
    """
    mean_validation_metrics = defaultdict(float)
    for loss_name, info in validation_metrics.items():
        for var in info[0].keys():
            mean_validation_metrics[f"val/{loss_name}/{var}"] = np.mean(
                np.stack(
                    [info[var] for info in info],
                    axis=0,
                ),
                axis=0,
            )
    return mean_validation_metrics


def get_validation_metrics(
    device: str,
    experiment_name: str,
    results_path: str = "./_results",
    checkpoint_name: Optional[str] = None,
    lead_time: int = 4,
) -> Dict[str, float]:
    """
    Obtain validation metrics.

    Arguments:
        device: Device to run the evaluation on (e.g., 'cuda', 'cpu').
        experiment_name: Name of the existing experiment.
        results_path: Name of the directory with results.
        checkpoint_name: Name of the checkpoint you want to evaluate.
        lead_time: Number of time steps to use in the validation rollout.

    Returns:
        validation_metrics: Dictionary with validation metrics.
    """
    # Load experiment.
    experiment = initialize_evaluation(
        device,
        experiment_name=experiment_name,
        results_path=results_path,
        lead_time=lead_time,
    )
    experiment.model = experiment.model.to(device)
    # load_checkpointed_weights_from_experiment(
    #     model=experiment.model,
    #     experiment_name=experiment_name,
    #     checkpoint_name=checkpoint_name,
    # )

    # Perform validation.
    _, validation_metrics = run_validation_epoch(
        model=experiment.model,
        device=device,
        dataloader=experiment.valid_dataloader,
        loss_fns=experiment.val_loss_fns,
        scale_scheduler=experiment.inference_scale_scheduler,
    )

    mean_validation_metrics = get_mean_metrics(validation_metrics)

    return mean_validation_metrics


def main(
    experiment_name: str,
    lead_time: int,
    results_path: str = "./_results",
    checkpoint_name: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate the model.
    validation_metrics = get_validation_metrics(
        device=device,
        experiment_name=experiment_name,
        results_path=results_path,
        checkpoint_name=checkpoint_name,
        lead_time=lead_time,
    )

    # If no checkpoint is specified, get the last checkpoint name.
    if checkpoint_name is None:
        checkpoint_name = get_last_ckpt_name(experiment_name, results_path)

    # Save validation_metrics dict.
    save_path = os.path.join(results_path, experiment_name)
    save_path = os.path.join(
        save_path, (f"{checkpoint_name}_LT{lead_time}.pkl")
    )
    with open(save_path, "wb") as f:
        pickle.dump(validation_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare models."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to evaluate.",
    )
    parser.add_argument(
        "--lead_time",
        type=int,
        help="Lead time in days for evaluation.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./_results",
        help="Path to store results (default: ./_results).",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the experiment.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device.")
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        results_path=args.results_path,
        checkpoint_name=args.checkpoint_name,
        lead_time=args.lead_time,
        device=args.device,
    )

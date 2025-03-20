import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
)

import pickle
import torch

from otter.experiment.train import (
    assert_single_device,
    run_validation_epoch,
)
from otter.experiment.utils import initialize_experiment
from otter.experiment.run_state import (
    get_latest_checkpoint,
)
from otter.models.losses.loss_fn import LossInfo
from otter.plots.plot_score_card import get_time_steps


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
    results_dir: str,
    checkpoint_path: Path,
    args_list: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Obtain validation metrics.

    Arguments:
        device: Device to run the evaluation on (e.g., 'cuda', 'cpu').
        results_dir: Name of the directory with results.
        checkpoint_path: Path to the checkpoint you want to evaluate.
        lead_time: Number of time steps to use in the validation rollout.

    Returns:
        validation_metrics: Dictionary with validation metrics.
    """
    # Load experiment
    experiment = initialize_experiment(
        device,
        args_list=args_list,
        config_dir=results_dir,
    )
    experiment.model = experiment.model.to(device)

    # Load the model weights from latest checkpoint
    experiment.model.load_state_dict(
        torch.load(checkpoint_path)["model_state"]
    )

    # Perform validation
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
    results_path: str = "./_results",
    checkpoint_name: Optional[str] = None,
    device: Optional[str] = None,
    args_list: Optional[Sequence[str]] = None,
) -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the directory with the experiment results
    results_dir = os.path.join(results_path, experiment_name)
    assert os.path.exists(results_dir), (
        f"No results for experiment {experiment_name} found."
    )
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    assert os.path.exists(checkpoint_dir), (
        f"No checkpoints for experiment {experiment_name} found."
    )

    # If no checkpoint is specified, get the last checkpoint name.
    if checkpoint_name is None:
        checkpoint_path = get_latest_checkpoint(Path(checkpoint_dir))
    else:
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    # Evaluate the model.
    validation_metrics = get_validation_metrics(
        device=device,
        results_dir=os.path.abspath(results_dir),
        checkpoint_path=checkpoint_path,
        args_list=args_list,
    )

    # Save validation_metrics dict.
    num_val_steps = len(get_time_steps(validation_metrics=validation_metrics))
    save_path = os.path.join(
        results_dir, (f"{checkpoint_name}_steps-{num_val_steps}.pkl")
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

    args, extra_args = parser.parse_known_args()

    main(
        experiment_name=args.experiment_name,
        results_path=args.results_path,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        args_list=extra_args,
    )

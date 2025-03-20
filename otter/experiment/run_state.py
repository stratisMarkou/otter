import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
)

import torch
from torch.amp.grad_scaler import GradScaler


@dataclass
class TrainState:
    epoch: int
    step: int
    best_val_loss: float = float("inf")


@dataclass
class RunState:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    scaler_state: Dict[str, Any]
    epoch: int
    step: int
    best_val_loss: float = float("inf")

    @staticmethod
    def from_objects(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        epoch: int,
        step: int,
        best_val_loss: float = float("inf"),
    ) -> "RunState":
        """Create RunState from torch objects."""
        return RunState(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=lr_scheduler.state_dict(),  # type: ignore
            scaler_state=scaler.state_dict(),
            epoch=epoch,
            step=step,
            best_val_loss=best_val_loss,
        )

    @staticmethod
    def from_dict(state_dict: Dict[str, Any]) -> "RunState":
        """Create RunState from a dictionary loaded from disk."""
        return RunState(
            model_state=state_dict["model_state"],
            optimizer_state=state_dict["optimizer_state"],
            scheduler_state=state_dict["scheduler_state"],
            scaler_state=state_dict["scaler_state"],
            epoch=state_dict["epoch"],
            step=state_dict["step"],
            best_val_loss=state_dict["best_val_loss"],
        )

    @staticmethod
    def load(path: Path) -> "RunState":
        """Load RunState from disk."""
        state_dict = torch.load(path, weights_only=False)
        return RunState.from_dict(state_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RunState to a dictionary that can be saved to disk."""
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
        }

    def save(self, path: Path) -> None:
        """Save RunState to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
    ) -> None:
        """Restore state to provided objects."""
        model.load_state_dict(self.model_state)
        optimizer.load_state_dict(self.optimizer_state)
        lr_scheduler.load_state_dict(self.scheduler_state)
        scaler.load_state_dict(self.scaler_state)


def load_run_state_from_latest_checkpoint(
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
) -> TrainState:
    """
    Load the initial state of the run from the given experiment_name.

    Args:
        experiment: The experiment object, loaded from the yaml config.
        optimizer: The torch optimizer.
        lr_scheduler: The torch learning rate scheduler.
        scaler: The torch amp gradient scaler object.
    Returns:
        The TrainState object with the initial epoch and step number.
    """

    path = get_latest_checkpoint(get_checkpoint_dir(experiment))

    # Load complete run state if available
    run_state = RunState.load(path)
    run_state.restore(
        model=experiment.model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
    )
    step = run_state.step
    epoch = run_state.epoch
    best_val_loss = run_state.best_val_loss
    logging.info(f"Successfully restored state from {path} at step {step}")

    return TrainState(epoch=epoch, step=step, best_val_loss=best_val_loss)


def get_checkpoint_dir(experiment: Any) -> Path:
    return Path(experiment.info.experiment_path) / "checkpoints"


def get_latest_checkpoint(path: Path) -> Path:
    """Get the latest checkpoint file in the given directory."""
    if (path / "latest.pt").exists():
        return path / "latest.pt"
    checkpoints = list(path.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {path}")
    return max(checkpoints, key=lambda x: str(x))


def save_run_state(
    experiment: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    step: int,
    checkpoint_name: str,
    best_val_loss: float = float("inf"),
) -> None:
    """Save the complete run state to disk."""
    logging.info(f"Saving checkpoint {checkpoint_name} at step {step}")
    run_state = RunState.from_objects(
        model=experiment.model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        epoch=epoch,
        step=step,
        best_val_loss=best_val_loss,
    )
    logging.info("Checkpoint saved.")

    path = get_checkpoint_dir(experiment) / checkpoint_name
    run_state.save(path)

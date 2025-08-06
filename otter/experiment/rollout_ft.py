import logging
from typing import (
    Callable,
    Tuple,
)

import torch
from omegaconf import DictConfig
from torch.amp import GradScaler  # type: ignore
from tqdm import tqdm

import wandb

from otter.experiment.monitoring import (
    WandbLogger,
)
from otter.experiment.run_state import (
    TrainState,
    load_run_state_from_latest_checkpoint,
    save_run_state,
)
from otter.experiment.train import (
    train_step,
    assert_single_device,
    validation_epoch,
    log_num_params,
)
from otter.experiment.utils import (
    initialize_experiment,
    initialize_wandb,
    load_model_weights_from_experiment,
)
from otter.models.architectures.schedulers import (
    ForecastHorizonScheduler,
)
from otter.models.losses.loss_fn import LossInfo

LossFunction = Callable[..., Tuple[torch.Tensor, LossInfo]]


def main() -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Initializing experiment.")
    experiment, config = initialize_experiment(device)
    initialize_wandb(config)
    logging.info(f"Initialized experiment {experiment.info.experiment_name}")

    experiment.model = experiment.model.to(device)

    optimizer = experiment.optimizer(experiment.model)
    lr_scheduler = experiment.lr_scheduler(optimizer)
    forecast_horizon_scheduler = experiment.forecast_horizon_scheduler

    scaler = GradScaler(device=device)

    if experiment.constants.load_weights_from_experiment:
        logging.info("Loading weights from another experiment.")
        experiment.model = load_model_weights_from_experiment(
            path=experiment.constants.load_weights_from_experiment,
            model=experiment.model,
        )

    if experiment.info.start_from_checkpoint:
        logging.info("Loading initial state from checkpoint.")
        initial_state = load_run_state_from_latest_checkpoint(
            experiment=experiment,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            forecast_horizon_scheduler=forecast_horizon_scheduler,
        )
        logging.info("Successfully loaded initial state.")
    else:
        initial_state = TrainState(epoch=0, step=0)

    logging.info(f"Starting from: {initial_state}")

    rollout_fine_tuning_loop(
        step=initial_state.step,
        experiment=experiment,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        forecast_horizon_scheduler=forecast_horizon_scheduler,
        best_val_loss=initial_state.best_val_loss,
    )

    wandb.finish()


def update_optimizer_and_scheduler(
    experiment: DictConfig,
    forecast_horizon_scheduler: ForecastHorizonScheduler,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer_params_updates = (
        forecast_horizon_scheduler.get_current_optimizer_params()
    )
    optimizer = experiment.optimizer(
        experiment.model, **optimizer_params_updates
    )

    lr_scheduler_params_updates = (
        forecast_horizon_scheduler.get_current_lr_scheduler_params()
    )
    lr_scheduler = experiment.lr_scheduler(
        optimizer, **lr_scheduler_params_updates
    )

    return optimizer, lr_scheduler


def rollout_fine_tuning_loop(
    step: int,
    experiment: DictConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    forecast_horizon_scheduler: ForecastHorizonScheduler,
    best_val_loss: float = float("inf"),
) -> None:
    """
    The main training loop for the experiment.
    """
    mixed_precision_dtype = experiment.constants.training.mixed_precision_dtype
    clip_grad_norm = experiment.constants.training.clip_grad_norm

    num_training_steps = experiment.constants.training.num_training_steps

    log_num_params(experiment.model)

    train_loader = iter(tqdm(experiment.train_dataloader))

    while step < num_training_steps:
        wandb_logger = WandbLogger(step)
        if step % experiment.constants.training.validate_every == 0:
            current_val_loss = validation_epoch(
                model=experiment.model,
                device=device,
                dataloader=experiment.valid_dataloader,
                loss_fns=experiment.val_loss_fns,
                scale_scheduler=experiment.inference_scale_scheduler,
                wandb_logger=wandb_logger,
                mixed_precision_dtype=mixed_precision_dtype,
            )

            if (
                experiment.constants.save_best_checkpoint
                and current_val_loss < best_val_loss
            ):
                best_val_loss = current_val_loss

                save_run_state(
                    experiment=experiment,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=step // len(experiment.train_dataloader),
                    step=step,
                    best_val_loss=best_val_loss,
                    checkpoint_name="best.pt",
                    forecast_horizon_scheduler=forecast_horizon_scheduler,
                )

        try:
            wandb_logger.timed(
                "train_step",
                train_step,
                loss_fn=experiment.loss_fn,
                model=experiment.model,
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                scaler=scaler,
                scale_scheduler=experiment.train_scale_scheduler,
                wandb_logger=wandb_logger,
                clip_grad_norm=clip_grad_norm,
                mixed_precision_dtype=mixed_precision_dtype,
                forecast_horizon_scheduler=forecast_horizon_scheduler,
                grad_ckpt_start_timestep=experiment.constants.training.grad_ckpt_start_timestep,
            )

            wandb_logger.log(
                {
                    "misc/forecast_horizon": forecast_horizon_scheduler.current_frame_forecast_horizon
                }
            )

            increase_horizon = forecast_horizon_scheduler.step()
            if increase_horizon:
                logging.info(
                    f"Step {step}: Increasing forecast horizon to {forecast_horizon_scheduler.current_frame_forecast_horizon}."
                )
                optimizer, lr_scheduler = update_optimizer_and_scheduler(
                    experiment=experiment,
                    forecast_horizon_scheduler=forecast_horizon_scheduler,
                )
            step += 1

        except StopIteration:
            train_loader = iter(tqdm(experiment.train_dataloader))
            continue

        if step % experiment.constants.save_checkpoint_every == 0:
            save_run_state(
                experiment=experiment,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=step // len(experiment.train_dataloader),
                step=step,
                best_val_loss=best_val_loss,
                checkpoint_name="latest.pt",
                forecast_horizon_scheduler=forecast_horizon_scheduler,
            )

        interval = experiment.constants.save_persistent_checkpoint_every
        if step % interval == 0:
            save_run_state(
                experiment=experiment,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=step // len(experiment.train_dataloader),
                step=step,
                best_val_loss=best_val_loss,
                checkpoint_name=f"step_{step:08d}.pt",
                forecast_horizon_scheduler=forecast_horizon_scheduler,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

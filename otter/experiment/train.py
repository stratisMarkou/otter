import logging
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

import numpy as np
import torch
from omegaconf import DictConfig
from torch.amp import GradScaler  # type: ignore
from tqdm import tqdm

import wandb
from otter.data.datasets import ForecastingSample
from otter.experiment.downsample import downsample
from otter.experiment.run_state import (
    TrainState,
    load_run_state_from_latest_checkpoint,
    save_run_state,
)
from otter.experiment.utils import (
    initialize_experiment,
    initialize_wandb,
    log_loss_info_to_wandb,
    run_and_log_time_to_wandb,
)
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.losses.loss_fn import (
    LossInfo,
    gather_load_balancing_losses,
)
from otter.models.multiscale_model import MultiscaleInput

LossFunction = Callable[..., Tuple[torch.Tensor, LossInfo]]


def assert_single_device() -> None:
    if torch.cuda.device_count() != 1:
        raise Exception(
            "Single-device mode only is currently supported. Ensure exactly "
            "one device is available using the CUDA_VISIBLE_DEVICES "
            "environment variable."
        )


def train_step(
    model: torch.nn.Module,
    loss_fn: LossFunction,
    train_loader: Iterable[ForecastingSample],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    step_number: int,
    scaler: GradScaler,
    scale_scheduler: ScaleScheduler,
    mixed_precision_dtype: torch.dtype,
    device: str = "cuda",
) -> int:
    batch = run_and_log_time_to_wandb("make_next_batch", next, train_loader)
    batch = run_and_log_time_to_wandb(
        "batch.to", batch.to, device, non_blocking=True
    )

    data_scale = scale_scheduler.sample_data_scale()
    processed_batch = run_and_log_time_to_wandb(
        "preprocess_batch",
        preprocess_batch,
        batch=batch,
        scale=data_scale,
    )
    processed_batch = processed_batch.to(device)

    rollout_scales = scale_scheduler.get_rollout_scales()

    # Do forward pass with AMP autocasting. Only forward pass needs to be
    # autocasted, as the backward is computed with matching dtypes to the
    # forward pass: https://pytorch.org/docs/stable/amp.html#torch.autocast
    with torch.amp.autocast(  # type: ignore
        device_type=device,
        dtype=mixed_precision_dtype,
    ):
        prd_stats, load_balancing_losses = run_and_log_time_to_wandb(
            "model.forward",
            model.forward,
            processed_batch,
            scales=rollout_scales,
        )

        loss, loss_info = run_and_log_time_to_wandb(
            "loss_fn",
            loss_fn,
            prd_stats=prd_stats,
            trg=batch.trg,
            scales=_combine_scales(data_scale, rollout_scales),
        )

        load_balancing_loss, load_balancing_loss_info = (
            run_and_log_time_to_wandb(
                "gather_load_balancing_losses",
                gather_load_balancing_losses,
                load_balancing_losses=load_balancing_losses,
            )
        )

        if load_balancing_loss is not None:
            loss += load_balancing_loss

        if load_balancing_loss_info is not None:
            loss_info.update(load_balancing_loss_info)

    scaled_loss = run_and_log_time_to_wandb("scaler.scale", scaler.scale, loss)
    run_and_log_time_to_wandb("scaled_loss.backward", scaled_loss.backward)
    run_and_log_time_to_wandb("scaler.step", scaler.step, optimizer)
    run_and_log_time_to_wandb("scaler.update", scaler.update)
    run_and_log_time_to_wandb("optimizer.zero_grad", optimizer.zero_grad)
    run_and_log_time_to_wandb("scheduler.step", lr_scheduler.step)

    log_loss_info_to_wandb(
        loss_info=loss_info,
        prefix="train",
    )

    current_lr = lr_scheduler.get_last_lr()[0]
    wandb.log({"misc/learning_rate": current_lr}, commit=False)
    wandb.log({"misc/gradient_step": step_number}, commit=True)

    return step_number + 1


def preprocess_batch(batch: ForecastingSample, scale: int) -> MultiscaleInput:
    """
    Preprocess a batch of forecasting samples for input to the multiscale
    model.
    """
    downsampled_trg = downsample(batch.trg, scale)

    return MultiscaleInput(
        ctx=batch.ctx,
        zero_time=batch.zero_time,
        downsampled_trg=downsampled_trg,
        scale=scale,
        trg_timedelta_hours=batch.trg_timedelta_hours,
    )


def _combine_scales(
    data_scale: int, rollout_scales: torch.Tensor
) -> torch.Tensor:
    """
    Combine the data scale and rollout scales into a single tensor.
    """
    data_scale_tensor = torch.tensor([data_scale], dtype=torch.int32)
    return torch.cat([data_scale_tensor, rollout_scales])


def validation_epoch(
    model: torch.nn.Module,
    device: str,
    dataloader: Iterable[ForecastingSample],
    loss_fns: Dict[str, LossFunction],
    scale_scheduler: ScaleScheduler,
) -> float:
    # Ensure model is in evaluation mode.
    model = model.eval()

    with torch.no_grad():
        losses: Dict[str, List[float]] = defaultdict(list)
        loss_infos: Dict[str, List[LossInfo]] = defaultdict(list)
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            data_scale = scale_scheduler.sample_data_scale()
            rollout_scales = scale_scheduler.get_rollout_scales()
            all_scales = _combine_scales(data_scale, rollout_scales)
            # Test that the largest scale is greater than the number of lats,
            # ensuring the total erasure case is covered.
            assert torch.max(all_scales) > batch.trg.shape[1], (
                f"Maximum scale {torch.max(all_scales)} must be greater than "
                f"the number of lats {batch.trg.shape[1]}"
            )
            preprocessed_batch = preprocess_batch(batch, data_scale)
            preprocessed_batch = preprocessed_batch.to(device)

            prd_stats, load_balancing_losses = model(
                preprocessed_batch, scales=rollout_scales
            )

            # Compute the loss for each loss function and store results.
            for loss_name, loss_fn in loss_fns.items():
                if loss_name not in losses:
                    losses[loss_name] = []
                    loss_infos[loss_name] = []

                loss, loss_info = loss_fn(
                    trg=batch.trg, prd_stats=prd_stats, scales=all_scales
                )

                if loss_name == "train_loss":
                    load_loss, load_loss_info = gather_load_balancing_losses(
                        load_balancing_losses=load_balancing_losses
                    )
                    if load_loss is not None:
                        loss += load_loss
                    if load_loss_info is not None:
                        loss_info.update(load_loss_info)

                losses[loss_name].append(loss.item())
                loss_infos[loss_name].append(loss_info)

    # Average the losses over the validation set.
    for loss_name, loss_values in losses.items():
        mean_loss = np.mean(np.stack(loss_values, axis=0), axis=0)
        wandb.log({f"val/{loss_name}": mean_loss}, commit=False)

    # Average the loss info over the validation set.
    for loss_name, info in loss_infos.items():
        for var in info[0].keys():
            mean_info = np.mean(
                np.stack(
                    [info[var] for info in info],
                    axis=0,
                ),
                axis=0,
            )
            wandb.log(
                {f"val/{loss_name}/{var}": mean_info},
                commit=False,
            )

    # Ensure model is back in training mode after validation.
    model.train()

    train_val_loss: float = np.mean(
        np.stack(losses["train_loss"], axis=0), axis=0
    ).item()

    return train_val_loss


def main() -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Initializing experiment.")
    experiment = initialize_experiment(device)
    initialize_wandb(experiment.info.experiment_name)
    logging.info(f"Initialized experiment {experiment.info.experiment_name}")

    experiment.model = experiment.model.to(device)
    optimizer = experiment.optimizer(experiment.model)
    lr_scheduler = experiment.lr_scheduler(optimizer)

    scaler = GradScaler(device=device)

    if experiment.info.start_from_checkpoint:
        logging.info("Loading initial state from checkpoint.")
        initial_state = load_run_state_from_latest_checkpoint(
            experiment=experiment,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
        )
        logging.info("Successfully loaded initial state.")
    else:
        initial_state = TrainState(epoch=0, step=0)

    logging.info(f"Starting from: {initial_state}")

    train_loop(
        start_epoch=initial_state.epoch,
        step=initial_state.step,
        experiment=experiment,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        best_val_loss=initial_state.best_val_loss,
    )

    wandb.finish()


def train_loop(
    start_epoch: int,
    step: int,
    experiment: DictConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    best_val_loss: float = float("inf"),
) -> None:
    """
    The main training loop for the experiment.
    """
    mixed_precision_dtype = experiment.constants.training.mixed_precision_dtype
    for epoch in range(start_epoch, experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:
            if step % experiment.constants.training.validate_every == 0:
                current_val_loss = validation_epoch(
                    model=experiment.model,
                    device=device,
                    dataloader=experiment.valid_dataloader,
                    loss_fns=experiment.val_loss_fns,
                    scale_scheduler=experiment.inference_scale_scheduler,
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
                        epoch=epoch,
                        step=step,
                        best_val_loss=best_val_loss,
                        checkpoint_name="best.pt",
                    )

            try:
                step = run_and_log_time_to_wandb(
                    "train_step",
                    train_step,
                    loss_fn=experiment.loss_fn,
                    model=experiment.model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step_number=step,
                    device=device,
                    scaler=scaler,
                    scale_scheduler=experiment.train_scale_scheduler,
                    mixed_precision_dtype=mixed_precision_dtype,
                )
            except StopIteration:
                break

            if step % experiment.constants.save_checkpoint_every == 0:
                save_run_state(
                    experiment=experiment,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    best_val_loss=best_val_loss,
                    checkpoint_name="latest.pt",
                )

            interval = experiment.constants.save_persistent_checkpoint_every
            if step % interval == 0:
                save_run_state(
                    experiment=experiment,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    best_val_loss=best_val_loss,
                    checkpoint_name=f"step_{step:08d}.pt",
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

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
from otter.eval_utils.power_spectrum import power_spectrum_density_mse
from otter.experiment.downsample import downsample
from otter.experiment.monitoring import (
    WandbLogger,
    log_loss_info_to_wandb,
)
from otter.experiment.run_state import (
    TrainState,
    load_run_state_from_latest_checkpoint,
    save_run_state,
)
from otter.experiment.utils import initialize_experiment, initialize_wandb
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.losses.loss_fn import (
    LossInfo,
    LossInfoArray,
    gather_array_losses,
    gather_load_balancing_losses,
)
from otter.models.multiscale_model import MultiscaleInput, MultiscaleModel

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
    scaler: GradScaler,
    scale_scheduler: ScaleScheduler,
    wandb_logger: WandbLogger,
    clip_grad_norm: float,
    mixed_precision_dtype: torch.dtype,
    device: str = "cuda",
) -> None:
    w = wandb_logger
    batch = w.timed("make_next_batch", next, train_loader)
    batch = w.timed("batch.to", batch.to, device, non_blocking=True)

    data_scale = scale_scheduler.sample_data_scale()
    processed_batch = w.timed(
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
        prd_dist, load_balancing_losses = w.timed(
            "model.forward",
            model.forward,
            processed_batch,
            scales=rollout_scales,
        )

        loss, loss_info = w.timed(
            "loss_fn",
            loss_fn,
            prd_dist=prd_dist,
            trg=batch.trg,
            scales=_combine_scales(data_scale, rollout_scales),
        )

        load_balancing_loss, load_balancing_loss_info = w.timed(
            "gather_load_balancing_losses",
            gather_load_balancing_losses,
            load_balancing_losses=load_balancing_losses,
        )

        if load_balancing_loss is not None:
            loss += load_balancing_loss

    if load_balancing_loss_info is not None:
        loss_info.update(load_balancing_loss_info)

    scaled_loss = w.timed("scaler.scale", scaler.scale, loss)
    w.timed("scaled_loss.backward", scaled_loss.backward)

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    w.timed("scaler.step", scaler.step, optimizer)
    w.timed("scaler.update", scaler.update)
    w.timed("optimizer.zero_grad", optimizer.zero_grad)
    w.timed("scheduler.step", lr_scheduler.step)

    log_loss_info_to_wandb(loss_info=loss_info, prefix="train")

    current_lr = lr_scheduler.get_last_lr()[0]
    w.log({"misc/learning_rate": current_lr})
    w.log({"misc/gradient_scaler.scale": scaler.get_scale()})


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


@torch.no_grad()
def validation_epoch(
    model: MultiscaleModel,
    device: str,
    dataloader: Iterable[ForecastingSample],
    loss_fns: Dict[str, LossFunction],
    scale_scheduler: ScaleScheduler,
    wandb_logger: WandbLogger,
    mixed_precision_dtype: torch.dtype,
) -> float:
    # Run the validation epoch
    losses, losses_info, _ = run_validation_epoch(
        model=model,
        device=device,
        dataloader=dataloader,
        loss_fns=loss_fns,
        scale_scheduler=scale_scheduler,
        mixed_precision_dtype=mixed_precision_dtype,
    )

    # Log metrics to wandb
    log_validation_metrics_to_wandb(
        losses, losses_info, wandb_logger=wandb_logger
    )

    train_val_loss: float = np.mean(
        np.stack(losses["train_loss"], axis=0), axis=0
    ).item()

    return train_val_loss


@torch.no_grad()
def run_validation_epoch(
    model: torch.nn.Module,
    device: str,
    dataloader: Iterable[ForecastingSample],
    loss_fns: Dict[str, LossFunction],
    scale_scheduler: ScaleScheduler,
    mixed_precision_dtype: torch.dtype,
    compute_power_spectrum_mse: bool = False,
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[LossInfo]],
    Dict[str, List[LossInfoArray]],
]:
    # Ensure model is in evaluation mode.
    model = model.eval()

    with torch.amp.autocast(  # type: ignore
        device_type=device,
        dtype=mixed_precision_dtype,
    ):
        losses: Dict[str, List[float]] = defaultdict(list)
        loss_infos: Dict[str, List[LossInfo]] = defaultdict(list)

        # Dict to store metrics which are arrays, e.g. power spectrum MSE.
        metrics_array: Dict[str, List[LossInfoArray]] = defaultdict(list)

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

            prd_dist, load_balancing_losses = model(
                preprocessed_batch, scales=rollout_scales
            )

            # Compute the loss for each loss function and store results.
            for loss_name, loss_fn in loss_fns.items():
                if loss_name not in losses:
                    losses[loss_name] = []
                    loss_infos[loss_name] = []

                loss, loss_info = loss_fn(
                    trg=batch.trg, prd_dist=prd_dist, scales=all_scales
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

            if compute_power_spectrum_mse:
                psd_mean_mse_per_batch = power_spectrum_density_mse(
                    trg=batch.trg,
                    prd_dist=prd_dist,
                ).cpu()

                psd_mean_mse_per_batch_per_var = gather_array_losses(
                    array_losses=psd_mean_mse_per_batch,
                    trg_variables_and_levels=model.trg_variables_and_levels,
                )
                metrics_array["power_spectrum_mse"].append(
                    psd_mean_mse_per_batch_per_var
                )

    # Ensure model is back in training mode after validation.
    model.train()

    return losses, loss_infos, metrics_array


def log_validation_metrics_to_wandb(
    losses: Dict[str, List[float]],
    loss_infos: Dict[str, List[LossInfo]],
    wandb_logger: WandbLogger,
) -> None:
    # Average the losses over the validation set.
    for loss_name, loss_values in losses.items():
        mean_loss = np.mean(np.stack(loss_values, axis=0), axis=0)
        wandb_logger.log({f"val/{loss_name}": mean_loss})

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
            wandb_logger.log({f"val/{loss_name}/{var}": mean_info})


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
    experiment.model = torch.compile(experiment.model)
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


def log_num_params(model: torch.nn.Module) -> int:
    """
    Compute the number of parameters in a model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params / 1e6:.2f}M")
    return num_params


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
    clip_grad_norm = experiment.constants.training.clip_grad_norm

    log_num_params(experiment.model)

    for epoch in range(start_epoch, experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:
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
                        epoch=epoch,
                        step=step,
                        best_val_loss=best_val_loss,
                        checkpoint_name="best.pt",
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
                )
                step += 1
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

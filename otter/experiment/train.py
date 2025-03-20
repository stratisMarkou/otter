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
from otter.experiment.monitoring import (
    WandbLogger,
    get_min_stds_by_var,
    get_norm_and_abs_max,
    log_loss_info_to_wandb,
)
from otter.experiment.run_state import (
    TrainState,
    load_run_state_from_latest_checkpoint,
    save_run_state,
)
from otter.experiment.utils import initialize_experiment, initialize_wandb
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.distributions import Gaussian
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
    step: int,
    scaler: GradScaler,
    scale_scheduler: ScaleScheduler,
    wandb_logger: WandbLogger,
    device: str = "cuda",
) -> int:
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

    prd_dist, load_balancing_losses = w.timed(
        "model.forward",
        model.forward,
        processed_batch,
        scales=rollout_scales,
    )

    if isinstance(prd_dist, Gaussian):
        smallest_stds_by_var = get_min_stds_by_var(
            prd_dist, model.trg_variables_and_levels
        )
        w.log(smallest_stds_by_var)

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

    param_norm, biggest_abs_param = get_norm_and_abs_max(model, "parameters")

    w.log({"misc/biggest_abs_param": biggest_abs_param})
    w.log({"misc/parameter_norm": param_norm})

    grad_norm, biggest_abs_grad = get_norm_and_abs_max(model, "gradients")

    w.log({"misc/biggest_abs_grad": biggest_abs_grad})
    w.log({"misc/gradient_norm": grad_norm})

    w.timed("scaler.step", scaler.step, optimizer)
    w.timed("scaler.update", scaler.update)
    w.timed("optimizer.zero_grad", optimizer.zero_grad)
    w.timed("scheduler.step", lr_scheduler.step)

    log_loss_info_to_wandb(loss_info=loss_info, prefix="train")

    current_lr = lr_scheduler.get_last_lr()[0]
    w.log({"misc/learning_rate": current_lr})
    w.log({"misc/gradient_step": step})
    w.log({"misc/gradient_scaler.scale": scaler.get_scale()})

    return step + 1


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
    step: int,
) -> None:
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

    # Average the losses over the validation set.
    for loss_name, loss_values in losses.items():
        mean_loss = np.mean(np.stack(loss_values, axis=0), axis=0)
        wandb.log({f"val/{loss_name}": mean_loss}, commit=False, step=step)

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
                {f"val/{loss_name}/{var}": mean_info}, commit=False, step=step
            )

    # Ensure model is back in training mode after validation.
    model.train()


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

    wandb.watch(
        experiment.model,
        log_freq=experiment.constants.training.log_gradients_every,
    )

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
) -> None:
    """
    The main training loop for the experiment.
    """
    for epoch in range(start_epoch, experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:
            if step % experiment.constants.training.validate_every == 0:
                validation_epoch(
                    model=experiment.model,
                    device=device,
                    dataloader=experiment.valid_dataloader,
                    loss_fns=experiment.val_loss_fns,
                    scale_scheduler=experiment.inference_scale_scheduler,
                    step=step,
                )

            try:
                wandb_logger = WandbLogger(step)
                step = wandb_logger.timed(
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
                    wandb_logger=wandb_logger,
                )
            except StopIteration:
                break

            if step % experiment.constants.training.save_checkpoint_every == 0:
                logging.info("Saving checkpoint.")
                save_run_state(
                    experiment=experiment,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                )
                logging.info("Checkpoint saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

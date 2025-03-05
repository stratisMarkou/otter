import os
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
from torch.amp import GradScaler  # type: ignore
from tqdm import tqdm

import wandb
from otter.data.datasets import ForecastingSample
from otter.experiment.downsample import downsample
from otter.experiment.utils import (
    create_optimizer_and_lr_scheduler,
    initialize_experiment,
    log_code_to_wandb,
    log_loss_info_to_wandb,
    run_and_log_time_to_wandb,
)
from otter.models.architectures.schedulers import ScaleScheduler
from otter.models.losses.loss_fn import (
    LossInfo,
    PredictionStatistics,
)
from otter.models.multiscale_model import MultiscaleInput

LossFunction = Callable[[PredictionStatistics], Tuple[torch.Tensor, LossInfo]]


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

    scales = scale_scheduler.get_scales()
    processed_batch = run_and_log_time_to_wandb(
        "preprocess_batch",
        preprocess_batch,
        batch=batch,
        scale=scales[0],
    )
    processed_batch = processed_batch.to(device)

    # Do forward pass with AMP autocasting. Only forward pass needs to be
    # autocasted, as the backward is computed with matching dtypes to the
    # forward pass: https://pytorch.org/docs/stable/amp.html#torch.autocast
    with torch.amp.autocast(  # type: ignore
        device_type=device,
        dtype=mixed_precision_dtype,
    ):
        prd_stats = run_and_log_time_to_wandb(
            "model.forward",
            model.forward,
            processed_batch,
            scales=scales[1:],
        )

    loss, loss_info = run_and_log_time_to_wandb(
        "loss_fn",
        loss_fn,
        prd_stats=prd_stats,
        trg=batch.trg,
        scales=scales,
    )

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


def validation_epoch(
    model: torch.nn.Module,
    device: str,
    dataloader: Iterable[ForecastingSample],
    loss_fns: Dict[str, LossFunction],
    scale_scheduler: ScaleScheduler,
) -> None:
    # Ensure model is in evaluation mode.
    model = model.eval()

    with torch.no_grad():
        losses = defaultdict(list)
        loss_infos: Dict[str, List[LossInfo]] = defaultdict(list)
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            scales = scale_scheduler.get_scales()
            # Test that the largest scale is greater than the number of lats,
            # ensuring the total erasure case is covered.
            assert max(scales) > batch.trg.shape[1], (
                f"Maximum scale {max(scales)} must be greater than the number "
                f"of lats {batch.trg.shape[1]}"
            )
            preprocessed_batch: MultiscaleInput = preprocess_batch(
                batch, int(scales[0])
            )
            preprocessed_batch = preprocessed_batch.to(device)

            prd_stats = model(preprocessed_batch, scales=scales[1:])

            # Compute the loss for each loss function and store results.
            for loss_name, loss_fn in loss_fns.items():
                loss, loss_info = loss_fn(  # type: ignore
                    prd_stats=prd_stats,
                    trg=batch.trg,
                    scales=scales,
                )

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


def load_checkpointed_weights_from_experiment(
    model: torch.nn.Module,
    experiment_name: str,
    experiments_path: str = "./_results",
    checkpoint_name: Optional[str] = None,
) -> None:
    # Get list of all files in the directory.
    experiment_path = os.path.join(experiments_path, experiment_name)
    assert os.path.exists(experiment_path)
    checkpoints_path = os.path.join(experiment_path, "checkpoints")

    if checkpoint_name is None:
        # Get latest checkpoint.
        checkpoint_name = sorted(os.listdir(checkpoints_path))[-1]

    checkpoint_path = os.path.join(checkpoints_path, checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path))


def main() -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb.
    wandb.init(project="otter-deterministic", entity="otter-weather")
    log_code_to_wandb()

    # Initialize experiment, including data loaders, model, and optimizer.
    assert wandb.run is not None
    experiment = initialize_experiment(device, experiment_name=wandb.run.name)
    experiment.model = experiment.model.to(device)

    optimizer, lr_scheduler = create_optimizer_and_lr_scheduler(
        module=experiment.model,
        optimizer_kwargs=experiment.optimizer_kwargs,
        lr_scheduler_kwargs=experiment.lr_scheduler_kwargs,
    )

    if experiment.constants.load_weights_from_experiment:
        load_checkpointed_weights_from_experiment(
            model=experiment.model,
            experiment_name=experiment.constants.load_weights_from_experiment,
        )

    scaler = GradScaler(device=device)
    mixed_precision_dtype = experiment.constants.training.mixed_precision_dtype
    step = 0
    for _ in range(experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:
            if step % experiment.constants.training.validate_every == 0:
                validation_epoch(
                    model=experiment.model,
                    device=device,
                    dataloader=experiment.valid_dataloader,
                    loss_fns=experiment.val_loss_fns,
                    scale_scheduler=experiment.inference_scale_scheduler,
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
            step = step + 1

            if step % experiment.constants.training.save_checkpoint_every == 0:
                checkpoint_path = f"{experiment.info.path_results}/checkpoints"
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(
                    experiment.model.state_dict(),
                    f"{checkpoint_path}/step={step:07d}",
                )

    wandb.finish()


if __name__ == "__main__":
    main()

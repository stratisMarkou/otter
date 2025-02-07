import os
from typing import (
    Any,
    Callable,
    Iterable,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from tqdm import tqdm

import wandb
from otter.experiment.utils import (
    create_optimizer_and_lr_scheduler,
    initialize_experiment,
    load_std_dataset,
    log_code_to_wandb,
    log_loss_info_to_wandb,
    log_per_variable_per_lead_time_losses_to_wandb,
    run_and_log_time_to_wandb,
)
from otter.models.losses.loss_fn import (
    LossInfo,
    PredictionStatistics,
)


def assert_single_device() -> None:
    if torch.cuda.device_count() != 1:
        raise Exception(
            "Single-device mode only is currently supported. Ensure exactly "
            "one device is available using the CUDA_VISIBLE_DEVICES "
            "environment variable."
        )


def prepare_tensors_zero_time_and_trg_timedelta_hours(
    batch: Any,
    device: str,
) -> Tuple[
    torch.Tensor,
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    torch.Tensor,
]:
    ctx_tensor = torch.stack(
        [torch.tensor(example.ctx.data).to(device) for example in batch],
        dim=0,
    )
    ctx_times = np.stack(
        [example.ctx.coords[0].values for example in batch], axis=0
    )
    trg_tensor = torch.stack(
        [torch.tensor(example.trg.data).to(device) for example in batch],
        dim=0,
    )
    trg_times = np.stack(
        [example.trg.coords[0].values for example in batch], axis=0
    )
    ctx_tensor = ctx_tensor.permute(
        0, 3, 2, 1, 4
    )  # (batch, lat, lon, time, var_and_level)
    trg_tensor = trg_tensor.permute(
        0, 3, 2, 1, 4
    )  # (batch, lat, lon, time, var_and_level)

    zero_time = np.max(ctx_times, axis=-1)  # (batch,)
    trg_timedelta = trg_times - zero_time[:, None]
    trg_timedelta_hours = trg_timedelta.astype("timedelta64[h]").astype(int)

    return ctx_tensor, zero_time, trg_timedelta_hours, trg_tensor


def train_step(
    model: torch.nn.Module,
    loss_fn: Callable[
        [PredictionStatistics],
        Tuple[torch.Tensor, LossInfo],
    ],
    train_loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    step_number: int,
    std_dataset: xr.DataArray,
    device: str = "cuda",
) -> int:

    model = model.to(device)
    batch = run_and_log_time_to_wandb("make_next_batch", next, train_loader)
    ctx, zero_time, trg_timedelta_hours, trg = (
        prepare_tensors_zero_time_and_trg_timedelta_hours(batch, device)
    )
    prd_stats = run_and_log_time_to_wandb(
        "model.forward",
        model.forward,
        ctx=ctx,
        zero_time=zero_time,
        trg_timedelta_hours=trg_timedelta_hours,
        device=device,
    )

    loss, loss_info = run_and_log_time_to_wandb(
        "loss_fn",
        loss_fn,
        prd_stats=prd_stats,
        trg=trg,
    )

    run_and_log_time_to_wandb("loss.backward", loss.backward)
    run_and_log_time_to_wandb("optimizer.step", optimizer.step)
    run_and_log_time_to_wandb("optimizer.zero_grad", optimizer.zero_grad)
    run_and_log_time_to_wandb("scheduler.step", lr_scheduler.step)

    log_loss_info_to_wandb(
        loss_info=loss_info,
        prefix="train",
    )

    current_lr = lr_scheduler.get_last_lr()[0]
    wandb.log({"train/loss": loss.detach().cpu().numpy()}, commit=False)
    wandb.log({"misc/learning_rate": current_lr}, commit=False)
    wandb.log({"misc/gradient_step": step_number}, commit=True)

    return step_number + 1


def validation_epoch(
    model: torch.nn.Module,
    device: str,
    dataloader: Iterable[Any],
    std_dataset: xr.DataArray,
) -> None:

    # Ensure model is in evaluation mode.
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        losses = []
        loss_infos = []
        for batch in tqdm(dataloader):
            # Loss and loss info contain a time index dimension
            # as the first dimension.
            loss, loss_info = model.loss(
                batch=batch,
                device=device,
                mean_over_time_dimension=False,
            )
            losses.append(loss.detach().cpu().numpy())
            loss_infos.append(loss_info)

    # For each lead time index collect the mean loss and loss info,
    # averaged over the batch dimension.
    mean_loss = np.mean(np.stack(losses, axis=0), axis=0)
    mean_loss_info = {
        key: np.mean(
            np.stack([info[key] for info in loss_infos], axis=0),
            axis=0,
        )
        for key in loss_infos[0].keys()
    }

    log_per_variable_per_lead_time_losses_to_wandb(
        mean_loss=mean_loss,
        mean_loss_info=mean_loss_info,
        std_dataset=std_dataset,
    )

    # Ensure model is back in training mode.
    model.train()


def main() -> None:

    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize experiment, including data loaders, model, and optimizer.
    experiment = initialize_experiment()

    # Initialize wandb.
    wandb.init(project=experiment.info.wandb_project)
    log_code_to_wandb()

    std_dataset = load_std_dataset()
    optimizer, lr_scheduler = create_optimizer_and_lr_scheduler(
        module=experiment.model,
        optimizer_kwargs=experiment.optimizer_kwargs,
        lr_scheduler_kwargs=experiment.lr_scheduler_kwargs,
    )

    step = 0
    for _ in range(experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:

            # if step % experiment.constants.training.validate_every == 0:
            #     validation_epoch(
            #         model=experiment.model,
            #         device=device,
            #         dataloader=experiment.valid_dataloader,
            #         std_dataset=std_dataset,
            #     )

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
                    std_dataset=std_dataset,
                    device=device,
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

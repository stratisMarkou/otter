import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

import wandb
from otter.experiment.utils import (
    count_parameters,
    create_optimizer_and_lr_scheduler,
    initialize_experiment,
    log_code_to_wandb,
    log_loss_info_to_wandb,
    run_and_log_time_to_wandb,
)
from otter.models.losses.loss_fn import (
    LossInfo,
    PredictionStatistics,
)

LossFunction = Callable[[PredictionStatistics], Tuple[torch.Tensor, LossInfo]]


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
    loss_fn: LossFunction,
    train_loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    step_number: int,
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
    loss_fns: Dict[str, LossFunction],
) -> None:

    # Ensure model is in evaluation mode.
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        losses: Dict[str, List[npt.NDArray[np.float32]]] = dict()
        loss_infos: Dict[str, List[Dict[str, npt.NDArray[np.float32]]]] = (
            dict()
        )
        for batch in tqdm(dataloader):

            # Prepare tensors for the batch.
            ctx, zero_time, trg_timedelta_hours, trg = (
                prepare_tensors_zero_time_and_trg_timedelta_hours(
                    batch,
                    device,
                )
            )
            prd_stats = model(
                ctx=ctx,
                zero_time=zero_time,
                trg_timedelta_hours=trg_timedelta_hours,
                device=device,
            )

            # Compute the loss for each loss function and store results.
            for loss_name, loss_fn in loss_fns.items():
                if loss_name not in losses:
                    losses[loss_name] = []
                    loss_infos[loss_name] = []

                loss, loss_info = loss_fn(  # type: ignore
                    prd_stats=prd_stats,
                    trg=trg,
                )

                losses[loss_name].append(loss.cpu().detach().numpy())
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

    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb.
    wandb.init(project="otter-deterministic", entity="otter-weather")
    log_code_to_wandb()

    # Initialize experiment, including data loaders, model, and optimizer.
    assert wandb.run is not None
    experiment = initialize_experiment(experiment_name=wandb.run.name)
    print(f"Model has {count_parameters(experiment.model)} parameters")

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

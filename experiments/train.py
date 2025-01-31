from typing import Iterable

import torch
import xarray as xr
from tqdm import tqdm
from utils import (
    initialize_experiment,
    load_std_dataset,
    log_code_to_wandb,
    log_per_variable_losses_to_wandb,
    run_and_log_time_to_wandb,
)

import wandb


def assert_single_device():
    if torch.cuda.device_count() != 1:
        raise Exception(
            "Single-device mode only is currently supported. Ensure exactly "
            "one device is available using the CUDA_VISIBLE_DEVICES "
            "environment variable."
        )


def train_step(
    model: torch.nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    step_number: int,
    std_dataset: xr.Dataset,
    device: str = "cuda",
) -> int:

    model = model.to(device)
    batch = run_and_log_time_to_wandb("make_next_batch", next, train_loader)
    loss, loss_info = run_and_log_time_to_wandb(
        "model.loss",
        model.loss,
        batch=batch,
        device=device,
        mean_over_time_dimension=True,
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # run_and_log_time_to_wandb("loss.backward", loss.backward)
    # run_and_log_time_to_wandb("optimizer.step", optimizer.step)
    # run_and_log_time_to_wandb("optimizer.zero_grad", optimizer.zero_grad)
    # run_and_log_time_to_wandb("scheduler.step", lr_scheduler.step)

    log_per_variable_losses_to_wandb(
        trg_variables_and_levels=model.trg_variables_and_levels,
        loss_info=loss_info,
        std_dataset=std_dataset,
    )

    wandb.log(
        {"misc/leanring_rate": lr_scheduler.get_last_lr()[0]}, commit=False
    )
    wandb.log({"misc/gradient_step": step_number}, commit=True)

    return step_number + 1


def main():

    # Ensure that only one device is available.
    assert_single_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize experiment, including data loaders, model, and optimizer.
    experiment = initialize_experiment()

    # Initialize wandb.
    wandb.init(project=experiment.info.wandb_project)
    log_code_to_wandb()

    std_dataset = load_std_dataset()

    # wandb log gradients
    wandb.watch(experiment.model, log="all", log_freq=20)
    step = 0
    for _ in range(experiment.constants.training.epochs):
        train_loader = iter(tqdm(experiment.train_dataloader))

        while True:
            try:
                step = run_and_log_time_to_wandb(
                    "train_step",
                    train_step,
                    model=experiment.model,
                    train_loader=train_loader,
                    optimizer=experiment.optimizer,
                    lr_scheduler=experiment.lr_scheduler,
                    step_number=step,
                    std_dataset=std_dataset,
                    device=device,
                )
            except StopIteration:
                break
            step = step + 1

    wandb.finish()


if __name__ == "__main__":
    main()

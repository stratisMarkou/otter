from functools import partial
from typing import List

import torch
from munch import Munch  # type: ignore

from otter.data.datasets import (
    collate_fn,
    make_gridded_weather_task,
)
from otter.models.architectures.embeddings import TimeEmbedding
from otter.models.architectures.transformer import SwinTransformer
from otter.models.deterministic_model import make_deterministic_model
from otter.models.losses.loss_fn import (
    apply_loss,
    rmse,
)
from otter.models.losses.weighting import (
    get_pressure_weighting,
    get_surface_variable_weighting,
)

TRAIN_START_DATE = "1979-01-01 00:00:00"
TRAIN_END_DATE = "2019-12-31 18:00:00"
VALID_START_DATE = "2020-01-01 00:00:00"
VALID_END_DATE = "2020-12-31 18:00:00"
TEMPORAL_RESOLUTION_HOURS = 6
CTX_VARIABLES_TO_REMOVE = ["sea_surface_temperature", "vertical_velocity"]
TRG_VARIABLES_TO_REMOVE = ["sea_surface_temperature", "vertical_velocity"]
NUM_CONTEXT_FRAMES = 2
TRAIN_NUM_TARGET_FRAMES = 1
VALID_NUM_TARGET_FRAMES = 4
BATCH_SIZE = 4
NUM_WORKERS = 2
PATH_TO_LABELLED_ARRAY = "./_data/era5_240x121_single_array_float32"
VAL_TASK_SUBSAMPLING_FACTOR = 5

INITIAL_LEARNING_RATE = 1e-4
EPOCHS = 40
VALIDATE_EVERY = 5000
SAVE_CHECKPOINT_EVERY = 10_000

ROOT_RESULTS = "_results"
WANDB_PROJECT = "otter-deterministic"

TIME_EMBEDDING_NUM_SCALES = 16
DATA_GRID_NUM_LAT = 121
DATA_GRID_NUM_LON = 240
LR_SCHEDULER_ETA_MIN = 1e-6


def get_experiment(
    train_start_date: str = TRAIN_START_DATE,
    train_end_date: str = TRAIN_END_DATE,
    valid_start_date: str = VALID_START_DATE,
    valid_end_date: str = VALID_END_DATE,
    temporal_resolution_hours: int = TEMPORAL_RESOLUTION_HOURS,
    ctx_variables_to_remove: List[str] = CTX_VARIABLES_TO_REMOVE,
    trg_variables_to_remove: List[str] = TRG_VARIABLES_TO_REMOVE,
    num_context_frames: int = NUM_CONTEXT_FRAMES,
    train_num_target_frames: int = TRAIN_NUM_TARGET_FRAMES,
    valid_num_target_frames: int = VALID_NUM_TARGET_FRAMES,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    path_to_labelled_array: str = PATH_TO_LABELLED_ARRAY,
    val_task_subsampling_factor: int = VAL_TASK_SUBSAMPLING_FACTOR,
    initial_learning_rate: float = INITIAL_LEARNING_RATE,
    epochs: int = EPOCHS,
    validate_every: int = VALIDATE_EVERY,
    save_checkpoint_every: int = SAVE_CHECKPOINT_EVERY,
    root_results: str = ROOT_RESULTS,
    wandb_project: str = WANDB_PROJECT,
) -> Munch:

    train_dataloader = torch.utils.data.DataLoader(
        dataset=make_gridded_weather_task(
            path_to_labelled_array=path_to_labelled_array,
            start_date=train_start_date,
            end_date=train_end_date,
            num_context_frames=num_context_frames,
            num_target_frames=train_num_target_frames,
            temporal_resolution_hours=temporal_resolution_hours,
            ctx_variables_to_exclude=ctx_variables_to_remove,
            trg_variables_to_exclude=trg_variables_to_remove,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=make_gridded_weather_task(
            path_to_labelled_array=path_to_labelled_array,
            start_date=valid_start_date,
            end_date=valid_end_date,
            num_context_frames=num_context_frames,
            num_target_frames=valid_num_target_frames,
            temporal_resolution_hours=temporal_resolution_hours,
            ctx_variables_to_exclude=ctx_variables_to_remove,
            trg_variables_to_exclude=trg_variables_to_remove,
            task_sub_sampling_factor=val_task_subsampling_factor,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    time_embedding = TimeEmbedding(num_scales=TIME_EMBEDDING_NUM_SCALES)

    backbone_network = SwinTransformer(
        in_channels=148 + 2 * TIME_EMBEDDING_NUM_SCALES,
        height=128,
        width=256,
        patch_size=2,
        token_dimensions=[512, 512, 512, 512],
        output_dimension=69,
        window_size=4,
        mlp_num_hidden=256,
        mlp_num_layers=2,
        num_heads=8,
        num_blocks_per_stage=[4, 4, 4],
    )

    model = make_deterministic_model(
        backbone_network=backbone_network,
        temporal_resolution_hours=TEMPORAL_RESOLUTION_HOURS,
        time_embedding=time_embedding,
        ctx_variables_to_remove=CTX_VARIABLES_TO_REMOVE,
        trg_variables_to_remove=TRG_VARIABLES_TO_REMOVE,
        data_grid_num_lat=DATA_GRID_NUM_LAT,
        data_grid_num_lon=DATA_GRID_NUM_LON,
    )

    loss_partial = partial(
        apply_loss,
        loss_fn=rmse,
        weighting_functions=[
            get_surface_variable_weighting(
                variable_and_level=model.trg_variables_and_levels,
            ),
            get_pressure_weighting(
                variable_and_level=model.trg_variables_and_levels
            ),
        ],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_learning_rate,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=LR_SCHEDULER_ETA_MIN,
    )
    experiment = Munch()
    experiment.update(
        dict(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
            loss_partial=loss_partial,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
            validate_every=validate_every,
            save_checkpoint_every=save_checkpoint_every,
            root_results=root_results,
            wandb_project=wandb_project,
        )
    )
    return experiment

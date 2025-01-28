from dataclasses import dataclass
from typing import (
    List,
    Sequence,
)

import numpy as np
from torch.utils.data import Dataset

from otter.data.source.utils import (
    ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
    ALL_GRID_STATIC_VARIABLES,
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
    ALL_GRID_VARIABLE_LEVELS,
)
from otter.data.utils import (
    LabelledArray,
    open_memmap_as_labelled_array,
)

ALL_CTX_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    + ALL_GRID_STATIC_VARIABLES
)

ALL_TRG_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
)


@dataclass
class ContextTargetData:
    ctx: LabelledArray
    trg: LabelledArray


def collate_batch_wise(
    batch: List[ContextTargetData],
) -> List[ContextTargetData]:
    return batch


def _combine_variables_and_levels(
    variables: Sequence[str],
    levels: Sequence[int],
) -> List[str]:
    return [f"{var}/level_{level}" for var in variables for level in levels]


class GriddedWeatherTask(Dataset[ContextTargetData]):

    def __init__(
        self,
        path_to_labelled_array: str,
        start_date: str,
        end_date: str,
        num_context_frames: int,
        num_target_frames: int,
        temporal_resolution_hours: int,
        ctx_surface_dynamic_variables: List[
            str
        ] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        ctx_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        ctx_static_variables: List[str] = ALL_GRID_STATIC_VARIABLES,
        trg_surface_dynamic_variables: List[
            str
        ] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        trg_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        levels: List[int] = ALL_GRID_VARIABLE_LEVELS,
        task_sub_sampling_factor: int = 1,
    ):
        super().__init__()
        assert num_context_frames > 0
        assert num_target_frames > 0

        # Start and end dates that will be used to form training tasks,
        # i.e. no tasks will contain data outside of this range.
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)

        # Temporal resolution of the tasks in hours. This is used as the
        # lead time resolution as well as the context frame resolution.
        self.temporal_resolution = np.timedelta64(
            temporal_resolution_hours, "h"
        )

        # First zero-time of the collection of tasks. Since the tasks may have
        # more than one context frame, the first zero-time is not the same as
        # the start time, as we need to account for the first context frame
        self.first_zero_time = self.start_date + self.temporal_resolution * (
            num_context_frames - 1
        )

        # Time window spanned by a single task, including context and
        # target frames. For example, if the temporal resolution is 6 hours,
        # the number of context frames is 3 and the number of target frames
        # is 2, the single task time window will be 30 hours.
        num_frames = num_context_frames + num_target_frames
        self.single_task_time_window = np.timedelta64(
            temporal_resolution_hours * (num_frames - 1),
            "h",
        )

        # Timedeltas between each context frame and zero time.
        self.context_timedeltas = self.temporal_resolution * (
            -np.arange(num_context_frames)
        )

        # Timedeltas between each target frame and zero time.
        self.target_timedeltas = self.temporal_resolution * (
            np.arange(num_target_frames) + 1
        )

        # Specify data source zarr and select the time range that will be
        # used to form tasks.
        self.dataset = open_memmap_as_labelled_array(path_to_labelled_array)

        # Context and target variable names
        ctx_multilevel_dynamic_variables = _combine_variables_and_levels(
            ctx_multilevel_dynamic_variables, levels
        )
        trg_multilevel_dynamic_variables = _combine_variables_and_levels(
            trg_multilevel_dynamic_variables, levels
        )

        self.ctx_variables_and_levels = sorted(
            ctx_surface_dynamic_variables
            + ctx_multilevel_dynamic_variables
            + ctx_static_variables
        )

        self.trg_variables_and_levels = sorted(
            trg_surface_dynamic_variables + trg_multilevel_dynamic_variables
        )

        self.task_sub_sampling_factor = task_sub_sampling_factor

    def __len__(self) -> int:
        # Trailing time interval that is not enough to form a full task
        # and will be discarded.
        trailing_time_interval = (
            self.start_date - self.end_date
        ) % self.single_task_time_window

        # Return the number of full tasks that can be formed with the
        # parameters specified for the task.
        length = (
            int(
                (self.end_date - self.start_date - trailing_time_interval)
                // self.temporal_resolution
            )
            - 1
        )

        return length // self.task_sub_sampling_factor

    def __getitem__(self, idx: int) -> ContextTargetData:
        idx = idx * self.task_sub_sampling_factor
        zero_time = self.first_zero_time + idx * self.temporal_resolution
        context_times = zero_time + self.context_timedeltas
        target_times = zero_time + self.target_timedeltas

        assert all(ctx_time >= self.start_date for ctx_time in context_times)
        assert all(trg_time <= self.end_date for trg_time in target_times)

        ctx = self.dataset.sel(time=context_times).sel(
            variable_and_level=self.ctx_variables_and_levels
        )
        trg = self.dataset.sel(time=target_times).sel(
            variable_and_level=self.trg_variables_and_levels
        )

        return ContextTargetData(ctx=ctx, trg=trg)

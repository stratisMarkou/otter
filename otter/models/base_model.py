from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from otter.data.datasets import (  # type: ignore
    ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
    ALL_GRID_STATIC_VARIABLES,
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
    ALL_GRID_VARIABLE_LEVELS,
    ForecastingSample,
)
from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    filter_variables,
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)
from otter.models.architectures.embeddings import TimeEmbedding


def _get_variable_and_level_list(
    levels: Sequence[int],
    surface_variables: Sequence[str],
    multilevel_variables: Sequence[str],
    static_variables: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if static_variables is None:
        static_variables = []

    # Variables without levels do not need further annotation.
    vars_without_levels = list(surface_variables) + list(static_variables)

    # Variables with levels need to be annotated with the levels.
    vars_with_levels = [
        f"{var}/level_{level}"
        for var in multilevel_variables
        for level in levels
    ]

    return sorted(vars_without_levels + vars_with_levels)


def _get_normalisation_mean_and_std(
    variables_and_levels: Sequence[str],
) -> Tuple[xr.DataArray, xr.DataArray]:
    mean_ds = load_statistic("mean")
    mean_ds = split_into_different_variables_along_dim(mean_ds, dim="level")
    mean = stack_dataset_variable_and_levels(mean_ds)
    mean = mean.sel(variable_and_level=variables_and_levels)

    std_ds = load_statistic("std")
    std_ds = split_into_different_variables_along_dim(std_ds, dim="level")
    std = stack_dataset_variable_and_levels(std_ds)
    std = std.sel(variable_and_level=variables_and_levels)

    return mean, std


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        backbone_network: torch.nn.Module,
        time_embedding: TimeEmbedding,
        temporal_resolution_hours: int,
        ctx_surface_variables: Sequence[str],
        ctx_multilevel_variables: Sequence[str],
        ctx_static_variables: Sequence[str],
        trg_surface_variables: Sequence[str],
        trg_multilevel_variables: Sequence[str],
        ctx_levels: Sequence[int],
        trg_levels: Sequence[int],
        data_grid_num_lat: int,
        data_grid_num_lon: int,
        device: str,
    ) -> None:
        super().__init__()

        self.backbone_network = backbone_network
        self.time_embedding = time_embedding
        self.temporal_resolution_hours = temporal_resolution_hours

        # Ensure ctx surface vars are subset of trg surface vars
        if not set(ctx_surface_variables).issubset(set(trg_surface_variables)):
            raise ValueError(
                "Context surface variables must be a subset of target "
                "surface variables"
            )

        # Ensure ctx multilevel vars are subset of trg multilevel vars
        if not set(ctx_multilevel_variables).issubset(
            set(trg_multilevel_variables)
        ):
            raise ValueError(
                "Context multilevel variables must be a subset of target "
                "multilevel variables"
            )

        self.ctx_surface_variables = ctx_surface_variables
        self.ctx_multilevel_variables = ctx_multilevel_variables
        self.ctx_static_variables = ctx_static_variables
        self.ctx_variables_and_levels = _get_variable_and_level_list(
            levels=ctx_levels,
            surface_variables=ctx_surface_variables,
            multilevel_variables=ctx_multilevel_variables,
            static_variables=ctx_static_variables,
        )

        self.trg_surface_variables = trg_surface_variables
        self.trg_multilevel_variables = trg_multilevel_variables
        self.trg_variables_and_levels = _get_variable_and_level_list(
            levels=trg_levels,
            surface_variables=trg_surface_variables,
            multilevel_variables=trg_multilevel_variables,
        )

        self.data_grid_num_lat = data_grid_num_lat
        self.data_grid_num_lon = data_grid_num_lon

        self.idx_residual = [
            i
            for i, var in enumerate(self.ctx_variables_and_levels)
            if var in self.trg_variables_and_levels
        ]

        self.idx_ctx_for_update = [
            i
            for i, var in enumerate(self.ctx_variables_and_levels)
            if var in self.trg_variables_and_levels
        ]

        ctx_mean, ctx_std = _get_normalisation_mean_and_std(
            variables_and_levels=self.ctx_variables_and_levels,
        )

        trg_mean, trg_std = _get_normalisation_mean_and_std(
            variables_and_levels=self.trg_variables_and_levels,
        )

        # Preload normalisation statistics
        ctx_mean_tensor = torch.tensor(ctx_mean.values).float().to(device)
        ctx_std_tensor = torch.tensor(ctx_std.values).float().to(device)
        self.ctx_mean = ctx_mean_tensor[None, None, None, None, :]
        self.ctx_std = ctx_std_tensor[None, None, None, None, :]

        trg_mean_tensor = torch.tensor(trg_mean.values).float().to(device)
        trg_std_tensor = torch.tensor(trg_std.values).float().to(device)
        self.trg_mean = trg_mean_tensor[None, None, None, None, :]
        self.trg_std = trg_std_tensor[None, None, None, None, :]

    def preprocess_batch(
        self,
        batch: ForecastingSample,
    ) -> ForecastingSample:
        return batch

    def forward(
        self,
        batch: ForecastingSample,
        skip_model: bool = False,
    ) -> Dict[str, torch.Tensor]:
        ctx = batch.ctx
        zero_time = batch.zero_time
        trg_timedelta_hours = batch.trg_timedelta_hours
        if np.any(trg_timedelta_hours % self.temporal_resolution_hours != 0):
            raise ValueError(
                "All target times must be multiples of the temporal resolution"
            )

        num_rollout_steps = np.max(
            (trg_timedelta_hours // self.temporal_resolution_hours)
        )

        ctx = ctx - self.ctx_mean
        ctx = ctx / self.ctx_std

        expected_shape = tuple(
            (self.data_grid_num_lat, self.data_grid_num_lon)
        )
        assert ctx.shape[1:3] == torch.Size(expected_shape), (
            f"Expected lat-lon shapes: {expected_shape}, but got "
            f"{tuple(ctx.shape[1:3])}"
        )

        prd = self.rollout(
            ctx=ctx,
            zero_time=zero_time,
            trg_timedelta_hours=trg_timedelta_hours,
            num_rollout_steps=num_rollout_steps,
            skip_model=skip_model,
        )  # (batch, lat, lon, time_idx, trg_var_and_level)

        prd = prd * self.trg_std
        prd = prd + self.trg_mean
        assert isinstance(prd, torch.Tensor)  # Assert to silence mypy

        return dict(mean=prd)

    def rollout(
        self,
        ctx: torch.Tensor,
        zero_time: npt.NDArray[np.datetime64],
        trg_timedelta_hours: npt.NDArray[np.int32],
        num_rollout_steps: int,
        skip_model: bool = False,
    ) -> torch.Tensor:
        predictions: List[torch.Tensor] = list()
        for i in range(num_rollout_steps):
            # Get residual connection from context data. The dimension size
            # res_var_and_level refers to the number of variables that are
            # residual connection, i.e. both in the context and the target.
            #
            # TODO(stratisMarkou): currently, the residual is assumed to be the
            # same shape as the backbone network output. This may not always be
            # the case, for example if we are predicting more variables in the
            # target than we have in the context. This should be handled in a
            # future update.
            # NOTE: time dimension is indexed in reverse, so that the 0th
            # timestep is the latest.
            residual = ctx[
                ..., 0, self.idx_residual
            ]  # (batch, lat, lon, time_idx, res_var_and_level)

            # Prediction is residual plus backbone network output.
            if skip_model:
                prediction = residual

            else:
                ctx_reshaped = ctx.reshape(
                    *ctx.shape[:-2], -1
                )  # (batch, lat, lon, time_idx * var_and_level)
                time_embedding: torch.Tensor = self.time_embedding(
                    zero_time + i * self.temporal_resolution_hours
                )  # (batch, embedding_dim)

                # Repeat-tile the time embedding to match the dimensions of
                # ctx_reshaped except the batch dimension (first) and the
                # feature dimension (last).
                time_embedding = time_embedding[:, None, None, :]
                time_embedding = time_embedding.repeat(
                    1, *ctx_reshaped.shape[1:-1], 1
                )  # (batch, lat, lon, embedding_dim)
                time_embedding = time_embedding.to(ctx_reshaped.device)

                # Concatenate the time embedding with the reshaped context.
                ctx_with_time_embedding = torch.cat(
                    [ctx_reshaped, time_embedding], dim=-1
                )

                prediction = residual + self.backbone_network(
                    ctx_with_time_embedding,
                )  # (batch, lat, lon, trg_var_and_level)

            if (i + 1) * self.temporal_resolution_hours in trg_timedelta_hours:
                predictions.append(prediction)

            # Using this prediction, update the context data for the next step.
            # NOTE: We can't simply assign the prediction e.g. using
            # ctx[..., 0, :] = prediction.reshape(...)
            # because the prediction may contain fewer variables than the
            # context data (e.g. static/known data).
            ctx_next_time_frame = ctx[..., :1, :].clone()
            ctx_next_time_frame[..., 0, self.idx_ctx_for_update] = prediction

            ctx = torch.cat([ctx_next_time_frame, ctx[..., :-1, :]], dim=3)

        return torch.stack(predictions, dim=-2)


def make_base_model(
    backbone_network: torch.nn.Module,
    time_embedding: TimeEmbedding,
    temporal_resolution_hours: int,
    ctx_variables_to_remove: Sequence[str],
    trg_variables_to_remove: Sequence[str],
    data_grid_num_lat: int,
    data_grid_num_lon: int,
    device: str,
) -> BaseModel:
    return BaseModel(
        backbone_network=backbone_network,
        time_embedding=time_embedding,
        temporal_resolution_hours=temporal_resolution_hours,
        ctx_surface_variables=filter_variables(
            ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
            ctx_variables_to_remove,
        ),
        ctx_multilevel_variables=filter_variables(
            ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
            ctx_variables_to_remove,
        ),
        ctx_static_variables=filter_variables(
            ALL_GRID_STATIC_VARIABLES,
            ctx_variables_to_remove,
        ),
        trg_surface_variables=filter_variables(
            ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
            trg_variables_to_remove,
        ),
        trg_multilevel_variables=filter_variables(
            ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
            trg_variables_to_remove,
        ),
        ctx_levels=ALL_GRID_VARIABLE_LEVELS,
        trg_levels=ALL_GRID_VARIABLE_LEVELS,
        data_grid_num_lat=data_grid_num_lat,
        data_grid_num_lon=data_grid_num_lon,
        device=device,
    )

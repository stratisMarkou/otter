from dataclasses import dataclass
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
)
from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    filter_variables,
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)
from otter.experiment.downsample import downsample
from otter.models.architectures.embeddings import TimeEmbedding
from otter.models.architectures.moe import LoadBalancingLosses


def get_trg_vars_and_levels(
    trg_variables_to_remove: Sequence[str],
) -> Sequence[str]:
    trg_surface_variables = filter_variables(
        ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        trg_variables_to_remove,
    )

    trg_multilevel_variables = filter_variables(
        ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        trg_variables_to_remove,
    )

    return _get_variable_and_level_list(
        levels=ALL_GRID_VARIABLE_LEVELS,
        surface_variables=trg_surface_variables,
        multilevel_variables=trg_multilevel_variables,
    )


def get_ctx_vars_and_levels(
    ctx_variables_to_remove: Sequence[str],
) -> Sequence[str]:
    ctx_surface_variables = filter_variables(
        ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        ctx_variables_to_remove,
    )

    ctx_multilevel_variables = filter_variables(
        ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        ctx_variables_to_remove,
    )

    ctx_static_variables = filter_variables(
        ALL_GRID_STATIC_VARIABLES,
        ctx_variables_to_remove,
    )

    return _get_variable_and_level_list(
        levels=ALL_GRID_VARIABLE_LEVELS,
        surface_variables=ctx_surface_variables,
        multilevel_variables=ctx_multilevel_variables,
        static_variables=ctx_static_variables,
    )


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


@dataclass
class MultiscaleInput:
    ctx: torch.Tensor
    zero_time: npt.NDArray[np.datetime64]
    downsampled_trg: torch.Tensor
    scale: int
    trg_timedelta_hours: npt.NDArray[np.int32]

    def to(self, device: str, non_blocking: bool = False) -> "MultiscaleInput":
        self.ctx = self.ctx.to(device, non_blocking=non_blocking)
        self.downsampled_trg = self.downsampled_trg.to(
            device, non_blocking=non_blocking
        )
        return self


class MultiscaleModel(torch.nn.Module):
    def __init__(
        self,
        backbone_network: torch.nn.Module,
        time_embedding: TimeEmbedding,
        temporal_resolution_hours: int,
        ctx_variables_and_levels: Sequence[str],
        trg_variables_and_levels: Sequence[str],
        data_grid_num_lat: int,
        data_grid_num_lon: int,
        device: str,
    ) -> None:
        super().__init__()

        self.backbone_network = backbone_network
        self.time_embedding = time_embedding
        self.temporal_resolution_hours = temporal_resolution_hours

        self.ctx_variables_and_levels = ctx_variables_and_levels
        self.trg_variables_and_levels = trg_variables_and_levels

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

    def forward(
        self, batch: MultiscaleInput, scales: torch.Tensor
    ) -> tuple[Dict[str, torch.Tensor], List[LoadBalancingLosses]]:
        ctx = batch.ctx
        zero_time = batch.zero_time
        downsampled_trg = batch.downsampled_trg
        trg_timedelta_hours = batch.trg_timedelta_hours
        if np.any(trg_timedelta_hours % self.temporal_resolution_hours != 0):
            raise ValueError(
                "All target times must be multiples of the temporal resolution"
            )

        ctx = ctx - self.ctx_mean
        ctx = ctx / self.ctx_std

        # Special case: when the downsampled_trg tensor is erased, i.e. all
        # zeros, we don't want to apply normalisation to it.
        # Could check all elements, but checking just the first one is
        # sufficient in practice, as no real data will be all zeros.
        if (downsampled_trg[0, 0, 0, 0, :] != 0).sum() > 0:
            downsampled_trg = downsampled_trg - self.trg_mean
            downsampled_trg = downsampled_trg / self.trg_std

        expected_shape = tuple(
            (self.data_grid_num_lat, self.data_grid_num_lon)
        )
        assert ctx.shape[1:3] == torch.Size(expected_shape), (
            f"Expected lat-lon shapes: {expected_shape}, but got "
            f"{tuple(ctx.shape[1:3])}"
        )

        prd, load_balancing_losses = self.time_rollout(
            ctx=ctx,
            zero_time=zero_time,
            downsampled_trg=downsampled_trg,
            scales=scales,
            trg_timedelta_hours=trg_timedelta_hours,
        )  # (batch, lat, lon, trg_time_idx, scale, trg_var_and_level)

        prd = prd * self.trg_std
        prd = prd + self.trg_mean

        return dict(mean=prd), load_balancing_losses

    def single_scale_forward(
        self,
        ctx: torch.Tensor,
        downsampled_trg: torch.Tensor,
        zero_time: npt.NDArray[np.datetime64],
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        residual = ctx[
            ..., 0, self.idx_residual
        ]  # (batch, lat, lon, time_idx, res_var_and_level)

        ctx_reshaped = ctx.reshape(
            *ctx.shape[:-2], -1
        )  # (batch, lat, lon, ctx_time_idx * ctx_var_and_level)

        downsampled_trg_reshaped = downsampled_trg.reshape(
            *downsampled_trg.shape[:-2], -1
        )  # (batch, lat, lon, trg_time_idx * trg_var_and_level)

        ctx_and_downsampled_trg = torch.cat(
            [ctx_reshaped, downsampled_trg_reshaped], dim=-1
        )

        time_embedding: torch.Tensor = self.time_embedding(
            zero_time
        )  # (batch, embedding_dim)

        # Repeat-tile the time embedding to match the dimensions of
        # ctx_reshaped except the batch dimension (first) and the
        # feature dimension (last).
        time_embedding = time_embedding[:, None, None, :]
        time_embedding = time_embedding.repeat(
            1, *ctx_and_downsampled_trg.shape[1:-1], 1
        )  # (batch, lat, lon, embedding_dim)
        time_embedding = time_embedding.to(ctx_and_downsampled_trg.device)

        # Concatenate the time embedding with the reshaped context.
        ctx_with_time_embedding = torch.cat(
            [ctx_and_downsampled_trg, time_embedding], dim=-1
        )

        prediction, load_balancing_losses = self.backbone_network(
            ctx_with_time_embedding
        )  # (batch, lat, lon, trg_var_and_level)

        prediction += residual
        prediction = prediction[..., None, :]

        return prediction, load_balancing_losses

    def multiscale_forward(
        self,
        ctx: torch.Tensor,
        downsampled_trg: torch.Tensor,
        zero_time: npt.NDArray[np.datetime64],
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, list[LoadBalancingLosses]]:
        predictions = []
        all_load_balancing_losses = []

        prediction, load_balancing_losses = self.single_scale_forward(
            ctx=ctx,
            downsampled_trg=downsampled_trg,
            zero_time=zero_time,
        )  # (batch, lat, lon, 1, trg_var_and_level)
        predictions.append(prediction)
        if load_balancing_losses:
            all_load_balancing_losses.append(load_balancing_losses)

        # Optionally refine the prediction via rollout through scales.
        for scale in scales:
            downsampled_trg = downsample(
                prediction,
                int(scale),
            )  # (batch, lat, lon, 1, trg_var_and_level)

            prediction, load_balancing_losses = self.single_scale_forward(
                ctx=ctx,
                downsampled_trg=downsampled_trg,
                zero_time=zero_time,
            )
            predictions.append(prediction)
            if load_balancing_losses:
                all_load_balancing_losses.append(load_balancing_losses)

        prediction_tensor = torch.stack(
            predictions, dim=-2
        )  # (batch, lat, lon, 1, scale, trg_var_and_level)

        return prediction_tensor, all_load_balancing_losses

    def time_rollout(
        self,
        ctx: torch.Tensor,
        zero_time: npt.NDArray[np.datetime64],
        downsampled_trg: torch.Tensor,
        scales: torch.Tensor,
        trg_timedelta_hours: npt.NDArray[np.int32],
    ) -> tuple[torch.Tensor, list[LoadBalancingLosses]]:
        """
        Rolls out the model over time, predicting the target variables at
        each time step.

        Args:
            ctx: The context data tensor of shape (batch, lat, lon,
                ctx_time_idx, ctx_var_and_level).
            zero_time: The time of the latest context timestep.
            downsampled_trg: The downsampled target data tensor of shape
                (batch, lat, lon, trg_time_idx, trg_var_and_level).
                At validation, this should be a tensor of zeros.
            scales: The scales to use for the rollout through scales.
            trg_timedelta_hours: The target times at which to predict.
        Returns:
            The prediction tensor of shape (batch, lat, lon, trg_time_idx,
                scale, trg_var_and_level).
        """
        num_rollout_steps = np.max(
            (trg_timedelta_hours // self.temporal_resolution_hours)
        )

        predictions: List[torch.Tensor] = list()
        all_load_balancing_losses: List[LoadBalancingLosses] = list()
        for i in range(num_rollout_steps):
            # Choose the downsampled target slice for the current time step.
            downsampled_slice = downsampled_trg[..., i : i + 1, :]
            prediction, load_balancing_losses = self.multiscale_forward(
                ctx,
                downsampled_trg=downsampled_slice,
                zero_time=zero_time + i * self.temporal_resolution_hours,
                scales=scales,
            )  # (batch, lat, lon, 1, scale, trg_var_and_level)

            if (i + 1) * self.temporal_resolution_hours in trg_timedelta_hours:
                predictions.append(prediction)
            all_load_balancing_losses += load_balancing_losses

            # Using this prediction, update the context data for the next step.
            # NOTE: We can't simply assign the prediction e.g. using
            # ctx[..., 0, :] = prediction.reshape(...)
            # because the prediction may contain fewer variables than the
            # context data (e.g. static/known data).
            ctx_next_time_frame = ctx[..., :1, :].clone()

            # Autoregressively update the context with the prediction of the
            # latest scale.
            prediction = prediction[
                ..., -1, :
            ]  # (batch, lat, lon, 1, trg_var_and_level)
            ctx_next_time_frame[..., 0:1, self.idx_ctx_for_update] = prediction

            ctx = torch.cat([ctx_next_time_frame, ctx[..., :-1, :]], dim=3)

        # Concat the time rollout steps along the time dimension
        return torch.concat(predictions, dim=3), all_load_balancing_losses


def make_multiscale_model(
    backbone_network: torch.nn.Module,
    time_embedding: TimeEmbedding,
    temporal_resolution_hours: int,
    ctx_variables_to_remove: Sequence[str],
    trg_variables_to_remove: Sequence[str],
    data_grid_num_lat: int,
    data_grid_num_lon: int,
    device: str,
) -> MultiscaleModel:
    ctx_variables_and_levels = get_ctx_vars_and_levels(ctx_variables_to_remove)
    trg_variables_and_levels = get_trg_vars_and_levels(trg_variables_to_remove)

    ctx_static_variables = filter_variables(
        ALL_GRID_STATIC_VARIABLES,
        ctx_variables_to_remove,
    )

    # Ensure ctx surface vars are subset of trg surface vars
    dynamic_ctx = set(ctx_variables_and_levels) - set(ctx_static_variables)
    if not set(dynamic_ctx).issubset(set(trg_variables_and_levels)):
        raise ValueError(
            "Non-static context variables must be a subset of target variables"
        )

    return MultiscaleModel(
        backbone_network=backbone_network,
        time_embedding=time_embedding,
        temporal_resolution_hours=temporal_resolution_hours,
        ctx_variables_and_levels=ctx_variables_and_levels,
        trg_variables_and_levels=trg_variables_and_levels,
        data_grid_num_lat=data_grid_num_lat,
        data_grid_num_lon=data_grid_num_lon,
        device=device,
    )

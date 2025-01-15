from typing import (
    List,
    Optional,
    Sequence,
)

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from otter.data.normalisation.utils import normalise_dataset_zero_mean_unit_std
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)


def _standardise_data_array_dimensions(
    data_array: xr.DataArray,
) -> xr.DataArray:
    # Ensure that the dimensions are in the standard order
    return data_array.transpose(
        "batch",
        "latitude",
        "longitude",
        "time_index",
        "variable_and_level",
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


class DeterministicModel(torch.nn.Module):

    def __init__(
        self,
        backbone_network: torch.nn.Module,
        temporal_resolution_hours: int,
        ctx_surface_variables: Sequence[str],
        ctx_multilevel_variables: Sequence[str],
        ctx_static_variables: Sequence[str],
        trg_surface_variables: Sequence[str],
        trg_multilevel_variables: Sequence[str],
        ctx_levels: Sequence[int],
        trg_levels: Sequence[int],
    ) -> None:
        super().__init__()

        self.backbone_network = backbone_network
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

        self.idx_residual = [
            i
            for i, var in enumerate(self.ctx_variables_and_levels)
            if var in self.trg_variables_and_levels
        ]

        self.idx_ctx_for_update = [
            i
            for i, var in enumerate(self.trg_variables_and_levels)
            if var in self.ctx_variables_and_levels
        ]

    def forward(
        self,
        ctx_ds: xr.Dataset,
        trg_timedeltas: npt.NDArray[np.timedelta64],
        device: str = "cpu",
    ) -> Sequence[torch.Tensor]:

        # Get number of rollout steps to predict
        trg_timedelta_hours = trg_timedeltas.astype("timedelta64[h]").astype(
            int
        )

        if any(trg_timedelta_hours % self.temporal_resolution_hours != 0):
            raise ValueError(
                "All target times must be multiples of the temporal resolution"
            )

        num_rollout_steps = np.max(
            (trg_timedelta_hours // self.temporal_resolution_hours)
        )

        # First, normalise the ctx data and split into variables and levels
        ctx_norm = normalise_dataset_zero_mean_unit_std(ctx_ds)
        ctx_split = split_into_different_variables_along_dim(
            ctx_norm, dim="level"
        )
        ctx_stacked = stack_dataset_variable_and_levels(
            ctx_split, vars_to_drop=["time"]
        )

        # Standardise context dims (batch, lat, lon, time_idx, var_and_level)
        # and order variable and level dim alphabetically.
        ctx_stacked = _standardise_data_array_dimensions(ctx_stacked)
        ctx_stacked = ctx_stacked.sel(
            variable_and_level=self.ctx_variables_and_levels,
        )

        # Create context tensor.
        ctx = torch.tensor(ctx_stacked.values).float().to(device)

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
            residual = ctx[
                ..., 0, self.idx_residual
            ]  # (batch, lat, lon, time_idx, res_var_and_level)

            # Prediction is residual plus backbone network output.
            prediction = residual + self.backbone_network(
                ctx.reshape(*ctx.shape[:-2], -1)
            )  # (batch, lat, lon, trg_var_and_level)
            if (i + 1) * self.temporal_resolution_hours in trg_timedelta_hours:
                predictions.append(prediction)

            # Using this prediction, update the context data for the next step.
            ctx_next_time_frame = ctx[..., :1, :]
            ctx_next_time_frame[..., 0, self.idx_ctx_for_update] = prediction

            ctx = torch.cat([ctx_next_time_frame, ctx[..., 1:, :]], dim=3)

        return predictions

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

from otter.data.normalisation.utils import normalise_dataset_zero_mean_unit_std
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)
from otter.models.losses import (
    DEFAULT_SURFACE_VARIABLE_WEIGHTS,
    apply_difference_weighting,
    apply_latitude_weighting,
    apply_pressure_weighting,
    apply_surface_variable_weighting,
    load_difference_weighting,
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


def _get_target_timedelta_hours(
    ctx: xr.Dataset,
    trg: xr.Dataset,
) -> npt.NDArray[np.int32]:

    # First, get the number of frames to predict with the rollout
    ctx_zero_time = np.max(ctx.time.values, axis=-1)  # (batch,)
    trg_time = trg.time.values  # (batch, trg_time_idx)

    trg_timedelta = trg_time - ctx_zero_time[:, None]
    trg_timedelta_hours = trg_timedelta.astype("timedelta64[h]").astype(int)

    # Check that entries of trg_timedelta_hours are the same in each column
    if np.any(trg_timedelta_hours != trg_timedelta_hours[:1, :]):
        raise ValueError("All target times must be the same for each batch")

    return trg_timedelta_hours[0]  # type: ignore


def _prepare_target_tensor(
    trg: xr.Dataset,
    trg_variables_and_levels: Sequence[str],
    device: str,
) -> torch.Tensor:
    # Normalise and prepare the target data.
    trg_norm = normalise_dataset_zero_mean_unit_std(trg)
    trg_split = split_into_different_variables_along_dim(trg_norm, dim="level")
    trg_stacked = stack_dataset_variable_and_levels(
        trg_split, vars_to_drop=["time"]
    )
    trg_stacked = _standardise_data_array_dimensions(trg_stacked)
    trg_stacked = trg_stacked.sel(variable_and_level=trg_variables_and_levels)

    # Create target tensor.
    return torch.tensor(trg_stacked.values).float().to(device)


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
        apply_pressure_weighting_loss: bool,
        apply_latitude_weighting_loss: bool,
        apply_difference_weighting: bool,
        surface_variable_weights: Dict[
            str, float
        ] = DEFAULT_SURFACE_VARIABLE_WEIGHTS,
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

        self.apply_pressure_weighting_loss = apply_pressure_weighting_loss
        self.apply_latitude_weighting_loss = apply_latitude_weighting_loss
        self.surface_variable_weights = surface_variable_weights
        self.apply_difference_weighting = apply_difference_weighting

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

        self.std_ratio_weights = load_difference_weighting()

    def forward(
        self,
        ctx_ds: xr.Dataset,
        trg_timedelta_hours: npt.NDArray[np.int32],
        device: str,
        skip_model: bool = False,
    ) -> torch.Tensor:

        from time import time

        t0 = time()

        if any(trg_timedelta_hours % self.temporal_resolution_hours != 0):
            raise ValueError(
                "All target times must be multiples of the temporal resolution"
            )

        num_rollout_steps = np.max(
            (trg_timedelta_hours // self.temporal_resolution_hours)
        )

        # First, normalise the ctx data and split into variables and levels
        ctx_norm = normalise_dataset_zero_mean_unit_std(ctx_ds)
        t1 = time()
        ctx_split = split_into_different_variables_along_dim(
            ctx_norm, dim="level"
        )
        ctx_stacked = stack_dataset_variable_and_levels(
            ctx_split, vars_to_drop=["time"]
        )
        t1a = time()

        # Standardise context dims (batch, lat, lon, time_idx, var_and_level)
        # and order variable and level dim alphabetically.
        ctx_stacked = _standardise_data_array_dimensions(ctx_stacked).compute()
        t2a = time()
        ctx_stacked = ctx_stacked.sel(
            variable_and_level=self.ctx_variables_and_levels,
        )
        t2b = time()

        # Create context tensor.
        ctx = torch.tensor(ctx_stacked.values).float().to(device)
        t3 = time()

        result = self.rollout(
            ctx=ctx,
            trg_timedelta_hours=trg_timedelta_hours,
            num_rollout_steps=num_rollout_steps,
            skip_model=skip_model,
        )
        t4 = time()
        print(f"ctx_norm: {t1-t0:.2f}")
        print(f"split: {t1a-t1:.2f}")
        print(f"standardise: {t2a-t1a:.2f}")
        print(f"sel: {t2b-t2a:.2f}")
        print(f"to_tensor: {t3-t2b:.2f}")
        print(f"rollout: {t4-t3:.2f}")
        return result

    def rollout(
        self,
        ctx: torch.Tensor,
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
            residual = ctx[
                ..., 0, self.idx_residual
            ]  # (batch, lat, lon, time_idx, res_var_and_level)

            # Prediction is residual plus backbone network output.
            if skip_model:
                prediction = residual

            else:
                prediction = residual + self.backbone_network(
                    ctx.reshape(*ctx.shape[:-2], -1)
                )  # (batch, lat, lon, trg_var_and_level)

            if (i + 1) * self.temporal_resolution_hours in trg_timedelta_hours:
                predictions.append(prediction)

            # Using this prediction, update the context data for the next step.
            ctx_next_time_frame = ctx[..., :1, :].clone()
            ctx_next_time_frame[..., 0, self.idx_ctx_for_update] = prediction

            ctx = torch.cat([ctx_next_time_frame, ctx[..., 1:, :]], dim=3)

        return torch.stack(predictions, dim=-2)

    def loss(
        self,
        ctx: xr.Dataset,
        trg: xr.Dataset,
        device: str,
        mean_over_time_dimension: bool,
    ) -> Tuple[torch.Tensor, Dict[str, npt.NDArray[np.float32]]]:

        from time import time

        def _apply_square_root_and_averaging(
            squared_error: torch.Tensor,
            mean_over_time_dimension: bool,
        ) -> torch.Tensor:
            rmse = torch.mean(squared_error, dim=[1, 2]) ** 0.5
            if mean_over_time_dimension:
                return torch.mean(rmse, dim=[0, 1])  # (var_and_level,)
            else:
                return torch.mean(rmse, dim=[0])  # (time_idx, var_and_level)

        t0 = time()
        # Get the target timedelta hours.
        trg_timedelta_hours = _get_target_timedelta_hours(ctx=ctx, trg=trg)
        t1 = time()

        # Get the predictions from the model.
        predictions = self.forward(ctx, trg_timedelta_hours, device=device)
        t2 = time()

        # Prepare the target tensor from the target dataset.
        trg_tensor = _prepare_target_tensor(
            trg=trg,
            trg_variables_and_levels=self.trg_variables_and_levels,
            device=device,
        )
        t3 = time()

        # Calculate the loss.
        squared_error = (
            predictions - trg_tensor
        ) ** 2  # (batch, lat, lon, time_idx, var_and_level)
        squared_error = apply_latitude_weighting(squared_error)
        t4 = time()

        lwrmse_per_variable = _apply_square_root_and_averaging(
            squared_error,
            mean_over_time_dimension=mean_over_time_dimension,
        )
        t5 = time()

        if self.apply_pressure_weighting_loss:
            squared_error = apply_pressure_weighting(
                squared_error,
                self.trg_variables_and_levels,
            )
        t6 = time()

        if self.surface_variable_weights:
            squared_error = apply_surface_variable_weighting(
                squared_error,
                self.trg_variables_and_levels,
                self.surface_variable_weights,
            )
        t7 = time()

        if self.apply_difference_weighting:
            squared_error = apply_difference_weighting(
                squared_error,
                self.trg_variables_and_levels,
                self.std_ratio_weights,
            )
        t8 = time()

        # Compute mean across (lat, lon) dimensions and take square root.
        rmse_loss = torch.mean(squared_error, dim=[1, 2]) ** 0.5
        t9 = time()

        if mean_over_time_dimension:
            per_variable_loss = torch.mean(rmse_loss, dim=[0, 1])

        else:
            per_variable_loss = torch.mean(rmse_loss, dim=[0])
        t10 = time()

        # Total loss is the sum of the per-variable losses.
        loss = torch.sum(per_variable_loss, dim=-1)
        t11 = time()

        # Per-variable losses as numpy arrays in a dictionary.
        training_loss_info = {
            var: per_variable_loss[..., i].cpu().detach().numpy()
            for i, var in enumerate(self.trg_variables_and_levels)
        }
        t12 = time()

        auxiliary_loss_info = {
            f"lwrmse/{var}": lwrmse_per_variable[..., i].cpu().detach().numpy()
            for i, var in enumerate(self.trg_variables_and_levels)
        }
        t13 = time()

        print("In loss " + "-" * 50)
        print(f"get_target_timedelta: {t1-t0:.2f}")
        print(f"forward: {t2-t1:.2f}")
        print(f"prepare_target: {t3-t2:.2f}")
        print(f"apply_lat: {t4-t3:.2f}")
        print(f"apply_lat_weighting: {t5-t4:.2f}")
        print(f"apply_pressure_weighting: {t6-t5:.2f}")
        print(f"apply_surface_weighting: {t7-t6:.2f}")
        print(f"apply_diff_weighting: {t8-t7:.2f}")
        print(f"rmse_loss: {t9-t8:.2f}")
        print(f"per_var_loss: {t10-t9:.2f}")
        print(f"total_loss: {t11-t10:.2f}")
        print(f"training_loss_info: {t12-t11:.2f}")
        print(f"auxiliary_loss_info: {t13-t12:.2f}")
        print(f"Total time: {t13-t0:.2f}")
        print("-" * 58)

        return loss, dict(**training_loss_info, **auxiliary_loss_info)

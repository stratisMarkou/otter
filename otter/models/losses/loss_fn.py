from functools import partial
from typing import (
    Callable,
    Dict,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr

from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)

PredictionStatistics = Dict[str, torch.Tensor]
LossInfo = Dict[str, torch.Tensor | npt.NDArray[np.float32]]
LossWeightingFunction = Callable[[torch.Tensor, Sequence[str]], torch.Tensor]

DEFAULT_SURFACE_VARIABLE_WEIGHTS = {
    "2m_temperature": 1e0,
    "10m_u_component_of_wind": 1e-1,
    "10m_v_component_of_wind": 1e-1,
    "mean_sea_level_pressure": 1e-1,
}


def load_stacked_std_data_array(diff: bool) -> xr.DataArray:
    std = load_statistic("diff_std" if diff else "std")
    std_split = split_into_different_variables_along_dim(std, "level")
    return stack_dataset_variable_and_levels(std_split)


VAR_STD_DATA_ARRAY = load_stacked_std_data_array(diff=False)
VAR_DIFF_STD_DATA_ARRAY = load_stacked_std_data_array(diff=True)


def apply_surface_variable_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
    variable_to_weight: Dict[str, float],
) -> torch.Tensor:
    """Create function to apply surface variable weighting.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
        variable_and_level: list of variable names and levels
        variable_to_weight: dictionary of surface variable name to weights

    Returns:
        callable: function that applies surface variable weighting to a tensor.
    """

    weights = []
    for variable in variable_and_level:
        if "/level_" in variable or variable not in variable_to_weight:
            weights.append(1.0)
        else:
            weights.append(variable_to_weight[variable])

    weight_tensor = torch.tensor(weights).float().to(tensor.device)
    weight_tensor = weight_tensor[None, None, :]

    return tensor * weight_tensor.to(tensor.device)


def apply_pressure_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
) -> torch.Tensor:
    """Create function to apply pressure weighting.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)

    Returns:
        callable: function that applies pressure weighting to a tensor.
    """

    pressure_levels = set()
    for var_and_level in variable_and_level:
        if "/level_" in var_and_level:
            pressure_levels.add(int(var_and_level.split("_")[-1]))

    # Pressure weight is proportional to the level, and each multilevel
    # variable has a total weight of one.
    pressure_weights = {
        level: level / sum(pressure_levels) for level in pressure_levels
    }

    # Loop again to create the weights tensor.
    weights = []

    for var_and_level in variable_and_level:
        if "/level_" in var_and_level:
            level = int(var_and_level.split("_")[-1])
            weights.append(pressure_weights[level])
        else:
            weights.append(1.0)

    weight_tensor = torch.tensor(weights).float().to(tensor.device)
    weight_tensor = weight_tensor[None, None, :]

    return tensor * weight_tensor.to(tensor.device)


def apply_division_by_variable_diff_std(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
) -> torch.Tensor:

    std_array = VAR_STD_DATA_ARRAY[variable_and_level].values
    std_tensor = torch.tensor(std_array).float().to(tensor.device)
    std_tensor = std_tensor[None, None, :]

    return tensor / (std_tensor + 1e-12).to(tensor.device)


def apply_division_by_variable_std(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
) -> torch.Tensor:

    std_array = VAR_DIFF_STD_DATA_ARRAY[variable_and_level].values
    std_tensor = torch.tensor(std_array).float().to(tensor.device)
    std_tensor = std_tensor[None, None, :]

    return tensor / (std_tensor + 1e-12).to(tensor.device)


def _apply_latitude_weighting(tensor: torch.Tensor) -> torch.Tensor:
    """Apply latitude weighting to a tensor.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)

    Returns:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
            with latitude weighting applied.
    """

    # Create latitude weights. This assumes that the latitude dimension is
    # the second dimension of the tensor, and that the poles are included.
    lat_deg = np.linspace(-90, 90, tensor.shape[1])
    lat_rad = np.deg2rad(lat_deg)

    # We use the cosine of the latitude as the weight.
    lat_weight = torch.tensor(np.cos(lat_rad)).float().to(tensor.device)
    lat_weight = lat_weight[None, :, None, None, None] / torch.mean(lat_weight)

    return tensor * lat_weight


def _rmse(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
    apply_latitude_weighting: bool,
) -> torch.Tensor:
    error = (
        prd_stats["mean"] - trg
    ) ** 2  # (batch, lat, lon, time_idx, var_and_level)

    if apply_latitude_weighting:
        error = _apply_latitude_weighting(error)

    # Compute the root mean squared error.
    error = (
        torch.mean(error, dim=[1, 2]) ** 0.5
    )  # (batch, time_idx, var_and_level)
    assert isinstance(error, torch.Tensor)

    return error


def rmse(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
) -> torch.Tensor:
    return _rmse(trg, prd_stats, apply_latitude_weighting=False)


def lwrmse(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
) -> torch.Tensor:
    return _rmse(trg, prd_stats, apply_latitude_weighting=True)


def apply_loss(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
    loss_fn: Callable[[torch.Tensor, PredictionStatistics], torch.Tensor],
    weighting_fns: Sequence[LossWeightingFunction],
    trg_variables_and_levels: Sequence[str],
) -> Tuple[torch.Tensor, LossInfo]:
    """Apply a loss function to target tensor using prediction statistics.

    NOTE: both target and prediction statistics should correspond to the
    variable values in physical space, i.e. in physical units, and not be
    normalised.
    """

    # Calculate the squared error. We apply specified weightings to it below.
    loss = loss_fn(trg, prd_stats)  # (batch, time, var_and_level)

    # Apply the specified weightings to the loss tensor.
    for weighting_fn in weighting_fns:
        loss = weighting_fn(
            loss, trg_variables_and_levels
        )  # (batch, time, var_and_level)

    loss_per_time_per_var = torch.mean(loss, dim=0)  # (time, var_and_level)

    # Total loss is the sum of the per-variable losses.
    loss_per_var = torch.mean(loss_per_time_per_var, dim=0)
    aggregated_loss = torch.sum(loss_per_var)

    loss_info = {
        f"{v}/time_{t:02d}": loss_per_time_per_var[t, i].cpu().detach().numpy()
        for i, v in enumerate(trg_variables_and_levels)
        for t in range(loss_per_time_per_var.shape[0])
    }

    return aggregated_loss, loss_info  # type: ignore


def make_loss_fn(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, PredictionStatistics], torch.Tensor],
    weighting_fns: Sequence[LossWeightingFunction],
) -> Callable[
    [torch.Tensor, PredictionStatistics], Tuple[torch.Tensor, LossInfo]
]:

    return partial(
        apply_loss,
        loss_fn=loss_fn,
        weighting_fns=weighting_fns,
        trg_variables_and_levels=model.trg_variables_and_levels,
    )

from typing import (
    Callable,
    Dict,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch

from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)

PredictionStatistics = Dict[str, torch.Tensor]
LossInfo = Dict[str, torch.Tensor | npt.NDArray[np.float32]]
LossWeightingFunction = Callable[[torch.Tensor], torch.Tensor]

DEFAULT_SURFACE_VARIABLE_WEIGHTS = {
    "2m_temperature": 1e0,
    "10m_u_component_of_wind": 1e-1,
    "10m_v_component_of_wind": 1e-1,
    "mean_sea_level_pressure": 1e-1,
}


def get_surface_variable_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
    variable_to_weight: Dict[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
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

    return lambda tensor: tensor * weight_tensor.to(tensor.device)


def get_pressure_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
) -> Callable[[torch.Tensor], torch.Tensor]:
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

    return lambda tensor: tensor * weight_tensor.to(tensor.device)


def get_division_by_variable_std(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
    diff_weighting: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply division by variable difference standard deviation to a tensor.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
        variable_and_level: list of variable names and levels
        diff_weighting: whether to apply weighting based on variable difference
            standard deviation or raw variable.

    Returns:
        callable: function that applies division by variable difference
            standard deviation to a tensor.
    """

    std = load_statistic("diff_std" if diff_weighting else "std")
    std_split = split_into_different_variables_along_dim(std, "level")
    std_stacked = stack_dataset_variable_and_levels(std_split)

    std_array = std_stacked[variable_and_level].values
    std_tensor = torch.tensor(std_array).float().to(tensor.device)
    std_tensor = std_tensor[None, None, :]

    return lambda tensor: tensor / (std_tensor + 1e-12).to(tensor.device)


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
    weighting_functions: Sequence[LossWeightingFunction],
) -> Tuple[torch.Tensor, LossInfo]:
    """Apply a loss function to target tensor using prediction statistics.

    NOTE: both target and prediction statistics should correspond to the
    variable values in physical space, i.e. in physical units, and not be
    normalised.
    """

    # Calculate the squared error. We apply specified weightings to it below.
    loss = loss_fn(trg, prd_stats)  # (batch, time, var_and_level)

    # Apply the specified weightings to the loss tensor.
    for weighting_fn in weighting_functions:
        loss = weighting_fn(loss)  # (batch, time, var_and_level)

    loss_per_time_per_var = torch.mean(loss, dim=0)  # (time, var_and_level)

    # Total loss is the sum of the per-variable losses.
    per_variable_loss = torch.mean(loss_per_time_per_var, dim=0)
    aggregated_loss = torch.sum(per_variable_loss)

    loss_info = {
        f"loss/{v}_{t:02d}": loss_per_time_per_var[t, i].cpu().detach().numpy()
        for i, v in enumerate(prd_stats.keys())
        for t in range(loss_per_time_per_var.shape[0])
    }

    return aggregated_loss, loss_info  # type: ignore

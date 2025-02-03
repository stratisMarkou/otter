from functools import lru_cache
from typing import (
    Dict,
    Sequence,
)

import numpy as np
import torch
import xarray as xr

from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)

DEFAULT_SURFACE_VARIABLE_WEIGHTS = {
    "2m_temperature": 1e0,
    "10m_u_component_of_wind": 1e-1,
    "10m_v_component_of_wind": 1e-1,
    "mean_sea_level_pressure": 1e-1,
}


def apply_latitude_weighting(tensor: torch.Tensor) -> torch.Tensor:
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


def apply_surface_variable_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
    variable_to_weight: Dict[str, float],
) -> torch.Tensor:
    """Apply surface variable weighting to a tensor.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
        variable_and_level: list of variable names and levels
        variable_to_weight: dictionary of surface variable name to weights

    Returns:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
            with surface variable weighting applied.
    """

    weights = []
    for variable in variable_and_level:
        if "/level_" in variable or variable not in variable_to_weight:
            weights.append(1.0)
        else:
            weights.append(variable_to_weight[variable])

    weight_tensor = torch.tensor(weights).float().to(tensor.device)
    weight_tensor = weight_tensor[None, None, None, None, :]

    return tensor * weight_tensor


def apply_pressure_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
) -> torch.Tensor:
    """Apply pressure weighting to a tensor.

    Args:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)

    Returns:
        tensor: tensor of shape (batch, lat, lon, time_idx, var_and_level)
            with pressure weighting applied.
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
    weight_tensor = weight_tensor[None, None, None, None, :]

    return tensor * weight_tensor


@lru_cache
def _load_difference_weighting(
    repo_base_path: str = ".",
    zarr_name: str = "era5_240x121",
) -> xr.DataArray:

    std = load_statistic("std", repo_base_path, zarr_name)
    std_split = split_into_different_variables_along_dim(std, "level")
    std_stacked = stack_dataset_variable_and_levels(std_split)

    std_diff = load_statistic("diff_std", repo_base_path, zarr_name)
    std_diff_split = split_into_different_variables_along_dim(
        std_diff, "level"
    )
    std_diff_stacked = stack_dataset_variable_and_levels(std_diff_split)

    return std_stacked / (std_diff_stacked + 1e-12)


def apply_difference_weighting(
    tensor: torch.Tensor,
    variable_and_level: Sequence[str],
    repo_base_path: str = ".",
    zarr_name: str = "era5_240x121",
) -> torch.Tensor:

    std_diff_ratio = _load_difference_weighting(repo_base_path, zarr_name)

    # Create weights tensor.
    weights = std_diff_ratio.sel(variable_and_level=variable_and_level).values
    weight_tensor = torch.tensor(weights).float().to(tensor.device)
    weight_tensor = weight_tensor[None, None, None, None, :]

    return tensor * weight_tensor

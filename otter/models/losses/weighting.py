from typing import (
    Callable,
    Dict,
    Sequence,
)

import torch

from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)

PredictionStatistics = Dict[str, torch.Tensor]
LossWeightingFunction = Callable[[torch.Tensor], torch.Tensor]

DEFAULT_SURFACE_VARIABLE_WEIGHTS = {
    "2m_temperature": 1e0,
    "10m_u_component_of_wind": 1e-1,
    "10m_v_component_of_wind": 1e-1,
    "mean_sea_level_pressure": 1e-1,
}


def get_surface_variable_weighting(
    variable_and_level: Sequence[str],
    variable_to_weight: Dict[str, float] = DEFAULT_SURFACE_VARIABLE_WEIGHTS,
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

    weight_tensor = torch.tensor(weights).float()
    weight_tensor = weight_tensor[None, None, :]

    return lambda tensor: tensor * weight_tensor.to(tensor.device)


def get_pressure_weighting(
    variable_and_level: Sequence[str],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create function to apply pressure weighting.

    Args:

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

    weight_tensor = torch.tensor(weights).float()
    weight_tensor = weight_tensor[None, None, :]

    return lambda tensor: tensor * weight_tensor.to(tensor.device)


def get_division_by_variable_std(
    variable_and_level: Sequence[str],
    diff_weighting: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Apply division by variable difference standard deviation to a tensor.

    Args:
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
    std_tensor = torch.tensor(std_array).float()
    std_tensor = std_tensor[None, None, :]

    return lambda tensor: tensor / (std_tensor + 1e-12).to(tensor.device)

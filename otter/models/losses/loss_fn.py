from functools import partial
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.architectures.utils import aggregate_load_balancing_losses
from otter.models.distributions import Distribution, Delta, Gaussian

DistributionWithMean = Union[Delta, Gaussian]
PredictionStatistics = Dict[str, torch.Tensor]
LossInfo = Dict[str, float]
LossWeightingFunction = Callable[[torch.Tensor, Sequence[str]], torch.Tensor]

DEFAULT_SURFACE_VARIABLE_WEIGHTS = {
    "2m_temperature": 1e0,
    "10m_u_component_of_wind": 1e-1,
    "10m_v_component_of_wind": 1e-1,
    "mean_sea_level_pressure": 1e-1,
}


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
    lat_weight = lat_weight[None, :, None, None, None, None] / torch.mean(
        lat_weight
    )

    return tensor * lat_weight


def _rmse(
    trg: torch.Tensor,
    prd_dist: DistributionWithMean,
    apply_latitude_weighting: bool,
) -> torch.Tensor:
    # Expand along the scale dimension to match the target tensor.
    trg = trg.unsqueeze(-2)  # (batch, lat, lon, time_idx, 1, var_and_level)

    error = torch.square(
        prd_dist.mean - trg
    )  # (batch, lat, lon, time_idx, scale, var_and_level)

    if apply_latitude_weighting:
        error = _apply_latitude_weighting(error)

    # Compute the root mean squared error.
    error = torch.sqrt(
        torch.mean(error, dim=[1, 2])
    )  # (batch, time_idx, scale, var_and_level)
    assert isinstance(error, torch.Tensor)

    return error


def rmse(
    trg: torch.Tensor,
    prd_dist: DistributionWithMean,
) -> torch.Tensor:
    return _rmse(trg, prd_dist, apply_latitude_weighting=False)


def lwrmse(
    trg: torch.Tensor,
    prd_dist: DistributionWithMean,
) -> torch.Tensor:
    return _rmse(trg, prd_dist, apply_latitude_weighting=True)


def gather_load_balancing_losses(
    load_balancing_losses: List[LoadBalancingLosses],
) -> Tuple[Optional[torch.Tensor], Optional[LossInfo]]:
    """Gather load balancing losses from the list of LoadBalancingLosses."""
    # Computes the mean loss across time steps
    losses = aggregate_load_balancing_losses(
        load_balancing_losses,
        mean=True,
    )
    if losses is not None:
        total_loss = losses.importance_loss + losses.load_loss
        loss_info = {
            "importance_loss": losses.importance_loss.item(),
            "load_loss": losses.load_loss.item(),
            "total_load_balancing_loss": total_loss.item(),
        }
        return total_loss, loss_info
    return None, None


def apply_loss(
    trg: torch.Tensor,
    prd_dist: Distribution,
    loss_fn: Callable[[torch.Tensor, Distribution], torch.Tensor],
    weighting_fns: Optional[Sequence[LossWeightingFunction]],
    trg_variables_and_levels: Sequence[str],
    scales: Sequence[int],
) -> Tuple[torch.Tensor, LossInfo]:
    """Apply a loss function to target tensor using prediction statistics.

    NOTE: both target and prediction statistics should correspond to the
    variable values in physical space, i.e. in physical units, and not be
    normalised.

    Args:
        trg: target tensor of shape (batch, lat, lon, time_idx, var_and_level)
        prd_dist: distribution of shape
            (batch, lat, lon, time_idx, scale, var_and_level)
        loss_fn: loss function to apply
        weighting_fns: list of weighting functions to apply to the loss tensor
        trg_variables_and_levels: list of target variables and levels
        scales: list of scales corresponding to the scale dimension of the
            prediction statistics tensor(s)
    """

    # Calculate the squared error. We apply specified weightings to it below.
    loss = loss_fn(trg, prd_dist)  # (batch, time, scale, var_and_level)

    # Apply the specified weightings to the loss tensor.
    if weighting_fns is not None:
        for weighting_fn in weighting_fns:
            loss = weighting_fn(
                loss, trg_variables_and_levels
            )  # (batch, time, scale, var_and_level)

    loss_per_time_var_scale = torch.mean(
        loss, dim=0
    )  # (time, scale, var_and_level)

    loss_info = {}
    for i_var, var in enumerate(trg_variables_and_levels):
        for t in range(loss_per_time_var_scale.shape[0]):
            for i_scale, scale in enumerate(scales):
                loss_val = loss_per_time_var_scale[t, i_scale, i_var].item()
                loss_key = f"{var}/time_{t:02d}/scale_{scale}"
                loss_info[loss_key] = float(loss_val)

    loss_per_var_scale = torch.mean(
        loss_per_time_var_scale, dim=0
    )  # (scale, var_and_level)

    loss_per_scale = torch.sum(loss_per_var_scale, dim=-1)  # (scale,)

    # Final loss is the loss associated with the latest scale.
    aggregated_loss = loss_per_scale[-1]

    # Also log aggregate losses along scales.
    for i_scale, scale in enumerate(scales):
        loss_val = loss_per_scale[i_scale].item()
        loss_key = f"loss/scale_{scale}"
        loss_info[loss_key] = float(loss_val)

    return aggregated_loss, loss_info


def make_loss_fn(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, Distribution], torch.Tensor],
    weighting_fns: Optional[Sequence[LossWeightingFunction]] = None,
) -> Callable[..., Tuple[torch.Tensor, LossInfo]]:
    return partial(
        apply_loss,
        loss_fn=loss_fn,
        weighting_fns=weighting_fns,
        trg_variables_and_levels=model.trg_variables_and_levels,
    )

from functools import partial
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch

from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.architectures.utils import aggregate_load_balancing_losses

PredictionStatistics = Dict[str, torch.Tensor]
LossInfo = Dict[str, npt.NDArray[np.float32]]
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
    lat_weight = lat_weight[None, :, None, None, None] / torch.mean(lat_weight)

    return tensor * lat_weight


def _rmse(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
    apply_latitude_weighting: bool,
) -> torch.Tensor:
    error = torch.square(
        prd_stats["mean"] - trg
    )  # (batch, lat, lon, time_idx, var_and_level)

    if apply_latitude_weighting:
        error = _apply_latitude_weighting(error)

    # Compute the root mean squared error.
    error = torch.sqrt(
        torch.mean(error, dim=[1, 2])
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
            "importance_loss": losses.importance_loss.cpu().detach().numpy(),
            "load_loss": losses.load_loss.cpu().detach().numpy(),
            "total_load_balancing_loss": total_loss.cpu().detach().numpy(),
        }
        return total_loss, loss_info
    return None, None


def apply_loss(
    trg: torch.Tensor,
    prd_stats: PredictionStatistics,
    loss_fn: Callable[[torch.Tensor, PredictionStatistics], torch.Tensor],
    weighting_fns: Optional[Sequence[LossWeightingFunction]],
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
    if weighting_fns is not None:
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

    return aggregated_loss, loss_info


def make_loss_fn(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, PredictionStatistics], torch.Tensor],
    weighting_fns: Optional[Sequence[LossWeightingFunction]] = None,
) -> Callable[
    [torch.Tensor, PredictionStatistics], Tuple[torch.Tensor, LossInfo]
]:
    return partial(
        apply_loss,
        loss_fn=loss_fn,
        weighting_fns=weighting_fns,
        trg_variables_and_levels=model.trg_variables_and_levels,
    )

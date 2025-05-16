from dataclasses import fields
from typing import (
    Dict,
    List,
)

import torch
import torch.nn.functional as F
from torch import nn

from otter.models.architectures.moe import LoadBalancingLosses


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out


def pad_bottom_right(
    x: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """
    Pads a channel-last tensor (B, H, W, C) to target_height and target_width
    by adding padding only to the bottom and right sides.
    """
    input_height: int = x.shape[1]
    input_width: int = x.shape[2]

    pad_h: int = target_height - input_height
    pad_w: int = target_width - input_width

    # F.pad expects padding in reverse order of dimensions:
    # (..., pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
    # We only pad right (W) and bottom (H).
    # Input is (B, H, W, C) -> Dims (0, 1, 2, 3)
    # Padding order for F.pad: (pad_C_left, pad_C_right,
    # pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
    padding_tuple: tuple[int, ...] = (
        0,
        0,  # Pad C (dim 3) - No padding
        0,
        pad_w,  # Pad W (dim 2) - Pad only right
        0,
        pad_h,  # Pad H (dim 1) - Pad only bottom
    )

    out: torch.Tensor = F.pad(x, pad=padding_tuple, mode="constant", value=0.0)
    return out


def aggregate_load_balancing_losses(
    x: List[LoadBalancingLosses],
    mean: bool = False,
) -> LoadBalancingLosses | None:
    """Aggregates the list of LoadBalanscingLosses by taking a sum or a mean.

    Arguments:
        x: list of LoadBalancingLosses to aggregate
        mean: whether to take the mean or the sum

    Returns:
        Aggregated LoadBalancingLosses or None if the input list is empty
    """
    if x:
        output: Dict[str, torch.Tensor] = {}
        for field in fields(x[0]):
            gathered_loss = [getattr(loss_item, field.name) for loss_item in x]
            output[field.name] = torch.stack(gathered_loss, dim=0)
            if mean:
                output[field.name] = output[field.name].mean(dim=0)
            else:
                output[field.name] = output[field.name].sum(dim=0)
        return LoadBalancingLosses(**output)
    return None

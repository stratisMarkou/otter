from dataclasses import fields
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from otter.models.architectures.moe import LoadBalancingLosses


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out


def geopad(
    x: torch.Tensor, target_height: int, target_width: int
) -> tuple[torch.Tensor, int, int]:
    """
    Pads a channel-last tensor (B, H, W, C) to target_height and target_width by adding
    wrap-around padding to the left and right, and reflective padding to the bottom.
    """

    x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W) for padding

    _, _, input_height, input_width = x.shape

    # Validate inputs
    if target_height < input_height or target_width < input_width:
        raise ValueError(
            "Target dimensions must be greater than or equal to input dimensions."
        )

    # 1. Pad the width dimension with wrap-around padding
    pad_w = target_width - input_width
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_left < 1 or pad_right < 1:
        raise ValueError(
            "Width padding must be at least 1 pixel on both sides."
        )

    pad = (pad_left, pad_right, 0, 0)
    x = F.pad(x, pad=pad, mode="circular")

    # 2. Pad the height dimension with reflective padding at the bottom/top
    # TODO: This isn't quite right, e.g. Argentina should wrap to reflected Australia.
    pad_h = target_height - input_height
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if pad_top < 1 or pad_bottom < 1:
        raise ValueError(
            "Height padding must be at least 1 pixel on both sides."
        )

    pad = (0, 0, pad_top, pad_bottom)  # 0, 0 => no more padding on width
    x = F.pad(x, pad=pad, mode="reflect")

    result = x.permute(0, 2, 3, 1)  # Convert back to (B, H, W, C)

    return result, pad_left, pad_top


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

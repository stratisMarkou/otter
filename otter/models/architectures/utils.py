from typing import (
    Dict,
    List,
)

import torch
from torch import nn


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out


def interpolate_bilinear_channel_last(
    x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    out: torch.Tensor = torch.nn.functional.interpolate(
        x.permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return out


def merge_output_dicts(
    output_dicts: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Merge a list of output dictionaries.

    Arguments:
        output_dicts: list of output dictionaries

    Returns:
        merged output dictionary
    """
    merged_dict: dict[str, torch.Tensor] = {}
    for key in output_dicts[0].keys():
        merged_dict[key] = torch.stack([d[key] for d in output_dicts], dim=0)
    return merged_dict

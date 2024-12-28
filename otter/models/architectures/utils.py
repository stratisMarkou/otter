import torch  # type: ignore
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

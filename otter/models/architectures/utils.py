import torch
from torch import nn


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out

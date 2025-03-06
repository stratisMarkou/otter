import torch
from torch import nn

from otter.models.architectures.utils import conv_channel_last


class ImageTokeniser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        token_dimension: int,
        patch_size: int,
    ):
        super().__init__()

        assert (
            patch_size == 1 or patch_size % 2 == 0
        ), "Patch size must be one or even"

        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=token_dimension,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenise the image `x`, applying a strided convolution.
        This is equivalent to splitting the image into patches,
        and then linearly projecting each one of these using a
        shared linear projection.

        Arguments:
            x: image input tensor of shape (B, W, H, C)

        Returns:
            output tensor of shape (B, N, D)
        """
        _, H, W, _ = x.shape
        Hk, Wk = self.conv.kernel_size

        assert H % Hk == 0 and W % Wk == 0, (
            f"Input shape dims {x.shape=} must be divisible by "
            f"corresponding kernel dims, found {self.conv.kernel_size}."
        )

        return conv_channel_last(self.conv, x)

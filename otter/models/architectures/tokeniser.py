import torch
from torch import nn


class ImageTokeniser(nn.Module):

    def __init__(self, token_dimension: int, patch_size: int):
        super().__init__()

        assert (
            patch_size == 1 or patch_size % 2 == 0
        ), "Patch size must be one or even"

        self.conv = nn.Conv2d(
            filters=token_dimension,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            activation=None,
            data_format="channels_last",
        )

    def call(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenise the image `x`, applying a strided convolution.
        This is equivalent to splitting the image into patches,
        and then linearly projecting each one of these using a
        shared linear projection.

        Arguments:
            x: image input tensor of shape (B, W, H, C)

        Returns:
            output tensor of shape (B, N, D)
        """
        _, W, H, _ = x.shape
        Wk, Hk = self.conv.kernel_size

        assert W % Wk == 0 and H % Hk == 0, (
            f"Input shape dims {x.shape=} must be divisible by "
            f"corresponding kernel dims, found {self.conv.kernel_size}."
        )

        return self.conv(x)

import torch
from torch import nn

from otter.models.architectures.utils import (
    conv_channel_last,
    extract_subimages_and_fold_into_batch_dim,
)

from otter.models.architectures.attention import (  # isort: skip
    EfficientMultiHeadCrossAttention,
)


class LinearImageTokeniser(nn.Module):

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
            x: image input tensor of shape (B, H, W, C)

        Returns:
            output tensor of shape (B, H // S, W // S, D)
        """
        _, H, W, _ = x.shape
        Hk, Wk = self.conv.kernel_size

        assert H % Hk == 0 and W % Wk == 0, (
            f"Input shape dims {x.shape=} must be divisible by "
            f"corresponding kernel dims, found {self.conv.kernel_size}."
        )

        return conv_channel_last(self.conv, x)


class CrossAttentionImageTokeniser(nn.Module):

    def __init__(
        self,
        in_channels: int,
        token_dimension: int,
        patch_size: int,
        num_heads: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.token_dimension = token_dimension
        self.patch_size = patch_size

        self.varwise_q = nn.Parameter(
            torch.randn(token_dimension, patch_size**2),
            requires_grad=True,
        )
        self.varwise_attention = EfficientMultiHeadCrossAttention(
            num_heads=1,
            kq_token_dim=patch_size**2,
            v_token_dim=patch_size**2,
        )

        self.gridwise_q = nn.Parameter(
            torch.randn(1, token_dimension),
            requires_grad=True,
        )
        self.gridwise_attention = EfficientMultiHeadCrossAttention(
            num_heads=num_heads,
            kq_token_dim=token_dimension,
            v_token_dim=token_dimension,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, H, W, _ = x.shape
        S = self.patch_size
        D = self.token_dimension
        output_shape = (B, H // S, W // S, D)

        # First, extract patches and fold into token dimension.
        x = extract_subimages_and_fold_into_batch_dim(
            x, subimage_size=self.patch_size
        )  # (B * H//S * W//S, S**2, V)

        # Repeat variable-wise query and grid-wise query in the batch dim.
        Vq = self.varwise_q[None, :, :].expand(x.shape[0], -1, -1)
        Gq = self.gridwise_q[None, :, :].expand(x.shape[0], -1, -1)

        x = x.permute(0, 2, 1)  # (B * H//S * W//S, V, S**2)

        # Apply variable-wise cross attention.
        x = self.varwise_attention(
            q=Vq, k=x, v=x
        )  # (B * H//S * W//S, D, S**2)
        x = x.permute(0, 2, 1)  # (B * H//S * W//S, S**2, D)

        # Apply grid-wise cross attention.
        x = self.gridwise_attention(q=Gq, k=x, v=x)  # (B * H//S * W//S, 1, V)

        # Unfold from batch dimension and combine subimages.
        x = x.reshape(output_shape)

        return x

from typing import Tuple

import torch
from torch import nn

from otter.models.architectures.attention import MultiHeadSelfAttention
from otter.models.architectures.mlp import MLP


def extract_subimages(x: torch.Tensor, subimage_size: int) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x`.

    Arguments:
        x: input tensor of shape
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B*(H//2)*(W//2), 4, D)
    """
    B, H, W, D = x.shape
    S = subimage_size

    assert (
        H % S == 0
    ), f"Height {H} must be divisible by window size {S} ({x.shape=})"
    assert (
        W % S == 0
    ), f"Width {W} must be divisible by window size {S} ({x.shape=})"

    return torch.reshape(x, [B, H // S, S, W // S, S, D])


def extract_subimages_and_fold_into_batch_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the batch dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B*(H//S)*(W//S), S**2, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B * (H // S) * (W // S), S**2, D])


def extract_subimages_and_fold_into_token_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the token dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B, H // S, W // S, S**2 * D])


def combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Combine subimages extracted from input tensor `x`.

    Arguments:
        x: input tensor of shape (B, H//S, S, W//S, S, D)
        original_shape: shape of original tensor

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = original_shape
    S = subimage_size

    assert H % S == 0, f"Height {H} must be divisible by window size {S}"
    assert W % S == 0, f"Width {W} must be divisible by window size {S}"

    return torch.reshape(x, [B, H, W, D])


def unfold_from_batch_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the batch dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B*(H//S)*(W//S), S**2, D)
        original_shape: shape of original tensor
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def unfold_from_token_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the token dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B, H // S, W // S, S**2 * D)
        original_shape: shape of original tensor
        window_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def shift_horizontally_and_vertically(
    x: torch.Tensor, shift: int
) -> torch.Tensor:
    """Shift windows in the input tensor `x` by shift along its width and
    height. For example, using shift == 1 (and fixing an index for the B and D
    dimensions), the corresponding image would change as follows:

                      Original                Shifted
                 -----------------       -----------------
                |  x   x   x   o  |     |  *   +   +   +  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  +   +   +   *  |     |  o   x   x   x  |
                 -----------------       -----------------

    Arguments:
        x: input tensor of shape (B, H, W, D)
        shift: amount of shift to apply

    Returns:
        output tensor of shape (B, H, W, D)
    """
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have shape (B, H, W, D)")
    return torch.roll(torch.roll(x, shift, dims=1), shift, dims=2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
        token_proj_dim_factor: int,
    ):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            divide_projection_dim_by_factor=token_proj_dim_factor,
        )

        self.mlp = MLP(
            in_features=token_dim,
            hidden_features=mlp_num_hidden,
            num_hidden_layers=mlp_num_layers,
            output_features=token_dim,
        )

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformer block to input tokens `x`.

        Args:
            x: input tensor of shape (B, N, D)

        Returns:
            output tensor of shape (B, N, D)
        """
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class SwinTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
        num_block_pairs: int,
        window_size: int,
        token_proj_dim_factor: int,
    ):

        super().__init__()
        self.token_dim = token_dim

        self.first_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    token_dim=token_dim,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    token_proj_dim_factor=token_proj_dim_factor,
                )
                for _ in range(num_block_pairs)
            ]
        )

        self.second_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    token_dim=token_dim,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    token_proj_dim_factor=token_proj_dim_factor,
                )
                for _ in range(num_block_pairs)
            ]
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swin Transformer block to input tokens `x`.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        original_shape = x.shape
        S = self.window_size

        for first_block, second_block in zip(
            self.first_blocks, self.second_blocks
        ):

            # Apply first transformer block, extracting windows, applying
            # the transformer block to them, and re-combining them to the
            # original image.
            x = extract_subimages_and_fold_into_batch_dim(
                x, S
            )  # (B*H//S*W//S, S**2, D)
            x = first_block(x)  # (B*H//S*W//S, S**2, D)
            x = unfold_from_batch_dim_and_combine_subimages(
                x, original_shape, S
            )  # (B, H, W, D)

            # Apply second transformer block same as the first block, but
            # shifting the windows before and after the block.
            x = shift_horizontally_and_vertically(x, S // 2)  # (B, H, W, D)
            x = extract_subimages_and_fold_into_batch_dim(
                x, S
            )  # (B*H//S*W//S, S**2, D)
            x = second_block(x)  # (B*H//S*W//S, S**2, D)
            x = unfold_from_batch_dim_and_combine_subimages(
                x, original_shape, S
            )  # (B, H, W, D)
            x = shift_horizontally_and_vertically(x, -(S // 2))  # (B, H, W, D)

        # x = (x, original_shape, S)  # (B, H, W, D)

        return x

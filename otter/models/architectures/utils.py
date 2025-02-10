from typing import Tuple

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


def extract_subimages(x: torch.Tensor, subimage_size: int) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x`.

    Arguments:
        x: input tensor of shape
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H//S, S, W//S, S, D)
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

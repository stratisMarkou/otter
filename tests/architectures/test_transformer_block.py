"""Tests forward passes through transformer block."""

import pytest
import torch

from otter.models.architectures.transformer_blocks import (
    TransformerBlock,
    combine_windows,
    extract_windows,
)


# Tests that combine_windows is the inverse of the extract_windows.
@pytest.mark.parametrize(
    "batch_dim, height, width, channels, window_size",
    [
        (7, 4, 4, 3, 1),
        (7, 4, 4, 3, 2),
        (7, 8, 8, 3, 1),
        (7, 8, 8, 3, 4),
    ],
)
def test_extract_and_combine_windows(
    batch_dim: int, height: int, width: int, channels: int, window_size: int
) -> None:
    x = torch.rand((batch_dim, height, width, channels))
    extracted_windows = extract_windows(x, window_size)
    combined_windows = combine_windows(extracted_windows, x.shape, window_size)
    assert torch.allclose(x, combined_windows)


# Tests that the forward pass of the transformer block runs without error for
# different token dimensions, MLP hidden dimensions, MLP number of layers,
# number of heads, batch dimensions, and sequence lengths.
@pytest.mark.parametrize(
    "token_dim, mlp_num_hidden, mlp_num_layers, "
    + "num_heads, batch_dim, sequence_length",
    [
        (32, 64, 2, 8, 64, 128),
        (64, 128, 4, 16, 128, 256),
    ],
)
def test_transformer_block(
    token_dim: int,
    mlp_num_hidden: int,
    mlp_num_layers: int,
    num_heads: int,
    batch_dim: int,
    sequence_length: int,
) -> None:
    transformer_block = TransformerBlock(
        token_dim=token_dim,
        mlp_num_hidden=mlp_num_hidden,
        mlp_num_layers=mlp_num_layers,
        num_heads=num_heads,
    )
    x = torch.rand((batch_dim, sequence_length, token_dim))
    y = transformer_block(x)
    assert y.shape == (batch_dim, sequence_length, token_dim)

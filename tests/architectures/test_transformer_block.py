"""Tests forward passes through transformer block."""

import pytest
import torch

from otter.models.architectures.transformer import TransformerBlock


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
):
    transformer_block = TransformerBlock(
        token_dim=token_dim,
        mlp_num_hidden=mlp_num_hidden,
        mlp_num_layers=mlp_num_layers,
        num_heads=num_heads,
    )
    x = torch.rand((batch_dim, sequence_length, token_dim))
    y = transformer_block(x)
    assert y.shape == (batch_dim, sequence_length, token_dim)

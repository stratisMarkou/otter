import pytest
import torch

from otter.models.architectures import attention


@pytest.mark.parametrize(
    "token_dim, projection_dim, batch_dim, sequence_length",
    [
        (32, 16, 64, 128),
        (64, 32, 128, 256),
    ],
)
def test_self_attention_forward(
    token_dim: int,
    projection_dim: int,
    batch_dim: int,
    sequence_length: int,
):
    self_attention = attention.SelfAttention(token_dim, projection_dim)
    x = torch.rand((batch_dim, sequence_length, token_dim))
    y = self_attention(x)
    assert y.shape == (batch_dim, sequence_length, token_dim)


@pytest.mark.parametrize(
    "token_dim, projection_dim, batch_dim, sequence_length, num_heads",
    [
        (32, 8, 64, 128, 8),
        (64, 16, 128, 256, 16),
    ],
)
def test_multi_head_self_attention_forward(
    token_dim: int,
    projection_dim: int,
    batch_dim: int,
    sequence_length: int,
    num_heads: int,
):
    multi_head_self_attention = attention.MultiHeadSelfAttention(
        token_dim=token_dim,
        projection_dim=projection_dim,
        num_heads=num_heads,
    )
    x = torch.rand((batch_dim, sequence_length, token_dim))
    y = multi_head_self_attention(x)
    assert y.shape == (batch_dim, sequence_length, token_dim)

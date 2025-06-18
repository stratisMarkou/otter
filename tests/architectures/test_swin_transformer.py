from functools import partial
from typing import List

import pytest
import torch

from otter.models.architectures.embeddings import (
    PositionEmbedding,
    SpatialEmbedding,
)
from otter.models.architectures.mlp import MLP
from otter.models.architectures.transformer import SwinTransformer


# Tests that the forward pass of the transformer block runs without error for
# different token dims, input / output dims and number of blocks per stage.
@pytest.mark.parametrize(
    "grid_width, grid_height, grid_channel_dim, token_dim, patch_size, "
    "window_size, num_blocks_per_stage, num_bottom_blocks, pos_embedding",
    [
        (64, 32, 7, 12, 2, 2, [1, 1], 4, PositionEmbedding(12, 16, 32)),
        (64, 32, 7, 12, 2, 4, [2, 2], 4, SpatialEmbedding(32, 0.1, 720, 12)),
        (64, 32, 9, 16, 4, 2, [1, 1], 4, PositionEmbedding(16, 8, 16)),
        (
            128,
            64,
            9,
            16,
            4,
            4,
            [2, 2],
            8,
            SpatialEmbedding(64, 0.1, 720, 16),
        ),
        (64, 32, 7, 12, 2, 2, [1, 1], 4, PositionEmbedding(12, 16, 32)),
    ],
)
def test_swin_transformer(
    grid_width: int,
    grid_height: int,
    grid_channel_dim: int,
    token_dim: int,
    patch_size: int,
    window_size: int,
    num_blocks_per_stage: List[int],
    num_bottom_blocks: int,
    pos_embedding: PositionEmbedding | SpatialEmbedding,
) -> None:
    batch_dim = 2
    mlp_num_hidden = 32
    num_mlp_layers = 2
    num_heads = 4

    feedforward_network = partial(
        MLP,
        hidden_features=mlp_num_hidden,
        num_hidden_layers=num_mlp_layers,
    )

    swin_transformer = SwinTransformer(
        in_channels=grid_channel_dim,
        height=grid_height,
        width=grid_width,
        patch_size=patch_size,
        token_dimensions=token_dim,
        output_dimension=grid_channel_dim,
        window_size=window_size,
        num_heads=num_heads,
        num_blocks_per_stage=num_blocks_per_stage,
        num_bottom_blocks=num_bottom_blocks,
        pos_embedding=pos_embedding,
        # Seting use_efficient attention to False uses nn.MultiheadAttention
        # instead of xformers.ops.memory_efficient_attention, which is not
        # runnable in the test environment when we don't have a GPU.
        use_efficient_attention=False,
        feedforward_network=feedforward_network,
    )

    # Create a random input tensor
    x = torch.randn(batch_dim, grid_height, grid_width, grid_channel_dim)

    # Run the forward pass
    y, _ = swin_transformer(x)

    # Check that the output tensor has the correct shape
    assert y.shape == (batch_dim, grid_height, grid_width, grid_channel_dim)

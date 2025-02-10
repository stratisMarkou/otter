from typing import Callable

import torch
from torch import nn

from otter.models.architectures.mlp import MLP
from otter.models.architectures.utils import (
    extract_subimages_and_fold_into_batch_dim,
    shift_horizontally_and_vertically,
    unfold_from_batch_dim_and_combine_subimages,
)

from otter.models.architectures.attention import (  # isort:skip
    EfficientMultiHeadSelfAttention,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        self.mhsa: Callable[[torch.Tensor], torch.Tensor]

        if use_efficient_attention:
            self.efficient_mha_layer = EfficientMultiHeadSelfAttention(
                num_heads=num_heads,
                token_dim=token_dim,
            )
            self.mhsa = lambda x: (self.efficient_mha_layer(x, x, x))

        else:
            self.mha_layer = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=num_heads,
            )
            self.mhsa = lambda x: (self.mha_layer(x, x, x)[0])

        self.mlp = MLP(
            in_features=token_dim,
            hidden_features=mlp_num_hidden,
            num_hidden_layers=mlp_num_layers,
            output_features=token_dim,
        )

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer block to input tokens `x`.

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
        window_size: int,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        # Transformer block applied before the shift operation.
        self.first_block = TransformerBlock(
            token_dim=token_dim,
            mlp_num_hidden=mlp_num_hidden,
            mlp_num_layers=mlp_num_layers,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
        )

        # Transformer block applied after the shift operation.
        self.second_block = TransformerBlock(
            token_dim=token_dim,
            mlp_num_hidden=mlp_num_hidden,
            mlp_num_layers=mlp_num_layers,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
        )

        # Window size for extracting subimages.
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

        # Apply first transformer block, extracting windows, applying
        # the transformer block to them, and re-combining them to the
        # original image.
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)
        x = self.first_block(x)  # (B*H//S*W//S, S**2, D)
        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)

        # Apply second transformer block same as the first block, but
        # shifting the windows before and after the block.
        x = shift_horizontally_and_vertically(x, S // 2)  # (B, H, W, D)
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)
        x = self.second_block(x)  # (B*H//S*W//S, S**2, D)
        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)
        return shift_horizontally_and_vertically(x, -(S // 2))  # (B, H, W, D)


def make_swin_stage(
    token_dim: int,
    mlp_num_hidden: int,
    mlp_num_layers: int,
    num_heads: int,
    num_swin_blocks: int,
    window_size: int,
    use_efficient_attention: bool = True,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        *[
            SwinTransformerBlock(
                token_dim=token_dim,
                mlp_num_hidden=mlp_num_hidden,
                mlp_num_layers=mlp_num_layers,
                num_heads=num_heads,
                window_size=window_size,
                use_efficient_attention=use_efficient_attention,
            )
            for _ in range(num_swin_blocks)
        ]
    )

from typing import List

import torch
from torch import nn

from otter.models.architectures.embeddings import PositionEmbedding
from otter.models.architectures.tokeniser import ImageTokeniser
from otter.models.architectures.transformer_blocks import SwinTransformerBlock
from otter.models.architectures.utils import (
    conv_channel_last,
    interpolate_bilinear_channel_last,
)


class PatchConv2d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        return out


class PatchConvTranspose2d(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.transpose_conv(x)
        return out


class SwinTransformer(nn.Module):

    def __init__(
        self,
        token_dimension: int,
        output_dimension: int,
        window_size: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int,
        num_blocks_per_stage: List[int],
    ):
        super().__init__()

        assert len(num_blocks_per_stage) > 0

        down_swin_stages = []
        up_swin_stages = []
        patch_downsampling_layers = []
        patch_upsampling_layers = []

        current_token_dimension = token_dimension
        for num_blocks in num_blocks_per_stage:
            patch_downsampling_layers.append(
                PatchConv2d(in_channels=current_token_dimension)
            )
            current_token_dimension = current_token_dimension * 2

            down_swin_stages.append(
                SwinTransformerBlock(
                    token_dim=current_token_dimension,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    num_block_pairs=num_blocks,
                    window_size=window_size,
                )
            )

        for num_blocks in num_blocks_per_stage[::-1]:
            up_swin_stages.append(
                SwinTransformerBlock(
                    token_dim=current_token_dimension,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    num_block_pairs=num_blocks,
                    window_size=window_size,
                )
            )
            patch_upsampling_layers.append(
                PatchConvTranspose2d(out_channels=current_token_dimension // 2)
            )
            current_token_dimension = current_token_dimension // 2

        self.down_swin_stages = torch.nn.ModuleList(down_swin_stages)
        self.up_swin_stages = torch.nn.ModuleList(up_swin_stages)
        self.patch_downsampling_layers = torch.nn.ModuleList(
            patch_downsampling_layers
        )
        self.patch_upsampling_layers = torch.nn.ModuleList(
            patch_upsampling_layers
        )

        self.tokeniser = ImageTokeniser(
            in_channels=in_channels,
            token_dimension=token_dimension,
            patch_size=patch_size,
        )

        self.embedding = PositionEmbedding(
            token_dimension=token_dimension,
            height=height // patch_size,
            width=width // patch_size,
        )

        self.final_conv = nn.ConvTranspose2d(
            in_channels=token_dimension,
            out_channels=output_dimension,
            stride=(patch_size, patch_size),
            kernel_size=(patch_size, patch_size),
        )

        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vision transformer to batch of images.

        Arguments:
            x: input image tensor of shape (B, H, W, C)

        Returns:
            output logits tensor of shape (B, H, W, C)
        """
        _, original_height, original_width, _ = x.shape

        # Interpolate image to target dimensions
        x = interpolate_bilinear_channel_last(
            x, height=self.height, width=self.width
        )

        x = self.tokeniser(x)
        x = self.embedding(x)
        skips = []

        for downsampling_conv, down_swin_stage in zip(
            self.patch_downsampling_layers,
            self.down_swin_stages,
        ):
            skips.append(x)
            x = downsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = down_swin_stage(x)

        for upsampling_conv, up_swin_stage, skip in zip(
            self.patch_upsampling_layers,
            self.up_swin_stages,
            skips[::-1],
        ):
            x = up_swin_stage(x)
            x = upsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + skip

        x = conv_channel_last(self.final_conv, x)

        # Interpolate image to original dimensions
        return interpolate_bilinear_channel_last(
            x, height=original_height, width=original_width
        )

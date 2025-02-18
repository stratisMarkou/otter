from typing import Sequence

import torch
from torch import nn

from otter.models.architectures.embeddings import PositionEmbedding
from otter.models.architectures.tokeniser import ImageTokeniser
from otter.models.architectures.transformer_blocks import make_swin_stage
from otter.models.architectures.utils import (
    conv_channel_last,
    interpolate_bilinear_channel_last,
)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        token_dimensions: Sequence[int] | int,
        output_dimension: int,
        window_size: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int,
        num_blocks_per_stage: Sequence[int],
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        assert len(num_blocks_per_stage) > 0

        if isinstance(token_dimensions, int):
            num_stages = len(num_blocks_per_stage)
            token_dimensions = [token_dimensions] * (num_stages + 1)

        down_swin_stages = []
        up_swin_stages = []
        patch_downsampling_layers = []
        patch_upsampling_layers = []
        token_in_out_dims = list(
            zip(token_dimensions[:-1], token_dimensions[1:])
        )

        for num_blocks, (t1, t2) in zip(
            num_blocks_per_stage,
            token_in_out_dims,
        ):
            patch_downsampling_layers.append(
                nn.Conv2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )

            down_swin_stages.append(
                make_swin_stage(
                    token_dim=t2,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                )
            )

        for num_blocks, (t2, t1) in zip(
            num_blocks_per_stage[::-1],
            token_in_out_dims[::-1],
        ):
            up_swin_stages.append(
                make_swin_stage(
                    token_dim=t1,
                    mlp_num_hidden=mlp_num_hidden,
                    mlp_num_layers=mlp_num_layers,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                )
            )
            patch_upsampling_layers.append(
                nn.ConvTranspose2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )

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
            token_dimension=token_dimensions[0],
            patch_size=patch_size,
        )

        self.embedding = PositionEmbedding(
            token_dimension=token_dimensions[0],
            height=height // patch_size,
            width=width // patch_size,
        )

        self.final_conv = nn.ConvTranspose2d(
            in_channels=token_dimensions[0],
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

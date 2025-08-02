from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)

import torch
from torch import nn

from otter.models.architectures.attention import EfficientMultiHeadAttention
from otter.models.architectures.moe import LoadBalancingLosses


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

    assert H % S == 0, (
        f"Height {H} must be divisible by window size {S} ({x.shape=})"
    )
    assert W % S == 0, (
        f"Width {W} must be divisible by window size {S} ({x.shape=})"
    )

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
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        dropout_rate: float,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        self.mhsa: Callable[
            [torch.Tensor, Optional[torch.Tensor]], torch.Tensor
        ]

        if use_efficient_attention:
            self.efficient_mha_layer = EfficientMultiHeadAttention(
                num_heads=num_heads,
                token_dim=token_dim,
                attn_drop=dropout_rate,
                proj_drop=dropout_rate,
            )
            self.mhsa = lambda x, attn_mask: (
                self.efficient_mha_layer(x, x, x, attn_mask=attn_mask)
            )

        else:
            self.mha_layer = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.mhsa = lambda x, attn_mask: (
                self.mha_layer(x, x, x, attn_mask=attn_mask)[0]
            )
            # Just skip using attention mask for the CPU-based attention.

        self.ffn = feedforward_network(token_dim, token_dim)

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        """Apply the transformer block to input tokens `x`.

        Args:
            x: input tensor of shape (B, N, D)

        Returns:
            output tensor of shape (B, N, D)
        """
        x = x + self.mhsa(self.ln1(x), attention_mask)
        x_ffn, load_balancing_losses = self.ffn(self.ln2(x))
        x = x + x_ffn

        return x, load_balancing_losses


class SwinTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        window_size: int,
        dropout_rate: float,
        height: int,
        width: int,
        batch_size: int,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        # Transformer block applied before the shift operation.
        self.first_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
            dropout_rate=dropout_rate,
        )

        # Transformer block applied after the shift operation.
        self.second_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
            dropout_rate=dropout_rate,
        )

        # Window size for extracting subimages.
        self.window_size = window_size

        attention_mask = self._compute_attention_mask(
            window_size=window_size,
            batch_size=batch_size,
            height=height,
            width=width,
            num_heads=num_heads,
        )
        # (num_windows * B, num_heads, window_size**2, window_size**2)
        if not use_efficient_attention:
            # nn.MultiheadAttention expects the attention mask to be
            # (num_windows * B * num_heads, window_size**2, window_size**2)
            # See https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            attention_mask = attention_mask.reshape(
                -1, window_size**2, window_size**2
            )
        self.register_buffer("attention_mask", attention_mask)

    def _compute_attention_mask(
        self,
        window_size: int,
        batch_size: int,
        height: int,
        width: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Compute the attention mask for the Swin Transformer block.

        This mask ensures that attention is only applied across the East/West
        boundary within each swin window, and not across the North/South.
        """
        shift = window_size // 2

        # Note: it's slightly memory inefficient to create the batch size
        # dimension here, but it allows us to use the same code for folding
        # and unfolding the subimages.
        image_mask = torch.zeros((batch_size, height, width, 1))

        # Divide the image into 3 regions from top to bottom (north to south):
        # 1. Top region: all of these windows should share information via attention,
        #   so we won't need a mask here.
        # 2. Middle region: this is the the last shift-sized region at the very bottom
        #   of the original image, which should not share information with the top region.
        # 3. Bottom region: this is the first shift-sized region at the very top of the
        #   original image, which has been rolled to the bottom of the image. It should
        #   not share information with the middle region.
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift),
            slice(-shift, None),
        )
        for count, h_slice in enumerate(h_slices):
            image_mask[:, h_slice, :, :] = count

        im_mask_windows = extract_subimages_and_fold_into_batch_dim(
            image_mask, self.window_size
        )  # (num_windows * B, window_size**2, 1)

        # This mask ensures that attention is only applied for regions with the same
        # value in the mask, i.e., the top region can attend to itself, but
        # not to the middle or bottom regions.
        attention_mask = im_mask_windows.unsqueeze(
            1
        ) - im_mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, float("-inf")
        )
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0, float(0.0)
        )
        attention_mask = attention_mask.squeeze().unsqueeze(1)
        # (num_windows * B, 1, window_size**2, window_size**2)

        return attention_mask.expand(-1, num_heads, -1, -1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, List[LoadBalancingLosses]]:
        """Apply the Swin Transformer block to input tokens `x`.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        original_shape = x.shape
        S = self.window_size

        load_balancing_losses_list: List[LoadBalancingLosses] = list()
        # Apply first transformer block, extracting windows, applying
        # the transformer block to them, and re-combining them to the
        # original image.
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)
        x, load_balancing_losses = self.first_block(
            x
        )  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)

        # Apply second transformer block same as the first block, but
        # shifting the windows before and after the block.
        x = shift_horizontally_and_vertically(x, -S // 2)  # (B, H, W, D)
        x = extract_subimages_and_fold_into_batch_dim(
            x, S
        )  # (B*H//S*W//S, S**2, D)

        x, load_balancing_losses = self.second_block(
            x, self.attention_mask.to(x.dtype)
        )  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)
        x = shift_horizontally_and_vertically(x, S // 2)  # (B, H, W, D)

        return x, load_balancing_losses_list


def make_swin_stage(
    token_dim: int,
    feedforward_network: Callable[[int, int], nn.Module],
    num_heads: int,
    num_swin_blocks: int,
    window_size: int,
    dropout_rate: float,
    height: int,
    width: int,
    batch_size: int,
    use_efficient_attention: bool = True,
) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [
            SwinTransformerBlock(
                token_dim=token_dim,
                feedforward_network=feedforward_network,
                num_heads=num_heads,
                window_size=window_size,
                use_efficient_attention=use_efficient_attention,
                dropout_rate=dropout_rate,
                height=height,
                width=width,
                batch_size=batch_size,
            )
            for _ in range(num_swin_blocks)
        ]
    )

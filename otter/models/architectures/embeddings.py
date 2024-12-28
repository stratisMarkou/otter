import torch
from torch import nn


class PositionEmbedding(nn.Module):

    def __init__(
        self,
        token_dimension: int,
        height: int,
        width: int,
        init_scale: float = 1e-1,
    ):
        super().__init__()

        self.embeddings = nn.Parameter(
            init_scale * torch.randn(height, width, token_dimension),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input tensor.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        return x + self.embeddings[None, :, :, :]

import torch
from torch import nn

from otter.models.architectures.attention import MultiHeadSelfAttention
from otter.models.architectures.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        mlp_num_hidden: int,
        mlp_num_layers: int,
        num_heads: int,
    ):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(
            token_dim=token_dim,
            projection_dim=token_dim,
            num_heads=num_heads,
        )

        self.mlp = MLP(
            in_features=token_dim,
            hidden_features=mlp_num_hidden,
            num_hidden_layers=mlp_num_layers,
            output_features=token_dim,
        )

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformer block to input tokens `x`.

        Args:
            x: input tensor of shape (B, N, D)

        Returns:
            output tensor of shape (B, N, D)
        """
        x = x + self.mhsa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

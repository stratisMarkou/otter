import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, token_dim: int, projection_dim: int):
        super().__init__()
        self.Uk = nn.Linear(token_dim, projection_dim, bias=False)
        self.Uq = nn.Linear(token_dim, projection_dim, bias=False)

    def self_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention weights for tokens in a sequence

        Args:
            x: input sequence of tokens, shape (B, N, D)

        Returns:
            attention weights, shape (B, N, N)
        """
        B, N, _ = x.shape
        k = self.Uk(x)
        q = self.Uq(x)

        dot_product = torch.matmul(k, q.transpose(1, 2))
        dot_product /= torch.sqrt(
            torch.tensor(k.shape[-1], dtype=torch.float32)
        )
        assert dot_product.shape == (B, N, N)

        self_attention_weights = torch.nn.functional.softmax(
            dot_product,
            dim=2,
        )
        assert self_attention_weights.shape == (B, N, N)

        return self_attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to a sequence of tokens

        Args:
            x: input sequence of tokens, shape (B, N, D)

        Returns:
            output sequence of tokens, shape (B, N, D)
        """
        B, N, D = x.shape
        attended_tokens = torch.matmul(self.self_attention_weights(x), x)
        assert attended_tokens.shape == (B, N, D)

        return attended_tokens


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        divide_projection_dim_by_factor: int,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"{token_dim=} must be divisible by {num_heads=}."
            )
        self.num_heads = num_heads
        projection_dim = token_dim // num_heads
        projection_dim = projection_dim // divide_projection_dim_by_factor
        self.heads = nn.ModuleList(
            [
                SelfAttention(token_dim, projection_dim)
                for _ in range(num_heads)
            ]
        )
        import logging

        logging.warning(
            f"About to initialise linear layer with {token_dim**2/1e6:.1f}M "
            f"params.\n\t\tDo we really need linear layers in MHSA? "
            f"The result is passed through a linear layer in the MLP anyway.\n"
        )
        self.linear = nn.Linear(token_dim, token_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention to a sequence of tokens

        Args:
            x: input sequence of tokens, shape (B, N, D)

        Returns:
            output sequence of tokens, shape (B, N, D)
        """

        x = torch.stack([head(x) for head in self.heads], dim=-1)
        return self.linear(torch.mean(x, dim=-1))  # type: ignore

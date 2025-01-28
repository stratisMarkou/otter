import torch
from torch import nn
from xformers.ops import memory_efficient_attention  # type: ignore


class EfficientMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        token_dim: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = token_dim // num_heads

        self.scale = head_dim**-0.5
        self.Uq = nn.Linear(token_dim, token_dim, bias=qkv_bias)
        self.Uk = nn.Linear(token_dim, token_dim, bias=qkv_bias)
        self.Uv = nn.Linear(token_dim, token_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(token_dim, token_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        x: torch.Tensor = memory_efficient_attention(
            self.Uq(q), self.Uk(k), self.Uv(v)
        )
        x = self.proj_drop(self.proj(x))

        return x

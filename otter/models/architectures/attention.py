import torch
from torch import nn
from xformers.ops import (  # type: ignore
    memory_efficient_attention,
    unbind,
)


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
        self.qkv = nn.Linear(token_dim, token_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(token_dim, token_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, C = x.shape

        qkv: torch.Tensor = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return x

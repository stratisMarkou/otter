import torch
from torch import nn
from xformers.ops import memory_efficient_attention  # type: ignore


class EfficientMultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        kq_token_dim: int,
        v_token_dim: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.Uq = nn.Linear(kq_token_dim, kq_token_dim, bias=qkv_bias)
        self.Uk = nn.Linear(kq_token_dim, kq_token_dim, bias=qkv_bias)
        self.Uv = nn.Linear(v_token_dim, v_token_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_token_dim, v_token_dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:

        B, Nq, Dqk = q.shape
        _, Nkv, _ = k.shape
        _, Nkv, Dv = v.shape

        q = self.Uq(q).reshape(B, Nq, self.num_heads, Dqk // self.num_heads)
        k = self.Uk(k).reshape(B, Nkv, self.num_heads, Dqk // self.num_heads)
        v = self.Uv(v).reshape(B, Nkv, self.num_heads, Dv // self.num_heads)

        x: torch.Tensor = memory_efficient_attention(q, k, v)
        x = x.reshape(B, Nq, Dv)
        x = self.proj_drop(self.proj(x))

        assert x.shape == (B, Nq, Dv)

        return x


class EfficientMultiHeadSelfAttention(EfficientMultiHeadCrossAttention):
    def __init__(
        self,
        num_heads: int,
        token_dim: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            kq_token_dim=token_dim,
            v_token_dim=token_dim,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

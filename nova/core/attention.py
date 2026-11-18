"""Causal self-attention and multi-head attention for NOVA Tiny.

Implements scaled dot-product causal self-attention with a lower-triangular
mask so each position can only attend to itself and earlier positions. All
operations run on CPU by default.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Scaled dot-product causal self-attention for a single head."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Single linear projection produces query, key, and value together.
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads: (B, T, H, HD) -> (B, H, T, HD)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask: forbid attending to future positions.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(
            1, 1, seq_len, seq_len
        )
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, T, HD)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        out = self.out_proj(out)
        return out

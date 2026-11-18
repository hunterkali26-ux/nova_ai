"""Transformer decoder block for NOVA Tiny.

Each block applies causal multi-head self-attention followed by a position-wise
feed-forward network, with residual connections and layer normalization around
each sub-layer (pre-norm style).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class DecoderBlock(nn.Module):
    """A single decoder-only Transformer block (pre-norm)."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention block with residual connection.
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        # Pre-norm feed-forward block with residual connection.
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        return x

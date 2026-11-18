"""NOVA Tiny: a small decoder-only Transformer language model built from scratch.

The model is intentionally small and educational. It is *not* pretrained and
should not be expected to behave like an instruction-tuned assistant. It is
trained from scratch on a user-provided corpus.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, get_preset
from .transformer import DecoderBlock


class NOVATinyModel(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.context_length = config.context_length
        self.vocab_size = config.vocab_size

        # Token and positional embeddings.
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer decoder blocks.
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config.embedding_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm + language modeling head.
        self.ln_final = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the output head.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Mark the projection layers for a different (scaled) init if desired.
        for module in self.modules():
            if isinstance(module, (nn.Linear,)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass.

        Args:
            input_ids: (B, T) token ids.
            targets: (B, T) target token ids for cross-entropy loss. If None,
                no loss is computed.
            mask: optional causal mask override.

        Returns:
            (logits, loss) where logits is (B, T, vocab_size) and loss is a
            scalar tensor or None.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift so we predict the next token.
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @classmethod
    def from_preset(cls, name: str) -> "NOVATinyModel":
        """Build a model from a named preset."""
        config = get_preset(name)
        return cls(config)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "num_parameters": self.num_parameters,
            "estimated_parameters": self.config.estimated_params,
        }

    def save(self, path: str | Path) -> None:  # type: ignore[name-defined]
        """Save model weights + config to a .pt checkpoint."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
                "num_parameters": self.num_parameters,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NOVATinyModel":
        """Load a model from a checkpoint written by ``save``."""
        from pathlib import Path

        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

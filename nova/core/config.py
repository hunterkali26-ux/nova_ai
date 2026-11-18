"""Project-wide configuration, paths, and model presets for NOVA AI.

All paths resolve relative to the project root so the application works
regardless of the current working directory. Defaults are conservative to
fit an 8 GB RAM, CPU-only laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
CONFIG_DIR: Path = PROJECT_ROOT / "config"
EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"
DATABASES_DIR: Path = PROJECT_ROOT / "databases"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

CONVERSATIONS_DB: Path = DATABASES_DIR / "conversations.db"
SETTINGS_DB: Path = DATABASES_DIR / "settings.db"
KNOWLEDGE_DB: Path = DATABASES_DIR / "knowledge.db"
TRAINING_DB: Path = DATABASES_DIR / "training.db"

# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------

DEFAULT_DEVICE: str = "cpu"
DEFAULT_CONTEXT_LENGTH: int = 256
DEFAULT_BATCH_SIZE: int = 2
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_CPU_THREADS: int = 4
DEFAULT_MAX_NEW_TOKENS: int = 64
DEFAULT_TEMPERATURE: float = 0.8
DEFAULT_TOP_K: int = 50
DEFAULT_TOP_P: float = 0.9
DEFAULT_REPETITION_PENALTY: float = 1.1
GRADIENT_ACCUMULATION_STEPS: int = 4
GRAD_CLIP_NORM: float = 1.0

MAX_RECOMMENDED_PARAMS: int = 8_000_000
WARN_RAM_GB: float = 6.0


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        CONFIG_DIR,
        EMBEDDINGS_DIR,
        DATABASES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a NOVA Tiny decoder-only Transformer."""

    name: str
    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    context_length: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    grad_clip_norm: float = GRAD_CLIP_NORM
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    device: str = DEFAULT_DEVICE

    @property
    def estimated_params(self) -> int:
        """Rough parameter count (dominated by embeddings + output head)."""
        d = self.embedding_dim
        v = self.vocab_size
        l = self.num_layers
        embed_params = 2 * v * d
        per_layer = 4 * d * d + 8 * d * d + 4 * d
        lm_head = v * d
        return embed_params + l * per_layer + lm_head

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "context_length": self.context_length,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "grad_clip_norm": self.grad_clip_norm,
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ModelConfig":
        return cls(
            name=str(data.get("name", "nova-tiny")),
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data["embedding_dim"]),
            num_heads=int(data["num_heads"]),
            num_layers=int(data["num_layers"]),
            context_length=int(data["context_length"]),
            dropout=float(data["dropout"]),
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            weight_decay=float(data["weight_decay"]),
            gradient_accumulation_steps=int(
                data.get("gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS)
            ),
            grad_clip_norm=float(data.get("grad_clip_norm", GRAD_CLIP_NORM)),
            max_new_tokens=int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
            device=str(data.get("device", DEFAULT_DEVICE)),
        )


MODEL_PRESETS: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        name="nova-tiny",
        vocab_size=400,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        context_length=DEFAULT_CONTEXT_LENGTH,
        dropout=0.1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    "small": ModelConfig(
        name="nova-small",
        vocab_size=800,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        context_length=DEFAULT_CONTEXT_LENGTH,
        dropout=0.1,
        batch_size=2,
        learning_rate=8e-4,
        weight_decay=0.01,
    ),
    "experimental": ModelConfig(
        name="nova-experimental",
        vocab_size=1600,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        context_length=512,
        dropout=0.1,
        batch_size=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
    ),
}


def get_preset(name: str) -> ModelConfig:
    """Return a model preset by name, raising a clear error if unknown."""
    key = name.lower()
    if key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset '{name}'. Available: {sorted(MODEL_PRESETS)}"
        )
    return MODEL_PRESETS[key]


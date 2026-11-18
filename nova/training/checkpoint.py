"""Checkpoint persistence: save/load model, optimizer, and training state.

A checkpoint contains everything needed to resume training:
- model state_dict
- optimizer state_dict
- model configuration
- tokenizer configuration
- training step, epoch, learning rate
- training history + metadata
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..core.config import CHECKPOINTS_DIR, ModelConfig
from ..core.tokenizer import TokenizerConfig
from ..training.metrics import TrainingHistory


def checkpoint_path(model_name: str, step: int, base_dir: Path | None = None) -> Path:
    base = base_dir or CHECKPOINTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{model_name}_step{step}.pt"


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    epoch: int,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    history: TrainingHistory,
    path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "epoch": int(epoch),
        "model_config": model_config.to_dict(),
        "tokenizer_config": tokenizer_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history.to_dict(),
        "metadata": metadata or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    # Atomic replace.
    shutil.move(str(tmp), str(path))
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint file and return its payload dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location)
    return payload


def latest_checkpoint(model_name: str, base_dir: Path | None = None) -> Optional[Path]:
    """Return the most recent checkpoint file for a model name, if any."""
    base = base_dir or CHECKPOINTS_DIR
    if not base.exists():
        return None
    candidates = sorted(base.glob(f"{model_name}_step*.pt"))
    if not candidates:
        return None
    return candidates[-1]

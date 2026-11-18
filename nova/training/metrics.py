"""Training metrics tracking for NOVA Tiny."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepMetrics:
    step: int
    epoch: int
    train_loss: float
    learning_rate: float
    elapsed_seconds: float


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    perplexity: Optional[float]
    learning_rate: float
    elapsed_seconds: float


@dataclass
class TrainingHistory:
    steps: List[StepMetrics] = field(default_factory=list)
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None

    def record_step(self, step, epoch, train_loss, learning_rate, elapsed) -> None:
        self.steps.append(StepMetrics(step, epoch, train_loss, learning_rate, elapsed))

    def record_epoch(self, epoch, train_loss, val_loss, learning_rate, elapsed) -> None:
        perplexity = None
        if val_loss is not None:
            try:
                import math

                perplexity = math.exp(min(val_loss, 20.0))
            except OverflowError:
                perplexity = float("inf")
        self.epochs.append(
            EpochMetrics(epoch, train_loss, val_loss, perplexity, learning_rate, elapsed)
        )
        if val_loss is not None and (self.best_val_loss is None or val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def to_dict(self) -> dict:
        return {
            "steps": [
                {"step": s.step, "epoch": s.epoch, "train_loss": s.train_loss, "learning_rate": s.learning_rate, "elapsed_seconds": s.elapsed_seconds}
                for s in self.steps
            ],
            "epochs": [
                {
                    "epoch": e.epoch,
                    "train_loss": e.train_loss,
                    "val_loss": e.val_loss,
                    "perplexity": e.perplexity,
                    "learning_rate": e.learning_rate,
                    "elapsed_seconds": e.elapsed_seconds,
                }
                for e in self.epochs
            ],
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

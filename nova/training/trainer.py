"""Training loop for NOVA Tiny.

Runs on CPU by default with gradient accumulation, gradient clipping, AdamW,
and an optional cosine learning-rate scheduler. Designed to run in a
background thread so training can be started/stopped without blocking chat.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from . import evaluator
from .metrics import TrainingHistory

from ..core.config import DEFAULT_DEVICE
from ..core.tokenizer import TokenizerConfig
from ..core.logging import get_logger

logger = get_logger("trainer")


@dataclass
class TrainConfig:
    model_name: str = "nova-tiny"
    model_preset: str = "tiny"
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    grad_clip_norm: float = 1.0
    context_length: int = 256
    warmup_steps: int = 50
    lr_scheduler: str = "cosine"  # "cosine", "constant", "linear"
    early_stopping: bool = False
    early_stopping_patience: int = 3
    save_every_steps: int = 100
    device: str = DEFAULT_DEVICE
    seed: int = 42


class Trainer:
    """Orchestrates a training run.

    Attributes:
        stop_requested: set this flag (e.g. from another thread) to request a
            graceful stop at the next checkpoint boundary.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        config: TrainConfig,
        model_config,
        tokenizer_config: TokenizerConfig,
        on_step: Optional[Callable[[int, float], None]] = None,
        on_epoch: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.on_step = on_step
        self.on_epoch = on_epoch

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.history = TrainingHistory()

        self.step = 0
        self.epoch = 0
        self.start_epoch = 0
        self.start_step = 0
        self.stop_requested = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _build_scheduler(self):
        total_steps = max(1, len(self.train_dataloader) * self.config.epochs)
        if self.config.lr_scheduler == "constant":
            return LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        if self.config.lr_scheduler == "linear":
            return LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps),
            )

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = (step - self.config.warmup_steps) / float(
                max(1, total_steps - self.config.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)


    # ------------------------------------------------------------------ #
    # Checkpoint / resume
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            path = checkpoint_path_name(self.config.model_name, self.step)
        from .checkpoint import save_checkpoint as _save

        return _save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=self.epoch,
            model_config=self.model_config,
            tokenizer_config=self.tokenizer_config,
            history=self.history,
            path=path,
            metadata={
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            },
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint state into this trainer (resume training)."""
        from .checkpoint import load_checkpoint as _load

        payload = _load(path, map_location=str(self.device))
        self.model.load_state_dict(payload["model_state_dict"])
        if self.optimizer is not None and payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and payload.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.step = int(payload.get("step", 0))
        self.epoch = int(payload.get("epoch", 0))
        self.start_epoch = self.epoch
        self.start_step = self.step
        if payload.get("history"):
            StepMetrics = __import__(
                "nova.training.metrics", fromlist=["StepMetrics"]
            ).StepMetrics
            self.history = TrainingHistory()
            self.history.steps = [
                StepMetrics(**s) for s in payload["history"].get("steps", [])
            ]
            self.history.epochs = payload["history"].get("epochs", [])
            self.history.best_val_loss = payload["history"].get("best_val_loss")
            self.history.best_epoch = payload["history"].get("best_epoch")
        logger.info("Resumed training from %s (step=%d, epoch=%d)", path, self.step, self.epoch)



    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, resume_from: Optional[str | Path] = None) -> TrainingHistory:
        """Run the full training loop. Blocks until finished or stopped."""
        if resume_from:
            self.load_checkpoint(resume_from)

        self.model.train()
        self._running = True
        self.stop_requested = False
        self._stop_event.clear()
        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                self.epoch = epoch
                if hasattr(self.train_dataloader.dataset, "reshuffle"):
                    self.train_dataloader.dataset.reshuffle()
                epoch_loss = 0.0
                epoch_batches = 0

                self.optimizer.zero_grad(set_to_none=True)
                for batch_idx, batch in enumerate(self.train_dataloader):
                    if self.stop_requested or self._stop_event.is_set():
                        logger.info("Stop requested at epoch %d step %d", epoch, self.step)
                        break

                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    _, loss = self.model(inputs, targets=targets)
                    if loss is None:
                        continue
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    epoch_loss += float(loss.item()) * self.config.gradient_accumulation_steps
                    epoch_batches += 1

                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.step += 1

                        lr = self.optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - start_time
                        self.history.record_step(
                            self.step,
                            epoch,
                            float(loss.item()) * self.config.gradient_accumulation_steps,
                            lr,
                            elapsed,
                        )
                        if self.on_step:
                            self.on_step(self.step, self.history.steps[-1].train_loss)

                        if self.step % self.config.save_every_steps == 0:
                            self.save_checkpoint()

                if self.stop_requested or self._stop_event.is_set():
                    break

                avg_epoch_loss = epoch_loss / max(1, epoch_batches)
                val_loss, val_ppl = evaluator.evaluate(
                    self.model, self.val_dataloader, device=str(self.device)
                )
                elapsed = time.time() - start_time
                self.history.record_epoch(
                    epoch, avg_epoch_loss, val_loss, self.optimizer.param_groups[0]["lr"], elapsed
                )
                epoch_metrics = self.history.epochs[-1]
                if self.on_epoch:
                    self.on_epoch(epoch_metrics)

                logger.info(
                    "Epoch %d train_loss=%.4f val_loss=%s lr=%.6f",
                    epoch,
                    avg_epoch_loss,
                    f"{val_loss:.4f}" if val_loss == val_loss else "nan",
                    self.optimizer.param_groups[0]["lr"],
                )

                if self.config.early_stopping:
                    if (
                        self.history.best_epoch is not None
                        and epoch - self.history.best_epoch >= self.config.early_stopping_patience
                    ):
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

        finally:
            self._running = False
            self._stop_event.clear()

        return self.history



    # ------------------------------------------------------------------ #
    # Background execution
    # ------------------------------------------------------------------ #
    def start_background(self, resume_from: Optional[str | Path] = None) -> threading.Thread:
        """Start training in a background daemon thread."""

        def _run():
            self.train(resume_from=resume_from)

        thread = threading.Thread(target=_run, daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def request_stop(self) -> None:
        self.stop_requested = True
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._running


def checkpoint_path_name(model_name: str, step: int) -> Path:
    from .checkpoint import checkpoint_path

    return checkpoint_path(model_name, step)


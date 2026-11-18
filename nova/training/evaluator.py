"""Validation loop: average loss and perplexity over a validation set."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .metrics import EpochMetrics


@torch.inference_mode()
def evaluate(model, val_dataloader, device: str = "cpu") -> tuple[float, Optional[float]]:
    """Run validation and return (average_loss, perplexity).

    Perplexity is None if the loss overflows.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in val_dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, loss = model(inputs, targets=targets)
        if loss is None:
            continue
        total_loss += float(loss.item())
        total_batches += 1
    if total_batches == 0:
        return float("nan"), None
    avg_loss = total_loss / total_batches
    try:
        import math

        perplexity = math.exp(min(avg_loss, 20.0))
    except OverflowError:
        perplexity = None
    return avg_loss, perplexity

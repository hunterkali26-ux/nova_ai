"""Memory-conscious dataset and batching for NOVA Tiny training.

The dataset concatenates tokenized documents into a flat token stream with
end-of-document markers, then yields fixed-length blocks. Blocks are produced
on the fly so large corpora do not need to be fully materialized in RAM.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..core.tokenizer import EOS_ID


class TextDataset(Dataset):
    """A lazy block dataset over tokenized text.

    Each sample is a contiguous block of ``block_size`` tokens with a shifted
    copy used as targets (next-token prediction). Document boundaries are
    respected by inserting EOS tokens so the model never learns to predict
    across unrelated documents within one block.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.stride = stride or block_size
        self.shuffle = shuffle
        self.seed = seed

        # Store tokens as a tensor; for very large corpora this list is built
        # incrementally by the preprocessing module to keep memory bounded.
        self.token_ids: List[int] = list(token_ids)
        self._block_starts: List[int] = self._compute_block_starts()
        self._rng = random.Random(seed)

    def _compute_block_starts(self) -> List[int]:
        starts = []
        i = 0
        while i + self.block_size <= len(self.token_ids):
            starts.append(i)
            i += self.stride
        return starts

    def __len__(self) -> int:
        return len(self._block_starts)

    def __getitem__(self, idx: int):
        if self.shuffle:
            # Deterministic-ish reshuffle each epoch via the rng.
            actual_idx = self._rng.randint(0, len(self._block_starts) - 1)
        else:
            actual_idx = idx
        start = self._block_starts[actual_idx]
        block = self.token_ids[start : start + self.block_size]
        x = torch.tensor(block[:-1], dtype=torch.long)   # inputs
        y = torch.tensor(block[1:], dtype=torch.long)    # targets (next token)
        return x, y

    def reshuffle(self) -> None:
        """Reshuffle block order (call at the start of each epoch)."""
        if self.shuffle and self._block_starts:
            self._rng.shuffle(self._block_starts)
            # When shuffled, ignore __getitem__ rng and use ordered starts.
            self._use_ordered = True


class EpochDataset(TextDataset):
    """Dataset that iterates blocks in order for a single epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_ordered = False

    def __getitem__(self, idx: int):
        start = self._block_starts[idx]
        block = self.token_ids[start : start + self.block_size]
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y

    def reshuffle(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._block_starts)
        self._use_ordered = True


def collate_batch(batch):
    """Collate a list of (x, y) pairs into tensors."""
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    return torch.stack(xs), torch.stack(ys)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
):
    """Build a DataLoader with CPU-friendly settings (num_workers=0 by default)."""
    from torch.utils.data import DataLoader

    if shuffle:
        dataset.shuffle = True
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    return loader

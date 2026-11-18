"""Text cleaning, train/validation split, and tokenization for training.

Produces a flat token-id stream saved to disk, plus a fitted tokenizer, ready
for the trainer. Memory-conscious: documents are tokenized and streamed rather
than keeping multiple copies of the full corpus in RAM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from ..core.config import PROCESSED_DATA_DIR
from ..core.tokenizer import CharTokenizer, EOS_ID, build_tokenizer
from ..core.logging import get_logger

logger = get_logger("preprocessing")


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, strip control characters."""
    if not text:
        return ""
    # Replace tabs and multiple spaces/newlines with a single space/newline.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters except newlines.
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch == "\n")
    # Collapse 3+ newlines to 2, collapse runs of spaces.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_train_val(texts: List[str], val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split texts into train and validation lists deterministically."""
    import random

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    n_val = max(1, int(len(texts) * val_fraction))
    val_indices = set(indices[:n_val])
    train = [t for i, t in enumerate(texts) if i not in val_indices]
    val = [t for i, t in enumerate(texts) if i in val_indices]
    return train, val


def tokenize_corpus(
    texts: List[str],
    tokenizer_type: str = "char",
    tokenizer_path: str | Path | None = None,
    max_vocab_size: int | None = None,
):
    """Fit a tokenizer on ``texts`` and return (token_ids, tokenizer).

    ``token_ids`` is a flat list of token ids with EOS between documents.
    """
    tokenizer = build_tokenizer(tokenizer_type)
    cleaned = [clean_text(t) for t in texts if clean_text(t)]
    if not cleaned:
        raise ValueError("No non-empty texts after cleaning; cannot fit tokenizer.")
    tokenizer.fit(cleaned, max_vocab_size=max_vocab_size)

    token_ids: List[int] = []
    for text in cleaned:
        ids = tokenizer.encode(text, add_special=True)  # includes BOS/EOS
        token_ids.extend(ids)

    if tokenizer_path is not None:
        tokenizer_path = Path(tokenizer_path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)
        logger.info("Tokenizer persisted to %s", tokenizer_path)

    return token_ids, tokenizer


def save_processed(token_ids: List[int], path: str | Path) -> None:
    """Save the flat token stream as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token_ids), encoding="utf-8")
    logger.info("Saved %d token ids to %s", len(token_ids), path)


def load_processed(path: str | Path) -> List[int]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

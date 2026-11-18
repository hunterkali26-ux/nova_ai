"""Text chunking with configurable size and overlap.

Splits text into overlapping chunks while respecting sentence boundaries where
possible. Each chunk carries its start/end character offsets in the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    text: str
    start: int
    end: int
    index: int


def _split_sentences(text: str) -> List[str]:
    # Naive sentence split on period/question/exclamation followed by space or newline.
    parts = re_split = __import__("re").split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[TextChunk]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: Source text.
        chunk_size: Max characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of TextChunk objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text:
        return []

    import re

    chunks: List[TextChunk] = []
    start = 0
    index = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # Try to end at a sentence boundary within the chunk.
        if end < n:
            slice_text = text[start:end]
            last_sentence = slice_text.rfind(". ")
            if last_sentence != -1 and last_sentence > chunk_size // 2:
                end = start + last_sentence + 2
        chunk = TextChunk(text=text[start:end], start=start, end=end, index=index)
        chunks.append(chunk)
        index += 1
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

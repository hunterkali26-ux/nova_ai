"""Local embedding generation for the RAG engine.

Prefers ``sentence-transformers`` when installed (model downloaded only on
explicit request). Falls back to a deterministic TF-IDF embedding so the RAG
engine remains functional offline without any external download.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from ..core.logging import get_logger

logger = get_logger("knowledge.embeddings")

_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")


class EmbeddingError(Exception):
    pass


class EmbeddingModel(ABC):
    """Interface: embed a list of strings into vectors."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformersEmbedding(EmbeddingModel):
    """Wrapper around sentence-transformers.

    The underlying model is NOT downloaded on construction. It is loaded
    lazily on the first ``embed`` call, which triggers a download only if the
    model is not already cached locally.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it or use TfidfEmbedding."
            ) from exc
        logger.info("Loading sentence-transformers model %s (downloads if not cached)", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        # Determine dimension from a probe encode.
        vec = self._model.encode(["__dimension_probe__"], convert_to_numpy=True)
        self._dimension = int(vec.shape[1])

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure_loaded()
        if not texts:
            return []
        arr = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return arr.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        return self._dimension or 384


class TfidfEmbedding(EmbeddingModel):
    """Deterministic TF-IDF embedding (no downloads, fully offline)."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension_value = dimension
        self.vocab: dict = {}
        self.idf: dict = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return _WORD_RE.findall(text.lower())

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedding":
        doc_freq: dict = {}
        total_docs = 0
        for text in texts:
            tokens = set(self._tokenize(text))
            if not tokens:
                continue
            total_docs += 1
            for tok in tokens:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        # Build vocab sorted by frequency, capped at dimension.
        sorted_terms = sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        self.vocab = {term: i for i, (term, _) in enumerate(sorted_terms[: self.dimension_value])}
        if total_docs > 0:
            self.idf = {
                term: math.log((1 + total_docs) / (1 + df)) + 1.0
                for term, df in doc_freq.items()
            }
        self._fitted = True
        logger.info("TfidfEmbedding fitted with %d terms over %d docs", len(self.vocab), total_docs)
        return self

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(list(texts))
        vectors: List[List[float]] = []
        vocab = self.vocab
        idf = self.idf
        dim = self.dimension_value
        for text in texts:
            tokens = self._tokenize(text)
            tf: dict = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            vec = [0.0] * dim
            norm = 0.0
            for tok, count in tf.items():
                idx = vocab.get(tok)
                if idx is None:
                    continue
                weight = count * idf.get(tok, 1.0)
                vec[idx] = weight
                norm += weight * weight
            norm = math.sqrt(norm) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @property
    def dimension(self) -> int:
        return self.dimension_value

"""Vector similarity store with a FAISS backend and a NumPy fallback.

Vectors are stored as float32 NumPy arrays. Search returns the top-K most
similar items by cosine similarity along with their ids and metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger

logger = get_logger("knowledge.vector_store")


@dataclass
class VectorItem:
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Abstract vector store interface."""

    def add(self, items: Sequence[VectorItem]) -> None:
        raise NotImplementedError

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[SearchResult]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError


class NumpyVectorStore(VectorStore):
    """Cosine-similarity vector store using pure NumPy (no external deps)."""

    def __init__(self, dimension: int) -> None:
        self.dimension_value = dimension
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def add(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        matrix = np.array([item.vector for item in items], dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if self._vectors is None:
            self._vectors = matrix
        else:
            self._vectors = np.vstack([self._vectors, matrix])
        for item in items:
            self._ids.append(item.id)
            self._metadata.append(item.metadata)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        v = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-12)
        scores = (v @ q_norm.T).ravel()
        top_k = min(top_k, len(self._ids))
        top_idx = np.argsort(-scores)[:top_k]
        results: List[SearchResult] = []
        for idx in top_idx:
            results.append(
                SearchResult(
                    id=self._ids[int(idx)],
                    score=float(scores[int(idx)]),
                    metadata=self._metadata[int(idx)],
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dimension": self.dimension_value,
            "ids": self._ids,
            "metadata": self._metadata,
        }
        if self._vectors is not None:
            data["vectors"] = self._vectors.tolist()
        path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Saved NumpyVectorStore (%d items) to %s", self.count, path)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(dimension=int(data["dimension"]))
        store._ids = list(data["ids"])
        store._metadata = list(data.get("metadata", [{}] * len(store._ids)))
        if data.get("vectors"):
            store._vectors = np.array(data["vectors"], dtype=np.float32)
        else:
            store._vectors = None
        logger.info("Loaded NumpyVectorStore (%d items) from %s", store.count, path)
        return store

    @property
    def dimension(self) -> int:
        return self.dimension_value

    @property
    def count(self) -> int:
        return len(self._ids)


class FaissVectorStore(VectorStore):
    """FAISS-backed vector store (IndexFlatIP with L2-normalized vectors)."""

    def __init__(self, dimension: int) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("faiss-cpu is not installed") from exc
        self.dimension_value = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def add(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        import faiss

        matrix = np.array([item.vector for item in items], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        matrix = matrix / norms
        self._index.add(matrix)
        for item in items:
            self._ids.append(item.id)
            self._metadata.append(item.metadata)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[SearchResult]:
        if self._index.ntotal == 0:
            return []
        import faiss

        q = np.array([query_vector], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, top_k)
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    id=self._ids[int(idx)],
                    score=float(score),
                    metadata=self._metadata[int(idx)],
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        meta = {"ids": self._ids, "metadata": self._metadata, "dimension": self.dimension_value}
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        logger.info("Saved FaissVectorStore (%d items) to %s", self.count, path)

    @classmethod
    def load(cls, path: str | Path) -> "FaissVectorStore":
        import faiss

        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dimension=int(meta["dimension"]))
        store._index = faiss.read_index(str(path))
        store._ids = list(meta["ids"])
        store._metadata = list(meta.get("metadata", [{}] * len(store._ids)))
        logger.info("Loaded FaissVectorStore (%d items) from %s", store.count, path)
        return store

    @property
    def dimension(self) -> int:
        return self.dimension_value

    @property
    def count(self) -> int:
        return self._index.ntotal


def create_vector_store(dimension: int, prefer_faiss: bool = True) -> VectorStore:
    """Factory: create a FAISS store if available and requested, else NumPy."""
    if prefer_faiss:
        try:
            return FaissVectorStore(dimension)
        except ImportError:
            logger.info("faiss-cpu unavailable; using NumPy vector store fallback")
    return NumpyVectorStore(dimension)


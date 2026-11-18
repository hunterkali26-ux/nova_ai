"""Knowledge retrieval: embed a query and search the vector store.

Returns ranked chunks with source metadata for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .embeddings import EmbeddingModel
from .vector_store import SearchResult, VectorStore


@dataclass
class Chunk:
    text: str
    score: float
    source: str
    source_type: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Chunk]:
        """Retrieve the top-K most relevant chunks for a query."""
        query_vector = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        chunks: List[Chunk] = []
        for result in results:
            if result.score < min_score:
                continue
            chunks.append(
                Chunk(
                    text=result.metadata.get("text", ""),
                    score=result.score,
                    source=result.metadata.get("source", result.id),
                    source_type=result.metadata.get("source_type", ""),
                    doc_id=result.metadata.get("doc_id", ""),
                    chunk_index=result.metadata.get("chunk_index", 0),
                    metadata=result.metadata,
                )
            )
        return chunks

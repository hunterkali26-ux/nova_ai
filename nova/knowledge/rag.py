"""Retrieval-Augmented Generation pipeline.

Builds a context string from retrieved chunks and tracks source citations for
attribution. If no relevant context is found, reports insufficiency clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .retriever import Chunk, Retriever


@dataclass
class RagContext:
    query: str
    context: str
    sources: List[dict] = field(default_factory=list)
    has_sufficient_context: bool = False


class RAGPipeline:
    def __init__(self, retriever: Retriever, max_context_chunks: int = 5, min_score: float = 0.0) -> None:
        self.retriever = retriever
        self.max_context_chunks = max_context_chunks
        self.min_score = min_score

    def build_context(self, query: str) -> RagContext:
        chunks = self.retriever.retrieve(query, top_k=self.max_context_chunks, min_score=self.min_score)
        if not chunks:
            return RagContext(
                query=query,
                context="",
                sources=[],
                has_sufficient_context=False,
            )

        parts: List[str] = []
        sources: List[dict] = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[Source {i}] {chunk.text}")
            sources.append(
                {
                    "index": i,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "score": round(chunk.score, 4),
                }
            )
        context = "\n\n".join(parts)
        return RagContext(
            query=query,
            context=context,
            sources=sources,
            has_sufficient_context=True,
        )

    def format_prompt(self, query: str, context: RagContext, system_prompt: str = "") -> str:
        """Construct a RAG prompt. No source citations are fabricated."""
        if context.has_sufficient_context:
            return (
                f"{system_prompt}\n\n"
                f"Use the following retrieved context to answer the question. "
                f"If the context does not contain enough information, say so.\n\n"
                f"Context:\n{context.context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
        return (
            f"{system_prompt}\n\n"
            f"No relevant information was found in the knowledge base. "
            f"Answer based only on what you know, and state that no supporting "
            f"documents were found.\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

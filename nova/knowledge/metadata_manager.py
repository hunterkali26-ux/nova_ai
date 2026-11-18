"""Metadata tracking for ingested documents: source URLs, timestamps, licenses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    source_type: str  # txt, md, pdf, html, url
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    license: Optional[str] = None
    chunk_count: int = 0
    indexed: bool = False
    byte_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

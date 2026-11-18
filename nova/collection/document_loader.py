"""Document loaders for TXT, Markdown, HTML, and PDF files.

Each loader returns (text, metadata). Malformed documents are handled
gracefully: they return empty text with an error note in metadata rather than
crashing the ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from ..core.logging import get_logger
from ..knowledge.cleaner import clean_text
from .html_parser import parse_html
from .pdf_parser import parse_pdf

logger = get_logger("collection.document_loader")


@dataclass
class LoadedDocument:
    text: str
    metadata: dict = field(default_factory=dict)


def load_txt(path: Path) -> LoadedDocument:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")
    return LoadedDocument(text=clean_text(raw), metadata={"encoding": "utf-8"})


def load_markdown(path: Path) -> LoadedDocument:
    return load_txt(path)  # Markdown is plain text; cleaning is sufficient.


def load_html(path: Path) -> LoadedDocument:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Failed to read HTML %s: %s", path, exc)
        return LoadedDocument(text="", metadata={"error": str(exc)})
    parsed = parse_html(raw)
    return LoadedDocument(
        text=parsed.text,
        metadata={"format": "html", "title": parsed.title, "links": parsed.links},
    )


def load_pdf(path: Path) -> LoadedDocument:
    try:
        parsed = parse_pdf(str(path))
    except Exception as exc:  # pragma: no cover - malformed PDF
        logger.error("Failed to parse PDF %s: %s", path, exc)
        return LoadedDocument(text="", metadata={"error": str(exc)})
    return LoadedDocument(
        text=parsed.text,
        metadata={"format": "pdf", "pages": parsed.pages},
    )


SUPPORTED_EXTENSIONS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_document(path: str | Path) -> LoadedDocument:
    """Load a single document by file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    loader = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    try:
        doc = loader(path)
    except Exception as exc:  # never crash ingestion on one bad file
        logger.exception("Failed to load %s: %s", path, exc)
        return LoadedDocument(text="", metadata={"error": str(exc), "path": str(path)})
    doc.metadata["path"] = str(path)
    doc.metadata["name"] = path.name
    return doc


def load_directory(directory: str | Path, extensions=None) -> List[LoadedDocument]:
    """Load all supported documents in a directory (non-recursive)."""
    directory = Path(directory)
    if extensions is None:
        extensions = set(SUPPORTED_EXTENSIONS)
    docs: List[LoadedDocument] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            docs.append(load_document(path))
    return docs

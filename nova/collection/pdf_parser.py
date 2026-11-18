"""PDF text extraction using pypdf.

Returns the extracted text plus page count. Malformed PDFs are handled
gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..knowledge.cleaner import clean_text


@dataclass
class ParsedPDF:
    text: str
    pages: int


def parse_pdf(path: str) -> ParsedPDF:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pypdf is required to parse PDF files") from exc

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    raw = "\n".join(pages_text)
    return ParsedPDF(text=clean_text(raw), pages=len(reader.pages))

"""Text and HTML cleaning utilities for document ingestion.

HTML parsing uses BeautifulSoup when available; otherwise a lightweight regex
fallback strips tags. Never raises on malformed input.
"""

from __future__ import annotations

import re
from typing import Optional

from ..core.logging import get_logger

logger = get_logger("knowledge.cleaner")

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"[ \t\r\f\v]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: Optional[str]) -> str:
    """Clean raw text: normalize whitespace, strip control characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep printable characters plus newline/tab.
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


def clean_html(html: Optional[str]) -> str:
    """Convert HTML to plain text, preserving basic structure."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style elements.
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()
        # Get text with newlines for block elements.
        text = soup.get_text(separator="\n")
    except ImportError:  # pragma: no cover - fallback when bs4 missing
        logger.warning("beautifulsoup4 not installed; using regex HTML stripping")
        text = _HTML_TAG.sub(" ", html)
    return clean_text(text)


def extract_links(html: Optional[str]) -> list:
    """Extract href URLs from HTML (used by the crawler)."""
    if not html:
        return []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href:
                links.append(href)
        return links
    except ImportError:
        return re.findall(r'href=["\']([^"\']+)["\']', html or "")

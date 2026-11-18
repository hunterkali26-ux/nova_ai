"""HTML parsing: extract plain text and links from HTML documents.

Uses BeautifulSoup when available; falls back to regex tag stripping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..knowledge.cleaner import clean_text

_HTML_TAG = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


@dataclass
class ParsedHTML:
    text: str
    links: List[str]
    title: str = ""


def parse_html(html: str) -> ParsedHTML:
    """Parse HTML into plain text, links, and title."""
    if not html:
        return ParsedHTML(text="", links=[])
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator="\n")
        links = [a.get("href", "").strip() for a in soup.find_all("a", href=True) if a.get("href")]
        return ParsedHTML(text=clean_text(text), links=links, title=title)
    except ImportError:
        text = clean_text(_HTML_TAG.sub(" ", html))
        links = _HREF_RE.findall(html)
        return ParsedHTML(text=text, links=links)

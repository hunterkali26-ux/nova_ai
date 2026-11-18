"""Allowlist-based web crawler.

Only explicitly configured sources are processed. External crawling is opt-in
and respects robots.txt, rate limits, and a configurable page cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .downloader import Downloader, DownloadResult
from ..core.logging import get_logger

logger = get_logger("collection.crawler")


@dataclass
class CrawlSource:
    url: str
    max_pages: int = 1
    label: str = ""


@dataclass
class CrawlResult:
    source: str
    downloaded: List[DownloadResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class Crawler:
    def __init__(self, downloader: Optional[Downloader] = None) -> None:
        self.downloader = downloader or Downloader()

    def crawl(self, sources: List[CrawlSource]) -> List[CrawlResult]:
        """Crawl the configured sources. Nothing happens if sources is empty."""
        results: List[CrawlResult] = []
        for source in sources:
            result = CrawlResult(source=source.url)
            for i in range(source.max_pages):
                # For simplicity, fetch the configured URL once per source.
                # Pagination/following links is intentionally limited to the
                # explicitly listed URLs to avoid unbounded crawling.
                try:
                    downloaded = self.downloader.download(source.url)
                    if downloaded.error:
                        result.errors.append(f"{source.url}: {downloaded.error}")
                    else:
                        result.downloaded.append(downloaded)
                        logger.info("Crawled %s (%d bytes)", source.url, len(downloaded.content))
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Crawl error for %s: %s", source.url, exc)
                    result.errors.append(str(exc))
                    break
            results.append(result)
        return results

"""Polite HTTP downloader with user-agent, rate limiting, and robots.txt respect.

Downloads are intentional and configured by the user; nothing is downloaded
automatically by the application.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import urllib.request
import urllib.error

from ..core.logging import get_logger

logger = get_logger("collection.downloader")

DEFAULT_USER_AGENT = "NOVA-AI/0.1 (local knowledge engine; contact: operator@localhost)"
DEFAULT_REQUEST_DELAY = 1.0  # seconds between requests to the same host


@dataclass
class DownloadResult:
    url: str
    content: str
    status_code: int
    content_type: str
    fetched_at: str
    error: Optional[str] = None


class Downloader:
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        respect_robots: bool = True,
        timeout: float = 15.0,
    ) -> None:
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.respect_robots = respect_robots
        self.timeout = timeout
        self._last_request: dict = {}
        self._robots_cache: dict = {}

    def _host(self, url: str) -> str:
        return urlparse(url).netloc

    def _rate_limit(self, url: str) -> None:
        host = self._host(url)
        now = time.time()
        last = self._last_request.get(host, 0.0)
        elapsed = now - last
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request[host] = time.time()

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        host = self._host(url)
        if host in self._robots_cache:
            rp = self._robots_cache[host]
        else:
            try:
                robots_url = urljoin(url, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[host] = rp
            except Exception as exc:
                logger.warning("Could not read robots.txt for %s: %s", host, exc)
                return True
        return rp.can_fetch(self.user_agent, url)

    def download(self, url: str) -> DownloadResult:
        """Download a single URL. Returns content as text (best-effort decode)."""
        from datetime import datetime

        if not self._allowed_by_robots(url):
            return DownloadResult(
                url=url,
                content="",
                status_code=0,
                content_type="",
                fetched_at=datetime.utcnow().isoformat() + "Z",
                error="Blocked by robots.txt",
            )

        self._rate_limit(url)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.getcode()
                content_type = resp.headers.get("Content-Type", "")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            logger.error("Download failed for %s: %s", url, exc)
            return DownloadResult(
                url=url,
                content="",
                status_code=0,
                content_type="",
                fetched_at=datetime.utcnow().isoformat() + "Z",
                error=str(exc),
            )

        try:
            content = raw.decode("utf-8", errors="replace")
        except Exception:
            content = raw.decode("latin-1", errors="replace")

        return DownloadResult(
            url=url,
            content=content,
            status_code=status,
            content_type=content_type,
            fetched_at=datetime.utcnow().isoformat() + "Z",
        )

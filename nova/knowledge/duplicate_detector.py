"""Duplicate document detection using content hashing."""

from __future__ import annotations

import hashlib
from typing import Optional

from ..core.logging import get_logger

logger = get_logger("knowledge.duplicate_detector")


class DuplicateDetector:
    def __init__(self) -> None:
        self._seen: dict = {}

    def fingerprint(self, text: str, source: str = "") -> str:
        """Return a stable hash fingerprint for a document."""
        return hashlib.sha256((source + "\n" + text).encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str, source: str = "") -> bool:
        fp = self.fingerprint(text, source)
        if fp in self._seen:
            logger.info("Duplicate detected for source %s", source)
            return True
        self._seen[fp] = source
        return False

    def reset(self) -> None:
        self._seen.clear()

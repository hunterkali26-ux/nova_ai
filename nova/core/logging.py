"""Application logging configuration for NOVA AI.

Logs are written to the project logs directory and to stdout. The default
level is INFO; it can be overridden with the NOVA_LOG_LEVEL environment
variable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import LOGS_DIR

_LOGGER_NAME = "nova"
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(f"{_LOGGER_NAME}.{name}")
    if logger.handlers:
        return logger

    logger.setLevel(
        getattr(logging, os_log_level(), logging.INFO)
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGS_DIR / "nova.log", encoding="utf-8")
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_FORMATTER)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def os_log_level() -> str:
    """Read the log level from the environment, defaulting to INFO."""
    import os

    return os.environ.get("NOVA_LOG_LEVEL", "INFO").upper()

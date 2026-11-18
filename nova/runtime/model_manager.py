"""Model runtime manager: registry, loading/unloading, selection, memory monitoring.

Keeps track of registered models and the currently loaded backend. Only one
model is loaded at a time to conserve RAM on an 8 GB machine.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import MODELS_DIR
from ..core.logging import get_logger
from ..database.models import ModelRegistryStore
from .base import ModelInfo
from .llamacpp_backend import LlamaCppBackend
from .pytorch_backend import PyTorchBackend

logger = get_logger("runtime.model_manager")


class ModelManager:
    def __init__(self, databases_dir=None, models_dir=None) -> None:
        from ..core.config import DATABASES_DIR
        self.databases_dir = Path(databases_dir or DATABASES_DIR)
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.databases_dir.mkdir(parents=True, exist_ok=True)

        registry_db = self.databases_dir / "settings.db"
        registry_db.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(registry_db), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                name TEXT PRIMARY KEY,
                identifier TEXT NOT NULL,
                model_type TEXT NOT NULL,
                architecture TEXT NOT NULL,
                file_path TEXT,
                quantization TEXT,
                context_length INTEGER,
                model_format TEXT,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self.registry = ModelRegistryStore(self._conn)

        self._backends: Dict[str, object] = {}
        self._active_name: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #
    def register(self, info: ModelInfo) -> None:
        self.registry.add(
            {
                "name": info.name,
                "identifier": info.name,
                "model_type": info.model_type,
                "architecture": info.architecture,
                "file_path": info.file_path,
                "quantization": info.quantization,
                "context_length": info.context_length,
                "model_format": info.model_format,
                "description": info.description,
            }
        )
        logger.info("Registered model %s (%s)", info.name, info.model_type)

    def list_models(self) -> List[dict]:
        rows = self.registry.list()
        return [dict(r) for r in rows]

    def get_model(self, name: str) -> Optional[dict]:
        row = self.registry.get(name)
        return dict(row) if row else None

    def unregister(self, name: str) -> None:
        self.registry.delete(name)

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _backend_for(self, entry: dict) -> object:
        mtype = entry["model_type"]
        if mtype == "nova-tiny":
            return PyTorchBackend(
                name=entry["name"],
                checkpoint_path=entry["file_path"],
                tokenizer_path=entry.get("tokenizer_path") or entry["file_path"],
                context_length=entry.get("context_length") or 256,
            )
        if mtype == "llama.cpp":
            return LlamaCppBackend(
                name=entry["name"],
                model_path=entry["file_path"],
                n_ctx=entry.get("context_length") or 512,
                n_threads=self.cpu_thread_count(),
            )
        raise ValueError(f"Unsupported model type: {mtype}")

    def load_model(self, name: str) -> object:
        with self._lock:
            self.unload_model()
            entry = self.get_model(name)
            if entry is None:
                raise ValueError(f"Model not registered: {name}")
            backend = self._backend_for(entry)
            backend.load_model()
            self._backends[name] = backend
            self._active_name = name
            logger.info("Loaded model %s", name)
            return backend

    def unload_model(self) -> None:
        with self._lock:
            if self._active_name is not None:
                backend = self._backends.pop(self._active_name, None)
                if backend is not None:
                    backend.unload_model()
                logger.info("Unloaded model %s", self._active_name)
                self._active_name = None

    def get_active_backend(self):
        if self._active_name is None:
            return None
        return self._backends.get(self._active_name)

    def is_loaded(self, name: Optional[str] = None) -> bool:
        if name is None:
            return self._active_name is not None
        return name in self._backends

    def active_model_name(self) -> Optional[str]:
        return self._active_name

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #
    def cpu_thread_count(self) -> int:
        try:
            return max(1, os.cpu_count() or 4)
        except Exception:
            return 4

    def memory_info(self) -> dict:
        total = available = used = 0
        try:
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        meminfo[key.strip()] = int(val.split()[0]) * 1024  # kB -> bytes
            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            free = meminfo.get("MemFree", 0)
            buffers = meminfo.get("Buffers", 0)
            cached = meminfo.get("Cached", 0)
            used = total - free - buffers - cached
        except Exception:
            pass
        return {
            "total_gb": round(total / (1024 ** 3), 2),
            "available_gb": round(available / (1024 ** 3), 2),
            "used_gb": round(max(0, used) / (1024 ** 3), 2),
        }

    def close(self) -> None:
        self.unload_model()
        try:
            self._conn.close()
        except Exception:
            pass

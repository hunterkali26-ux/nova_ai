"""Abstract backend interface for local model inference.

Every model backend (NOVA Tiny PyTorch, optional llama.cpp) implements this
interface so the orchestrator and desktop app stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelInfo:
    name: str
    model_type: str  # "nova-tiny" | "llama.cpp" | "embedding"
    architecture: str
    file_path: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    model_format: Optional[str] = None
    description: str = ""
    loaded: bool = False


@dataclass
class GenerationResult:
    text: str
    model: str
    tokens: int = 0
    finish_reason: str = "stop"


class BaseBackend:
    def load_model(self) -> None:
        raise NotImplementedError

    def unload_model(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        raise NotImplementedError

    def stream_generate(self, prompt: str, **kwargs):
        """Yield text chunks. Default implementation wraps generate()."""
        raise NotImplementedError

    def get_model_info(self) -> ModelInfo:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError

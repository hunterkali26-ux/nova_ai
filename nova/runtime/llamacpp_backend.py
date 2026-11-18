"""Optional llama.cpp backend for GGUF models.

This backend is only usable when the ``llama-cpp-python`` package is installed
and a compatible GGUF model file is provided. It is intentionally optional and
never downloaded automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from .base import BaseBackend, GenerationResult, ModelInfo


class LlamaCppBackend(BaseBackend):
    """Backend for llama.cpp GGUF models."""

    def __init__(
        self,
        name: str,
        model_path: str | Path,
        n_ctx: int = 512,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers  # 0 = CPU only
        self.verbose = verbose
        self._llm = None
        self._loaded = False

    def _ensure_llama(self):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. Install it explicitly to use "
                "llama.cpp GGUF models."
            ) from exc
        return Llama

    def load_model(self) -> None:
        if self._loaded:
            return
        Llama = self._ensure_llama()
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        self._loaded = True

    def unload_model(self) -> None:
        self._llm = None
        self._loaded = False

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        if not self._loaded:
            self.load_model()
        assert self._llm is not None
        max_tokens = int(kwargs.get("max_tokens", 128))
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = int(kwargs.get("top_k", 50))
        top_p = float(kwargs.get("top_p", 0.9))
        repeat_penalty = float(kwargs.get("repeat_penalty", 1.1))
        stop = kwargs.get("stop", None)

        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )
        text = output["choices"][0]["text"]
        return GenerationResult(text=text, model=self.name)

    def stream_generate(self, prompt: str, **kwargs):
        if not self._loaded:
            self.load_model()
        assert self._llm is not None
        max_tokens = int(kwargs.get("max_tokens", 128))
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = int(kwargs.get("top_k", 50))
        top_p = float(kwargs.get("top_p", 0.9))
        repeat_penalty = float(kwargs.get("repeat_penalty", 1.1))
        stop = kwargs.get("stop", None)

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=True,
        ):
            yield chunk["choices"][0]["text"]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_type="llama.cpp",
            architecture="llama.cpp GGUF",
            file_path=str(self.model_path),
            context_length=self.n_ctx,
            model_format="gguf",
            description="llama.cpp compatible GGUF model",
            loaded=self._loaded,
        )

    def is_loaded(self) -> bool:
        return self._loaded

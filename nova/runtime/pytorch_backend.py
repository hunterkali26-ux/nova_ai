"""NOVA Tiny PyTorch backend for local CPU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch

from ..core.config import DEFAULT_DEVICE
from ..core.generation import generate
from ..core.model import NOVATinyModel
from ..core.tokenizer import CharTokenizer, load_tokenizer
from .base import BaseBackend, GenerationResult, ModelInfo


class PyTorchBackend(BaseBackend):
    """Backend for a NOVA Tiny model loaded from a checkpoint."""

    def __init__(
        self,
        name: str,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = DEFAULT_DEVICE,
        context_length: int = 256,
    ) -> None:
        self.name = name
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device
        self.context_length = context_length
        self._model: Optional[NOVATinyModel] = None
        self._tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        if self._loaded:
            return
        self._model = NOVATinyModel.load(self.checkpoint_path, device=self.device)
        self._tokenizer = load_tokenizer(self.tokenizer_path)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        if not self._loaded:
            self.load_model()
        assert self._model is not None and self._tokenizer is not None
        max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        temperature = float(kwargs.get("temperature", 0.8))
        top_k = int(kwargs.get("top_k", 50))
        top_p = float(kwargs.get("top_p", 0.9))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.1))

        text = generate(
            self._model,
            self._tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=self.device,
        )
        return GenerationResult(text=text, model=self.name)

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Default streaming: generate whole text then yield in pieces."""
        result = self.generate(prompt, **kwargs)
        text = result.text
        # Yield the prompt portion once, then stream the continuation.
        yield text
        return
        yield  # pragma: no cover - makes this a generator

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_type="nova-tiny",
            architecture="decoder-only Transformer",
            file_path=str(self.checkpoint_path),
            context_length=self.context_length,
            model_format="pytorch",
            description="NOVA Tiny custom decoder-only Transformer",
            loaded=self._loaded,
        )

    def is_loaded(self) -> bool:
        return self._loaded

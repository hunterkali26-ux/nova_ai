"""Character-level tokenizer with optional Byte Pair Encoding (BPE).

The default tokenizer is a simple character-level tokenizer that builds a
vocabulary from a corpus. It is reliable, dependency-free, and well suited to
initial CPU experiments with NOVA Tiny. An optional BPE tokenizer wraps the
``tokenizers`` library when it is installed.

Special tokens:
    <pad>   - padding
    <tool_call>   - unknown token
    <bos>   - beginning of sequence
    <eos>   - end of sequence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..core.logging import get_logger

logger = get_logger("tokenizer")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<tool_call>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


@dataclass
class TokenizerConfig:
    vocab_size: int
    tokenizer_type: str = "char"
    special_tokens: List[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "tokenizer_type": self.tokenizer_type,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerConfig":
        return cls(
            vocab_size=int(data["vocab_size"]),
            tokenizer_type=str(data.get("tokenizer_type", "char")),
            special_tokens=list(data.get("special_tokens", SPECIAL_TOKENS)),
        )


class CharTokenizer:
    """Character-level tokenizer.

    Encodes each character as a single token id. Vocabulary is built from the
    unique characters found in the training corpus plus the special tokens.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig(vocab_size=0, tokenizer_type="char")
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        self._fitted: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    def fit(self, texts: List[str], max_vocab_size: Optional[int] = None) -> "CharTokenizer":
        """Build the vocabulary from an iterable of text strings."""
        chars: set = set()
        for text in texts:
            chars.update(text)

        ordered = sorted(chars)
        token_list = list(self.config.special_tokens) + ordered
        if max_vocab_size is not None:
            token_list = token_list[: max_vocab_size]

        self._char_to_id = {tok: i for i, tok in enumerate(token_list)}
        self._id_to_char = {i: tok for tok, i in self._char_to_id.items()}
        self.config.vocab_size = self.vocab_size
        self._fitted = True
        logger.info(
            "CharTokenizer fitted with %d tokens (max_vocab_size=%s)",
            self.vocab_size,
            max_vocab_size,
        )
        return self

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode a string into a list of token ids."""
        if not self._fitted:
            raise RuntimeError("Tokenizer has not been fitted. Call fit() first.")
        ids: List[int] = []
        if add_special:
            ids.append(BOS_ID)
        for ch in text:
            ids.append(self._char_to_id.get(ch, UNK_ID))
        if add_special:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token ids back into a string."""
        special = {PAD_ID, UNK_ID, BOS_ID, EOS_ID} if skip_special else set()
        chars = []
        for i in ids:
            if i in special:
                continue
            tok = self._id_to_char.get(i, UNK_TOKEN)
            chars.append(tok)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Persist the vocabulary and config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "char_to_id": self._char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Tokenizer saved to %s (%d tokens)", path, self.vocab_size)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer from a JSON file written by ``save``."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = TokenizerConfig.from_dict(payload["config"])
        tokenizer = cls(config=config)
        tokenizer._char_to_id = {k: int(v) for k, v in payload["char_to_id"].items()}
        tokenizer._id_to_char = {int(i): ch for ch, i in tokenizer._char_to_id.items()}
        tokenizer._fitted = True
        logger.info("Tokenizer loaded from %s (%d tokens)", path, tokenizer.vocab_size)
        return tokenizer



class BPETokenizer:
    """Optional Byte Pair Encoding tokenizer wrapping the ``tokenizers`` library.

    Falls back to raising ``ImportError`` if the ``tokenizers`` package is not
    installed. The BPE tokenizer is useful for larger experiments but is not
    required for the default char-level pipeline.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig(vocab_size=0, tokenizer_type="bpe")
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Whitespace
        except ImportError as exc:
            raise ImportError(
                "BPE tokenizer requires the 'tokenizers' package. "
                "Install it with: pip install tokenizers"
            ) from exc

        self._hf = HFTokenizer(BPE(unk_token=UNK_TOKEN))
        self._hf.pre_tokenizer = Whitespace()
        self._trainer = BpeTrainer(
            vocab_size=self.config.vocab_size or 5000,
            special_tokens=self.config.special_tokens,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "BPETokenizer":
        self._hf.train_from_iterator(texts, self._trainer)
        self._fitted = True
        self.config.vocab_size = self._hf.get_vocab_size()
        logger.info("BPETokenizer fitted with %d tokens", self.config.vocab_size)
        return self

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if not self._fitted:
            raise RuntimeError("Tokenizer has not been fitted. Call fit() first.")
        ids = self._hf.encode(text).ids
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self._hf.decode(ids, skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._hf.save(str(path))
        config_path = path.with_suffix(path.suffix + ".config.json")
        config_path.write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("BPE tokenizer saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        from tokenizers import Tokenizer as HFTokenizer

        tokenizer = cls()
        tokenizer._hf = HFTokenizer.from_file(str(path))
        config_path = path.with_suffix(path.suffix + ".config.json")
        if config_path.exists():
            tokenizer.config = TokenizerConfig.from_dict(
                json.loads(config_path.read_text(encoding="utf-8"))
            )
        tokenizer._fitted = True
        return tokenizer


def build_tokenizer(
    tokenizer_type: str = "char", config: Optional[TokenizerConfig] = None
):
    """Factory: build a tokenizer by type ('char' or 'bpe')."""
    config = config or TokenizerConfig(vocab_size=0, tokenizer_type=tokenizer_type)
    if tokenizer_type == "char":
        return CharTokenizer(config=config)
    if tokenizer_type == "bpe":
        return BPETokenizer(config=config)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use 'char' or 'bpe'.")


def load_tokenizer(path: str | Path):
    """Load a tokenizer, detecting char vs BPE from the saved config."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "config" in payload and "char_to_id" in payload:
            return CharTokenizer.load(path)
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    config_path = Path(str(path) + ".config.json")
    if config_path.exists():
        return BPETokenizer.load(path)
    return CharTokenizer.load(path)


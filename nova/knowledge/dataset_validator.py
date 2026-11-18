"""Dataset validation: checks that a training corpus is usable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ValidationReport:
    valid: bool
    document_count: int
    total_chars: int
    empty_documents: int
    errors: List[str]
    warnings: List[str]


def validate_texts(texts: List[str], min_chars: int = 10) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []
    empty = 0
    total_chars = 0
    for i, text in enumerate(texts):
        if not text or not text.strip():
            empty += 1
            errors.append(f"Document {i} is empty")
        else:
            total_chars += len(text)
    if not texts:
        errors.append("Corpus contains no documents")
    if total_chars < min_chars:
        warnings.append(f"Corpus is small ({total_chars} chars); training may not converge")
    valid = len(errors) == 0 and len(texts) > 0
    return ValidationReport(
        valid=valid,
        document_count=len(texts),
        total_chars=total_chars,
        empty_documents=empty,
        errors=errors,
        warnings=warnings,
    )

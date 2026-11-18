"""Text generation strategies for NOVA Tiny.

Supports greedy decoding, temperature sampling, top-K filtering, top-P (nucleus)
filtering, and repetition penalty. Designed for CPU inference with small models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .model import NOVATinyModel


def _top_k_top_p_filter(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """Apply top-K and top-P filtering to a 1-D logits vector in place."""
    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size) if top_k and top_k > 0 else vocab_size
    if top_k < vocab_size:
        # Remove all tokens with a probability below the top-K threshold.
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < threshold, torch.tensor(float("-inf"), device=logits.device), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above top_p.
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift so the first token above threshold is kept (the one that crosses it).
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


@torch.inference_mode()
def generate(
    model: NOVATinyModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model: The NOVA Tiny model.
        tokenizer: A fitted tokenizer with encode/decode.
        prompt: Starting text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (1.0 = neutral, lower = more deterministic).
        top_k: Keep only the top-K most likely tokens (0 = disabled).
        top_p: Nucleus sampling cumulative probability (1.0 = disabled).
        repetition_penalty: Penalty applied to already-generated tokens (>1 reduces repetition).
        device: Torch device.

    Returns:
        The generated text (prompt + continuation).
    """
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(prompt, add_special=True)
    # Work with a copy so we don't mutate the tokenizer output list.
    tokens = list(input_ids)

    for _ in range(max_new_tokens):
        # Truncate context to the model's context length.
        context = tokens[-model.context_length :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)

        logits, _ = model(input_tensor)
        # Focus on the last position's logits.
        next_logits = logits[0, -1, :].clone()

        # Repetition penalty: divide logits of already-seen tokens.
        if repetition_penalty != 1.0 and len(tokens) > 0:
            for token in set(tokens):
                if next_logits[token] < 0:
                    next_logits[token] *= repetition_penalty
                else:
                    next_logits[token] /= repetition_penalty

        if temperature == 0.0:
            # Greedy decoding.
            next_token = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            next_logits = _top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next_token)

    return tokenizer.decode(tokens, skip_special=True)

"""Tokenizer-independent prompt truncation primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


class Tokenizer(Protocol):
    """The subset of the Hugging Face tokenizer API that we use."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...


Strategy = Literal["head", "tail"]


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """The truncated text and the accounting used to produce it."""

    text: str
    original_tokens: int
    kept_tokens: int
    max_prompt_tokens: int

    @property
    def truncated(self) -> bool:
        return self.kept_tokens < self.original_tokens

    @property
    def removed_tokens(self) -> int:
        return self.original_tokens - self.kept_tokens


def truncate_prompt(
    prompt: str,
    tokenizer: Tokenizer,
    *,
    context_size: int = 4096,
    reserve_tokens: int = 512,
    template_tokens: int = 128,
    strategy: Strategy = "tail",
) -> TruncationResult:
    """Fit a prompt into a context window using actual tokenizer token IDs.

    ``reserve_tokens`` leaves room for model output and ``template_tokens`` is a
    safety allowance for the model template Ollama adds around the raw prompt.
    ``tail`` retains the end of a long prompt; ``head`` retains its beginning.
    Special tokens are deliberately excluded because Ollama applies its own model
    template when the prompt is submitted.
    """
    if context_size <= 0:
        raise ValueError("context_size must be greater than zero")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens cannot be negative")
    if template_tokens < 0:
        raise ValueError("template_tokens cannot be negative")
    if reserve_tokens + template_tokens >= context_size:
        raise ValueError("reserve_tokens plus template_tokens must be smaller than context_size")
    if strategy not in ("head", "tail"):
        raise ValueError("strategy must be 'head' or 'tail'")

    max_prompt_tokens = context_size - reserve_tokens - template_tokens
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    original_tokens = len(token_ids)

    if original_tokens <= max_prompt_tokens:
        return TruncationResult(prompt, original_tokens, original_tokens, max_prompt_tokens)

    kept_ids = (
        token_ids[:max_prompt_tokens] if strategy == "head" else token_ids[-max_prompt_tokens:]
    )
    text = tokenizer.decode(kept_ids, skip_special_tokens=True)
    return TruncationResult(text, original_tokens, len(kept_ids), max_prompt_tokens)


def load_tokenizer(
    model_name_or_path: str,
    *,
    use_fast: bool = True,
    local_files_only: bool = False,
) -> Any:
    """Load a Hugging Face tokenizer without enabling remote model code."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "transformers is required; install the project with `pip install -e .`"
        ) from exc

    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )

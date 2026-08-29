"""Token-aware fitting for Ollama chat message arrays."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .core import Strategy, Tokenizer


@dataclass(frozen=True, slots=True)
class ChatTruncationResult:
    messages: list[dict[str, Any]]
    original_tokens: int
    kept_tokens: int
    max_input_tokens: int
    dropped_messages: int

    @property
    def truncated(self) -> bool:
        return self.dropped_messages > 0 or self.kept_tokens < self.original_tokens


def count_chat_tokens(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Count a rendered chat, preferring the tokenizer's native chat template."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    template_messages = deepcopy([dict(message) for message in messages])
    if callable(apply_template):
        try:
            kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = list(tools)
            ids = apply_template(template_messages, **kwargs)
            if isinstance(ids, Mapping) and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except (ImportError, TypeError, ValueError, IndexError):
            # Some tokenizers have no template or require model-specific fields.
            pass

    rendered = json.dumps(
        {"messages": list(messages), "tools": list(tools or [])},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def fit_chat_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Tokenizer,
    *,
    context_size: int,
    reserve_tokens: int,
    template_tokens: int = 128,
    strategy: Strategy = "tail",
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> ChatTruncationResult:
    """Fit chat history while preserving system messages and the newest turn.

    Old non-system messages are removed first. If that is insufficient, the text
    content of the newest message is truncated at tokenizer boundaries.
    """
    if not messages:
        raise ValueError("messages cannot be empty")
    if context_size <= 0:
        raise ValueError("context_size must be greater than zero")
    if reserve_tokens < 0 or template_tokens < 0:
        raise ValueError("token reserves cannot be negative")
    if reserve_tokens + template_tokens >= context_size:
        raise ValueError("token reserves must be smaller than context_size")
    if strategy not in ("head", "tail"):
        raise ValueError("strategy must be 'head' or 'tail'")

    max_input = context_size - reserve_tokens - template_tokens
    fitted = deepcopy([dict(message) for message in messages])
    original_tokens = count_chat_tokens(fitted, tokenizer, tools=tools)
    dropped = 0

    while count_chat_tokens(fitted, tokenizer, tools=tools) > max_input:
        removable_start = next(
            (index for index, message in enumerate(fitted[:-1]) if message.get("role") != "system"),
            None,
        )
        if removable_start is None:
            break
        removable_end = removable_start + 1
        while removable_end < len(fitted) - 1:
            if fitted[removable_end].get("role") == "user":
                break
            removable_end += 1
        dropped += removable_end - removable_start
        del fitted[removable_start:removable_end]

    kept_tokens = count_chat_tokens(fitted, tokenizer, tools=tools)
    if kept_tokens > max_input:
        target = next(
            (index for index in range(len(fitted) - 1, -1, -1) if _text_content(fitted[index])),
            None,
        )
        if target is None:
            raise ValueError("non-text chat content exceeds the configured token budget")
        original_content = _text_content(fitted[target])
        content_ids = list(tokenizer.encode(original_content, add_special_tokens=False))

        low, high = 0, len(content_ids)
        best: list[dict[str, Any]] | None = None
        best_count = 0
        while low <= high:
            length = (low + high) // 2
            selected = content_ids[:length] if strategy == "head" else content_ids[-length:]
            if length == 0:
                selected = []
            candidate = deepcopy(fitted)
            candidate[target]["content"] = tokenizer.decode(selected, skip_special_tokens=True)
            candidate_count = count_chat_tokens(candidate, tokenizer, tools=tools)
            if candidate_count <= max_input:
                best, best_count = candidate, candidate_count
                low = length + 1
            else:
                high = length - 1

        if best is None:
            raise ValueError("chat template and preserved messages exceed the token budget")
        fitted, kept_tokens = best, best_count

    return ChatTruncationResult(
        messages=fitted,
        original_tokens=original_tokens,
        kept_tokens=kept_tokens,
        max_input_tokens=max_input,
        dropped_messages=dropped,
    )


def _text_content(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    return content if isinstance(content, str) else ""

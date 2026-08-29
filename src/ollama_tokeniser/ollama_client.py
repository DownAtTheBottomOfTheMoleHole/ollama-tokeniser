"""Small Ollama wrapper that truncates before sending a generation request."""

from __future__ import annotations

from typing import Any

from .core import Strategy, Tokenizer, truncate_prompt


def generate(
    prompt: str,
    *,
    model: str,
    tokenizer: Tokenizer,
    context_size: int = 4096,
    reserve_tokens: int = 512,
    template_tokens: int = 128,
    strategy: Strategy = "tail",
    host: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Truncate ``prompt`` and make a non-streaming Ollama generate request."""
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ollama is required; install the project with `pip install -e .`"
        ) from exc

    result = truncate_prompt(
        prompt,
        tokenizer,
        context_size=context_size,
        reserve_tokens=reserve_tokens,
        template_tokens=template_tokens,
        strategy=strategy,
    )
    client = ollama.Client(host=host) if host else ollama.Client()
    response = client.generate(
        model=model,
        prompt=result.text,
        stream=False,
        options={"num_ctx": context_size, "num_predict": reserve_tokens, **options},
    )

    # Ollama's response objects expose model_dump(); older clients are mapping-like.
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    payload["truncation"] = {
        "truncated": result.truncated,
        "original_tokens": result.original_tokens,
        "kept_tokens": result.kept_tokens,
        "removed_tokens": result.removed_tokens,
        "max_prompt_tokens": result.max_prompt_tokens,
    }
    return payload

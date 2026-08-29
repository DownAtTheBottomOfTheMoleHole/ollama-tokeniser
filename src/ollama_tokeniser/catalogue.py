"""Load and resolve the built-in Ollama model tokenizer catalogue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCatalogue:
    """Resolved tokenizer mappings plus documented unsupported families."""

    tokenizers: dict[str, str]
    unsupported: dict[str, str]
    model_count: int
    generated_at: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ModelCatalogue:
        entries = payload.get("models")
        if not isinstance(entries, dict):
            raise ValueError("catalogue must contain a models object")

        tokenizers: dict[str, str] = {}
        unsupported: dict[str, str] = {}
        for family, raw_entry in entries.items():
            if not isinstance(raw_entry, dict):
                raise ValueError(f"catalogue entry {family!r} must be an object")
            status = raw_entry.get("status")
            if status == "mapped" and raw_entry.get("tokenizer"):
                tokenizers[str(family)] = str(raw_entry["tokenizer"])
            elif status == "unsupported" and raw_entry.get("reason"):
                unsupported[str(family)] = str(raw_entry["reason"])
            else:
                raise ValueError(f"catalogue entry {family!r} is incomplete")

        return cls(
            tokenizers=tokenizers,
            unsupported=unsupported,
            model_count=len(entries),
            generated_at=str(payload.get("generated_at", "unknown")),
        )


def load_catalogue(reference: str, config_dir: Path) -> ModelCatalogue:
    """Load the packaged catalogue or a config-relative JSON catalogue."""
    if reference == "builtin":
        resource = files("ollama_tokeniser").joinpath("data/ollama-model-catalogue.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    else:
        path = Path(reference)
        if not path.is_absolute():
            path = config_dir / path
        payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelCatalogue.from_payload(payload)


def model_family(model: str) -> str:
    """Remove an Ollama tag or digest while retaining an optional namespace."""
    return model.split("@", 1)[0].rsplit(":", 1)[0]


def tokenizer_for_model(model: str, mappings: dict[str, str]) -> str | None:
    """Resolve an exact model override before falling back to its family."""
    if model in mappings:
        return mappings[model]
    if ":" not in model and f"{model}:latest" in mappings:
        return mappings[f"{model}:latest"]
    return mappings.get(model_family(model))

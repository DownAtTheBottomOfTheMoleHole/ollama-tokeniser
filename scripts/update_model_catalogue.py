#!/usr/bin/env python3
"""Generate the dated Ollama library catalogue from its public index."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

LIBRARY_URL = "https://ollama.com/library"
LINK_PATTERN = re.compile(r'href="/library/([a-zA-Z0-9._-]+)')

# Conservative family mappings: every tag in a family must share this tokenizer.
# Families with tag-dependent bases remain unsupported until tag rules are added.
TOKENIZERS = {
    "codegeex4": "THUDM/codegeex4-all-9b",
    "codestral": "mistralai/Codestral-22B-v0.1",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-llm": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-v2": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-v2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "falcon": "tiiuae/falcon-7b-instruct",
    "falcon2": "tiiuae/falcon-11B",
    "falcon3": "tiiuae/Falcon3-7B-Instruct",
    "glm4": "THUDM/glm-4-9b-chat",
    "granite-code": "ibm-granite/granite-8b-code-instruct",
    "granite3-dense": "ibm-granite/granite-3.0-8b-instruct",
    "granite3-moe": "ibm-granite/granite-3.0-3b-a800m-instruct",
    "granite3.1-dense": "ibm-granite/granite-3.1-8b-instruct",
    "granite3.1-moe": "ibm-granite/granite-3.1-3b-a800m-instruct",
    "granite3.2": "ibm-granite/granite-3.2-8b-instruct",
    "granite3.3": "ibm-granite/granite-3.3-8b-instruct",
    "internlm2": "internlm/internlm2-chat-7b",
    "llama2": "NousResearch/Llama-2-7b-chat-hf",
    "llama3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama3.1": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "llama3.2": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.3": "unsloth/Llama-3.3-70B-Instruct",
    "mathstral": "mistralai/Mathstral-7B-v0.1",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "olmo2": "allenai/OLMo-2-1124-7B-Instruct",
    "openchat": "openchat/openchat-3.5-0106",
    "phi": "microsoft/phi-2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "phi4": "microsoft/phi-4",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "phi4-mini-reasoning": "microsoft/Phi-4-mini-reasoning",
    "phi4-reasoning": "microsoft/Phi-4-reasoning",
    "qwen": "Qwen/Qwen-7B-Chat",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen2-math": "Qwen/Qwen2-Math-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
    "qwq": "Qwen/QwQ-32B",
    "smollm": "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "stable-code": "stabilityai/stable-code-3b",
    "stablelm2": "stabilityai/stablelm-2-zephyr-1_6b",
    "starcoder": "bigcode/tiny_starcoder_py",
    "starcoder2": "bigcode/starcoder2-7b",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "yi": "01-ai/Yi-6B-Chat",
    "yi-coder": "01-ai/Yi-Coder-9B-Chat",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
}

VISION_MARKERS = ("vision", "llava", "bakllava", "moondream", "minicpm-v", "medgemma")
EMBEDDING_MARKERS = ("embed", "bge", "minilm", "paraphrase")


def unsupported_reason(family: str) -> str:
    if any(marker in family for marker in VISION_MARKERS):
        return "vision input requires model-specific image-token accounting"
    if any(marker in family for marker in EMBEDDING_MARKERS):
        return "embedding routes are outside this chat and generate proxy"
    if family == "deepseek-r1":
        return "tags use different Qwen, Llama, and DeepSeek tokenizer families"
    return "exact public tokenizer mapping has not yet been confirmed"


def build_catalogue(html: str, generated_at: str | None = None) -> dict[str, object]:
    families = sorted(set(LINK_PATTERN.findall(html)))
    models: dict[str, dict[str, str]] = {}
    for family in families:
        if family in TOKENIZERS:
            models[family] = {"status": "mapped", "tokenizer": TOKENIZERS[family]}
        else:
            models[family] = {"status": "unsupported", "reason": unsupported_reason(family)}
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(UTC).date().isoformat(),
        "source": LIBRARY_URL,
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-html", type=Path, help="Use a saved library page")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_html:
        html = args.input_html.read_text(encoding="utf-8")
    else:
        request = urllib.request.Request(LIBRARY_URL, headers={"User-Agent": "ollama-tokeniser"})
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode()
    payload = build_catalogue(html)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("models") == payload["models"]:
            payload["generated_at"] = existing.get("generated_at", payload["generated_at"])
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['models'])} model families to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

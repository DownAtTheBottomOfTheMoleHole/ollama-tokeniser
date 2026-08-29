#!/usr/bin/env bash
set -eu

minimum_vscode="1.127.0"
extension_id="Ollama.ollama"

if ! command -v code >/dev/null 2>&1; then
  echo "error: the VS Code 'code' command is not available" >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "error: Ollama is not installed" >&2
  exit 1
fi

vscode_version="$(code --version | head -n 1)"
first_version="$(printf '%s\n%s\n' "$minimum_vscode" "$vscode_version" | sort -V | head -n 1)"
if [ "$first_version" != "$minimum_vscode" ]; then
  echo "error: VS Code $minimum_vscode or newer is required (found $vscode_version)" >&2
  exit 1
fi

if ! code --list-extensions | grep -Fxiq "$extension_id"; then
  echo "error: install the official provider with:" >&2
  echo "  code --install-extension $extension_id" >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 2 \
  http://127.0.0.1:11434/api/version >/dev/null; then
  echo "error: Ollama is not responding at http://127.0.0.1:11434" >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 2 \
  http://127.0.0.1:11435/api/version >/dev/null; then
  echo "error: the tokenizing proxy is not responding at http://127.0.0.1:11435" >&2
  echo "start it with: uv run --offline ollama-tokeniser-proxy --config tokenizers.json" >&2
  exit 1
fi

model_count="$(ollama list | tail -n +2 | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
if [ "$model_count" -eq 0 ]; then
  echo "error: no local Ollama models are installed" >&2
  echo "example: ollama pull qwen3.6" >&2
  exit 1
fi

echo "VS Code: $vscode_version"
echo "Provider: $extension_id"
echo "Ollama: reachable with $model_count local model(s)"
echo "Tokenizer proxy: reachable"
echo "Local token-aware VS Code Chat prerequisites are ready."

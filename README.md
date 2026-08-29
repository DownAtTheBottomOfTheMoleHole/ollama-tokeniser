# Ollama Tokeniser

![Down At The Bottom Of The Mole Hole banner][org-banner]

[![CI][ci-badge]][ci-link]
[![Python][python-badge]][python-link]
[![License: MIT][licence-badge]][licence-link]

> **Note:** This is a community-maintained integration. It is not an official
> Ollama, Hugging Face, Microsoft, GitHub, or VS Code project.

A local, model-aware tokenizer proxy for using Ollama models in VS Code Chat
without consuming GitHub Copilot credits. It counts rendered chat messages and
tool definitions with the matching Hugging Face tokenizer, trims oversized input,
and forwards the fitted request to a locally running Ollama server.

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Security and privacy](#security-and-privacy)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [VS Code Chat](#vs-code-chat)
- [Configuration](#configuration)
- [Python API](#python-api)
- [Command-line usage](#command-line-usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [Limitations](#limitations)
- [Licence](#licence)

## Overview

VS Code's official Ollama extension normally connects directly to Ollama. This
repository inserts a local reverse proxy between them so the tokenizer code is in
the request path rather than being an unrelated utility.

The proxy:

- uses an explicit Ollama-model to Hugging Face-tokenizer mapping;
- counts the model's rendered chat template, messages, tool definitions, and tool
  call data;
- removes the oldest complete conversation turns first;
- truncates the newest text at tokenizer boundaries only when still required;
- reserves separate budgets for generated output and template safety overhead;
- hides models without a configured tokenizer mapping; and
- adds response headers that show whether tokenization was applied.

## Architecture

```text
VS Code Chat
    -> official Ollama extension
    -> http://127.0.0.1:11435
       ollama-tokeniser proxy
       - load the matching cached tokenizer
       - count the rendered chat and tools
       - fit messages to the configured context budget
    -> http://127.0.0.1:11434
       local Ollama server
    -> local model
```

For a processed request, the proxy returns:

```text
X-Ollama-Tokeniser: applied
X-Ollama-Tokeniser-Truncated: false
```

## Security and privacy

Normal Chat inference and tokenization stay on the local machine:

- both services bind to loopback addresses;
- the proxy refuses non-loopback listen addresses;
- only `/api/tags`, `/api/version`, `/api/show`, `/api/chat`, and
  `/api/generate` are allowed;
- model mutation and publishing routes are blocked;
- prompt content is not written to logs;
- tokenizer remote code is disabled; and
- normal runtime uses cached tokenizer files only.

The first model, extension, dependency, and tokenizer downloads require internet
access. After setup, the proxy can run offline. No GitHub, Copilot, Ollama Cloud,
or Hugging Face access token is required for the configured public model.

See [SECURITY.md](SECURITY.md) before changing network or trust boundaries.

## Getting started

### Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- VS Code 1.127 or newer
- the official `Ollama.ollama` VS Code extension

### Installation

```bash
git clone https://github.com/DownAtTheBottomOfTheMoleHole/ollama-tokeniser.git
cd ollama-tokeniser
uv sync --extra test
code --install-extension Ollama.ollama
ollama pull qwen2.5-coder:1.5b
uv run ollama-tokeniser-cache --config tokenizers.json
```

The cache command downloads the configured public tokenizer once. Proxy runtime
then uses `local_files_only` mode.

### VS Code Chat

1. Open this repository in VS Code.
2. Trust the workspace and allow its automatic task when prompted.
3. Reload VS Code if the Ollama extension was installed during setup.
4. Open Chat and select `qwen2.5-coder:1.5b` from the **Ollama** section.

The workspace task starts the proxy with:

```bash
uv run --offline ollama-tokeniser-proxy --config tokenizers.json
```

VS Code remembers the selected model. Workspace settings also route supported
utility requests through the selected local model, disable Chat session sync, and
enable local models for agent-host sessions.

Run the readiness check while the proxy is active:

```bash
./scripts/check-vscode-local.sh
```

## Configuration

Edit [tokenizers.json](tokenizers.json) to configure supported model mappings and
token budgets.

<!-- markdownlint-disable MD013 -->

| Setting | Default | Description |
| --- | --- | --- |
| `model_tokenizers` | Qwen coder mapping | Exact Ollama model to Hugging Face tokenizer mappings |
| `context_size` | `32768` | Total model context budget |
| `reserve_tokens` | `4096` | Tokens reserved for generated output |
| `template_tokens` | `128` | Additional template safety allowance |
| `strategy` | `tail` | Preserve the beginning (`head`) or end (`tail`) when truncating |
| `local_files_only` | `true` | Prevent tokenizer network access during proxy runtime |

<!-- markdownlint-enable MD013 -->

Add another model only after identifying its exact original tokenizer. Then rerun:

```bash
uv run ollama-tokeniser-cache --config tokenizers.json
```

Unmapped models are deliberately omitted from VS Code model discovery.

## Python API

```python
from ollama_tokeniser import load_tokenizer, truncate_prompt

tokenizer = load_tokenizer(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    local_files_only=True,
)
result = truncate_prompt(
    long_prompt,
    tokenizer,
    context_size=32768,
    reserve_tokens=4096,
    template_tokens=128,
    strategy="tail",
)

print(result.text)
print(result.original_tokens, result.kept_tokens)
```

The package also exposes `fit_chat_messages` for multi-message requests and
`generate` for direct non-streaming Ollama generation.

## Command-line usage

Truncate a file without calling Ollama:

```bash
uv run ollama-tokeniser truncate \
  --tokenizer Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --context-size 32768 \
  --reserve-tokens 4096 \
  --template-tokens 128 \
  --file prompt.txt
```

Start the cache-only local proxy manually:

```bash
uv run --offline ollama-tokeniser-proxy --config tokenizers.json
```

## Testing

```bash
uv sync --extra test
uv run --offline pytest
uv build --offline
git diff --check
```

Tests use deterministic fake tokenizers and do not require inference, credentials,
or network access.

## Contributing

Contributions are welcome. Follow the organisation's
[contribution guidance][contributing-link], keep changes focused, use Conventional
Commits, and include tests for behavioural changes.

Repository-specific Copilot guidance is available in
`.github/copilot-instructions.md`, `.github/instructions/`, and `.github/agents/`.

## Limitations

- A tokenizer mapping must match the Ollama model. Similar vocabularies are not
  a safe substitute.
- Image token costs are model-specific and are not currently calculated. Do not
  map vision models without implementing their image-token accounting.
- VS Code ghost-text completions, semantic search, and cloud embeddings are
  separate GitHub-backed features and are not routed through this proxy.
- The proxy is intended for a single-user local workstation, not shared hosting.

## Licence

Released under the [MIT Licence](LICENSE).

[ci-badge]: https://github.com/DownAtTheBottomOfTheMoleHole/ollama-tokeniser/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/DownAtTheBottomOfTheMoleHole/ollama-tokeniser/actions/workflows/ci.yml
[contributing-link]: https://github.com/DownAtTheBottomOfTheMoleHole/.github/blob/main/CONTRIBUTING.md
[licence-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[licence-link]: LICENSE
[org-banner]: https://raw.githubusercontent.com/DownAtTheBottomOfTheMoleHole/.github/main/assets/banners/datbmh_banner_v7a_base.png
[python-badge]: https://img.shields.io/badge/python-%3E%3D3.10-3776AB
[python-link]: https://www.python.org/

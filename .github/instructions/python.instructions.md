---
applyTo: "**/*.py,pyproject.toml,tests/**/*.py"
description: "Python implementation and testing guidance for ollama-tokeniser."
---

# Python instructions

- Support Python 3.10 and newer.
- Prefer the standard library unless a dependency materially improves tokenizer
  correctness or Ollama compatibility.
- Keep functions focused and typed; avoid hidden global state other than bounded
  tokenizer caches.
- Validate network inputs, configuration values, request sizes, and token budgets.
- Return concise client errors without exposing stack traces or prompt content.
- Preserve loopback-only network defaults and least-privilege API routing.
- Cover success, boundary, and failure paths with deterministic tests.
- Do not require model inference, internet access, or credentials in unit tests.

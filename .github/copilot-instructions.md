# Repository instructions

Apply these rules to all work in this repository.

- Use British English in documentation, comments, and user-facing messages.
- Prefer small, reversible changes that preserve public API compatibility.
- Keep inference and prompt processing local by default.
- Never add credentials, access tokens, private paths, prompt content, or machine
  state to source control or logs.
- Keep the proxy bound to loopback and retain its Ollama API route allowlist.
- Require an explicit, model-matched tokenizer mapping. Never silently fall back
  to character counts or an unrelated tokenizer.
- Keep `trust_remote_code` disabled and normal runtime in cache-only mode.
- Add or update tests whenever behaviour changes.
- Run `uv run --offline pytest`, `uv build --offline`, and `git diff --check`
  before claiming completion.
- Use Conventional Commits with an imperative, lower-case subject.
- Document security impact, compatibility impact, and validation evidence in
  pull requests.

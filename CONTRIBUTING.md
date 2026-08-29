# Contributing

Thank you for improving Ollama Tokeniser. Keep changes small, reviewable, and
covered by tests where behaviour changes.

## GitHub Flow

1. Create a short-lived branch from `main`.
2. Make focused, atomic commits.
3. Open a pull request and let CI complete.
4. Squash-merge the approved pull request into `main`.
5. Delete the branch after merging.

`main` must remain releasable. Do not develop directly on a long-lived release or
development branch.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) with a lower-case,
imperative description no longer than 72 characters:

```text
feat(catalogue): add a verified model mapping
fix(proxy): preserve system messages while trimming
docs: explain local tokenizer caching
```

Use `!` and a `BREAKING CHANGE:` footer for incompatible changes. Each commit
should represent one coherent change and should pass its relevant tests.

GitVersion 6 derives release versions from these messages:

- `feat` increments the minor version;
- `fix`, `perf`, and `refactor` increment the patch version;
- a breaking change increments the major version; and
- documentation, test, build, CI, style, and chore commits do not increment it.

## Pull requests

Run the local checks before opening a pull request:

```bash
uv sync --extra test
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
python scripts/check_conventional_commits.py --base origin/main
```

Never include credentials, private prompts, model files, tokenizer caches, or
machine-specific configuration.

## Releases

Maintainers release the current `main` commit with an annotated `vMAJOR.MINOR.PATCH`
tag. GitVersion 6 verifies the version, and the release workflow tests, builds,
attests, and attaches the Python distributions to a GitHub Release.

PyPI publishing uses OpenID Connect Trusted Publishing and is disabled until the
`pypi` environment and `PUBLISH_TO_PYPI=true` repository variable are configured.
No long-lived PyPI token belongs in GitHub.

"""Command-line interface for truncating prompts and calling Ollama."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import load_tokenizer, truncate_prompt
from .ollama_client import generate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-tokeniser",
        description="Fit prompts into an Ollama model's token context window.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--tokenizer", required=True, help="Hugging Face tokenizer name or path"
        )
        command.add_argument("--context-size", type=int, default=4096)
        command.add_argument("--reserve-tokens", type=int, default=512)
        command.add_argument("--template-tokens", type=int, default=128)
        command.add_argument("--strategy", choices=("head", "tail"), default="tail")
        command.add_argument("--file", type=Path, help="Read the prompt from a UTF-8 file")
        command.add_argument("prompt", nargs="?", help="Prompt text; stdin is used when omitted")

    truncate = subcommands.add_parser("truncate", help="Print a fitted prompt")
    add_common_arguments(truncate)

    generate_parser = subcommands.add_parser("generate", help="Send the fitted prompt to Ollama")
    add_common_arguments(generate_parser)
    generate_parser.add_argument("--model", required=True, help="Installed Ollama model name")
    generate_parser.add_argument("--host", help="Ollama host URL")
    return parser


def _read_prompt(args: argparse.Namespace) -> str:
    sources = sum((args.file is not None, args.prompt is not None, not sys.stdin.isatty()))
    if sources > 1:
        raise ValueError("provide the prompt using exactly one of --file, an argument, or stdin")
    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("provide a prompt argument, --file, or piped stdin")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        prompt = _read_prompt(args)
        tokenizer = load_tokenizer(args.tokenizer)
        common = {
            "context_size": args.context_size,
            "reserve_tokens": args.reserve_tokens,
            "template_tokens": args.template_tokens,
            "strategy": args.strategy,
        }
        if args.command == "generate":
            response = generate(
                prompt,
                model=args.model,
                tokenizer=tokenizer,
                host=args.host,
                **common,
            )
            print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
        else:  # truncate
            result = truncate_prompt(prompt, tokenizer, **common)
            print(result.text)
            print(
                f"tokens: {result.kept_tokens}/{result.original_tokens} "
                f"(budget {result.max_prompt_tokens})",
                file=sys.stderr,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

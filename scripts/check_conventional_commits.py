#!/usr/bin/env python3
"""Validate commit subjects for a GitHub Flow branch or push."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SUBJECT_PATTERN = re.compile(
    r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|security|style|test)"
    r"(?:\([a-z0-9][a-z0-9._/-]*\))?!?: [a-z0-9].*$"
)
ZERO_SHA = "0" * 40


def commit_subjects(base: str | None, head: str) -> list[tuple[str, str]]:
    """Return abbreviated SHA and subject for commits in the requested range."""
    revision = head
    if base and base != ZERO_SHA and _commit_exists(base):
        revision = f"{base}..{head}"

    output = subprocess.run(
        ["git", "log", "--format=%h%x09%s", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [tuple(line.split("\t", 1)) for line in output.splitlines() if line]


def _commit_exists(revision: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def invalid_subjects(subjects: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return subjects that do not follow this repository's commit convention."""
    return [
        (sha, subject)
        for sha, subject in subjects
        if len(subject) > 72 or not SUBJECT_PATTERN.fullmatch(subject)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Exclusive base commit; omitted for all reachable commits")
    parser.add_argument("--head", default="HEAD", help="Inclusive head commit")
    parser.add_argument("--subject", help="Validate one pull request or commit subject")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.subject:
        subjects = [("pull request", args.subject)]
    else:
        subjects = commit_subjects(args.base, args.head)
    invalid = invalid_subjects(subjects)
    if invalid:
        print("Commit subjects must use Conventional Commits and be at most 72 characters:")
        for sha, subject in invalid:
            print(f"  {sha} {subject}")
        return 1

    print(f"Validated {len(subjects)} conventional commit subject(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

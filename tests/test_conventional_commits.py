import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_conventional_commits.py"
SPEC = importlib.util.spec_from_file_location("check_conventional_commits", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
invalid_subjects = MODULE.invalid_subjects


def test_accepts_supported_conventional_subjects() -> None:
    subjects = [
        ("abc1234", "feat(proxy): add local routing"),
        ("def5678", "fix!: remove unsafe remote access"),
        ("987abcd", "docs: explain offline operation"),
    ]

    assert invalid_subjects(subjects) == []


def test_rejects_non_conventional_or_overlong_subjects() -> None:
    subjects = [
        ("abc1234", "Added local routing"),
        ("def5678", "feat: Add upper-case description"),
        ("987abcd", f"fix: {'x' * 68}"),
    ]

    assert invalid_subjects(subjects) == subjects

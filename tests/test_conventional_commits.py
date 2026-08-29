import importlib.util
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_conventional_commits.py"
SPEC = importlib.util.spec_from_file_location("check_conventional_commits", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
invalid_subjects = MODULE.invalid_subjects
main = MODULE.main


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


def test_main_validates_pull_request_subject_without_reading_git() -> None:
    with (
        patch("sys.argv", ["check", "--subject", "build(deps): update checkout action"]),
        patch.object(MODULE, "commit_subjects") as commit_subjects,
    ):
        assert main() == 0

    commit_subjects.assert_not_called()


def test_main_rejects_non_conventional_pull_request_subject() -> None:
    with patch("sys.argv", ["check", "--subject", "Configure Renovate"]):
        assert main() == 1

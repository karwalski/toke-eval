"""Guard: no tracked code may DEFAULT to a hidden_tests/ directory (story 128.30).

AGENTS.md section 5: agents must never read any ``hidden_tests/`` directory.
A script whose path argument *defaults* to one reads hidden tests whenever it
is run bare, so this guard fails if any tracked Python or shell file under
``benchmark/``, ``scripts/`` or ``toke_eval/`` holds a default value or an
assignment pointing at ``hidden_tests``.  Usage text and refusal checks are
fine; only defaults and assignments are flagged.

The scan is a text search over files listed by ``git ls-files`` with
``benchmark/hidden_tests`` excluded by pathspec, so it never opens a hidden
test file even if one were tracked.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SELF = Path(__file__).resolve()
_TOKEN = "hidden" + "_tests"  # spelled so this file's own source is not a hit

_PATTERNS = [
    # argparse / function keyword defaults: default=... hidden_tests
    re.compile(r"\bdefault\s*=\s*[^,)\n]*" + _TOKEN),
    # Python or shell assignment whose right-hand side names hidden_tests
    re.compile(r"^\s*(?:export\s+|local\s+)?[A-Za-z_][\w.]*\s*(?::[^=\n]+)?=(?!=)[^\n]*" + _TOKEN),
    # shell parameter default: ${VAR:-...hidden_tests} / ${VAR:=...}
    re.compile(r"\$\{\w+:?[-=][^}]*" + _TOKEN),
    # environ.get("X", "...hidden_tests") / dict.get / setdefault fallbacks
    re.compile(r"\b(?:get|setdefault)\([^)\n]*,[^)\n]*" + _TOKEN),
]

# Reviewed exceptions: (repo-relative path, stripped line).  Each one must be
# unreachable without the owner's explicit opt-in.
_ALLOW = {
    # Owner-only schema check; runs only with TOKE_OWNER_HIDDEN_TESTS=1.
    ("benchmark/tests/test_harness_128_1c.py",
     'hidden = _BENCH / "' + _TOKEN + '"'),
}

_SCAN_DIRS = ("benchmark", "scripts", "toke_eval")
_SUFFIXES = (".py", ".sh")


def find_defaults(root: Path, rel_paths: list[str]) -> list[str]:
    hits = []
    for rel in rel_paths:
        if not rel.endswith(_SUFFIXES) or (root / rel).resolve() == _SELF:
            continue
        parts = Path(rel).parts
        if _TOKEN in parts:  # never open anything under a hidden_tests/ dir
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if _TOKEN not in line or line.lstrip().startswith("#"):
                continue
            if (rel, line.strip()) in _ALLOW:
                continue
            if any(p.search(line) for p in _PATTERNS):
                hits.append(f"{rel}:{n}: {line.strip()}")
    return hits


def _tracked() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "ls-files", "--", *_SCAN_DIRS,
             f":!benchmark/{_TOKEN}"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout; cannot list tracked files")
    return out.splitlines()


def test_no_tracked_code_defaults_to_hidden_tests() -> None:
    hits = find_defaults(_REPO, _tracked())
    assert not hits, (
        "code defaults to a hidden tests directory (AGENTS.md section 5); "
        "require an explicit path instead:\n" + "\n".join(hits)
    )


@pytest.mark.parametrize("planted", [
    'parser.add_argument("--tasks-dir", default=Path("' + _TOKEN + '/"))',
    'TASKS_DIR = "' + _TOKEN + '"',
    'TESTS_DIR="$REPO_ROOT/' + _TOKEN + '"',
    'DIR="${TESTS_DIR:-' + _TOKEN + '}"',
    'd = os.environ.get("TASKS", "' + _TOKEN + '")',
    'tests: Path = ROOT / "' + _TOKEN + '"',
])
def test_negative_control_planted_default_is_caught(tmp_path: Path,
                                                   planted: str) -> None:
    ext = ".sh" if "$" in planted else ".py"
    rel = f"scripts/planted{ext}"
    (tmp_path / "scripts").mkdir()
    (tmp_path / rel).write_text(planted + "\n")
    assert find_defaults(tmp_path, [rel]) == [f"{rel}:1: {planted}"]


@pytest.mark.parametrize("benign", [
    "# never point this at a " + _TOKEN + "/ directory",
    '    if "' + _TOKEN + '" in p.parts:',
    '        --tasks-dir <tasks-dir> \\\\  (not ' + _TOKEN + ')',
    'assert "--tasks-dir ' + _TOKEN + '/" not in readme',
    'x == "' + _TOKEN + '"',
])
def test_usage_text_and_refusals_are_not_flagged(tmp_path: Path,
                                                 benign: str) -> None:
    rel = "scripts/benign.py"
    (tmp_path / "scripts").mkdir()
    (tmp_path / rel).write_text(benign + "\n")
    assert find_defaults(tmp_path, [rel]) == []

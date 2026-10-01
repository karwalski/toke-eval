"""Story 128.32: benchmark CLIs fail with a clear message when --tasks-dir is missing.

(1) run_inference_mlx.py must report its required-argument error (and --help)
    without mlx-lm installed, so mlx_lm is imported lazily inside main().
(2) run_benchmark.py's --tasks-dir is required; its old default, tasks/, holds
    only schema.json.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_BENCH / script), *args],
                          capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.parametrize("script", ["run_inference_mlx.py", "run_benchmark.py"])
def test_missing_tasks_dir_is_a_clear_usage_error(script: str) -> None:
    r = _run(script)
    assert r.returncode == 2
    assert "the following arguments are required: --tasks-dir" in r.stderr


@pytest.mark.parametrize("script", ["run_inference_mlx.py", "run_benchmark.py"])
def test_help_works(script: str) -> None:
    r = _run(script, "--help")
    assert r.returncode == 0
    assert "--tasks-dir" in r.stdout


def test_mlx_lm_not_imported_at_module_top() -> None:
    tree = ast.parse((_BENCH / "run_inference_mlx.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlx"), node.module
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("mlx") for a in node.names)


def test_run_benchmark_tasks_dir_has_no_default() -> None:
    src = (_BENCH / "run_benchmark.py").read_text()
    assert 'default=Path("tasks/")' not in src

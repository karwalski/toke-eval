"""Tests for scripts/check_gate1_attribution.py (story 128.22)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_gate1_attribution.py"
_spec = importlib.util.spec_from_file_location("check_gate1_attribution", SCRIPT)
assert _spec is not None and _spec.loader is not None
cga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cga)

HARNESS = '''
def generate_report(results, language, timeout):
    return None

def score_model_pass_at_1(tasks_dir, endpoint, n_samples=1, timeout=10):
    task_results = []
    return generate_report(task_results, "toke-model", timeout)

LANGUAGE_LOADERS = {"python": None, "c": None, "toke": None}
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True)


def _make_repo(tmp_path: Path, *, language: str = "toke",
               solution_names: tuple[str, ...] = ("task-a-0001", "task-a-0002"),
               task_ids: tuple[str, ...] = ("task-a-0001",),
               harness: str = HARNESS) -> Path:
    repo = tmp_path / "repo"
    (repo / "solutions").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "run_benchmark.py").write_text(harness)
    for n in solution_names:
        (repo / "solutions" / f"{n}.toke").write_text("m=x;\n")
    (repo / "solutions" / "inference_stats.json").write_text(json.dumps(
        {"total": len(solution_names), "generated": len(solution_names)}))
    tasks = [{"task_id": t, "pass_count": 1, "total_count": 1, "pass_at_1": 1.0}
             for t in task_ids]
    (repo / "results" / "gate1_x.json").write_text(json.dumps(
        {"language": language, "tasks_evaluated": len(tasks), "tasks": tasks}))
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    return repo


def _failed(result: dict) -> list[str]:
    return [k for k, c in result["checks"].items() if not c["ok"]]


def test_solutions_path_is_attributed(tmp_path: Path) -> None:
    r = cga.check_file(_make_repo(tmp_path), "gate1_x.json")
    assert r["verdict"] == "SOLUTIONS_PATH", r


def test_model_path_language_is_rejected(tmp_path: Path) -> None:
    r = cga.check_file(_make_repo(tmp_path, language="toke-model"), "gate1_x.json")
    assert r["verdict"] == "NOT_ATTRIBUTED"
    assert "A1" in _failed(r)


def test_harness_whose_model_path_writes_toke_is_rejected(tmp_path: Path) -> None:
    harness = HARNESS.replace('"toke-model", timeout', '"toke", timeout')
    r = cga.check_file(_make_repo(tmp_path, harness=harness), "gate1_x.json")
    assert "A2" in _failed(r)


def test_sample_indexed_solutions_are_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, solution_names=("task-a-0001", "task-a-0001-s1"))
    r = cga.check_file(repo, "gate1_x.json")
    assert "A3" in _failed(r)


def test_task_without_solution_file_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, task_ids=("task-a-0001", "task-a-0099"))
    r = cga.check_file(repo, "gate1_x.json")
    assert "A4" in _failed(r)


def test_real_gate1_files_are_attributed() -> None:
    repo = SCRIPT.parent.parent
    probe = subprocess.run(["git", "-C", str(repo), "cat-file", "-e",
                            "a9bde666d9fe4cd8cfd2cfd3f1fd53d50e29dd4d"],
                           capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("introducing commit not in this clone (shallow checkout)")
    assert cga.main(["--repo", str(repo)]) == 0

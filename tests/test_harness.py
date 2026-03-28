"""Tests for the benchmark evaluation harness (run_benchmark.py)."""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# Ensure the repo root is importable.
import sys

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import run_benchmark  # noqa: E402
from run_benchmark import (
    BenchmarkReport,
    TaskResult,
    TestCaseResult,
    _compare,
    discover_tasks,
    dry_run,
    generate_report,
    load_python_solutions,
    main,
    report_to_dict,
    run_benchmark as run_benchmark_fn,
    score_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_task_yaml(directory: Path, task_id: str, test_inputs: list[dict]) -> Path:
    """Write a minimal task YAML file and return its path."""
    task = {
        "id": task_id,
        "phase": "A",
        "category": "test",
        "description": f"Test task {task_id}",
        "input_type": "i64",
        "output_type": "i64",
        "test_inputs": test_inputs,
    }
    p = directory / f"{task_id}.yaml"
    p.write_text(yaml.dump(task, default_flow_style=False))
    return p


def _write_solutions_py(directory: Path, code: str) -> Path:
    """Write a solutions.py into *directory* and return the dir."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "solutions.py").write_text(code)
    return directory


# ---------------------------------------------------------------------------
# Test: task discovery
# ---------------------------------------------------------------------------

class TestDiscoverTasks:
    def test_finds_yaml_files(self, tmp_path: Path) -> None:
        _write_task_yaml(tmp_path, "task-a-0001", [{"input": 1, "expected": 1}])
        _write_task_yaml(tmp_path, "task-a-0002", [{"input": 2, "expected": 2}])
        found = discover_tasks(tmp_path)
        assert len(found) == 2
        assert found[0].stem == "task-a-0001"
        assert found[1].stem == "task-a-0002"

    def test_returns_sorted(self, tmp_path: Path) -> None:
        _write_task_yaml(tmp_path, "task-a-0010", [{"input": 1, "expected": 1}])
        _write_task_yaml(tmp_path, "task-a-0002", [{"input": 1, "expected": 1}])
        _write_task_yaml(tmp_path, "task-a-0005", [{"input": 1, "expected": 1}])
        found = discover_tasks(tmp_path)
        stems = [f.stem for f in found]
        assert stems == ["task-a-0002", "task-a-0005", "task-a-0010"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        found = discover_tasks(tmp_path)
        assert found == []

    def test_ignores_non_task_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.yaml").write_text("hello: world")
        (tmp_path / "schema.json").write_text("{}")
        _write_task_yaml(tmp_path, "task-a-0001", [{"input": 1, "expected": 1}])
        found = discover_tasks(tmp_path)
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Test: comparison helper
# ---------------------------------------------------------------------------

class TestCompare:
    def test_int_equal(self) -> None:
        assert _compare(42, 42)

    def test_int_not_equal(self) -> None:
        assert not _compare(42, 43)

    def test_bool_match(self) -> None:
        assert _compare(True, True)
        assert _compare(False, False)

    def test_bool_mismatch(self) -> None:
        assert not _compare(True, False)

    def test_bool_vs_int(self) -> None:
        # True == 1 in Python, but our comparator rejects mixed types
        assert not _compare(True, 1)
        assert not _compare(1, True)

    def test_list_equal(self) -> None:
        assert _compare([1, 2, 3], [1, 2, 3])

    def test_list_not_equal(self) -> None:
        assert not _compare([1, 2], [1, 3])


# ---------------------------------------------------------------------------
# Test: scoring logic
# ---------------------------------------------------------------------------

class TestScoring:
    def test_all_pass(self) -> None:
        fn = lambda x: x * 2  # noqa: E731
        cases = [
            {"input": 1, "expected": 2},
            {"input": 3, "expected": 6},
            {"input": 0, "expected": 0},
        ]
        result = score_task("task-a-test", fn, cases, timeout=5)
        assert result.task_id == "task-a-test"
        assert result.pass_count == 3
        assert result.total_count == 3
        assert result.pass_at_1 == 1.0

    def test_partial_fail(self) -> None:
        fn = lambda x: x + 1  # noqa: E731
        cases = [
            {"input": 1, "expected": 2},   # pass
            {"input": 2, "expected": 99},   # fail
            {"input": 3, "expected": 4},    # pass
        ]
        result = score_task("task-a-test", fn, cases, timeout=5)
        assert result.pass_count == 2
        assert result.total_count == 3
        assert result.pass_at_1 == 0.0  # not all passed

    def test_exception_in_solution(self) -> None:
        def fn(x: Any) -> Any:
            raise ValueError("boom")

        cases = [{"input": 1, "expected": 1}]
        result = score_task("task-a-test", fn, cases, timeout=5)
        assert result.pass_count == 0
        assert result.cases[0].error is not None
        assert "boom" in result.cases[0].error

    def test_all_fail(self) -> None:
        fn = lambda x: -1  # noqa: E731  -- always wrong
        cases = [
            {"input": 1, "expected": 1},
            {"input": 2, "expected": 2},
        ]
        result = score_task("task-a-test", fn, cases, timeout=5)
        assert result.pass_count == 0
        assert result.pass_at_1 == 0.0


# ---------------------------------------------------------------------------
# Test: report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def _make_results(self) -> list[TaskResult]:
        return [
            TaskResult(task_id="task-a-0001", pass_count=10, total_count=10, pass_at_1=1.0),
            TaskResult(task_id="task-a-0002", pass_count=8, total_count=10, pass_at_1=0.0),
            TaskResult(task_id="task-a-0003", pass_count=10, total_count=10, pass_at_1=1.0),
        ]

    def test_aggregate_counts(self) -> None:
        results = self._make_results()
        report = generate_report(results, language="python", timeout=10)
        assert report.tasks_evaluated == 3
        assert report.total_pass_at_1 == 2
        assert report.mean_pass_at_1 == round(2 / 3, 4)

    def test_empty_results(self) -> None:
        report = generate_report([], language="python", timeout=10)
        assert report.tasks_evaluated == 0
        assert report.total_pass_at_1 == 0
        assert report.mean_pass_at_1 == 0.0

    def test_report_to_dict_schema(self) -> None:
        results = self._make_results()
        report = generate_report(results, language="python", timeout=10)
        d = report_to_dict(report)

        assert "total_pass_at_1" in d
        assert "mean_pass_at_1" in d
        assert "tasks_evaluated" in d
        assert "language" in d
        assert "timeout" in d
        assert "tasks" in d
        assert len(d["tasks"]) == 3

        for task in d["tasks"]:
            assert "task_id" in task
            assert "pass_count" in task
            assert "total_count" in task
            assert "pass_at_1" in task

    def test_report_json_serialisable(self) -> None:
        results = self._make_results()
        report = generate_report(results, language="python", timeout=10)
        d = report_to_dict(report)
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["tasks_evaluated"] == 3


# ---------------------------------------------------------------------------
# Test: dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_prints_config(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_task_yaml(tasks_dir, "task-a-0001", [{"input": 1, "expected": 1}])
        _write_task_yaml(tasks_dir, "task-a-0002", [{"input": 2, "expected": 2}])

        sol_dir = _write_solutions_py(tmp_path / "sol", textwrap.dedent("""\
            SOLUTIONS = {}
        """))

        dry_run(tasks_dir, sol_dir, "python", 10)
        captured = capsys.readouterr()
        assert "Dry Run" in captured.out
        assert "task-a-0001" in captured.out
        assert "task-a-0002" in captured.out
        assert "python" in captured.out
        assert "10s" in captured.out
        assert "2" in captured.out  # tasks found: 2

    def test_dry_run_cli_returns_zero(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_task_yaml(tasks_dir, "task-a-0001", [{"input": 1, "expected": 1}])

        sol_dir = _write_solutions_py(tmp_path / "sol", "SOLUTIONS = {}")

        rc = main([
            "--solutions-dir", str(sol_dir),
            "--tasks-dir", str(tasks_dir),
            "--dry-run",
        ])
        assert rc == 0


# ---------------------------------------------------------------------------
# Test: timeout handling
# ---------------------------------------------------------------------------

class TestTimeout:
    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="SIGALRM not available on this platform",
    )
    def test_timeout_triggers(self) -> None:
        def slow_fn(x: Any) -> Any:
            time.sleep(5)
            return x

        cases = [{"input": 1, "expected": 1}]
        result = score_task("task-a-slow", slow_fn, cases, timeout=1)
        assert result.pass_count == 0
        assert result.cases[0].error is not None
        assert "timeout" in result.cases[0].error.lower()

    def test_fast_fn_not_timed_out(self) -> None:
        fn = lambda x: x  # noqa: E731
        cases = [{"input": 42, "expected": 42}]
        result = score_task("task-a-fast", fn, cases, timeout=10)
        assert result.pass_count == 1
        assert result.cases[0].error is None


# ---------------------------------------------------------------------------
# Test: full pipeline (integration)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _setup_mini_benchmark(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a minimal benchmark with two tasks and matching solutions."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        _write_task_yaml(tasks_dir, "task-a-0001", [
            {"input": [1, 2, 3], "expected": 6},
            {"input": [0], "expected": 0},
            {"input": [-1, 1], "expected": 0},
        ])
        _write_task_yaml(tasks_dir, "task-a-0002", [
            {"input": 5, "expected": 25},
            {"input": 0, "expected": 0},
            {"input": -3, "expected": 9},
        ])

        sol_dir = _write_solutions_py(tmp_path / "sol", textwrap.dedent("""\
            SOLUTIONS = {
                "task-a-0001": lambda x: sum(x),
                "task-a-0002": lambda x: x * x,
            }
        """))
        return tasks_dir, sol_dir

    def test_run_benchmark_all_pass(self, tmp_path: Path) -> None:
        tasks_dir, sol_dir = self._setup_mini_benchmark(tmp_path)
        report = run_benchmark_fn(sol_dir, tasks_dir, "python", timeout=10)

        assert report.tasks_evaluated == 2
        assert report.total_pass_at_1 == 2
        assert report.mean_pass_at_1 == 1.0

    def test_run_benchmark_partial(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_task_yaml(tasks_dir, "task-a-0001", [
            {"input": 1, "expected": 2},
            {"input": 2, "expected": 4},
        ])

        sol_dir = _write_solutions_py(tmp_path / "sol", textwrap.dedent("""\
            SOLUTIONS = {
                "task-a-0001": lambda x: x * 2,
            }
        """))

        report = run_benchmark_fn(sol_dir, tasks_dir, "python", timeout=10)
        assert report.tasks_evaluated == 1
        assert report.total_pass_at_1 == 1

    def test_run_benchmark_skips_missing_solutions(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_task_yaml(tasks_dir, "task-a-0001", [{"input": 1, "expected": 1}])
        _write_task_yaml(tasks_dir, "task-a-0099", [{"input": 1, "expected": 1}])

        sol_dir = _write_solutions_py(tmp_path / "sol", textwrap.dedent("""\
            SOLUTIONS = {
                "task-a-0001": lambda x: x,
            }
        """))

        report = run_benchmark_fn(sol_dir, tasks_dir, "python", timeout=10)
        assert report.tasks_evaluated == 1  # only task-a-0001

    def test_cli_output_json(self, tmp_path: Path) -> None:
        tasks_dir, sol_dir = self._setup_mini_benchmark(tmp_path)
        output_path = tmp_path / "report.json"

        rc = main([
            "--solutions-dir", str(sol_dir),
            "--tasks-dir", str(tasks_dir),
            "--language", "python",
            "--output", str(output_path),
        ])
        assert rc == 0
        assert output_path.exists()

        report = json.loads(output_path.read_text())
        assert report["tasks_evaluated"] == 2
        assert report["total_pass_at_1"] == 2

    def test_cli_bad_tasks_dir(self, tmp_path: Path) -> None:
        sol_dir = _write_solutions_py(tmp_path / "sol", "SOLUTIONS = {}")
        rc = main([
            "--solutions-dir", str(sol_dir),
            "--tasks-dir", str(tmp_path / "nonexistent"),
        ])
        assert rc == 1

    def test_cli_bad_solutions_dir(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_task_yaml(tasks_dir, "task-a-0001", [{"input": 1, "expected": 1}])

        rc = main([
            "--solutions-dir", str(tmp_path / "nonexistent"),
            "--tasks-dir", str(tasks_dir),
        ])
        assert rc == 1


# ---------------------------------------------------------------------------
# Test: solution loading
# ---------------------------------------------------------------------------

class TestLoadPythonSolutions:
    def test_loads_solutions_dict(self, tmp_path: Path) -> None:
        sol_dir = _write_solutions_py(tmp_path, textwrap.dedent("""\
            SOLUTIONS = {"task-a-0001": lambda x: x}
        """))
        solutions = load_python_solutions(sol_dir)
        assert "task-a-0001" in solutions

    def test_missing_solutions_py(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_python_solutions(tmp_path)

    def test_missing_solutions_dict(self, tmp_path: Path) -> None:
        _write_solutions_py(tmp_path, "X = 1\n")
        # Overwrite to remove SOLUTIONS
        (tmp_path / "solutions.py").write_text("X = 1\n")
        with pytest.raises(ImportError):
            load_python_solutions(tmp_path)

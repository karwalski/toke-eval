#!/usr/bin/env python3
"""Benchmark evaluation harness for the toke project.

Runs solutions against task YAML files, scores pass/fail per test case,
and generates a JSON report with per-task and aggregate metrics.

Usage:
    python run_benchmark.py --solutions-dir baselines/python --tasks-dir hidden_tests/ \\
        [--output report.json] [--language python] [--timeout 10] [--dry-run]

Exit codes:
    0  success
    1  error (missing dirs, no tasks, import failure, etc.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is required.  Install with:  pip install pyyaml")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TestCaseResult:
    """Result of a single test case execution."""
    index: int
    passed: bool
    input: Any = None
    expected: Any = None
    actual: Any = None
    error: str | None = None


@dataclass
class TaskResult:
    """Result of evaluating a single task."""
    task_id: str
    pass_count: int
    total_count: int
    pass_at_1: float
    cases: list[TestCaseResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    total_pass_at_1: int
    mean_pass_at_1: float
    tasks_evaluated: int
    language: str
    timeout: int
    tasks: list[TaskResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparison helper (matches run_baselines.py semantics)
# ---------------------------------------------------------------------------

def _compare(actual: object, expected: object) -> bool:
    """Deep equality that normalises booleans stored as strings in YAML."""
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        return False
    if isinstance(actual, bool) and not isinstance(expected, bool):
        return False
    return actual == expected


# ---------------------------------------------------------------------------
# Timeout support
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    """Raised when a solution exceeds its time budget."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Solution exceeded time limit")


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_tasks(tasks_dir: Path) -> list[Path]:
    """Return sorted list of task YAML files under *tasks_dir*."""
    yamls = sorted(tasks_dir.glob("task-*.yaml"))
    return yamls


# ---------------------------------------------------------------------------
# Solution loading
# ---------------------------------------------------------------------------

def load_python_solutions(solutions_dir: Path) -> dict[str, Any]:
    """Import solutions.py from *solutions_dir* and return its SOLUTIONS dict."""
    sol_path = solutions_dir / "solutions.py"
    if not sol_path.exists():
        raise FileNotFoundError(f"No solutions.py found in {solutions_dir}")

    spec = importlib.util.spec_from_file_location("solutions", str(sol_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {sol_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    solutions: dict[str, Any] = getattr(mod, "SOLUTIONS", None)  # type: ignore[assignment]
    if solutions is None:
        raise ImportError(f"{sol_path} does not export a SOLUTIONS dict")
    return solutions


def load_c_solutions(solutions_dir: Path) -> dict[str, Any]:
    """Stub for C solutions -- not yet implemented."""
    raise NotImplementedError("C solution runner not yet implemented")


def load_toke_solutions(solutions_dir: Path) -> dict[str, Any]:
    """Stub for toke solutions -- not yet implemented."""
    raise NotImplementedError("toke solution runner not yet implemented")


LANGUAGE_LOADERS = {
    "python": load_python_solutions,
    "c": load_c_solutions,
    "toke": load_toke_solutions,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_task(
    task_id: str,
    fn: Any,
    test_cases: list[dict[str, Any]],
    timeout: int,
) -> TaskResult:
    """Run *fn* against every test case and return a TaskResult."""
    case_results: list[TestCaseResult] = []
    pass_count = 0

    for i, tc in enumerate(test_cases):
        inp = tc["input"]
        expected = tc["expected"]
        result = TestCaseResult(index=i, passed=False, input=inp, expected=expected)

        # Set alarm-based timeout (Unix only; on Windows we skip alarm)
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            actual = fn(inp)
            result.actual = actual
            if _compare(actual, expected):
                result.passed = True
                pass_count += 1
        except TimeoutError:
            result.error = f"timeout ({timeout}s)"
        except Exception as exc:
            result.error = repr(exc)
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        case_results.append(result)

    total = len(test_cases)
    pass_at_1 = 1.0 if pass_count == total else 0.0

    return TaskResult(
        task_id=task_id,
        pass_count=pass_count,
        total_count=total,
        pass_at_1=pass_at_1,
        cases=case_results,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    task_results: list[TaskResult],
    language: str,
    timeout: int,
) -> BenchmarkReport:
    """Build an aggregate BenchmarkReport from individual task results."""
    total_pass_at_1 = sum(1 for t in task_results if t.pass_at_1 == 1.0)
    n = len(task_results)
    mean = total_pass_at_1 / n if n else 0.0

    return BenchmarkReport(
        total_pass_at_1=total_pass_at_1,
        mean_pass_at_1=round(mean, 4),
        tasks_evaluated=n,
        language=language,
        timeout=timeout,
        tasks=task_results,
    )


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """Serialise report to a JSON-friendly dict (drop per-case detail for brevity)."""
    tasks_out = []
    for t in report.tasks:
        tasks_out.append({
            "task_id": t.task_id,
            "pass_count": t.pass_count,
            "total_count": t.total_count,
            "pass_at_1": t.pass_at_1,
        })
    return {
        "total_pass_at_1": report.total_pass_at_1,
        "mean_pass_at_1": report.mean_pass_at_1,
        "tasks_evaluated": report.tasks_evaluated,
        "language": report.language,
        "timeout": report.timeout,
        "tasks": tasks_out,
    }


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(
    tasks_dir: Path,
    solutions_dir: Path,
    language: str,
    timeout: int,
) -> None:
    """Print configuration and discovered tasks, then exit."""
    task_files = discover_tasks(tasks_dir)
    print("=== Dry Run ===")
    print(f"  tasks_dir:     {tasks_dir}")
    print(f"  solutions_dir: {solutions_dir}")
    print(f"  language:      {language}")
    print(f"  timeout:       {timeout}s")
    print(f"  tasks found:   {len(task_files)}")
    for tf in task_files:
        print(f"    {tf.stem}")
    print("=== End Dry Run ===")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_benchmark(
    solutions_dir: Path,
    tasks_dir: Path,
    language: str,
    timeout: int,
) -> BenchmarkReport:
    """Execute the full benchmark and return a report."""
    # Load solutions
    loader = LANGUAGE_LOADERS.get(language)
    if loader is None:
        raise ValueError(f"Unsupported language: {language!r}")
    solutions = loader(solutions_dir)

    # Discover tasks
    task_files = discover_tasks(tasks_dir)
    if not task_files:
        raise FileNotFoundError(f"No task YAML files found in {tasks_dir}")

    task_results: list[TaskResult] = []

    for tf in task_files:
        with open(tf) as f:
            task = yaml.safe_load(f)

        task_id: str = task["id"]

        if task_id not in solutions:
            continue

        fn = solutions[task_id]
        test_cases = task["test_inputs"]
        result = score_task(task_id, fn, test_cases, timeout)
        task_results.append(result)

    return generate_report(task_results, language, timeout)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on error."""
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation harness for the toke project.",
    )
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        required=True,
        help="Directory containing solutions (e.g. baselines/python)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("tasks/"),
        help="Directory containing task YAML files (default: tasks/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to this path (default: stdout)",
    )
    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGE_LOADERS.keys()),
        default="python",
        help="Language baseline to evaluate (default: python)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Per-task execution timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks and config without executing",
    )

    args = parser.parse_args(argv)

    # Validate directories
    if not args.tasks_dir.is_dir():
        print(f"ERROR: tasks directory not found: {args.tasks_dir}", file=sys.stderr)
        return 1
    if not args.solutions_dir.is_dir():
        print(f"ERROR: solutions directory not found: {args.solutions_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        dry_run(args.tasks_dir, args.solutions_dir, args.language, args.timeout)
        return 0

    try:
        report = run_benchmark(
            solutions_dir=args.solutions_dir,
            tasks_dir=args.tasks_dir,
            language=args.language,
            timeout=args.timeout,
        )
    except (FileNotFoundError, ImportError, NotImplementedError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report_dict = report_to_dict(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report_dict, indent=2))

    # Print summary
    print(
        f"\n{'=' * 60}\n"
        f"  Language:       {report.language}\n"
        f"  Tasks evaluated:{report.tasks_evaluated:>4}\n"
        f"  Pass@1:         {report.total_pass_at_1}/{report.tasks_evaluated}\n"
        f"  Mean pass@1:    {report.mean_pass_at_1:.4f}\n"
        f"{'=' * 60}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

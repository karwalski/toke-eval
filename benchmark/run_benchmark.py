#!/usr/bin/env python3
"""Benchmark evaluation harness for the toke project.

Runs solutions against task YAML files, scores pass/fail per test case,
and generates a JSON report with per-task and aggregate metrics.

Usage (baseline evaluation):
    python run_benchmark.py --solutions-dir baselines/python --tasks-dir hidden_tests/ \\
        [--output report.json] [--language python] [--timeout 10] [--dry-run]

    python run_benchmark.py --solutions-dir baselines/c --tasks-dir hidden_tests/ \\
        --language c

    python run_benchmark.py --solutions-dir path/to/toke-files/ --tasks-dir hidden_tests/ \\
        --language toke

Usage (model inference / pass@1):
    python run_benchmark.py --tasks-dir hidden_tests/ \\
        --model-endpoint http://localhost:8000/generate \\
        [--n-samples 5] [--api-key KEY]

Exit codes:
    0  success
    1  error (missing dirs, no tasks, import failure, etc.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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


# ---------------------------------------------------------------------------
# Subprocess-based solution runner helper
# ---------------------------------------------------------------------------

# Default timeout for subprocess calls (seconds); overridden by --timeout.
_SUBPROCESS_TIMEOUT = 10


def _make_subprocess_runner(binary: str | Path, task_id: str | None = None) -> Any:
    """Return a callable(input) that runs a binary with JSON I/O.

    If *task_id* is given the binary is invoked as:
        <binary> <task-id> <json-input>        (C-style multi-task binary)
    Otherwise:
        <binary> <json-input>                  (toke-style single-task binary)

    The binary must print JSON on stdout.
    """
    binary = str(binary)

    def runner(inp: Any) -> Any:
        json_input = json.dumps(inp, separators=(",", ":"))
        cmd = [binary, task_id, json_input] if task_id else [binary, json_input]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Binary exited with code {proc.returncode}: "
                f"{proc.stderr.strip()[:200]}"
            )
        raw = proc.stdout.strip()
        if not raw:
            raise RuntimeError("Binary produced no output")
        return json.loads(raw)

    return runner


# ---------------------------------------------------------------------------
# C solutions
# ---------------------------------------------------------------------------

def load_c_solutions(solutions_dir: Path) -> dict[str, Any]:
    """Build the C solutions binary and return a dict of task-id -> callable.

    The C baseline uses a single multi-task binary:
        ./solutions <task-id> <json-input>
    We build it (if needed) and return one callable per supported task.
    """
    makefile = solutions_dir / "Makefile"
    binary = solutions_dir / "solutions"
    source = solutions_dir / "solutions.c"

    if not source.exists():
        raise FileNotFoundError(f"No solutions.c found in {solutions_dir}")

    # Build if binary is missing or stale.
    if not binary.exists() or (
        source.exists() and source.stat().st_mtime > binary.stat().st_mtime
    ):
        if makefile.exists():
            subprocess.run(
                ["make", "-C", str(solutions_dir), "-s"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            # Fallback: compile directly.
            subprocess.run(
                ["cc", "-std=c11", "-O2", "-Wall", "-o", str(binary), str(source), "-lm"],
                check=True,
                capture_output=True,
                text=True,
            )

    if not binary.exists():
        raise FileNotFoundError(f"Failed to build C binary at {binary}")

    # Discover which task-ids the binary supports by scanning the source
    # for the dispatch table entries.  The C source uses string comparisons
    # like:  strcmp(task_id, "task-a-0001") == 0
    import re
    task_ids: list[str] = []
    c_text = source.read_text()
    for m in re.finditer(r'"(task-[abc]-\d{4})"', c_text):
        tid = m.group(1)
        if tid not in task_ids:
            task_ids.append(tid)

    solutions: dict[str, Any] = {}
    for tid in sorted(task_ids):
        solutions[tid] = _make_subprocess_runner(binary, task_id=tid)

    return solutions


# ---------------------------------------------------------------------------
# Toke solutions
# ---------------------------------------------------------------------------

# Search order for the tkc compiler.
_TKC_SEARCH_PATHS = [
    os.environ.get("TKC", ""),
    os.path.expanduser("~/tk/toke/tkc"),
    os.path.expanduser("~/tk/toke/bin/tkc"),
]


def _find_tkc() -> str:
    """Locate the tkc compiler binary."""
    for candidate in _TKC_SEARCH_PATHS:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # Last resort: check PATH.
    which = shutil.which("tkc")
    if which:
        return which

    raise FileNotFoundError(
        "Cannot find tkc compiler.  Set TKC env var or ensure it is on PATH."
    )


def _compile_toke(
    source_path: Path,
    output_path: Path,
    tkc: str,
) -> tuple[bool, str]:
    """Compile a .toke source file to a native binary.

    Returns (success, error_message).
    """
    cmd = [tkc, "--out", str(output_path), str(source_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        diag = proc.stderr.strip() or proc.stdout.strip()
        return False, diag[:500]
    if not output_path.exists():
        return False, "Compiler produced no output binary"
    # Make executable.
    output_path.chmod(output_path.stat().st_mode | 0o111)
    return True, ""


def load_toke_solutions(solutions_dir: Path) -> dict[str, Any]:
    """Compile toke source files and return a dict of task-id -> callable.

    Expected layout in *solutions_dir*:
        task-a-0001.toke
        task-a-0002.toke
        ...

    Each file is compiled with tkc to produce a binary.  The binary is
    invoked as:
        ./<binary> <json-input>
    and must print JSON on stdout.

    A build directory (.build/) is created inside solutions_dir to hold
    compiled binaries.
    """
    toke_files = sorted(solutions_dir.glob("task-*.toke"))
    if not toke_files:
        raise FileNotFoundError(
            f"No .toke solution files found in {solutions_dir}"
        )

    tkc = _find_tkc()
    build_dir = solutions_dir / ".build"
    build_dir.mkdir(exist_ok=True)

    solutions: dict[str, Any] = {}
    compile_errors: list[str] = []

    for src in toke_files:
        task_id = src.stem  # e.g. "task-a-0001"
        binary = build_dir / task_id

        # Recompile if source is newer than binary.
        needs_build = (
            not binary.exists()
            or src.stat().st_mtime > binary.stat().st_mtime
        )

        if needs_build:
            ok, err = _compile_toke(src, binary, tkc)
            if not ok:
                compile_errors.append(f"  {task_id}: {err}")
                continue

        solutions[task_id] = _make_subprocess_runner(binary)

    if compile_errors:
        print(
            f"WARNING: {len(compile_errors)} toke file(s) failed to compile:",
            file=sys.stderr,
        )
        for line in compile_errors:
            print(line, file=sys.stderr)

    if not solutions:
        raise RuntimeError(
            f"All {len(toke_files)} toke files failed to compile.  "
            f"Check tkc at {tkc}"
        )

    return solutions


# ---------------------------------------------------------------------------
# Model inference mode (Pass@1 evaluation)
# ---------------------------------------------------------------------------

def load_model_solutions(
    tasks_dir: Path,
    model_endpoint: str,
    n_samples: int = 1,
    api_key: str | None = None,
) -> dict[str, list[Any]]:
    """Generate toke solutions from a model and return pass@1 callables.

    For each task YAML in *tasks_dir*, send the task description to the model
    endpoint, receive toke source code, compile it, and wrap the binary as a
    callable.

    Returns a dict of task-id -> list[callable] where each callable
    represents one sample (for pass@N evaluation).

    The model endpoint is called via HTTP POST with JSON body:
        {"prompt": "<task description>", "task_id": "<id>",
         "input_type": "<type>", "output_type": "<type>"}
    Expected response:
        {"source": "<toke source code>"}
    """
    import urllib.request

    tkc = _find_tkc()
    tmp_dir = Path(tempfile.mkdtemp(prefix="toke-model-"))
    task_files = sorted(tasks_dir.glob("task-*.yaml"))

    results: dict[str, list[Any]] = {}
    stats = {"generated": 0, "compiled": 0, "failed_gen": 0, "failed_compile": 0}

    for tf in task_files:
        with open(tf) as f:
            task = yaml.safe_load(f)

        task_id: str = task["id"]
        task_dir = tmp_dir / task_id
        task_dir.mkdir(exist_ok=True)

        samples: list[Any] = []

        for sample_idx in range(n_samples):
            # Build prompt payload.
            payload = json.dumps({
                "prompt": task["description"],
                "task_id": task_id,
                "input_type": task.get("input_type", ""),
                "output_type": task.get("output_type", ""),
            }).encode()

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            req = urllib.request.Request(
                model_endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read())
                source = body.get("source", "")
                if not source:
                    stats["failed_gen"] += 1
                    continue
                stats["generated"] += 1
            except Exception as exc:
                print(
                    f"WARNING: model request failed for {task_id} "
                    f"sample {sample_idx}: {exc}",
                    file=sys.stderr,
                )
                stats["failed_gen"] += 1
                continue

            # Write source to temp file and compile.
            src_path = task_dir / f"sample-{sample_idx}.toke"
            src_path.write_text(source)
            bin_path = task_dir / f"sample-{sample_idx}"

            ok, err = _compile_toke(src_path, bin_path, tkc)
            if not ok:
                stats["failed_compile"] += 1
                continue

            stats["compiled"] += 1
            samples.append(_make_subprocess_runner(bin_path))

        if samples:
            results[task_id] = samples

    print(
        f"Model inference stats: "
        f"{stats['generated']} generated, {stats['compiled']} compiled, "
        f"{stats['failed_gen']} gen failures, "
        f"{stats['failed_compile']} compile failures",
        file=sys.stderr,
    )

    return results


def score_model_pass_at_1(
    tasks_dir: Path,
    model_endpoint: str,
    n_samples: int = 1,
    timeout: int = 10,
    api_key: str | None = None,
) -> BenchmarkReport:
    """Run pass@1 evaluation using model-generated toke solutions.

    For each task, generates N toke solutions from the model, compiles each,
    and runs against the hidden test cases.  A task passes if any sample
    passes all test cases.
    """
    global _SUBPROCESS_TIMEOUT
    _SUBPROCESS_TIMEOUT = timeout

    model_solutions = load_model_solutions(
        tasks_dir, model_endpoint, n_samples=n_samples, api_key=api_key,
    )

    task_files = discover_tasks(tasks_dir)
    task_results: list[TaskResult] = []

    for tf in task_files:
        with open(tf) as f:
            task = yaml.safe_load(f)

        task_id: str = task["id"]
        if task_id not in model_solutions:
            continue

        test_cases = task["test_inputs"]
        best_result: TaskResult | None = None

        for fn in model_solutions[task_id]:
            result = score_task(task_id, fn, test_cases, timeout)
            if best_result is None or result.pass_count > best_result.pass_count:
                best_result = result
            if result.pass_at_1 == 1.0:
                break  # All tests passed, no need to try more samples.

        if best_result is not None:
            task_results.append(best_result)

    return generate_report(task_results, "toke-model", timeout)


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
    global _SUBPROCESS_TIMEOUT
    _SUBPROCESS_TIMEOUT = timeout

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
        default=None,
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

    # Model inference mode (pass@1 evaluation)
    model_group = parser.add_argument_group("model inference (pass@1)")
    model_group.add_argument(
        "--model-endpoint",
        type=str,
        default=None,
        help="HTTP endpoint for model inference (enables pass@1 mode)",
    )
    model_group.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples per task for pass@N (default: 1)",
    )
    model_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for model endpoint (or set TOKE_API_KEY env var)",
    )

    args = parser.parse_args(argv)

    # Validate directories
    if not args.tasks_dir.is_dir():
        print(f"ERROR: tasks directory not found: {args.tasks_dir}", file=sys.stderr)
        return 1

    # Model inference mode
    if args.model_endpoint:
        api_key = args.api_key or os.environ.get("TOKE_API_KEY")
        try:
            report = score_model_pass_at_1(
                tasks_dir=args.tasks_dir,
                model_endpoint=args.model_endpoint,
                n_samples=args.n_samples,
                timeout=args.timeout,
                api_key=api_key,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        # Standard baseline mode: require solutions-dir.
        if args.solutions_dir is None:
            print("ERROR: --solutions-dir is required (unless using --model-endpoint)", file=sys.stderr)
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
        except (FileNotFoundError, ImportError, NotImplementedError,
                RuntimeError, ValueError) as exc:
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

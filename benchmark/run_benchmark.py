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

Usage (model inference / Pass@1):
    python run_benchmark.py --tasks-dir hidden_tests/ \\
        --model-endpoint http://localhost:8000/generate \\
        [--n-samples 5] [--api-key KEY]

Metric definitions (128.1c)
---------------------------
``--n-samples N`` draws N independent samples per task.  Three DIFFERENT
numbers come out of that and they are reported under three different names:

    pass_at_1    mean over tasks of (correct samples / N).  The probability a
                 single sample solves the task.  With N = 1 this is literally
                 one sample, one attempt.
    best_of_n    mean over tasks of (1 if ANY sample is correct else 0).  This
                 is an oracle metric: it assumes a perfect selector that knows
                 which sample passes the hidden tests.  It is >= pass_at_1 and
                 rises with N.
    pass_at_k    the unbiased Chen et al. (2021) estimator,
                 1 - C(N-c, k)/C(N, k).

Before 128.1c this harness computed best_of_n (keep the best sample, `break`
on the first perfect one) and wrote it into the field named `pass_at_1`.
Every figure produced with `--n-samples > 1` was therefore inflated, and the
inflation grows with N.  Such figures must be re-derived, never carried
forward.

Exit codes:
    0  success
    1  error (missing dirs, no tasks, import failure, etc.)
    2  benchmark schema violation -- NO score is emitted
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    """Result of evaluating a single task.

    For the deterministic baseline path there is exactly one solution per
    task, so ``n_samples == 1`` and ``pass_at_1`` is 1.0 or 0.0 exactly.
    For the model path ``n_samples`` may exceed 1; see the module docstring
    for how the three metrics differ.
    """
    task_id: str
    pass_count: int
    total_count: int
    #: correct samples / n_samples -- the Pass@1 contribution of this task.
    pass_at_1: float
    cases: list[TestCaseResult] = field(default_factory=list)
    n_samples: int = 1
    #: samples that passed every test case.  Derived from pass_at_1 when the
    #: caller does not supply it (single-sample path).
    n_correct: int | None = None
    #: 1.0 if ANY sample was correct.  Oracle metric -- never a Pass@1.
    best_of_n: float | None = None
    #: Unbiased Chen et al. Pass@k, keyed by k.  Empty for n_samples == 1.
    pass_at_k: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_correct is None:
            self.n_correct = int(round(self.pass_at_1 * self.n_samples))
        if self.best_of_n is None:
            self.best_of_n = 1.0 if self.n_correct > 0 else 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report.

    ``mean_pass_at_1`` is always Pass@1 -- one sample, one attempt.
    ``mean_best_of_n`` is the oracle best-of-N figure and is ``None`` unless
    more than one sample was drawn.  They are never the same field.
    """
    total_pass_at_1: int | None
    mean_pass_at_1: float
    tasks_evaluated: int
    language: str
    timeout: int
    tasks: list[TaskResult] = field(default_factory=list)
    n_samples: int = 1
    mean_best_of_n: float | None = None
    mean_pass_at_k: dict[int, float] = field(default_factory=dict)
    metric_note: str = ""


class BenchmarkSchemaError(Exception):
    """A task file does not match ``benchmark/tasks/schema.json``.

    Raised instead of degrading to a 0/0 (or, for an empty ``test_inputs``
    list, a free 1.0) score.
    """


#: Canonical test-case key, per ``benchmark/tasks/schema.json``.
TEST_CASES_KEY = "test_inputs"

#: Keys earlier harness revisions looked for; flagged explicitly so a
#: mismatch can never read as "zero test cases".
LEGACY_TEST_CASES_KEYS = ("test_cases", "tests", "cases", "examples")


def load_task(path: Path) -> dict[str, Any]:
    """Load and schema-validate a task YAML file.

    Raises:
        BenchmarkSchemaError: on anything that would otherwise be scored as
            0/0 or as a vacuous pass.
    """
    try:
        with open(path) as f:
            task = yaml.safe_load(f)
    except OSError as exc:
        raise BenchmarkSchemaError(f"{path}: cannot read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise BenchmarkSchemaError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(task, dict):
        raise BenchmarkSchemaError(
            f"{path}: expected a mapping at the top level, "
            f"got {type(task).__name__}"
        )
    if not task.get("id"):
        raise BenchmarkSchemaError(f"{path}: missing required key 'id'")

    if TEST_CASES_KEY not in task:
        legacy = [k for k in LEGACY_TEST_CASES_KEYS if k in task]
        hint = (
            f" (found legacy key(s) {legacy!r}; the schema key is "
            f"{TEST_CASES_KEY!r})"
            if legacy else
            f" (keys present: {sorted(task)!r})"
        )
        raise BenchmarkSchemaError(
            f"{path}: missing required key {TEST_CASES_KEY!r}{hint}"
        )

    cases = task[TEST_CASES_KEY]
    if not isinstance(cases, list):
        raise BenchmarkSchemaError(
            f"{path}: {TEST_CASES_KEY!r} must be a list, "
            f"got {type(cases).__name__}"
        )
    if not cases:
        raise BenchmarkSchemaError(
            f"{path}: {TEST_CASES_KEY!r} is empty; a task with no test cases "
            f"cannot be scored (0/0 is not a pass)"
        )
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkSchemaError(
                f"{path}: {TEST_CASES_KEY}[{i}] must be a mapping, "
                f"got {type(case).__name__}"
            )
        missing = [k for k in ("input", "expected") if k not in case]
        if missing:
            raise BenchmarkSchemaError(
                f"{path}: {TEST_CASES_KEY}[{i}] is missing {missing!r}"
            )

    return task


def validate_tasks_dir(tasks_dir: Path) -> list[Path]:
    """Validate every task file up front; raise before any score is computed."""
    task_files = discover_tasks(tasks_dir)
    if not task_files:
        raise FileNotFoundError(f"No task YAML files found in {tasks_dir}")

    errors: list[str] = []
    for tf in task_files:
        try:
            load_task(tf)
        except BenchmarkSchemaError as exc:
            errors.append(str(exc))

    if errors:
        shown = "\n  ".join(errors[:20])
        more = (f"\n  ... and {len(errors) - 20} more"
                if len(errors) > 20 else "")
        raise BenchmarkSchemaError(
            f"{len(errors)} of {len(task_files)} task file(s) violate the "
            f"benchmark schema; refusing to emit a score:\n  {shown}{more}"
        )

    return task_files


def _comb(n: int, k: int) -> int:
    """Binomial coefficient, 0 when k is out of range."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k -- Chen et al. (2021).

    Args:
        n: samples drawn, c: samples correct, k: attempts allowed.
    """
    if k > n:
        raise ValueError(f"pass@{k} is undefined from only {n} sample(s)")
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


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

#: Task ids whose .toke solution failed to compile during the most recent
#: `load_toke_solutions` call.  Populated by that loader and consumed by
#: `run_benchmark` so a compile failure scores 0 rather than vanishing from
#: the denominator.
#:
#: 128.1c: dropping them is how the published Gate-1 figure became
#: 588/923 = 63.7% when 1,000 solutions were generated.  588/1000 = 58.8% is
#: the Pass@1; 63.7% is Pass@1 *given the solution compiled*, a different and
#: strictly more generous quantity.
_COMPILE_FAILURES: list[str] = []


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

# Search order for the toke compiler.
_TKC_SEARCH_PATHS = [
    os.environ.get("TKC", ""),
    os.path.expanduser("~/tk/toke/toke"),
    os.path.expanduser("~/tk/toke/bin/toke"),
]


def _find_tkc() -> str:
    """Locate the toke compiler binary."""
    for candidate in _TKC_SEARCH_PATHS:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # Last resort: check PATH.
    which = shutil.which("toke")
    if which:
        return which

    raise FileNotFoundError(
        "Cannot find toke compiler.  Set TKC env var or ensure it is on PATH."
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

    Each file is compiled with the toke compiler to produce a binary.  The binary is
    invoked as:
        ./<binary> <json-input>
    and must print JSON on stdout.

    A build directory (.build/) is created inside solutions_dir to hold
    compiled binaries.
    """
    global _COMPILE_FAILURES
    _COMPILE_FAILURES = []

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
                # A solution that does not compile is a FAILED task, not an
                # absent one.  Record it so run_benchmark() can score it zero
                # instead of dropping it out of the denominator.
                _COMPILE_FAILURES.append(task_id)
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


def score_model_samples(
    tasks_dir: Path,
    model_endpoint: str,
    n_samples: int = 1,
    timeout: int = 10,
    api_key: str | None = None,
) -> BenchmarkReport:
    """Evaluate model-generated toke solutions and report each metric by name.

    For each task, generate ``n_samples`` solutions, compile each, and run
    EVERY compiled sample against the hidden test cases.  No sample is
    skipped: the old early ``break`` on the first perfect sample both made
    the metric best-of-N and biased the correct-sample count downwards.

    The returned report carries:
      * ``mean_pass_at_1``  -- Pass@1, mean of (correct / n_samples).
      * ``mean_best_of_n``  -- the oracle best-of-N rate (``None`` at N = 1,
        where it is identical to Pass@1 by construction).
      * ``mean_pass_at_k``  -- unbiased Chen et al. Pass@k for k <= N.

    A generation or compile failure counts as an incorrect sample, not as a
    missing one: ``n_samples`` is always the number requested.
    """
    global _SUBPROCESS_TIMEOUT
    _SUBPROCESS_TIMEOUT = timeout

    if n_samples < 1:
        raise ValueError(f"--n-samples must be >= 1, got {n_samples}")

    # Schema gate: fail before a single sample is drawn or scored.
    task_files = validate_tasks_dir(tasks_dir)

    model_solutions = load_model_solutions(
        tasks_dir, model_endpoint, n_samples=n_samples, api_key=api_key,
    )

    k_values = sorted({k for k in (1, 5, 10) if k <= n_samples})
    task_results: list[TaskResult] = []

    for tf in task_files:
        task = load_task(tf)
        task_id: str = task["id"]
        test_cases = task[TEST_CASES_KEY]

        # Samples that failed to generate or compile are still samples; they
        # are simply incorrect ones.  Dropping them would inflate Pass@1.
        samples = model_solutions.get(task_id, [])

        n_correct = 0
        best_result: TaskResult | None = None
        for fn in samples:
            result = score_task(task_id, fn, test_cases, timeout)
            if result.pass_count == result.total_count:
                n_correct += 1
            if best_result is None or result.pass_count > best_result.pass_count:
                best_result = result

        agg = TaskResult(
            task_id=task_id,
            # Per-case detail from the single best sample, kept for triage
            # only.  It is NOT the basis of any headline number.
            pass_count=best_result.pass_count if best_result else 0,
            total_count=len(test_cases),
            pass_at_1=n_correct / n_samples,
            cases=best_result.cases if best_result else [],
            n_samples=n_samples,
            n_correct=n_correct,
            best_of_n=1.0 if n_correct > 0 else 0.0,
            pass_at_k={
                k: pass_at_k_estimator(n_samples, n_correct, k)
                for k in k_values
            },
        )
        task_results.append(agg)

    return generate_report(
        task_results, "toke-model", timeout, n_samples=n_samples,
    )


def score_model_pass_at_1(
    tasks_dir: Path,
    model_endpoint: str,
    timeout: int = 10,
    api_key: str | None = None,
) -> BenchmarkReport:
    """Strict Pass@1: exactly one sample per task, one attempt.

    Kept as a named entry point so "Pass@1" cannot be produced by a call that
    silently drew more than one sample.  For N > 1 call
    :func:`score_model_samples` and quote the field you actually mean.
    """
    return score_model_samples(
        tasks_dir, model_endpoint, n_samples=1, timeout=timeout,
        api_key=api_key,
    )


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
    """Run *fn* against every test case and return a TaskResult.

    Raises:
        BenchmarkSchemaError: if *test_cases* is empty.  ``pass_count ==
            total == 0`` would otherwise satisfy the all-cases-passed test and
            award a free 1.0.
    """
    if not test_cases:
        raise BenchmarkSchemaError(
            f"{task_id}: no test cases; 0/0 is not a pass"
        )

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
    n_samples: int = 1,
) -> BenchmarkReport:
    """Build an aggregate BenchmarkReport from individual task results.

    ``mean_pass_at_1`` is the mean of the per-task Pass@1 contributions.  At
    ``n_samples == 1`` that is exactly (tasks solved / tasks evaluated); at
    N > 1 it is the single-sample success probability, which is strictly what
    Pass@1 means.  The oracle best-of-N figure goes in ``mean_best_of_n`` and
    nowhere else.
    """
    n = len(task_results)
    mean_p1 = (sum(t.pass_at_1 for t in task_results) / n) if n else 0.0

    if n_samples == 1:
        total_pass_at_1: int | None = sum(
            1 for t in task_results if t.pass_at_1 == 1.0
        )
        mean_best_of_n: float | None = None
        note = "Pass@1: one sample, one attempt."
    else:
        # A count of solved tasks is not defined for an estimator over N
        # samples; forcing one is how best-of-N got reported as Pass@1.
        total_pass_at_1 = None
        mean_best_of_n = round(
            sum(t.best_of_n or 0.0 for t in task_results) / n, 4
        ) if n else 0.0
        note = (
            f"Pass@1 estimated from n={n_samples} samples per task. "
            f"mean_best_of_n is an ORACLE best-of-{n_samples} figure and must "
            f"never be quoted as Pass@1."
        )

    mean_pass_at_k: dict[int, float] = {}
    if n and task_results[0].pass_at_k:
        for k in sorted(task_results[0].pass_at_k):
            mean_pass_at_k[k] = round(
                sum(t.pass_at_k.get(k, 0.0) for t in task_results) / n, 4
            )

    return BenchmarkReport(
        total_pass_at_1=total_pass_at_1,
        mean_pass_at_1=round(mean_p1, 4),
        tasks_evaluated=n,
        language=language,
        timeout=timeout,
        tasks=task_results,
        n_samples=n_samples,
        mean_best_of_n=mean_best_of_n,
        mean_pass_at_k=mean_pass_at_k,
        metric_note=note,
    )


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """Serialise report to a JSON-friendly dict (drop per-case detail for brevity)."""
    tasks_out = []
    for t in report.tasks:
        entry: dict[str, Any] = {
            "task_id": t.task_id,
            "pass_count": t.pass_count,
            "total_count": t.total_count,
            "pass_at_1": t.pass_at_1,
        }
        if t.n_samples > 1:
            entry["n_samples"] = t.n_samples
            entry["n_correct"] = t.n_correct
            entry["best_of_n"] = t.best_of_n
            entry["pass_at_k"] = {str(k): v for k, v in t.pass_at_k.items()}
        tasks_out.append(entry)

    out: dict[str, Any] = {
        "total_pass_at_1": report.total_pass_at_1,
        "mean_pass_at_1": report.mean_pass_at_1,
        "tasks_evaluated": report.tasks_evaluated,
        "language": report.language,
        "timeout": report.timeout,
        "n_samples": report.n_samples,
        "metric_note": report.metric_note,
        "tasks": tasks_out,
    }
    if report.mean_best_of_n is not None:
        out["mean_best_of_n"] = report.mean_best_of_n
    if report.mean_pass_at_k:
        out["mean_pass_at_k"] = {
            str(k): v for k, v in report.mean_pass_at_k.items()
        }
    return out


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
    global _COMPILE_FAILURES
    _COMPILE_FAILURES = []

    loader = LANGUAGE_LOADERS.get(language)
    if loader is None:
        raise ValueError(f"Unsupported language: {language!r}")
    solutions = loader(solutions_dir)
    compile_failures = set(_COMPILE_FAILURES)

    # Discover + schema-gate tasks before scoring anything.
    task_files = validate_tasks_dir(tasks_dir)

    task_results: list[TaskResult] = []

    for tf in task_files:
        task = load_task(tf)
        task_id: str = task["id"]
        test_cases = task[TEST_CASES_KEY]

        if task_id in compile_failures:
            # Scored, and scored zero.  Not dropped.
            task_results.append(TaskResult(
                task_id=task_id,
                pass_count=0,
                total_count=len(test_cases),
                pass_at_1=0.0,
            ))
            continue

        if task_id not in solutions:
            continue

        fn = solutions[task_id]
        result = score_task(task_id, fn, test_cases, timeout)
        task_results.append(result)

    if not task_results:
        # Zero overlap between the solution set and the task set is an input
        # mismatch, not a score of zero.  Reporting "Mean Pass@1: 0.0000 over
        # 0 tasks" is the same disease as the 0/0 schema bug.
        raise BenchmarkSchemaError(
            f"no task in {tasks_dir} has a matching solution in "
            f"{solutions_dir} ({len(task_files)} task(s), "
            f"{len(solutions)} solution(s)); refusing to emit a score over "
            f"zero tasks"
        )

    return generate_report(task_results, language, timeout, n_samples=1)


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
        help=(
            "Samples drawn per task (default: 1). N=1 is strict Pass@1. "
            "N>1 still reports Pass@1 as the single-sample success rate; the "
            "oracle best-of-N figure is reported separately as "
            "mean_best_of_n and is NOT a Pass@1."
        ),
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
        if args.n_samples > 1:
            print(
                f"NOTE: --n-samples {args.n_samples}. The headline Pass@1 is "
                f"the single-sample success rate. The oracle best-of-"
                f"{args.n_samples} figure is reported as 'mean_best_of_n' and "
                f"must not be quoted as Pass@1.",
                file=sys.stderr,
            )
        try:
            report = score_model_samples(
                tasks_dir=args.tasks_dir,
                model_endpoint=args.model_endpoint,
                n_samples=args.n_samples,
                timeout=args.timeout,
                api_key=api_key,
            )
        except BenchmarkSchemaError as exc:
            print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
            print("\nNo score was computed and no report was written.",
                  file=sys.stderr)
            return 2
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
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
        except BenchmarkSchemaError as exc:
            print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
            print("\nNo score was computed and no report was written.",
                  file=sys.stderr)
            return 2
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
    lines = [
        "=" * 60,
        f"  Language:       {report.language}",
        f"  Tasks evaluated:{report.tasks_evaluated:>4}",
        f"  Samples/task:   {report.n_samples}",
    ]
    if report.total_pass_at_1 is not None:
        lines.append(
            f"  Pass@1:         {report.total_pass_at_1}"
            f"/{report.tasks_evaluated}"
        )
    lines.append(f"  Mean Pass@1:    {report.mean_pass_at_1:.4f}")
    if report.mean_best_of_n is not None:
        lines.append(
            f"  Best-of-{report.n_samples} (ORACLE, not Pass@1): "
            f"{report.mean_best_of_n:.4f}"
        )
    for k, v in report.mean_pass_at_k.items():
        lines.append(f"  Pass@{k} (Chen et al.):  {v:.4f}")
    if report.metric_note:
        lines.append(f"  {report.metric_note}")
    lines.append("=" * 60)
    print("\n" + "\n".join(lines), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""EvalPlus-compatible evaluation harness for toke code generation.

Loads tasks from benchmark format, generates N solutions per task
(stubbed model inference with temperature support), compiles each with tkc,
runs test cases, and computes Pass@k using the unbiased estimator from
Chen et al. (2021).

Usage:
    python scripts/evalplus_harness.py \
        --tasks-dir ../benchmark/hidden_tests/ \
        --solutions-dir ../benchmark/solutions/ \
        --compiler ../toke/tkc \
        --n-samples 10 \
        --temperatures 0.0 0.2 0.8 \
        --output results/evalplus_results.json

Exit codes:
    0  success
    1  error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scipy.special import comb

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml required. Install: pip install pyyaml")


# ---------------------------------------------------------------------------
# Pass@k estimator — Chen et al. (2021), "Evaluating Large Language Models
# Trained on Code", Codex paper, unbiased estimator.
#
#   pass@k = 1 - C(n-c, k) / C(n, k)
#
# where n = total solutions, c = correct solutions, k = attempts.
# Uses scipy.special.comb for exact integer computation to avoid
# floating-point overflow with large n.
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator.

    Args:
        n: total number of generated solutions
        c: number of correct (all-tests-pass) solutions
        k: number of attempts (1, 5, 10, etc.)

    Returns:
        Estimated probability that at least one of k random samples is correct.
    """
    if n - c < k:
        return 1.0
    # exact=True uses integer arithmetic — no floating-point overflow
    return 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SolutionResult:
    """Result of compiling and running a single generated solution."""
    solution_index: int
    temperature: float
    compiled: bool = False
    compile_error: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    all_passed: bool = False
    runtime_ms: float = 0.0
    solution_text: str = ""


@dataclass
class TaskEvalResult:
    """EvalPlus-compatible result for a single task."""
    task_id: str
    n_samples: int = 0
    n_correct: int = 0
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    pass_at_10: float = 0.0
    solutions: list[SolutionResult] = field(default_factory=list)


@dataclass
class EvalPlusReport:
    """Top-level EvalPlus-compatible report."""
    timestamp: str = ""
    compiler: str = ""
    n_samples: int = 0
    temperatures: list[float] = field(default_factory=list)
    tasks_total: int = 0
    summary: dict[str, float] = field(default_factory=dict)
    results: list[TaskEvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compilation and execution (reuses logic from toke_eval.pass_at_k)
# ---------------------------------------------------------------------------

def compile_toke(source_path: Path, compiler: str, output: Path,
                 timeout: int = 30) -> tuple[bool, str]:
    """Compile a .toke source file to a native binary via tkc + clang."""
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
        ll_path = tmp.name

    try:
        result = subprocess.run(
            [compiler, str(source_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout

        Path(ll_path).write_text(result.stdout)

        result = subprocess.run(
            ["clang", "-x", "ir", ll_path, "-o", str(output), "-lm"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "compilation timeout"
    finally:
        try:
            os.unlink(ll_path)
        except OSError:
            pass


def run_test_cases(binary: Path, test_file: Path,
                   timeout: int = 10) -> tuple[int, int]:
    """Run a compiled binary against YAML test cases.

    Returns (passed, total).
    """
    with open(test_file) as f:
        task = yaml.safe_load(f)

    # benchmark format uses 'test_inputs' at top level
    cases = task.get("test_inputs", [])
    if not cases:
        return 0, 0

    passed = 0
    total = len(cases)

    for case in cases:
        input_val = case["input"]
        if not isinstance(input_val, str):
            input_val = json.dumps(input_val)
        expected = str(case["expected"]).strip()

        try:
            result = subprocess.run(
                [str(binary), input_val],
                capture_output=True, text=True, timeout=timeout,
            )
            actual = result.stdout.strip()
            if actual == expected:
                passed += 1
        except (subprocess.TimeoutExpired, OSError):
            continue

    return passed, total


# ---------------------------------------------------------------------------
# Solution generation stub
# ---------------------------------------------------------------------------

def generate_solutions(
    task_id: str,
    prompt: str,
    n_samples: int,
    temperature: float,
    model_endpoint: str | None = None,
) -> list[str]:
    """Generate N candidate solutions for a task.

    Currently a stub: returns an empty list. When a model endpoint is
    provided, this will call the inference API with the given temperature.

    To use pre-generated solutions instead, pass --solutions-dir and the
    harness will load .toke files from disk.

    Args:
        task_id: benchmark task identifier (e.g. "task-a-0001")
        prompt: the task prompt / description
        n_samples: number of solutions to generate
        temperature: sampling temperature (0.0 = greedy)
        model_endpoint: URL of the inference API (None = stub mode)

    Returns:
        List of solution source strings.
    """
    if model_endpoint is not None:
        # TODO: implement actual model inference
        #   POST to model_endpoint with:
        #     {"prompt": prompt, "temperature": temperature,
        #      "n": n_samples, "max_tokens": 512}
        #   Parse response and return list of generated code strings.
        print(f"  [STUB] model inference not yet implemented for {task_id}, "
              f"T={temperature}", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    task_id: str,
    test_file: Path,
    solutions: list[Path | str],
    compiler: str,
    temperatures: list[float],
    compile_timeout: int = 30,
    run_timeout: int = 10,
) -> TaskEvalResult:
    """Evaluate all solutions for a single task.

    Args:
        task_id: task identifier
        test_file: path to YAML test file
        solutions: list of solution file paths or source strings
        compiler: path to tkc binary
        temperatures: list of temperatures used (for metadata)
        compile_timeout: max seconds for compilation
        run_timeout: max seconds per test execution

    Returns:
        TaskEvalResult with per-solution results and pass@k scores.
    """
    task_result = TaskEvalResult(task_id=task_id)
    n = len(solutions)
    task_result.n_samples = n
    correct = 0

    for i, sol in enumerate(solutions):
        sr = SolutionResult(
            solution_index=i,
            temperature=temperatures[0] if temperatures else 0.0,
        )

        # Write solution to temp file if it's a string
        if isinstance(sol, str):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toke", delete=False
            ) as tmp:
                tmp.write(sol)
                sol_path = Path(tmp.name)
            sr.solution_text = sol
            owns_source = True
        else:
            sol_path = Path(sol)
            sr.solution_text = sol_path.read_text()
            owns_source = False

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            binary_path = Path(tmp.name)

        try:
            compiled, err = compile_toke(sol_path, compiler, binary_path,
                                          timeout=compile_timeout)
            sr.compiled = compiled
            sr.compile_error = err

            if compiled:
                t0 = time.monotonic()
                passed, total = run_test_cases(binary_path, test_file,
                                               timeout=run_timeout)
                sr.runtime_ms = (time.monotonic() - t0) * 1000
                sr.tests_passed = passed
                sr.tests_total = total
                sr.all_passed = (passed == total and total > 0)
                if sr.all_passed:
                    correct += 1
        finally:
            try:
                os.unlink(binary_path)
            except OSError:
                pass
            if owns_source:
                try:
                    os.unlink(sol_path)
                except OSError:
                    pass

        task_result.solutions.append(sr)

    task_result.n_correct = correct

    # Compute pass@k for k in {1, 5, 10}
    if n > 0:
        task_result.pass_at_1 = pass_at_k(n, correct, 1)
        task_result.pass_at_5 = pass_at_k(n, correct, min(5, n))
        task_result.pass_at_10 = pass_at_k(n, correct, min(10, n))

    return task_result


def run_evaluation(
    tasks_dir: Path,
    solutions_dir: Path | None,
    compiler: str,
    n_samples: int,
    temperatures: list[float],
    model_endpoint: str | None = None,
    compile_timeout: int = 30,
    run_timeout: int = 10,
) -> EvalPlusReport:
    """Run full EvalPlus-compatible evaluation.

    In file mode (--solutions-dir): loads pre-generated .toke files.
    In inference mode (--model-endpoint): generates solutions via API stub.
    """
    report = EvalPlusReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        compiler=compiler,
        n_samples=n_samples,
        temperatures=temperatures,
    )

    # Discover tasks from test YAML files
    task_files = sorted(tasks_dir.glob("task-*.yaml"))
    report.tasks_total = len(task_files)

    if report.tasks_total == 0:
        print("WARNING: no task files found", file=sys.stderr)
        return report

    start = time.time()

    for idx, test_file in enumerate(task_files):
        task_id = test_file.stem

        # Collect solutions for this task
        solutions: list[Path | str] = []

        if solutions_dir is not None:
            # File mode: look for matching .toke files
            # Support both single file and numbered variants
            single = solutions_dir / f"{task_id}.toke"
            if single.exists():
                solutions.append(single)

            # Check for numbered variants: task-a-0001_0.toke, etc.
            for i in range(n_samples):
                numbered = solutions_dir / f"{task_id}_{i}.toke"
                if numbered.exists():
                    solutions.append(numbered)

        if model_endpoint is not None and len(solutions) < n_samples:
            # Load task description for prompt
            with open(test_file) as f:
                task_data = yaml.safe_load(f)
            prompt = task_data.get("description", "")

            for temp in temperatures:
                generated = generate_solutions(
                    task_id, prompt, n_samples - len(solutions),
                    temp, model_endpoint,
                )
                solutions.extend(generated)

        if not solutions:
            # No solutions available — record zero scores
            task_result = TaskEvalResult(task_id=task_id, n_samples=0)
            report.results.append(task_result)
            continue

        task_result = evaluate_task(
            task_id=task_id,
            test_file=test_file,
            solutions=solutions,
            compiler=compiler,
            temperatures=temperatures,
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
        )
        report.results.append(task_result)

        done = idx + 1
        if done % 50 == 0 or done == report.tasks_total:
            elapsed = time.time() - start
            rate = done / elapsed * 60 if elapsed > 0 else 0
            print(f"  [{done}/{report.tasks_total}] "
                  f"{rate:.0f} tasks/min", file=sys.stderr)

    # Compute summary statistics
    evaluated = [r for r in report.results if r.n_samples > 0]
    if evaluated:
        report.summary = {
            "tasks_evaluated": len(evaluated),
            "mean_pass_at_1": sum(r.pass_at_1 for r in evaluated) / len(evaluated),
            "mean_pass_at_5": sum(r.pass_at_5 for r in evaluated) / len(evaluated),
            "mean_pass_at_10": sum(r.pass_at_10 for r in evaluated) / len(evaluated),
            "tasks_with_any_correct": sum(
                1 for r in evaluated if r.n_correct > 0
            ),
            "total_solutions": sum(r.n_samples for r in evaluated),
            "total_correct": sum(r.n_correct for r in evaluated),
            "duration_s": round(time.time() - start, 2),
        }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EvalPlus-compatible evaluation harness for toke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks-dir", type=Path, required=True,
        help="Directory containing task YAML files (benchmark format)",
    )
    parser.add_argument(
        "--solutions-dir", type=Path, default=None,
        help="Directory with pre-generated .toke solution files",
    )
    parser.add_argument(
        "--compiler", default="tkc",
        help="Path to tkc compiler binary (default: tkc)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of solutions per task (default: 10)",
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.0],
        help="Sampling temperatures (default: 0.0 greedy)",
    )
    parser.add_argument(
        "--model-endpoint", type=str, default=None,
        help="Model inference API endpoint URL (stub — not yet implemented)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--compile-timeout", type=int, default=30,
        help="Compilation timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--run-timeout", type=int, default=10,
        help="Per-test execution timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    if not args.tasks_dir.is_dir():
        sys.exit(f"ERROR: tasks dir not found: {args.tasks_dir}")
    if args.solutions_dir and not args.solutions_dir.is_dir():
        sys.exit(f"ERROR: solutions dir not found: {args.solutions_dir}")
    if args.solutions_dir is None and args.model_endpoint is None:
        sys.exit("ERROR: provide --solutions-dir or --model-endpoint")

    report = run_evaluation(
        tasks_dir=args.tasks_dir,
        solutions_dir=args.solutions_dir,
        compiler=args.compiler,
        n_samples=args.n_samples,
        temperatures=args.temperatures,
        model_endpoint=args.model_endpoint,
        compile_timeout=args.compile_timeout,
        run_timeout=args.run_timeout,
    )

    # Print summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  EvalPlus Harness Results", file=sys.stderr)
    print(f"  Tasks:         {report.tasks_total}", file=sys.stderr)
    s = report.summary
    if s:
        print(f"  Evaluated:     {s.get('tasks_evaluated', 0)}", file=sys.stderr)
        print(f"  Mean Pass@1:   {s.get('mean_pass_at_1', 0):.4f}", file=sys.stderr)
        print(f"  Mean Pass@5:   {s.get('mean_pass_at_5', 0):.4f}", file=sys.stderr)
        print(f"  Mean Pass@10:  {s.get('mean_pass_at_10', 0):.4f}", file=sys.stderr)
        print(f"  Correct/Total: {s.get('total_correct', 0)}"
              f"/{s.get('total_solutions', 0)}", file=sys.stderr)
        print(f"  Duration:      {s.get('duration_s', 0):.1f}s", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # Serialize — strip solution_text from output to keep file size manageable
    report_dict = asdict(report)
    for task in report_dict.get("results", []):
        for sol in task.get("solutions", []):
            sol.pop("solution_text", None)

    output_json = json.dumps(report_dict, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()

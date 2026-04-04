#!/usr/bin/env python3
"""Pass@k evaluation with temperature sweeps.

Story 10.6.1: Extends the toke-eval harness with Pass@k metrics (k=1,5,10)
and temperature sweep support using the unbiased estimator from Chen et al.
(2021), "Evaluating Large Language Models Trained on Code" (Codex paper).

Input format (JSONL):
    {"task_id": "task-a-0001", "samples": ["M=main;...", ...], "temperature": 0.2}

The estimator:
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples, k = desired k.

Usage:
    # Dry-run with synthetic data:
    python scripts/pass_at_k.py \
        --benchmark-dir ../toke-benchmark/hidden_tests/ \
        --output-dir data/ \
        --dry-run --samples-per-task 10 --seed 42

    # Real evaluation from JSONL predictions:
    python scripts/pass_at_k.py \
        --predictions-dir predictions/ \
        --benchmark-dir ../toke-benchmark/hidden_tests/ \
        --output-dir data/ \
        --k-values 1,5,10

Exit codes:
    0  success
    1  error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pass@k estimator -- Chen et al. (2021), Codex paper, unbiased estimator.
#
#   pass@k = 1 - C(n-c, k) / C(n, k)
#
# Uses scipy.special.comb for numerical stability with large n.
# Falls back to math.comb if scipy is unavailable.
# ---------------------------------------------------------------------------

try:
    from scipy.special import comb as _scipy_comb

    def _comb(n: int, k: int) -> float:
        return float(_scipy_comb(n, k, exact=True))

except ImportError:
    import math

    def _comb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return 0.0
        return float(math.comb(n, k))


try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator.

    Args:
        n: total number of generated samples
        c: number of correct (all-tests-pass) samples
        k: number of attempts (1, 5, 10, etc.)

    Returns:
        Estimated probability that at least one of k random samples is correct.
    """
    if n < k:
        # Not enough samples to compute pass@k; fall back to pass@n
        if c > 0:
            return 1.0 - _comb(n - c, n) / _comb(n, n)
        return 0.0
    if n - c < k:
        return 1.0
    denom = _comb(n, k)
    if denom == 0:
        return 0.0
    return 1.0 - _comb(n - c, k) / denom


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskPassAtKResult:
    """Per-task Pass@k result for a single temperature."""
    task_id: str
    temperature: float
    n_samples: int = 0
    n_correct: int = 0
    pass_at_k: dict[int, float] = field(default_factory=dict)


@dataclass
class AggregateResult:
    """Aggregate Pass@k across all tasks for a single temperature."""
    temperature: float
    n_tasks: int = 0
    k_values: list[int] = field(default_factory=list)
    mean_pass_at_k: dict[int, float] = field(default_factory=dict)
    per_task: list[TaskPassAtKResult] = field(default_factory=list)


@dataclass
class PassAtKReport:
    """Top-level report for Pass@k evaluation with temperature sweeps."""
    timestamp: str = ""
    mode: str = ""  # "predictions" or "dry-run"
    samples_per_task: int = 0
    k_values: list[int] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    seed: int = 0
    n_tasks_total: int = 0
    aggregates: list[AggregateResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark task loading
# ---------------------------------------------------------------------------

def load_benchmark_tasks(benchmark_dir: Path) -> dict[str, Path]:
    """Discover benchmark task YAML files.

    Returns:
        Dict mapping task_id -> YAML path.
    """
    tasks: dict[str, Path] = {}
    for p in sorted(benchmark_dir.glob("task-*.yaml")):
        tasks[p.stem] = p
    return tasks


# ---------------------------------------------------------------------------
# Prediction loading from JSONL
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir: Path) -> dict[tuple[str, float], list[str]]:
    """Load predictions from JSONL files in predictions_dir.

    Expected JSONL format per line:
        {"task_id": "task-a-0001", "samples": ["code1", "code2", ...], "temperature": 0.2}

    Returns:
        Dict mapping (task_id, temperature) -> list of sample code strings.
    """
    preds: dict[tuple[str, float], list[str]] = {}

    for jsonl_path in sorted(predictions_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: {jsonl_path}:{line_num}: bad JSON: {e}",
                          file=sys.stderr)
                    continue

                task_id = obj.get("task_id", "")
                temperature = float(obj.get("temperature", 0.0))
                samples = obj.get("samples", [])

                if not task_id or not samples:
                    continue

                key = (task_id, temperature)
                if key not in preds:
                    preds[key] = []
                preds[key].extend(samples)

    return preds


# ---------------------------------------------------------------------------
# Compilation and execution
# ---------------------------------------------------------------------------

def compile_and_check(source_code: str, compiler: str,
                      timeout: int = 30) -> tuple[bool, str]:
    """Compile toke source via tkc --check (syntax/type check only).

    Returns (passed, error_message).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toke", delete=False
    ) as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [compiler, "--check", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "compilation timeout"
    except FileNotFoundError:
        return False, f"compiler not found: {compiler}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def compile_and_run(source_code: str, compiler: str, test_file: Path,
                    compile_timeout: int = 30,
                    run_timeout: int = 10) -> bool:
    """Compile toke source and run against benchmark test cases.

    Returns True if all test cases pass.
    """
    if yaml is None:
        # Cannot run without pyyaml; caller should use --dry-run
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toke", delete=False
    ) as src_tmp:
        src_tmp.write(source_code)
        src_path = src_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as ll_tmp:
        ll_path = ll_tmp.name

    with tempfile.NamedTemporaryFile(delete=False) as bin_tmp:
        bin_path = bin_tmp.name

    try:
        # Step 1: tkc -> LLVM IR
        result = subprocess.run(
            [compiler, src_path],
            capture_output=True, text=True, timeout=compile_timeout,
        )
        if result.returncode != 0:
            return False

        Path(ll_path).write_text(result.stdout)

        # Step 2: clang IR -> native binary
        result = subprocess.run(
            ["clang", "-x", "ir", ll_path, "-o", bin_path, "-lm"],
            capture_output=True, text=True, timeout=compile_timeout,
        )
        if result.returncode != 0:
            return False

        # Step 3: Run test cases
        with open(test_file) as f:
            task = yaml.safe_load(f)

        cases = task.get("test_inputs", [])
        if not cases:
            return False

        for case in cases:
            input_val = case["input"]
            if not isinstance(input_val, str):
                input_val = json.dumps(input_val)
            expected = str(case["expected"]).strip()

            try:
                result = subprocess.run(
                    [bin_path, input_val],
                    capture_output=True, text=True, timeout=run_timeout,
                )
                actual = result.stdout.strip()
                if actual != expected:
                    return False
            except (subprocess.TimeoutExpired, OSError):
                return False

        return True

    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        return False
    finally:
        for p in (src_path, ll_path, bin_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Dry-run: synthetic data generation
# ---------------------------------------------------------------------------

def generate_dry_run_predictions(
    task_ids: list[str],
    temperatures: list[float],
    samples_per_task: int,
    seed: int,
) -> dict[tuple[str, float], list[bool]]:
    """Generate synthetic pass/fail outcomes for dry-run mode.

    Uses deterministic random generation based on seed.  Higher temperatures
    yield slightly higher variance in pass rates; lower temperatures
    cluster around a per-task base rate.

    Returns:
        Dict mapping (task_id, temperature) -> list of bool (pass/fail per sample).
    """
    import hashlib

    results: dict[tuple[str, float], list[bool]] = {}

    for task_id in task_ids:
        # Deterministic base pass rate per task from hash
        h = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
        base_rate = (int(h[:8], 16) % 1000) / 1000.0
        # Scale to a realistic range: 0.05 to 0.85
        base_rate = 0.05 + base_rate * 0.80

        for temp in temperatures:
            # Higher temperature -> more variance in individual outcomes
            # but roughly same expected rate.  We use a simple deterministic
            # scheme: hash (seed, task_id, temp, sample_idx) to get each
            # outcome.
            outcomes: list[bool] = []
            for i in range(samples_per_task):
                sample_hash = hashlib.sha256(
                    f"{seed}:{task_id}:{temp}:{i}".encode()
                ).hexdigest()
                sample_val = (int(sample_hash[:8], 16) % 10000) / 10000.0
                # Adjust threshold by temperature: T=0 -> stricter,
                # T=0.8 -> base_rate used directly
                threshold = base_rate * (0.6 + 0.5 * temp)
                outcomes.append(sample_val < threshold)
            results[(task_id, temp)] = outcomes

    return results


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_pass_at_k(
    benchmark_tasks: dict[str, Path],
    predictions: dict[tuple[str, float], list[str]] | None,
    dry_run_outcomes: dict[tuple[str, float], list[bool]] | None,
    k_values: list[int],
    temperatures: list[float],
    samples_per_task: int,
    compiler: str = "tkc",
    dry_run: bool = False,
) -> PassAtKReport:
    """Run Pass@k evaluation across tasks and temperatures.

    In dry-run mode, uses pre-computed synthetic outcomes.
    In predictions mode, compiles and runs each sample.
    """
    report = PassAtKReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mode="dry-run" if dry_run else "predictions",
        samples_per_task=samples_per_task,
        k_values=sorted(k_values),
        temperatures=sorted(temperatures),
        n_tasks_total=len(benchmark_tasks),
    )

    task_ids = sorted(benchmark_tasks.keys())
    start = time.time()

    for temp in sorted(temperatures):
        agg = AggregateResult(
            temperature=temp,
            n_tasks=0,
            k_values=sorted(k_values),
        )

        for task_id in task_ids:
            test_file = benchmark_tasks[task_id]

            if dry_run and dry_run_outcomes is not None:
                key = (task_id, temp)
                outcomes = dry_run_outcomes.get(key, [])
                n = len(outcomes)
                c = sum(outcomes)
            elif predictions is not None and not dry_run:
                key = (task_id, temp)
                samples = predictions.get(key, [])
                n = len(samples)
                c = 0
                for sample_code in samples:
                    if compile_and_run(sample_code, compiler, test_file):
                        c += 1
            else:
                continue

            if n == 0:
                continue

            task_result = TaskPassAtKResult(
                task_id=task_id,
                temperature=temp,
                n_samples=n,
                n_correct=c,
            )

            for k in sorted(k_values):
                effective_k = min(k, n)
                task_result.pass_at_k[k] = pass_at_k(n, c, effective_k)

            agg.per_task.append(task_result)
            agg.n_tasks += 1

        # Compute aggregate means
        if agg.n_tasks > 0:
            for k in sorted(k_values):
                vals = [t.pass_at_k.get(k, 0.0) for t in agg.per_task]
                agg.mean_pass_at_k[k] = sum(vals) / len(vals)

        report.aggregates.append(agg)

    elapsed = time.time() - start
    print(f"Evaluation completed in {elapsed:.1f}s", file=sys.stderr)

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_summary_table(report: PassAtKReport) -> str:
    """Format a summary table to stdout."""
    lines = [
        "=" * 72,
        "  Pass@k Evaluation Results",
        f"  Mode: {report.mode}",
        f"  Tasks: {report.n_tasks_total}",
        f"  Samples per task: {report.samples_per_task}",
        f"  k values: {report.k_values}",
        "=" * 72,
        "",
    ]

    # Header row
    k_headers = [f"Pass@{k}" for k in report.k_values]
    header = f"{'Temperature':>12}  {'Tasks':>6}  " + "  ".join(
        f"{h:>10}" for h in k_headers
    )
    lines.append(header)
    lines.append("-" * len(header))

    for agg in report.aggregates:
        k_vals = "  ".join(
            f"{agg.mean_pass_at_k.get(k, 0.0):>10.4f}"
            for k in report.k_values
        )
        lines.append(f"{agg.temperature:>12.1f}  {agg.n_tasks:>6}  {k_vals}")

    lines.append("-" * len(header))
    lines.append("")

    # Best temperature per k
    if report.aggregates:
        lines.append("Best temperature per metric:")
        for k in report.k_values:
            best_agg = max(
                report.aggregates,
                key=lambda a: a.mean_pass_at_k.get(k, 0.0),
            )
            lines.append(
                f"  Pass@{k}: T={best_agg.temperature:.1f} "
                f"({best_agg.mean_pass_at_k.get(k, 0.0):.4f})"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


def write_json_report(report: PassAtKReport, output_path: Path) -> None:
    """Write the full report to JSON."""
    # Convert dataclass tree to dict, handling dict[int, float] keys
    def _to_serializable(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            d = {}
            for f_name in obj.__dataclass_fields__:
                val = getattr(obj, f_name)
                d[f_name] = _to_serializable(val)
            return d
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_serializable(report), indent=2),
        encoding="utf-8",
    )
    print(f"JSON report: {output_path}", file=sys.stderr)


def write_csv_summary(report: PassAtKReport, output_path: Path) -> None:
    """Write a CSV summary with one row per (temperature, metric)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["temperature", "n_tasks"] + [
            f"pass_at_{k}" for k in report.k_values
        ]
        writer.writerow(header)

        for agg in report.aggregates:
            row = [f"{agg.temperature:.1f}", str(agg.n_tasks)]
            for k in report.k_values:
                row.append(f"{agg.mean_pass_at_k.get(k, 0.0):.6f}")
            writer.writerow(row)

    print(f"CSV summary: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers: '1,5,10' -> [1, 5, 10]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    """Parse comma-separated floats: '0.0,0.2,0.8' -> [0.0, 0.2, 0.8]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pass@k evaluation with temperature sweeps for toke",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=None,
        help="Directory containing JSONL prediction files",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Directory containing benchmark task YAML files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Output directory for results (default: data/)",
    )
    parser.add_argument(
        "--k-values", type=str, default="1,5,10",
        help="Comma-separated k values (default: 1,5,10)",
    )
    parser.add_argument(
        "--temperatures", type=str, default="0.0,0.2,0.4,0.6,0.8",
        help="Comma-separated temperatures (default: 0.0,0.2,0.4,0.6,0.8)",
    )
    parser.add_argument(
        "--samples-per-task", type=int, default=20,
        help="Number of samples per task (default: 20)",
    )
    parser.add_argument(
        "--compiler", default="tkc",
        help="Path to tkc compiler binary (default: tkc)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate with synthetic pass rates (no compilation)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    k_values = parse_int_list(args.k_values)
    temperatures = parse_float_list(args.temperatures)

    if not k_values:
        print("ERROR: --k-values must specify at least one value",
              file=sys.stderr)
        return 1

    if not temperatures:
        print("ERROR: --temperatures must specify at least one value",
              file=sys.stderr)
        return 1

    # --- Load benchmark tasks ---
    if args.benchmark_dir and args.benchmark_dir.is_dir():
        benchmark_tasks = load_benchmark_tasks(args.benchmark_dir)
    elif args.dry_run:
        # In dry-run without benchmark dir, generate synthetic task IDs
        benchmark_tasks = {
            f"task-a-{i:04d}": Path(f"task-a-{i:04d}.yaml")
            for i in range(1, 51)
        }
        print(f"Dry-run: using {len(benchmark_tasks)} synthetic task IDs",
              file=sys.stderr)
    else:
        print("ERROR: --benchmark-dir is required (or use --dry-run)",
              file=sys.stderr)
        return 1

    if not benchmark_tasks:
        print("ERROR: no benchmark tasks found", file=sys.stderr)
        return 1

    print(f"Tasks: {len(benchmark_tasks)}", file=sys.stderr)
    print(f"k values: {k_values}", file=sys.stderr)
    print(f"Temperatures: {temperatures}", file=sys.stderr)
    print(f"Samples per task: {args.samples_per_task}", file=sys.stderr)

    # --- Evaluate ---
    predictions = None
    dry_run_outcomes = None

    if args.dry_run:
        print("Mode: dry-run (synthetic pass rates)", file=sys.stderr)
        dry_run_outcomes = generate_dry_run_predictions(
            task_ids=sorted(benchmark_tasks.keys()),
            temperatures=temperatures,
            samples_per_task=args.samples_per_task,
            seed=args.seed,
        )
    elif args.predictions_dir:
        if not args.predictions_dir.is_dir():
            print(f"ERROR: predictions dir not found: {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print("Mode: predictions (JSONL)", file=sys.stderr)
        predictions = load_predictions(args.predictions_dir)
        if not predictions:
            print("ERROR: no predictions loaded from JSONL files",
                  file=sys.stderr)
            return 1
        print(f"Loaded predictions for {len(predictions)} (task, temp) pairs",
              file=sys.stderr)
    else:
        print("ERROR: provide --predictions-dir or --dry-run",
              file=sys.stderr)
        return 1

    report = evaluate_pass_at_k(
        benchmark_tasks=benchmark_tasks,
        predictions=predictions,
        dry_run_outcomes=dry_run_outcomes,
        k_values=k_values,
        temperatures=temperatures,
        samples_per_task=args.samples_per_task,
        compiler=args.compiler,
        dry_run=args.dry_run,
    )
    report.seed = args.seed

    # --- Output ---
    summary = format_summary_table(report)
    print(summary)

    write_json_report(
        report, args.output_dir / "pass_at_k_results.json",
    )
    write_csv_summary(
        report, args.output_dir / "pass_at_k_summary.csv",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

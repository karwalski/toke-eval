#!/usr/bin/env python3
"""Cost and latency benchmarking: Python-mode vs toke-mode generation.

Story 9.4.3: Framework for comparing wall-clock generation time and API cost
between Python and toke generation modes on benchmark tasks.

Measures per-task: generation time, token counts (input + output), estimated
API cost.  Computes break-even analysis showing at what Pass@1 toke becomes
cheaper per correct solution than Python.

Usage:
    python scripts/cost_latency_benchmark.py \\
        --benchmark-dir ../benchmark \\
        --output-dir data \\
        --dry-run --tasks 50 --seed 42

Output:
    - cost_latency_results.json    — per-task metrics
    - cost_latency_summary.json    — aggregate comparison
    - break_even_analysis.json     — break-even curve data
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default pricing: GPT-4o (USD per 1M tokens)
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PRICE = 2.50   # $ per 1M input tokens
DEFAULT_OUTPUT_PRICE = 10.00  # $ per 1M output tokens


# ---------------------------------------------------------------------------
# Task discovery (mirrors token_economy.py pattern)
# ---------------------------------------------------------------------------

def _extract_python_functions(solutions_py: Path) -> dict[str, str]:
    """Extract per-task Python function source from solutions.py."""
    source = solutions_py.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    source_lines = source.splitlines(keepends=True)
    results: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        task_id = None
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "task"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)):
                task_id = dec.args[0].value
                break
        if task_id is None:
            continue

        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else start + 1
        func_lines = source_lines[start:end]
        func_source = textwrap.dedent("".join(func_lines))
        results[task_id] = func_source.strip()

    return results


@dataclass
class TaskInfo:
    """A benchmark task with metadata."""
    task_id: str
    toke_source: str
    python_source: str
    difficulty: str  # easy, medium, hard


def _infer_difficulty(task_id: str, python_source: str) -> str:
    """Infer difficulty tier from task ID numbering or code length.

    Convention: task-a-0001..0033 = easy, 0034..0066 = medium, 0067..0100 = hard.
    Falls back to code-length heuristic if ID does not follow convention.
    """
    try:
        num = int(task_id.split("-")[-1])
        if num <= 33:
            return "easy"
        elif num <= 66:
            return "medium"
        else:
            return "hard"
    except (ValueError, IndexError):
        pass

    # Fallback: line count heuristic
    lines = python_source.count("\n") + 1
    if lines <= 10:
        return "easy"
    elif lines <= 25:
        return "medium"
    return "hard"


def discover_tasks(benchmark_dir: Path) -> list[TaskInfo]:
    """Find tasks with both toke and Python solutions."""
    solutions_dir = benchmark_dir / "solutions"
    baselines_py = benchmark_dir / "baselines" / "python" / "solutions.py"

    if not solutions_dir.is_dir():
        sys.exit(f"ERROR: solutions directory not found: {solutions_dir}")
    if not baselines_py.exists():
        sys.exit(f"ERROR: Python baselines not found: {baselines_py}")

    python_solutions = _extract_python_functions(baselines_py)
    if not python_solutions:
        sys.exit(f"ERROR: no @task functions found in {baselines_py}")

    tasks: list[TaskInfo] = []
    for toke_file in sorted(solutions_dir.glob("*.toke")):
        task_id = toke_file.stem
        toke_source = toke_file.read_text(encoding="utf-8").strip()
        if not toke_source:
            continue

        python_source = python_solutions.get(task_id)
        if python_source is None:
            continue

        difficulty = _infer_difficulty(task_id, python_source)
        tasks.append(TaskInfo(
            task_id=task_id,
            toke_source=toke_source,
            python_source=python_source,
            difficulty=difficulty,
        ))

    return tasks


def stratified_sample(
    tasks: list[TaskInfo],
    n: int,
    rng,
) -> list[TaskInfo]:
    """Stratified sample by difficulty tier, up to n tasks total."""
    if n >= len(tasks):
        return list(tasks)

    by_diff: dict[str, list[TaskInfo]] = {"easy": [], "medium": [], "hard": []}
    for t in tasks:
        by_diff.setdefault(t.difficulty, []).append(t)

    # Proportional allocation per stratum
    selected: list[TaskInfo] = []
    remaining = n
    strata = [k for k in ("easy", "medium", "hard") if by_diff.get(k)]

    for i, key in enumerate(strata):
        pool = by_diff[key]
        if i == len(strata) - 1:
            take = remaining  # last stratum gets remainder
        else:
            take = max(1, round(n * len(pool) / len(tasks)))
        take = min(take, len(pool), remaining)

        indices = list(range(len(pool)))
        rng.shuffle(indices)
        for idx in indices[:take]:
            selected.append(pool[idx])
        remaining -= take
        if remaining <= 0:
            break

    return sorted(selected, key=lambda t: t.task_id)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens_cl100k(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 tokenizer)."""
    try:
        import tiktoken
    except ImportError:
        sys.exit("ERROR: tiktoken is required.\n  pip install tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Synthetic system prompts / few-shot examples (for input token estimation)
# ---------------------------------------------------------------------------

_PYTHON_SYSTEM_PROMPT = (
    "You are a code generation assistant. Write a Python function that "
    "solves the given programming task. Return only the function definition."
)

_TOKE_SYSTEM_PROMPT = (
    "You are a code generation assistant. Write a toke solution that "
    "solves the given programming task. Return only the toke source."
)

_PYTHON_FEW_SHOT = (
    "Example:\n"
    "Task: Return the sum of two numbers.\n"
    "def solve(a, b):\n"
    "    return a + b\n"
)

_TOKE_FEW_SHOT = (
    "Example:\n"
    "Task: Return the sum of two numbers.\n"
    "fn solve(a, b) -> a + b\n"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskCostResult:
    task_id: str
    difficulty: str
    mode: str  # "python" or "toke"
    # Token counts
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # Cost (USD)
    input_cost: float
    output_cost: float
    total_cost: float
    # Timing
    wall_clock_seconds: float
    # Pass (simulated in dry-run)
    passed: bool


@dataclass
class ModeSummary:
    mode: str
    n_tasks: int
    n_passed: int
    pass_rate: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float
    cost_per_task: float
    cost_per_correct: float  # total_cost / n_passed (inf if 0)
    total_time: float
    time_per_task: float
    time_per_correct: float  # total_time / n_passed (inf if 0)
    mean_input_tokens: float
    mean_output_tokens: float


@dataclass
class BreakEvenPoint:
    """A point on the break-even curve."""
    pass_at_1: float
    python_cost_per_correct: float
    toke_cost_per_correct: float
    toke_cheaper: bool


@dataclass
class BenchmarkSummary:
    python: ModeSummary
    toke: ModeSummary
    token_savings_total: int
    token_savings_pct: float
    cost_savings_total: float
    cost_savings_pct: float
    time_savings_total: float
    time_savings_pct: float
    break_even_pass_at_1: float | None  # toke Pass@1 where costs equal
    pricing: dict[str, float]


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def compute_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float,
    output_price: float,
) -> tuple[float, float, float]:
    """Compute API cost in USD.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        input_price: USD per 1M input tokens.
        output_price: USD per 1M output tokens.

    Returns:
        (input_cost, output_cost, total_cost)
    """
    ic = input_tokens * input_price / 1_000_000
    oc = output_tokens * output_price / 1_000_000
    return ic, oc, ic + oc


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    tasks: list[TaskInfo],
    input_price: float,
    output_price: float,
    dry_run: bool,
    seed: int,
) -> tuple[list[TaskCostResult], BenchmarkSummary]:
    """Run the cost/latency benchmark on all tasks.

    In dry-run mode, timing and pass/fail are simulated with deterministic
    synthetic data derived from the seed.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    results: list[TaskCostResult] = []

    for task in tasks:
        for mode in ("python", "toke"):
            if mode == "python":
                system_prompt = _PYTHON_SYSTEM_PROMPT
                few_shot = _PYTHON_FEW_SHOT
                output_source = task.python_source
            else:
                system_prompt = _TOKE_SYSTEM_PROMPT
                few_shot = _TOKE_FEW_SHOT
                output_source = task.toke_source

            # Input = system prompt + few-shot + task description (use task_id as proxy)
            input_text = system_prompt + "\n" + few_shot + "\nTask: " + task.task_id
            input_tokens = count_tokens_cl100k(input_text)
            output_tokens = count_tokens_cl100k(output_source)

            ic, oc, tc = compute_cost(input_tokens, output_tokens,
                                      input_price, output_price)

            if dry_run:
                # Synthetic latency: base + per-output-token delay + noise
                base_latency = 0.5  # seconds
                per_token_latency = 0.015  # seconds per output token
                noise = float(rng.normal(0, 0.1))
                wall_clock = max(0.1, base_latency + output_tokens * per_token_latency + noise)

                # Synthetic pass rate: toke slightly lower than python,
                # varies by difficulty
                diff_pass_rates = {
                    "python": {"easy": 0.95, "medium": 0.80, "hard": 0.60},
                    "toke": {"easy": 0.90, "medium": 0.72, "hard": 0.50},
                }
                p = diff_pass_rates[mode].get(task.difficulty, 0.70)
                passed = bool(rng.random() < p)
            else:
                # Real mode: just measure token counts, no actual API call
                wall_clock = 0.0
                passed = True  # assume pass when not simulating

            results.append(TaskCostResult(
                task_id=task.task_id,
                difficulty=task.difficulty,
                mode=mode,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost=ic,
                output_cost=oc,
                total_cost=tc,
                wall_clock_seconds=round(wall_clock, 4),
                passed=passed,
            ))

    # Split by mode
    py_results = [r for r in results if r.mode == "python"]
    tk_results = [r for r in results if r.mode == "toke"]

    py_summary = _build_mode_summary("python", py_results)
    tk_summary = _build_mode_summary("toke", tk_results)

    # Token and cost savings (python - toke; positive = toke saves)
    token_savings = py_summary.total_tokens - tk_summary.total_tokens
    token_savings_pct = (
        100.0 * token_savings / py_summary.total_tokens
        if py_summary.total_tokens > 0 else 0.0
    )
    cost_savings = py_summary.total_cost - tk_summary.total_cost
    cost_savings_pct = (
        100.0 * cost_savings / py_summary.total_cost
        if py_summary.total_cost > 0 else 0.0
    )
    time_savings = py_summary.total_time - tk_summary.total_time
    time_savings_pct = (
        100.0 * time_savings / py_summary.total_time
        if py_summary.total_time > 0 else 0.0
    )

    # Break-even analysis
    break_even = _compute_break_even(
        py_summary, tk_summary, input_price, output_price,
    )

    summary = BenchmarkSummary(
        python=py_summary,
        toke=tk_summary,
        token_savings_total=token_savings,
        token_savings_pct=round(token_savings_pct, 2),
        cost_savings_total=round(cost_savings, 6),
        cost_savings_pct=round(cost_savings_pct, 2),
        time_savings_total=round(time_savings, 4),
        time_savings_pct=round(time_savings_pct, 2),
        break_even_pass_at_1=break_even,
        pricing={
            "input_price_per_1M": input_price,
            "output_price_per_1M": output_price,
            "model": "GPT-4o (default)",
        },
    )

    return results, summary


def _build_mode_summary(mode: str, results: list[TaskCostResult]) -> ModeSummary:
    """Build aggregate summary for a single mode."""
    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.total_cost for r in results)
    total_time = sum(r.wall_clock_seconds for r in results)

    return ModeSummary(
        mode=mode,
        n_tasks=n,
        n_passed=n_passed,
        pass_rate=n_passed / n if n > 0 else 0.0,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        total_cost=round(total_cost, 6),
        cost_per_task=round(total_cost / n, 6) if n > 0 else 0.0,
        cost_per_correct=round(total_cost / n_passed, 6) if n_passed > 0 else float("inf"),
        total_time=round(total_time, 4),
        time_per_task=round(total_time / n, 4) if n > 0 else 0.0,
        time_per_correct=round(total_time / n_passed, 4) if n_passed > 0 else float("inf"),
        mean_input_tokens=round(total_input / n, 1) if n > 0 else 0.0,
        mean_output_tokens=round(total_output / n, 1) if n > 0 else 0.0,
    )


def _compute_break_even(
    py: ModeSummary,
    tk: ModeSummary,
    input_price: float,
    output_price: float,
) -> float | None:
    """Find the toke Pass@1 at which toke cost-per-correct equals Python's.

    Returns the break-even Pass@1 as a float in [0, 1], or None if no
    crossover exists within [0, 1].
    """
    if py.n_passed == 0 or py.total_cost == 0:
        return None

    py_cost_per_correct = py.total_cost / py.n_passed
    tk_total_cost = tk.total_cost

    # toke cost per correct = tk_total_cost / (pass@1 * n_tasks)
    # Set equal to python cost per correct:
    # tk_total_cost / (p * n) = py_cost_per_correct
    # p = tk_total_cost / (n * py_cost_per_correct)
    if py_cost_per_correct <= 0:
        return None

    p = tk_total_cost / (tk.n_tasks * py_cost_per_correct)

    if 0.0 < p <= 1.0:
        return round(p, 4)
    return None


# ---------------------------------------------------------------------------
# Break-even curve generation
# ---------------------------------------------------------------------------

def generate_break_even_curve(
    py_summary: ModeSummary,
    tk_summary: ModeSummary,
) -> list[BreakEvenPoint]:
    """Generate break-even curve data: cost-per-correct at various Pass@1 values."""
    points: list[BreakEvenPoint] = []

    py_cost_per_correct = (
        py_summary.total_cost / py_summary.n_passed
        if py_summary.n_passed > 0 else float("inf")
    )

    for pct in range(5, 101, 5):
        p = pct / 100.0
        # Hypothetical toke cost per correct at this pass rate
        simulated_passed = max(1, round(p * tk_summary.n_tasks))
        tk_cpc = tk_summary.total_cost / simulated_passed

        # Python cost per correct stays fixed (observed)
        points.append(BreakEvenPoint(
            pass_at_1=p,
            python_cost_per_correct=round(py_cost_per_correct, 6),
            toke_cost_per_correct=round(tk_cpc, 6),
            toke_cheaper=tk_cpc < py_cost_per_correct,
        ))

    return points


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_results_json(results: list[TaskCostResult], path: Path) -> None:
    """Write per-task cost/latency results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = [asdict(r) for r in results]
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def write_summary_json(summary: BenchmarkSummary, path: Path) -> None:
    """Write aggregate comparison summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = asdict(summary)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def write_break_even_json(points: list[BreakEvenPoint], path: Path) -> None:
    """Write break-even curve data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = [asdict(p) for p in points]
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def print_summary_table(summary: BenchmarkSummary) -> None:
    """Print human-readable summary table to stdout."""
    w = 78
    print("=" * w)
    print("COST AND LATENCY BENCHMARK — Python vs Toke")
    print("=" * w)
    print()
    print(f"Pricing: ${summary.pricing['input_price_per_1M']:.2f} / "
          f"${summary.pricing['output_price_per_1M']:.2f} per 1M "
          f"input/output tokens ({summary.pricing['model']})")
    print()

    # Mode comparison table
    header = f"{'Metric':<30} {'Python':>18} {'Toke':>18}"
    print(header)
    print("-" * len(header))

    py = summary.python
    tk = summary.toke

    rows = [
        ("Tasks", f"{py.n_tasks}", f"{tk.n_tasks}"),
        ("Passed", f"{py.n_passed}", f"{tk.n_passed}"),
        ("Pass rate", f"{py.pass_rate:.1%}", f"{tk.pass_rate:.1%}"),
        ("Total input tokens", f"{py.total_input_tokens:,}", f"{tk.total_input_tokens:,}"),
        ("Total output tokens", f"{py.total_output_tokens:,}", f"{tk.total_output_tokens:,}"),
        ("Total tokens", f"{py.total_tokens:,}", f"{tk.total_tokens:,}"),
        ("Total cost (USD)", f"${py.total_cost:.4f}", f"${tk.total_cost:.4f}"),
        ("Cost per task", f"${py.cost_per_task:.6f}", f"${tk.cost_per_task:.6f}"),
        ("Cost per correct", f"${py.cost_per_correct:.6f}", f"${tk.cost_per_correct:.6f}"),
        ("Total time (s)", f"{py.total_time:.2f}", f"{tk.total_time:.2f}"),
        ("Time per task (s)", f"{py.time_per_task:.4f}", f"{tk.time_per_task:.4f}"),
        ("Time per correct (s)", f"{py.time_per_correct:.4f}", f"{tk.time_per_correct:.4f}"),
        ("Mean input tokens", f"{py.mean_input_tokens:.1f}", f"{tk.mean_input_tokens:.1f}"),
        ("Mean output tokens", f"{py.mean_output_tokens:.1f}", f"{tk.mean_output_tokens:.1f}"),
    ]

    for label, pv, tv in rows:
        print(f"{label:<30} {pv:>18} {tv:>18}")

    print()
    print("--- Savings (toke vs Python) ---")
    print(f"  Token savings:  {summary.token_savings_total:,} tokens"
          f" ({summary.token_savings_pct:+.1f}%)")
    print(f"  Cost savings:   ${summary.cost_savings_total:.4f}"
          f" ({summary.cost_savings_pct:+.1f}%)")
    print(f"  Time savings:   {summary.time_savings_total:.2f}s"
          f" ({summary.time_savings_pct:+.1f}%)")

    if summary.break_even_pass_at_1 is not None:
        print(f"\n  Break-even Pass@1: {summary.break_even_pass_at_1:.1%}")
        print("  (toke is cheaper per correct solution above this Pass@1)")
    else:
        print("\n  Break-even Pass@1: N/A (no crossover in [0, 1])")

    print()
    print("=" * w)


# ---------------------------------------------------------------------------
# Dry-run synthetic task generation
# ---------------------------------------------------------------------------

def generate_synthetic_tasks(n: int, seed: int) -> list[TaskInfo]:
    """Generate synthetic tasks for dry-run mode when no benchmark dir exists."""
    import numpy as np
    rng = np.random.default_rng(seed)

    difficulties = ["easy", "medium", "hard"]
    tasks: list[TaskInfo] = []

    for i in range(1, n + 1):
        task_id = f"task-a-{i:04d}"

        # Determine difficulty from ID numbering
        if i <= n // 3:
            diff = "easy"
        elif i <= 2 * n // 3:
            diff = "medium"
        else:
            diff = "hard"

        # Synthetic Python: longer for harder tasks
        line_counts = {"easy": 8, "medium": 18, "hard": 35}
        n_lines = line_counts[diff] + int(rng.integers(-2, 3))
        py_lines = [f"def solve_{i}(x):"]
        py_lines.append(f'    """Solve task {i}."""')
        for j in range(max(1, n_lines - 2)):
            py_lines.append(f"    result_{j} = x + {j}")
        py_lines.append(f"    return result_0")
        python_source = "\n".join(py_lines)

        # Synthetic toke: shorter (simulating toke compression)
        compression = {"easy": 0.55, "medium": 0.50, "hard": 0.45}
        toke_lines_n = max(2, int(n_lines * compression[diff]))
        tk_lines = [f"fn solve_{i}(x):"]
        for j in range(max(1, toke_lines_n - 1)):
            tk_lines.append(f"  r{j} = x + {j}")
        tk_lines.append(f"  -> r0")
        toke_source = "\n".join(tk_lines)

        tasks.append(TaskInfo(
            task_id=task_id,
            toke_source=toke_source,
            python_source=python_source,
            difficulty=diff,
        ))

    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cost and latency benchmark: compare wall-clock generation time "
            "and API cost between Python-mode and toke-mode on benchmark tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Dry-run mode uses synthetic timing data and pass rates.
            Real mode measures token counts from actual benchmark tasks
            (no API calls are made — costs are estimated from token counts).
        """),
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Path to benchmark directory. Optional in --dry-run mode.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Output directory for JSON files (default: data).",
    )
    parser.add_argument(
        "--input-price", type=float, default=DEFAULT_INPUT_PRICE,
        help=f"USD per 1M input tokens (default: {DEFAULT_INPUT_PRICE}).",
    )
    parser.add_argument(
        "--output-price", type=float, default=DEFAULT_OUTPUT_PRICE,
        help=f"USD per 1M output tokens (default: {DEFAULT_OUTPUT_PRICE}).",
    )
    parser.add_argument(
        "--tasks", type=int, default=100,
        help="Number of tasks to benchmark (default: 100). Stratified sample.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate with synthetic timing data and pass rates.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args(argv)

    # Import numpy here so missing-dep error is clear
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required.\n  pip install numpy", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)

    # Discover or generate tasks
    if args.benchmark_dir is not None and args.benchmark_dir.is_dir():
        all_tasks = discover_tasks(args.benchmark_dir)
        if not all_tasks:
            print("ERROR: no paired tasks found in benchmark directory.",
                  file=sys.stderr)
            return 1
        tasks = stratified_sample(all_tasks, args.tasks, rng)
        print(f"Selected {len(tasks)} tasks (stratified) from "
              f"{len(all_tasks)} available.", file=sys.stderr)
    elif args.dry_run:
        tasks = generate_synthetic_tasks(args.tasks, args.seed)
        print(f"Generated {len(tasks)} synthetic tasks for dry-run.",
              file=sys.stderr)
    else:
        print("ERROR: --benchmark-dir is required unless --dry-run is set.",
              file=sys.stderr)
        return 1

    # Difficulty distribution
    diff_counts = {}
    for t in tasks:
        diff_counts[t.difficulty] = diff_counts.get(t.difficulty, 0) + 1
    print(f"Difficulty distribution: {diff_counts}", file=sys.stderr)
    print(f"Mode: {'dry-run (synthetic)' if args.dry_run else 'real (token counts)'}",
          file=sys.stderr)
    print(f"Pricing: ${args.input_price:.2f} / ${args.output_price:.2f} "
          f"per 1M input/output tokens", file=sys.stderr)
    print("", file=sys.stderr)

    # Run benchmark
    results, summary = run_benchmark(
        tasks=tasks,
        input_price=args.input_price,
        output_price=args.output_price,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    # Generate break-even curve
    break_even_points = generate_break_even_curve(summary.python, summary.toke)

    # Write outputs
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_results_json(results, out / "cost_latency_results.json")
    write_summary_json(summary, out / "cost_latency_summary.json")
    write_break_even_json(break_even_points, out / "break_even_analysis.json")

    # Summary to stdout
    print_summary_table(summary)

    print(f"\nOutputs written to: {out}", file=sys.stderr)
    print(f"  cost_latency_results.json    ({len(results)} entries)", file=sys.stderr)
    print(f"  cost_latency_summary.json    (aggregate comparison)", file=sys.stderr)
    print(f"  break_even_analysis.json     ({len(break_even_points)} points)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

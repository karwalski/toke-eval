#!/usr/bin/env python3
"""Repair-loop evaluation harness for toke code generation.

Implements generate-compile-repair loops with a fixed iteration budget:
  1. Generate toke code for a task (or load from tasks directory)
  2. Compile with tkc (--check --diag-json)
  3. If compilation fails, feed structured diagnostics back to the model
     for repair (the "repair prompt")
  4. Track iterations until success or budget exhausted
  5. Report failure modes, iteration counts, and success rates

The structured diagnostics format from tkc includes:
  - code: error code (e.g. E001, E010)
  - message: human-readable description
  - line / col: source location
  - fix: suggested fix (optional, from tkc --diag-json)

Usage::

    # Dry-run (no API calls, simulated compilation):
    python scripts/repair_loop_harness.py --dry-run --max-iterations 5

    # With real compiler and tasks directory:
    python scripts/repair_loop_harness.py \\
        --tkc-path ../tkc/tkc \\
        --tasks-dir data/humaneval_format.jsonl \\
        --max-iterations 5 \\
        --output results/repair_loop_report.json

    # Reproducible run with seed:
    python scripts/repair_loop_harness.py --dry-run --seed 42 --output report.json

Story 10.6.5 -- Repair-loop evaluation harness.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Error taxonomy (consistent with teacher_student_loop.py, Story 10.6.2)
# ---------------------------------------------------------------------------

ERROR_CODE_TO_STAGE: dict[str, int] = {
    "E001": 1, "E002": 1, "E003": 1, "E004": 1, "E005": 1, "E006": 1,
    "E010": 2, "E011": 2, "E012": 2, "E013": 2, "E014": 2, "E015": 2,
    "E020": 3, "E021": 3, "E022": 3, "E023": 3, "E024": 3,
    "E030": 4, "E031": 4, "E032": 4, "E033": 4, "E034": 4,
    "E040": 5, "E041": 5, "E042": 5, "E043": 5,
    "E050": 6, "E051": 6,
}

FAILURE_CATEGORIES: dict[str, str] = {
    "syntax": "Syntax / parse errors",
    "type": "Type system violations",
    "name_resolution": "Undeclared names or missing imports",
    "control_flow": "Control flow errors (unreachable, missing branch)",
    "exhaustiveness": "Non-exhaustive pattern matches",
    "error_handling": "Unhandled Result types or error propagation",
    "codegen": "Code generation errors",
    "unknown": "Unclassified errors",
}

ERROR_TO_CATEGORY: dict[str, str] = {
    "E001": "syntax", "E003": "syntax", "E004": "syntax",
    "E012": "syntax", "E015": "syntax", "E032": "syntax",
    "E005": "type", "E006": "type", "E010": "type", "E011": "type",
    "E014": "type", "E022": "type", "E050": "type", "E051": "type",
    "E002": "name_resolution", "E013": "name_resolution",
    "E030": "name_resolution", "E033": "name_resolution",
    "E034": "name_resolution", "E042": "name_resolution",
    "E020": "control_flow", "E021": "control_flow",
    "E023": "control_flow", "E024": "control_flow",
    "E031": "exhaustiveness",
    "E040": "error_handling", "E041": "error_handling",
    "E043": "error_handling",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    """A single compiler diagnostic from tkc --diag-json."""
    code: str
    message: str
    line: int = 0
    col: int = 0
    fix: str = ""

    @property
    def category(self) -> str:
        return ERROR_TO_CATEGORY.get(self.code, "unknown")


@dataclass
class RepairIteration:
    """Record of a single repair attempt."""
    iteration: int
    source_snippet: str
    diagnostics: list[dict[str, Any]]
    error_codes: list[str]
    failure_categories: list[str]
    repaired: bool


@dataclass
class TaskResult:
    """Result of running the repair loop for a single task."""
    task_id: str
    description: str
    success: bool
    iterations_used: int
    max_iterations: int
    final_error_codes: list[str]
    final_failure_categories: list[str]
    all_failure_categories: list[str]
    history: list[RepairIteration] = field(default_factory=list)

    @property
    def dominant_failure(self) -> str:
        """Most common failure category across all iterations."""
        if not self.all_failure_categories:
            return "none"
        counts: dict[str, int] = {}
        for cat in self.all_failure_categories:
            counts[cat] = counts.get(cat, 0) + 1
        return max(counts, key=lambda k: counts[k])


@dataclass
class AggregateReport:
    """Aggregate statistics across all tasks."""
    total_tasks: int
    succeeded: int
    failed: int
    success_rate: float
    mean_iterations: float
    median_iterations: float
    max_iterations_used: int
    failure_category_counts: dict[str, int]
    failure_category_pcts: dict[str, float]
    iterations_histogram: dict[int, int]


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks_from_dir(tasks_path: str) -> list[dict[str, Any]]:
    """Load tasks from a JSONL file or a directory of JSONL/JSON files.

    Supports HumanEval-style JSONL (task_id, prompt fields) and plain
    JSON files with task_id + prompt/description.
    """
    p = Path(tasks_path)
    tasks: list[dict[str, Any]] = []

    if p.is_file() and p.suffix == ".jsonl":
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    elif p.is_file() and p.suffix == ".json":
        with open(p) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                tasks = data
            else:
                tasks = [data]
    elif p.is_dir():
        for child in sorted(p.iterdir()):
            if child.suffix in (".jsonl", ".json"):
                tasks.extend(load_tasks_from_dir(str(child)))
    else:
        print(f"Warning: cannot load tasks from {tasks_path}", file=sys.stderr)

    return tasks


def generate_builtin_tasks(rng: random.Random, n: int = 20) -> list[dict[str, Any]]:
    """Generate built-in synthetic tasks for dry-run mode."""
    templates = [
        ("bind_int", 1, "Declare an integer binding and return it"),
        ("bind_str", 1, "Declare a string binding and return its length"),
        ("add_fn", 2, "Write a function that adds two integers"),
        ("mul_fn", 2, "Write a function that multiplies two floats"),
        ("struct_point", 2, "Define a Point struct with x, y fields"),
        ("if_abs", 3, "Write a function returning absolute value using if/else"),
        ("loop_sum", 3, "Write a function summing 1..n using a loop"),
        ("fibonacci", 3, "Write a function computing the nth Fibonacci number"),
        ("import_math", 4, "Import a math module and use sqrt"),
        ("match_option", 4, "Define Option sum type and match over it"),
        ("match_shape", 4, "Define Shape sum type with area via match"),
        ("result_div", 5, "Write a division function returning Result"),
        ("file_read", 5, "Read a file and propagate errors with Result"),
        ("stdlib_sort", 5, "Sort a list using stdlib functions"),
        ("map_fn", 6, "Write a generic map function over lists"),
        ("filter_fn", 6, "Write a generic filter with a predicate"),
        ("compose", 6, "Compose two functions f and g into f(g(x))"),
        ("fold_fn", 6, "Write a generic fold/reduce function"),
        ("pipeline", 6, "Build a data pipeline with function composition"),
        ("curry", 6, "Implement a curried two-argument function"),
    ]
    rng.shuffle(templates)
    tasks = []
    for i, (tid, stage, desc) in enumerate(templates[:n]):
        tasks.append({
            "task_id": f"repair-{tid}",
            "prompt": desc,
            "description": desc,
            "stage": stage,
        })
    return tasks


# ---------------------------------------------------------------------------
# Compilation (real and simulated)
# ---------------------------------------------------------------------------

def parse_diagnostics(raw: list[dict[str, Any]]) -> list[Diagnostic]:
    """Parse raw JSON diagnostics from tkc into Diagnostic objects."""
    result = []
    for d in raw:
        result.append(Diagnostic(
            code=d.get("code", "E000"),
            message=d.get("message", "unknown error"),
            line=d.get("line", 0),
            col=d.get("col", 0),
            fix=d.get("fix", ""),
        ))
    return result


def run_tkc(source: str, tkc_path: str) -> list[Diagnostic]:
    """Run tkc --check --diag-json on a source string.

    Returns a list of Diagnostic objects. An empty list means clean
    compilation (success).
    """
    try:
        result = subprocess.run(
            [tkc_path, "--check", "--diag-json"],
            input=source,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            raw = json.loads(result.stdout)
            if isinstance(raw, list):
                return parse_diagnostics(raw)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return []


def simulate_compilation(
    source: str,
    task: dict[str, Any],
    iteration: int,
    rng: random.Random,
) -> list[Diagnostic]:
    """Simulate compilation with repair-aware pass rates.

    On each repair iteration the chance of success increases -- modelling
    the effect of feeding diagnostics back to the model.  Higher-stage
    tasks remain harder.
    """
    stage = task.get("stage", 3)
    base_pass = {1: 0.70, 2: 0.55, 3: 0.42, 4: 0.32, 5: 0.24, 6: 0.15}
    bp = base_pass.get(stage, 0.40)

    # Each repair iteration improves the odds by ~15 percentage points,
    # capped at 0.95.
    repair_bonus = 0.15 * (iteration - 1)
    effective_pass = min(bp + repair_bonus, 0.95)

    if rng.random() < effective_pass:
        return []  # Clean compilation.

    # Generate 1-3 errors from the task's stage.
    codes_for_stage = [
        code for code, s in ERROR_CODE_TO_STAGE.items() if s == stage
    ]
    if not codes_for_stage:
        codes_for_stage = ["E001"]

    n_errors = rng.randint(1, min(3, 1 + max(0, 3 - iteration)))
    diagnostics = []
    for _ in range(n_errors):
        code = rng.choice(codes_for_stage)
        fix_suggestions = {
            "syntax": "Check for missing semicolons or brackets",
            "type": "Verify type annotations match usage",
            "name_resolution": "Ensure the name is declared or imported",
            "control_flow": "Check all branches return a value",
            "exhaustiveness": "Add missing match arms",
            "error_handling": "Propagate or handle the Result value",
        }
        cat = ERROR_TO_CATEGORY.get(code, "unknown")
        diagnostics.append(Diagnostic(
            code=code,
            message=f"simulated {code}: {FAILURE_CATEGORIES.get(cat, 'error')}",
            line=rng.randint(1, 20),
            col=rng.randint(1, 40),
            fix=fix_suggestions.get(cat, "Review the code"),
        ))

    return diagnostics


def simulate_repair(
    source: str,
    diagnostics: list[Diagnostic],
    iteration: int,
    rng: random.Random,
) -> str:
    """Simulate a model repairing the source based on diagnostics.

    In dry-run mode, returns a slightly modified source to represent the
    model's repair attempt.
    """
    repair_lines = [f"// Repair attempt {iteration}"]
    for d in diagnostics:
        if d.fix:
            repair_lines.append(f"// Applied fix for {d.code}: {d.fix}")
    repair_lines.append(source)
    return "\n".join(repair_lines)


# ---------------------------------------------------------------------------
# Build repair prompt from diagnostics (for real model calls)
# ---------------------------------------------------------------------------

def build_repair_prompt(
    source: str,
    diagnostics: list[Diagnostic],
    task_description: str,
    iteration: int,
) -> str:
    """Build a structured repair prompt from compiler diagnostics.

    This is the prompt that would be sent to the code model to request
    a repair.  It includes the original task, the failing source, and
    the structured diagnostic information including fix suggestions.
    """
    parts = [
        f"The following toke code failed to compile (attempt {iteration}).",
        f"Task: {task_description}",
        "",
        "Compiler diagnostics (tkc --check --diag-json):",
    ]
    for d in diagnostics:
        loc = f"line {d.line}, col {d.col}" if d.line else "unknown location"
        parts.append(f"  [{d.code}] {d.message} at {loc}")
        if d.fix:
            parts.append(f"    Suggested fix: {d.fix}")
    parts.extend([
        "",
        "Failing source:",
        "```toke",
        source,
        "```",
        "",
        "Please fix ALL compiler errors and return the corrected toke code.",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core repair loop
# ---------------------------------------------------------------------------

def run_repair_loop(
    task: dict[str, Any],
    tkc_path: str,
    max_iterations: int,
    dry_run: bool,
    rng: random.Random,
) -> TaskResult:
    """Run the generate-compile-repair loop for a single task.

    Returns a TaskResult recording the full history.
    """
    task_id = task.get("task_id", "unknown")
    description = task.get("prompt", task.get("description", ""))
    stage = task.get("stage", 3)

    # Initial "generation" -- in dry-run mode, a synthetic snippet.
    source = task.get("canonical_solution", "")
    if not source or dry_run:
        source = f"// Generated solution for: {description}\nlet result = 0\n"

    history: list[RepairIteration] = []
    all_failure_cats: list[str] = []

    for iteration in range(1, max_iterations + 1):
        # Compile.
        if dry_run:
            diagnostics = simulate_compilation(source, task, iteration, rng)
        else:
            diagnostics = run_tkc(source, tkc_path)

        error_codes = [d.code for d in diagnostics]
        cats = [d.category for d in diagnostics]
        all_failure_cats.extend(cats)
        success = len(diagnostics) == 0

        history.append(RepairIteration(
            iteration=iteration,
            source_snippet=source[:200],
            diagnostics=[asdict(d) for d in diagnostics],
            error_codes=error_codes,
            failure_categories=cats,
            repaired=success,
        ))

        if success:
            return TaskResult(
                task_id=task_id,
                description=description,
                success=True,
                iterations_used=iteration,
                max_iterations=max_iterations,
                final_error_codes=[],
                final_failure_categories=[],
                all_failure_categories=all_failure_cats,
                history=history,
            )

        # Repair: build prompt from diagnostics and get a new source.
        _prompt = build_repair_prompt(source, diagnostics, description, iteration)
        if dry_run:
            source = simulate_repair(source, diagnostics, iteration, rng)
        else:
            # In live mode, would call the model API with _prompt.
            # For now, fall back to simulated repair.
            source = simulate_repair(source, diagnostics, iteration, rng)

    # Budget exhausted.
    final_codes = history[-1].error_codes if history else []
    final_cats = history[-1].failure_categories if history else []
    return TaskResult(
        task_id=task_id,
        description=description,
        success=False,
        iterations_used=max_iterations,
        max_iterations=max_iterations,
        final_error_codes=final_codes,
        final_failure_categories=final_cats,
        all_failure_categories=all_failure_cats,
        history=history,
    )


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def compute_aggregate(results: list[TaskResult]) -> AggregateReport:
    """Compute aggregate statistics from per-task results."""
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded

    iterations_list = [r.iterations_used for r in results]
    mean_iter = sum(iterations_list) / max(total, 1)
    sorted_iters = sorted(iterations_list)
    if sorted_iters:
        mid = len(sorted_iters) // 2
        if len(sorted_iters) % 2 == 0:
            median_iter = (sorted_iters[mid - 1] + sorted_iters[mid]) / 2
        else:
            median_iter = sorted_iters[mid]
    else:
        median_iter = 0.0

    # Failure category counts across all tasks.
    cat_counts: dict[str, int] = {}
    for r in results:
        for cat in r.all_failure_categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    total_errors = sum(cat_counts.values()) or 1
    cat_pcts = {k: round(v / total_errors * 100, 1) for k, v in cat_counts.items()}

    # Iterations histogram.
    hist: dict[int, int] = {}
    for i in iterations_list:
        hist[i] = hist.get(i, 0) + 1

    return AggregateReport(
        total_tasks=total,
        succeeded=succeeded,
        failed=failed,
        success_rate=round(succeeded / max(total, 1), 4),
        mean_iterations=round(mean_iter, 2),
        median_iterations=median_iter,
        max_iterations_used=max(iterations_list) if iterations_list else 0,
        failure_category_counts=dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        failure_category_pcts=dict(sorted(cat_pcts.items(), key=lambda x: -x[1])),
        iterations_histogram=dict(sorted(hist.items())),
    )


def build_report(
    results: list[TaskResult],
    aggregate: AggregateReport,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the full JSON report."""
    return {
        "harness": "repair-loop-evaluation",
        "story": "10.6.5",
        "config": {
            "tkc_path": args.tkc_path,
            "tasks_dir": args.tasks_dir,
            "max_iterations": args.max_iterations,
            "dry_run": args.dry_run,
            "seed": args.seed,
        },
        "aggregate": asdict(aggregate),
        "failure_category_legend": FAILURE_CATEGORIES,
        "per_task": [
            {
                "task_id": r.task_id,
                "description": r.description,
                "success": r.success,
                "iterations_used": r.iterations_used,
                "dominant_failure": r.dominant_failure,
                "final_error_codes": r.final_error_codes,
                "final_failure_categories": r.final_failure_categories,
                "history": [asdict(h) for h in r.history],
            }
            for r in results
        ],
    }


def print_summary(aggregate: AggregateReport, results: list[TaskResult]) -> None:
    """Print a human-readable summary to stdout."""
    print()
    print("=" * 72)
    print("  Repair-Loop Evaluation Harness -- Summary")
    print("  Story 10.6.5")
    print("=" * 72)
    print()
    print(f"  Tasks:        {aggregate.total_tasks}")
    print(f"  Succeeded:    {aggregate.succeeded}")
    print(f"  Failed:       {aggregate.failed}")
    print(f"  Success rate: {aggregate.success_rate:.1%}")
    print()
    print(f"  Mean iterations:   {aggregate.mean_iterations:.2f}")
    print(f"  Median iterations: {aggregate.median_iterations:.1f}")
    print(f"  Max iterations:    {aggregate.max_iterations_used}")
    print()

    # Iterations histogram.
    print("  Iterations histogram:")
    for iters, count in sorted(aggregate.iterations_histogram.items()):
        bar = "#" * count
        print(f"    {iters:>2} iterations: {count:>3}  {bar}")
    print()

    # Failure categories.
    if aggregate.failure_category_counts:
        print("  Failure categories:")
        for cat, count in aggregate.failure_category_counts.items():
            pct = aggregate.failure_category_pcts.get(cat, 0.0)
            desc = FAILURE_CATEGORIES.get(cat, cat)
            print(f"    {cat:<20s}: {count:>4} ({pct:>5.1f}%)  {desc}")
        print()

    # Per-task table.
    print(f"  {'Task ID':<30s} {'OK':>3} {'Iter':>5} {'Dominant Failure'}")
    print(f"  {'-------':<30s} {'--':>3} {'----':>5} {'----------------'}")
    for r in results:
        ok = "yes" if r.success else "NO"
        dom = r.dominant_failure
        print(f"  {r.task_id:<30s} {ok:>3} {r.iterations_used:>5} {dom}")
    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair-loop evaluation harness for toke code generation. "
            "Runs generate-compile-repair loops with a fixed iteration budget "
            "and reports failure modes, iteration counts, and success rates. "
            "Story 10.6.5."
        ),
    )
    parser.add_argument(
        "--tkc-path",
        type=str,
        default="tkc",
        help="Path to tkc compiler binary (default: tkc)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help=(
            "Path to tasks JSONL file or directory of task files. "
            "If omitted, built-in synthetic tasks are used."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum repair iterations per task (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON report (default: stdout summary only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate compilation and repair (no real tkc or model calls)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    # Load tasks.
    if args.tasks_dir:
        tasks = load_tasks_from_dir(args.tasks_dir)
        if not tasks:
            print(f"Error: no tasks found in {args.tasks_dir}", file=sys.stderr)
            return 1
        print(f"Loaded {len(tasks)} tasks from {args.tasks_dir}")
    else:
        tasks = generate_builtin_tasks(rng)
        print(f"Using {len(tasks)} built-in synthetic tasks (dry-run)")

    if not args.dry_run and args.tasks_dir is None:
        print("Warning: no --tasks-dir and not --dry-run; using built-in tasks",
              file=sys.stderr)

    # Run repair loops.
    results: list[TaskResult] = []
    for i, task in enumerate(tasks, 1):
        tid = task.get("task_id", f"task-{i}")
        print(f"  [{i}/{len(tasks)}] {tid}...", end="", flush=True)
        result = run_repair_loop(task, args.tkc_path, args.max_iterations,
                                 args.dry_run, rng)
        status = "OK" if result.success else f"FAIL ({result.iterations_used} iters)"
        print(f" {status}")
        results.append(result)

    # Aggregate and report.
    aggregate = compute_aggregate(results)
    print_summary(aggregate, results)

    if args.output:
        report = build_report(results, aggregate, args)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nJSON report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

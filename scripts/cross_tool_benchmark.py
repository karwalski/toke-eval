#!/usr/bin/env python3
"""Cross-tool benchmark harness for Pass@1 comparison of AI coding tools.

Story 10.12.25: Head-to-head comparison of Claude Code, Codex, and Copilot
on toke coding tasks via MCP.  Computes Pass@1, repair loop metrics,
iteration counts, token usage, and error code distributions per tool.

Supports real predictions (from --predictions-dir) or dry-run mode with
simulated pass rates for methodology validation.

Usage:
    # Dry-run with simulated predictions
    python scripts/cross_tool_benchmark.py \\
        --dry-run --tasks 25 --seed 42

    # Real predictions from JSONL files
    python scripts/cross_tool_benchmark.py \\
        --predictions-dir results/predictions \\
        --benchmark-dir /path/to/toke-benchmark \\
        --output-dir data

Output:
    - data/cross_tool_results.json   -- per-task, per-tool results
    - data/cross_tool_summary.json   -- aggregate comparison
    - data/cross_tool_table.csv      -- LaTeX-ready comparison table
    - data/cross_tool_report.md      -- Human-readable report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """A single iteration within a repair loop."""
    iteration: int
    passed: bool
    error_codes: list[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class Prediction:
    """A tool's prediction for a single task."""
    task_id: str
    tool: str
    iterations: list[IterationRecord]
    final_source: str = ""
    passed: bool = False
    total_tokens: int = 0
    total_time_s: float = 0.0


@dataclass
class TaskResult:
    """Per-task, per-tool result."""
    task_id: str
    tool: str
    passed: bool
    first_attempt_passed: bool
    n_iterations: int
    total_tokens: int
    total_time_s: float
    error_codes: list[str]
    difficulty: str = "unknown"


@dataclass
class ToolSummary:
    """Aggregate metrics for one tool."""
    tool: str
    n_tasks: int
    pass_at_1: float
    first_attempt_pass_rate: float
    repair_loop_pass_rate: float
    mean_iterations: float
    median_iterations: float
    mean_iterations_to_fix: float
    mean_tokens: float
    median_tokens: float
    total_tokens: int
    mean_time_s: float
    error_code_distribution: dict[str, int]


@dataclass
class PairwiseComparison:
    """Statistical comparison between two tools."""
    tool_a: str
    tool_b: str
    mcnemar_statistic: float
    mcnemar_p_value: float
    wilcoxon_statistic: float | None
    wilcoxon_p_value: float | None
    pass_rate_diff: float
    iteration_diff: float


@dataclass
class CIResult:
    """Point estimate with bootstrap confidence interval."""
    estimate: float
    ci_lower: float
    ci_upper: float


# ---------------------------------------------------------------------------
# Task generation for dry-run mode
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
DIFFICULTY_WEIGHTS = [0.4, 0.4, 0.2]  # stratified sampling

TOOL_PROFILES = {
    "claude-code": {
        "first_attempt_rate": {"easy": 0.90, "medium": 0.75, "hard": 0.50},
        "repair_rate": {"easy": 0.98, "medium": 0.90, "hard": 0.70},
        "mean_iterations": {"easy": 1.2, "medium": 2.0, "hard": 3.5},
        "mean_tokens": {"easy": 150, "medium": 300, "hard": 500},
        "error_pool": ["E1003", "E2001", "E2003", "E3011", "E4010", "E4031"],
    },
    "codex": {
        "first_attempt_rate": {"easy": 0.80, "medium": 0.65, "hard": 0.40},
        "repair_rate": {"easy": 0.95, "medium": 0.80, "hard": 0.55},
        "mean_iterations": {"easy": 1.5, "medium": 2.5, "hard": 4.0},
        "mean_tokens": {"easy": 180, "medium": 350, "hard": 600},
        "error_pool": ["E1003", "E2001", "E2003", "E3011", "E4010",
                        "E4031", "E3012", "E1001"],
    },
    "copilot": {
        "first_attempt_rate": {"easy": 0.70, "medium": 0.55, "hard": 0.30},
        "repair_rate": {"easy": 0.90, "medium": 0.70, "hard": 0.45},
        "mean_iterations": {"easy": 1.8, "medium": 3.0, "hard": 4.5},
        "mean_tokens": {"easy": 200, "medium": 400, "hard": 700},
        "error_pool": ["E1003", "E2001", "E2003", "E3011", "E4010",
                        "E4031", "E3012", "E1001", "E1002", "E2004"],
    },
}


def generate_task_ids(n_tasks: int, rng: np.random.Generator) -> list[dict]:
    """Generate stratified task IDs with difficulty labels."""
    tasks = []
    for i in range(n_tasks):
        diff_idx = rng.choice(len(DIFFICULTY_LEVELS), p=DIFFICULTY_WEIGHTS)
        difficulty = DIFFICULTY_LEVELS[diff_idx]
        tasks.append({
            "task_id": f"task-a-{i + 1:04d}",
            "difficulty": difficulty,
        })
    return tasks


def simulate_prediction(
    task: dict,
    tool_name: str,
    rng: np.random.Generator,
    max_iterations: int = 5,
) -> Prediction:
    """Simulate a tool prediction with realistic pass/fail and repair loop."""
    profile = TOOL_PROFILES[tool_name]
    difficulty = task["difficulty"]

    first_rate = profile["first_attempt_rate"][difficulty]
    repair_rate = profile["repair_rate"][difficulty]
    mean_iters = profile["mean_iterations"][difficulty]
    mean_toks = profile["mean_tokens"][difficulty]
    error_pool = profile["error_pool"]

    iterations = []
    passed = False

    # First attempt
    first_passed = rng.random() < first_rate
    iter_tokens = max(10, int(rng.normal(mean_toks, mean_toks * 0.3)))
    if first_passed:
        iterations.append(IterationRecord(
            iteration=1, passed=True, error_codes=[], token_count=iter_tokens,
        ))
        passed = True
    else:
        n_errors = rng.integers(1, 4)
        errors = list(rng.choice(error_pool, size=min(n_errors, len(error_pool)),
                                  replace=False))
        iterations.append(IterationRecord(
            iteration=1, passed=False, error_codes=errors,
            token_count=iter_tokens,
        ))

        # Repair loop
        n_repair = max(1, int(rng.poisson(mean_iters - 1)))
        n_repair = min(n_repair, max_iterations - 1)

        for j in range(n_repair):
            iter_num = j + 2
            # Increasing chance of passing with each iteration
            repair_chance = repair_rate * (1 - 0.5 ** iter_num)
            iter_passed = rng.random() < repair_chance
            repair_tokens = max(10, int(rng.normal(
                mean_toks * 0.6, mean_toks * 0.2)))

            if iter_passed:
                iterations.append(IterationRecord(
                    iteration=iter_num, passed=True, error_codes=[],
                    token_count=repair_tokens,
                ))
                passed = True
                break
            else:
                n_errs = rng.integers(1, 3)
                errs = list(rng.choice(error_pool,
                                        size=min(n_errs, len(error_pool)),
                                        replace=False))
                iterations.append(IterationRecord(
                    iteration=iter_num, passed=False, error_codes=errs,
                    token_count=repair_tokens,
                ))

    total_tokens = sum(it.token_count for it in iterations)
    total_time = len(iterations) * rng.uniform(0.5, 3.0)

    return Prediction(
        task_id=task["task_id"],
        tool=tool_name,
        iterations=iterations,
        passed=passed,
        total_tokens=total_tokens,
        total_time_s=round(total_time, 2),
    )


# ---------------------------------------------------------------------------
# Prediction loading from JSONL
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir: Path, tool_name: str) -> list[Prediction]:
    """Load predictions from a JSONL file for a given tool."""
    jsonl_path = predictions_dir / f"{tool_name}.jsonl"
    if not jsonl_path.exists():
        print(f"WARNING: predictions file not found: {jsonl_path}",
              file=sys.stderr)
        return []

    predictions = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: {jsonl_path}:{line_num}: {e}", file=sys.stderr)
                continue

            iters = []
            for it in obj.get("iterations", []):
                iters.append(IterationRecord(
                    iteration=it.get("iteration", len(iters) + 1),
                    passed=it.get("passed", False),
                    error_codes=it.get("error_codes", []),
                    token_count=it.get("token_count", 0),
                ))

            predictions.append(Prediction(
                task_id=obj["task_id"],
                tool=obj.get("tool", tool_name),
                iterations=iters,
                final_source=obj.get("final_source", ""),
                passed=obj.get("passed", False),
                total_tokens=obj.get("total_tokens",
                                      sum(it.token_count for it in iters)),
                total_time_s=obj.get("total_time_s", 0.0),
            ))

    return predictions


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_task_results(
    predictions: list[Prediction],
    task_difficulties: dict[str, str] | None = None,
) -> list[TaskResult]:
    """Convert predictions to per-task results."""
    results = []
    for pred in predictions:
        all_errors = []
        for it in pred.iterations:
            all_errors.extend(it.error_codes)

        first_passed = (
            pred.iterations[0].passed if pred.iterations else False
        )

        results.append(TaskResult(
            task_id=pred.task_id,
            tool=pred.tool,
            passed=pred.passed,
            first_attempt_passed=first_passed,
            n_iterations=len(pred.iterations),
            total_tokens=pred.total_tokens,
            total_time_s=pred.total_time_s,
            error_codes=all_errors,
            difficulty=(task_difficulties or {}).get(
                pred.task_id, "unknown"),
        ))

    return results


def compute_tool_summary(
    results: list[TaskResult],
    tool_name: str,
) -> ToolSummary:
    """Compute aggregate metrics for a single tool."""
    tool_results = [r for r in results if r.tool == tool_name]
    n = len(tool_results)
    if n == 0:
        return ToolSummary(
            tool=tool_name, n_tasks=0, pass_at_1=0.0,
            first_attempt_pass_rate=0.0, repair_loop_pass_rate=0.0,
            mean_iterations=0.0, median_iterations=0.0,
            mean_iterations_to_fix=0.0, mean_tokens=0.0,
            median_tokens=0.0, total_tokens=0, mean_time_s=0.0,
            error_code_distribution={},
        )

    passed = [r for r in tool_results if r.passed]
    first_passed = [r for r in tool_results if r.first_attempt_passed]
    needed_repair = [r for r in tool_results if not r.first_attempt_passed]
    repaired = [r for r in needed_repair if r.passed]

    iterations = [r.n_iterations for r in tool_results]
    tokens = [r.total_tokens for r in tool_results]
    times = [r.total_time_s for r in tool_results]

    # Iterations to fix (for tasks that eventually passed after failing first)
    iters_to_fix = [r.n_iterations for r in repaired] if repaired else [0]

    # Error code distribution
    all_errors: list[str] = []
    for r in tool_results:
        all_errors.extend(r.error_codes)
    error_dist = dict(Counter(all_errors).most_common())

    return ToolSummary(
        tool=tool_name,
        n_tasks=n,
        pass_at_1=len(passed) / n,
        first_attempt_pass_rate=len(first_passed) / n,
        repair_loop_pass_rate=(
            len(repaired) / len(needed_repair)
            if needed_repair else 1.0
        ),
        mean_iterations=float(np.mean(iterations)),
        median_iterations=float(np.median(iterations)),
        mean_iterations_to_fix=float(np.mean(iters_to_fix)),
        mean_tokens=float(np.mean(tokens)),
        median_tokens=float(np.median(tokens)),
        total_tokens=sum(tokens),
        mean_time_s=float(np.mean(times)),
        error_code_distribution=error_dist,
    )


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def mcnemar_test(
    results_a: list[TaskResult],
    results_b: list[TaskResult],
) -> tuple[float, float]:
    """McNemar's test for pairwise pass/fail comparison.

    Compares whether two tools have significantly different pass rates
    on the same set of tasks.

    Returns (statistic, p_value).
    """
    # Build lookup by task_id
    a_by_task = {r.task_id: r.passed for r in results_a}
    b_by_task = {r.task_id: r.passed for r in results_b}

    common_tasks = set(a_by_task.keys()) & set(b_by_task.keys())
    if not common_tasks:
        return 0.0, 1.0

    # Contingency: b=discordant pairs
    # b = A passes, B fails; c = A fails, B passes
    b_count = 0  # A pass, B fail
    c_count = 0  # A fail, B pass

    for tid in common_tasks:
        a_pass = a_by_task[tid]
        b_pass = b_by_task[tid]
        if a_pass and not b_pass:
            b_count += 1
        elif not a_pass and b_pass:
            c_count += 1

    # McNemar's with continuity correction
    if b_count + c_count == 0:
        return 0.0, 1.0

    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1.0 - sp_stats.chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)


def wilcoxon_iterations(
    results_a: list[TaskResult],
    results_b: list[TaskResult],
) -> tuple[float | None, float | None]:
    """Wilcoxon signed-rank test on iteration counts for common tasks.

    Returns (statistic, p_value) or (None, None) if insufficient data.
    """
    a_by_task = {r.task_id: r.n_iterations for r in results_a}
    b_by_task = {r.task_id: r.n_iterations for r in results_b}

    common_tasks = sorted(set(a_by_task.keys()) & set(b_by_task.keys()))
    if len(common_tasks) < 6:
        return None, None

    x = np.array([a_by_task[t] for t in common_tasks], dtype=float)
    y = np.array([b_by_task[t] for t in common_tasks], dtype=float)

    # Remove ties (equal values)
    diffs = x - y
    non_zero = diffs != 0
    if np.sum(non_zero) < 6:
        return None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = sp_stats.wilcoxon(
            x[non_zero], y[non_zero], alternative="two-sided")

    return float(stat), float(p_val)


def bootstrap_ci(
    data: np.ndarray,
    stat_func,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CIResult:
    """Bootstrap confidence interval (percentile method)."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(data)
    if n < 3:
        est = float(stat_func(data))
        return CIResult(estimate=est, ci_lower=est, ci_upper=est)

    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_stats = np.array([stat_func(data[idx]) for idx in boot_indices])

    est = float(stat_func(data))
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return CIResult(estimate=est, ci_lower=ci_lo, ci_upper=ci_hi)


def compute_pairwise(
    all_results: list[TaskResult],
    tools: list[str],
) -> list[PairwiseComparison]:
    """Compute pairwise statistical comparisons between all tool pairs."""
    comparisons = []
    for i, tool_a in enumerate(tools):
        for tool_b in tools[i + 1:]:
            results_a = [r for r in all_results if r.tool == tool_a]
            results_b = [r for r in all_results if r.tool == tool_b]

            mc_stat, mc_p = mcnemar_test(results_a, results_b)
            wil_stat, wil_p = wilcoxon_iterations(results_a, results_b)

            sum_a = compute_tool_summary(all_results, tool_a)
            sum_b = compute_tool_summary(all_results, tool_b)

            comparisons.append(PairwiseComparison(
                tool_a=tool_a,
                tool_b=tool_b,
                mcnemar_statistic=mc_stat,
                mcnemar_p_value=mc_p,
                wilcoxon_statistic=wil_stat,
                wilcoxon_p_value=wil_p,
                pass_rate_diff=sum_a.pass_at_1 - sum_b.pass_at_1,
                iteration_diff=sum_a.mean_iterations - sum_b.mean_iterations,
            ))

    return comparisons


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_results_json(
    all_results: list[TaskResult],
    output_path: Path,
) -> None:
    """Write per-task, per-tool results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = [asdict(r) for r in all_results]
    output_path.write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_summary_json(
    summaries: list[ToolSummary],
    comparisons: list[PairwiseComparison],
    ci_pass_at_1: dict[str, dict],
    output_path: Path,
) -> None:
    """Write aggregate comparison summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "tool_summaries": [asdict(s) for s in summaries],
        "pairwise_comparisons": [asdict(c) for c in comparisons],
        "bootstrap_ci_pass_at_1": ci_pass_at_1,
    }
    output_path.write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_table_csv(
    summaries: list[ToolSummary],
    ci_pass_at_1: dict[str, dict],
    output_path: Path,
) -> None:
    """Write LaTeX-ready comparison table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tool",
            "n_tasks",
            "pass_at_1",
            "pass_at_1_ci_lower",
            "pass_at_1_ci_upper",
            "first_attempt_rate",
            "repair_loop_rate",
            "mean_iterations",
            "median_iterations",
            "mean_iterations_to_fix",
            "mean_tokens",
            "total_tokens",
            "mean_time_s",
        ])
        for s in summaries:
            ci = ci_pass_at_1.get(s.tool, {})
            writer.writerow([
                s.tool,
                s.n_tasks,
                f"{s.pass_at_1:.4f}",
                f"{ci.get('ci_lower', s.pass_at_1):.4f}",
                f"{ci.get('ci_upper', s.pass_at_1):.4f}",
                f"{s.first_attempt_pass_rate:.4f}",
                f"{s.repair_loop_pass_rate:.4f}",
                f"{s.mean_iterations:.2f}",
                f"{s.median_iterations:.1f}",
                f"{s.mean_iterations_to_fix:.2f}",
                f"{s.mean_tokens:.0f}",
                s.total_tokens,
                f"{s.mean_time_s:.2f}",
            ])


def write_report_md(
    summaries: list[ToolSummary],
    comparisons: list[PairwiseComparison],
    ci_pass_at_1: dict[str, dict],
    all_results: list[TaskResult],
    dry_run: bool,
    output_path: Path,
) -> None:
    """Write human-readable Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Cross-Tool Benchmark Report: Pass@1 on Toke Tasks",
        "",
    ]

    if dry_run:
        lines.extend([
            "> **Note:** This report was generated in dry-run mode with",
            "> simulated predictions. Results are for methodology validation",
            "> only and do not reflect actual tool performance.",
            "",
        ])

    # Summary table
    lines.extend([
        "## Summary",
        "",
        "| Tool | n | Pass@1 | 95% CI | First-attempt | Repair-loop |"
        " Mean iters | Mean tokens |",
        "|------|--:|-------:|-------:|--------------:|------------:|"
        "-----------:|------------:|",
    ])
    for s in summaries:
        ci = ci_pass_at_1.get(s.tool, {})
        ci_str = (f"[{ci.get('ci_lower', 0):.2f}, {ci.get('ci_upper', 0):.2f}]"
                  if ci else "---")
        lines.append(
            f"| {s.tool} | {s.n_tasks} | {s.pass_at_1:.2%} | {ci_str} |"
            f" {s.first_attempt_pass_rate:.2%} | {s.repair_loop_pass_rate:.2%} |"
            f" {s.mean_iterations:.1f} | {s.mean_tokens:.0f} |"
        )

    # Repair loop analysis
    lines.extend([
        "",
        "## Repair Loop Analysis",
        "",
        "| Tool | First-attempt pass | After repair | Improvement |"
        " Mean iters to fix |",
        "|------|-----------------:|-----------:|----------:|"
        "-----------------:|",
    ])
    for s in summaries:
        improvement = s.pass_at_1 - s.first_attempt_pass_rate
        lines.append(
            f"| {s.tool} | {s.first_attempt_pass_rate:.2%} |"
            f" {s.pass_at_1:.2%} | +{improvement:.2%} |"
            f" {s.mean_iterations_to_fix:.1f} |"
        )

    # Error code distribution
    lines.extend([
        "",
        "## Error Code Distribution",
        "",
        "| Tool | Top error codes |",
        "|------|-----------------|",
    ])
    for s in summaries:
        top_5 = sorted(s.error_code_distribution.items(),
                       key=lambda x: -x[1])[:5]
        codes_str = ", ".join(f"{k}: {v}" for k, v in top_5)
        lines.append(f"| {s.tool} | {codes_str} |")

    # Pairwise comparisons
    if comparisons:
        lines.extend([
            "",
            "## Pairwise Statistical Comparisons",
            "",
            "| Comparison | McNemar chi2 | p-value | Wilcoxon W | p-value |"
            " Pass@1 diff |",
            "|------------|------------:|--------:|-----------:|--------:|"
            "------------:|",
        ])
        for c in comparisons:
            wil_str = (f"{c.wilcoxon_statistic:.1f}"
                       if c.wilcoxon_statistic is not None else "---")
            wil_p_str = (f"{c.wilcoxon_p_value:.4f}"
                         if c.wilcoxon_p_value is not None else "---")
            sig = ""
            if c.mcnemar_p_value < 0.05:
                sig = " *"
            if c.mcnemar_p_value < 0.01:
                sig = " **"
            lines.append(
                f"| {c.tool_a} vs {c.tool_b} |"
                f" {c.mcnemar_statistic:.2f}{sig} |"
                f" {c.mcnemar_p_value:.4f} |"
                f" {wil_str} | {wil_p_str} |"
                f" {c.pass_rate_diff:+.2%} |"
            )
        lines.extend([
            "",
            "\\* p < 0.05, \\*\\* p < 0.01 (McNemar's test with continuity correction)",
        ])

    # Per-difficulty breakdown
    lines.extend([
        "",
        "## Per-Difficulty Breakdown",
        "",
        "| Tool | Difficulty | n | Pass@1 | Mean iters |",
        "|------|-----------|--:|-------:|-----------:|",
    ])
    for s in summaries:
        tool_results = [r for r in all_results if r.tool == s.tool]
        for diff in DIFFICULTY_LEVELS:
            diff_results = [r for r in tool_results
                            if r.difficulty == diff]
            if not diff_results:
                continue
            n_d = len(diff_results)
            pass_d = sum(1 for r in diff_results if r.passed) / n_d
            iters_d = np.mean([r.n_iterations for r in diff_results])
            lines.append(
                f"| {s.tool} | {diff} | {n_d} |"
                f" {pass_d:.2%} | {iters_d:.1f} |"
            )

    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_summary_table(summaries: list[ToolSummary]) -> str:
    """Format summary table for stdout."""
    header = (
        f"{'Tool':<14} {'n':>4} {'Pass@1':>8} {'1st-att':>8}"
        f" {'Repair':>8} {'MeanItr':>8} {'MeanTok':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for s in summaries:
        lines.append(
            f"{s.tool:<14} {s.n_tasks:>4} {s.pass_at_1:>8.2%}"
            f" {s.first_attempt_pass_rate:>8.2%}"
            f" {s.repair_loop_pass_rate:>8.2%}"
            f" {s.mean_iterations:>8.1f} {s.mean_tokens:>8.0f}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_benchmark(
    tools: list[str],
    predictions_dir: Path | None = None,
    benchmark_dir: Path | None = None,
    output_dir: Path | None = None,
    n_tasks: int = 25,
    dry_run: bool = False,
    seed: int = 42,
) -> tuple[list[TaskResult], list[ToolSummary], list[PairwiseComparison]]:
    """Run the full cross-tool benchmark.

    Returns (all_results, summaries, comparisons).
    """
    rng = np.random.default_rng(seed)
    all_predictions: list[Prediction] = []
    task_difficulties: dict[str, str] = {}

    if dry_run:
        print(f"DRY-RUN: Simulating {n_tasks} tasks for {len(tools)} tools"
              f" (seed={seed})", file=sys.stderr)
        tasks = generate_task_ids(n_tasks, rng)
        task_difficulties = {t["task_id"]: t["difficulty"] for t in tasks}

        for tool_name in tools:
            if tool_name not in TOOL_PROFILES:
                print(f"WARNING: No profile for tool '{tool_name}', skipping.",
                      file=sys.stderr)
                continue
            for task in tasks:
                pred = simulate_prediction(task, tool_name, rng)
                all_predictions.append(pred)
            print(f"  Simulated {len(tasks)} predictions for {tool_name}",
                  file=sys.stderr)
    else:
        if predictions_dir is None:
            sys.exit("ERROR: --predictions-dir required when not in --dry-run mode.")
        for tool_name in tools:
            preds = load_predictions(predictions_dir, tool_name)
            if preds:
                all_predictions.extend(preds)
                print(f"  Loaded {len(preds)} predictions for {tool_name}",
                      file=sys.stderr)
            else:
                print(f"  WARNING: No predictions for {tool_name}",
                      file=sys.stderr)

    if not all_predictions:
        sys.exit("ERROR: No predictions loaded for any tool.")

    # Compute results
    all_results = compute_task_results(all_predictions, task_difficulties)
    summaries = [compute_tool_summary(all_results, t) for t in tools]
    comparisons = compute_pairwise(all_results, tools)

    # Bootstrap CIs for Pass@1
    ci_pass_at_1: dict[str, dict] = {}
    for tool_name in tools:
        tool_results = [r for r in all_results if r.tool == tool_name]
        if not tool_results:
            continue
        pass_arr = np.array([1.0 if r.passed else 0.0 for r in tool_results])
        ci = bootstrap_ci(pass_arr, np.mean, rng=np.random.default_rng(seed))
        ci_pass_at_1[tool_name] = asdict(ci)

    # Write outputs
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        write_results_json(
            all_results,
            output_dir / "cross_tool_results.json",
        )
        write_summary_json(
            summaries, comparisons, ci_pass_at_1,
            output_dir / "cross_tool_summary.json",
        )
        write_table_csv(
            summaries, ci_pass_at_1,
            output_dir / "cross_tool_table.csv",
        )
        write_report_md(
            summaries, comparisons, ci_pass_at_1, all_results, dry_run,
            output_dir / "cross_tool_report.md",
        )

        print(f"\nOutputs written to: {output_dir}", file=sys.stderr)
        print(f"  cross_tool_results.json  ({len(all_results)} records)",
              file=sys.stderr)
        print(f"  cross_tool_summary.json", file=sys.stderr)
        print(f"  cross_tool_table.csv", file=sys.stderr)
        print(f"  cross_tool_report.md", file=sys.stderr)

    return all_results, summaries, comparisons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-tool benchmark harness: Pass@1 comparison of AI coding "
            "tools on toke tasks.  Supports real predictions from JSONL "
            "files or dry-run mode with simulated pass rates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Dry-run tool profiles:
              Claude Code: 75%% first-attempt, 90%% after repair
              Codex:       65%% first-attempt, 80%% after repair
              Copilot:     55%% first-attempt, 70%% after repair

            Example:
              python scripts/cross_tool_benchmark.py \\
                  --dry-run --tasks 25 --seed 42
        """),
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=None,
        help="Directory containing {tool_name}.jsonl prediction files.",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Path to toke-benchmark repo root (for task metadata).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output files (JSON, CSV, Markdown).",
    )
    parser.add_argument(
        "--tools", type=str, default="claude-code,codex,copilot",
        help="Comma-separated tool names (default: claude-code,codex,copilot).",
    )
    parser.add_argument(
        "--tasks", type=int, default=25,
        help="Number of tasks to benchmark (default: 25).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate tool predictions with built-in pass rate profiles.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args(argv)

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    if not tools:
        print("ERROR: No tools specified.", file=sys.stderr)
        return 1

    # Default output dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("data")

    print("=" * 60, file=sys.stderr)
    print("CROSS-TOOL BENCHMARK: Pass@1 on Toke Tasks", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Tools: {', '.join(tools)}", file=sys.stderr)
    print(f"Tasks: {args.tasks}", file=sys.stderr)
    print(f"Mode:  {'dry-run (simulated)' if args.dry_run else 'real'}",
          file=sys.stderr)
    print(f"Seed:  {args.seed}", file=sys.stderr)
    print("", file=sys.stderr)

    all_results, summaries, comparisons = run_benchmark(
        tools=tools,
        predictions_dir=args.predictions_dir,
        benchmark_dir=args.benchmark_dir,
        output_dir=output_dir,
        n_tasks=args.tasks,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    # Print summary to stdout
    print()
    print(format_summary_table(summaries))

    return 0


if __name__ == "__main__":
    sys.exit(main())

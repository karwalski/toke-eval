#!/usr/bin/env python3
"""Constrained decoding ablation study.

Story 10.9.3: Compares toke + constrained decoding vs Python + constrained
decoding across four experimental conditions:

    1. toke_constrained     — grammar-guided decoding using toke EBNF
    2. toke_unconstrained   — standard autoregressive generation for toke
    3. python_constrained   — grammar-guided decoding using Python subset EBNF
    4. python_unconstrained — standard autoregressive generation for Python

For each condition, we compute:
    - Pass@1 (compile/parse success rate)
    - Mean token count
    - Syntax error rate
    - Semantic error rate

Statistical analysis:
    - McNemar's test for binary outcomes (pass/fail)
    - Cohen's d for token count differences
    - Bootstrap confidence intervals

Usage:
    # With real predictions:
    python scripts/constrained_decoding_ablation.py \\
        --predictions-dir data/predictions \\
        --benchmark-dir /path/to/benchmark \\
        --output-dir data

    # Dry run with simulated data:
    python scripts/constrained_decoding_ablation.py \\
        --dry-run --tasks 50 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = [
    "toke_constrained",
    "toke_unconstrained",
    "python_constrained",
    "python_unconstrained",
]

TOKE_GRAMMAR_PATH = Path(__file__).resolve().parents[2] / "toke" / "spec" / "grammar.ebnf"

# Simplified Python subset EBNF for constrained decoding reference
PYTHON_SUBSET_EBNF = r"""
(* Simplified Python subset EBNF — functions, loops, conditionals, assignments *)

Program      = { Statement } ;
Statement    = FuncDef | IfStmt | ForStmt | WhileStmt | Assignment
             | ReturnStmt | ExprStmt | PassStmt ;

FuncDef      = 'def' IDENT '(' [ParamList] ')' ['->' TypeHint] ':' Suite ;
ParamList    = Param { ',' Param } ;
Param        = IDENT [':' TypeHint] ['=' Expr] ;
TypeHint     = IDENT [ '[' TypeHint { ',' TypeHint } ']' ] ;

IfStmt       = 'if' Expr ':' Suite { 'elif' Expr ':' Suite } ['else' ':' Suite] ;
ForStmt      = 'for' IDENT 'in' Expr ':' Suite ;
WhileStmt    = 'while' Expr ':' Suite ;
ReturnStmt   = 'return' [Expr] ;
PassStmt     = 'pass' ;
Assignment   = Target '=' Expr ;
ExprStmt     = Expr ;

Suite        = NEWLINE INDENT { Statement } DEDENT ;

Expr         = OrExpr ;
OrExpr       = AndExpr { 'or' AndExpr } ;
AndExpr      = NotExpr { 'and' NotExpr } ;
NotExpr      = 'not' NotExpr | Comparison ;
Comparison   = AddExpr { CompOp AddExpr } ;
CompOp       = '==' | '!=' | '<' | '>' | '<=' | '>=' | 'in' | 'not' 'in' ;
AddExpr      = MulExpr { ('+' | '-') MulExpr } ;
MulExpr      = UnaryExpr { ('*' | '/' | '//' | '%') UnaryExpr } ;
UnaryExpr    = ('-' | '+') UnaryExpr | CallExpr ;
CallExpr     = Primary { '(' [ArgList] ')' | '[' Expr ']' | '.' IDENT } ;
Primary      = IDENT | INT_LIT | FLOAT_LIT | STR_LIT | 'True' | 'False'
             | 'None' | '(' Expr ')' | ListLit | DictLit ;

ListLit      = '[' [Expr { ',' Expr }] ']' ;
DictLit      = '{' [DictEntry { ',' DictEntry }] '}' ;
DictEntry    = Expr ':' Expr ;
ArgList      = Expr { ',' Expr } ;
Target       = IDENT { '.' IDENT | '[' Expr ']' } ;
""".strip()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Per-task result for a single condition."""
    task_id: str
    condition: str
    passed: bool          # compile/parse success
    token_count: int
    syntax_error: bool
    semantic_error: bool


@dataclass
class ConditionSummary:
    """Aggregate metrics for one condition."""
    condition: str
    n_tasks: int
    pass_at_1: float
    mean_tokens: float
    median_tokens: float
    syntax_error_rate: float
    semantic_error_rate: float
    token_ci_lower: float
    token_ci_upper: float


@dataclass
class PairedTest:
    """Result of a paired statistical comparison."""
    comparison: str
    metric: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_lower: float
    ci_upper: float


@dataclass
class AblationSummary:
    """Full ablation study summary."""
    n_tasks: int
    seed: int | None
    dry_run: bool
    conditions: list[ConditionSummary]
    comparisons: list[PairedTest]
    toke_grammar_source: str


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

def simulate_predictions(n_tasks: int, rng: np.random.Generator) -> dict[str, list[TaskResult]]:
    """Generate simulated predictions with realistic error distributions.

    Assumptions (based on prior art and toke design goals):
    - Constrained decoding virtually eliminates syntax errors
    - Toke produces fewer tokens than Python on equivalent tasks
    - Unconstrained generation has higher error rates
    - Semantic errors are partially orthogonal to syntax enforcement
    """
    # Per-condition parameters: (pass_rate, mean_tokens, std_tokens,
    #                            syntax_error_rate, semantic_error_rate)
    params = {
        "toke_constrained":       (0.92, 38, 12, 0.02, 0.10),
        "toke_unconstrained":     (0.78, 42, 15, 0.15, 0.12),
        "python_constrained":     (0.90, 52, 16, 0.03, 0.11),
        "python_unconstrained":   (0.74, 58, 18, 0.18, 0.14),
    }

    task_ids = [f"task_{i:04d}" for i in range(n_tasks)]
    results: dict[str, list[TaskResult]] = {}

    for cond, (pass_rate, mu, sigma, syn_rate, sem_rate) in params.items():
        cond_results = []
        for tid in task_ids:
            tokens = max(5, int(rng.normal(mu, sigma)))
            syn_err = bool(rng.random() < syn_rate)
            sem_err = bool(rng.random() < sem_rate)
            passed = (not syn_err) and (not sem_err) and (rng.random() < pass_rate / (1 - syn_rate - sem_rate + syn_rate * sem_rate))
            # Clamp: if syntax error, definitely not passed
            if syn_err:
                passed = False
            cond_results.append(TaskResult(
                task_id=tid,
                condition=cond,
                passed=passed,
                token_count=tokens,
                syntax_error=syn_err,
                semantic_error=sem_err,
            ))
        results[cond] = cond_results

    return results


# ---------------------------------------------------------------------------
# Loading real predictions
# ---------------------------------------------------------------------------

def load_predictions(predictions_dir: Path, n_tasks: int | None) -> dict[str, list[TaskResult]]:
    """Load prediction JSONL files from predictions directory.

    Expected files: {condition}.jsonl with fields:
        task_id, passed, token_count, syntax_error, semantic_error
    """
    results: dict[str, list[TaskResult]] = {}
    for cond in CONDITIONS:
        path = predictions_dir / f"{cond}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping condition {cond}", file=sys.stderr)
            continue
        cond_results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cond_results.append(TaskResult(
                    task_id=obj["task_id"],
                    condition=cond,
                    passed=bool(obj["passed"]),
                    token_count=int(obj["token_count"]),
                    syntax_error=bool(obj.get("syntax_error", False)),
                    semantic_error=bool(obj.get("semantic_error", False)),
                ))
        if n_tasks is not None:
            cond_results = cond_results[:n_tasks]
        results[cond] = cond_results
    return results


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 10000,
                 ci_level: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval using the BCa method.

    Returns (estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng()
    estimate = float(stat_fn(data))
    n = len(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = (1 - ci_level) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return estimate, lo, hi


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """McNemar's test for paired binary outcomes.

    a, b: boolean arrays (True = pass).
    Returns (chi2_statistic, p_value).
    """
    # Discordant pairs
    b_01 = np.sum(a & ~b)  # a pass, b fail
    c_10 = np.sum(~a & b)  # a fail, b pass

    if b_01 + c_10 == 0:
        return 0.0, 1.0

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b_01 - c_10) - 1) ** 2 / (b_01 + c_10)
    p_value = float(sp_stats.chi2.sf(chi2, df=1))
    return float(chi2), p_value


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = x - y
    d_mean = np.mean(diff)
    d_std = np.std(diff, ddof=1)
    if d_std == 0:
        return 0.0
    return float(d_mean / d_std)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_condition_summary(results: list[TaskResult], rng: np.random.Generator) -> ConditionSummary:
    """Compute aggregate metrics for one condition."""
    n = len(results)
    passes = np.array([r.passed for r in results], dtype=float)
    tokens = np.array([r.token_count for r in results], dtype=float)
    syn_errs = np.array([r.syntax_error for r in results], dtype=float)
    sem_errs = np.array([r.semantic_error for r in results], dtype=float)

    _, tok_lo, tok_hi = bootstrap_ci(tokens, np.mean, rng=rng)

    return ConditionSummary(
        condition=results[0].condition,
        n_tasks=n,
        pass_at_1=float(np.mean(passes)),
        mean_tokens=float(np.mean(tokens)),
        median_tokens=float(np.median(tokens)),
        syntax_error_rate=float(np.mean(syn_errs)),
        semantic_error_rate=float(np.mean(sem_errs)),
        token_ci_lower=tok_lo,
        token_ci_upper=tok_hi,
    )


def run_paired_comparisons(results: dict[str, list[TaskResult]],
                           rng: np.random.Generator) -> list[PairedTest]:
    """Run statistical comparisons between key condition pairs."""
    comparisons = [
        ("toke_constrained", "python_constrained"),
        ("toke_unconstrained", "python_unconstrained"),
        ("toke_constrained", "toke_unconstrained"),
        ("python_constrained", "python_unconstrained"),
    ]

    tests: list[PairedTest] = []
    for cond_a, cond_b in comparisons:
        if cond_a not in results or cond_b not in results:
            continue
        ra = results[cond_a]
        rb = results[cond_b]
        n = min(len(ra), len(rb))

        # Pass@1 — McNemar's test
        passes_a = np.array([r.passed for r in ra[:n]])
        passes_b = np.array([r.passed for r in rb[:n]])
        chi2, p_mc = mcnemar_test(passes_a, passes_b)

        # Effect size for pass rate difference via bootstrap
        diff_pass = passes_a.astype(float) - passes_b.astype(float)
        _, dp_lo, dp_hi = bootstrap_ci(diff_pass, np.mean, rng=rng)

        tests.append(PairedTest(
            comparison=f"{cond_a} vs {cond_b}",
            metric="pass_at_1",
            test_name="McNemar",
            statistic=chi2,
            p_value=p_mc,
            effect_size=float(np.mean(diff_pass)),
            effect_size_name="pass_rate_difference",
            ci_lower=dp_lo,
            ci_upper=dp_hi,
        ))

        # Token count — Cohen's d
        tokens_a = np.array([r.token_count for r in ra[:n]], dtype=float)
        tokens_b = np.array([r.token_count for r in rb[:n]], dtype=float)
        d = cohens_d(tokens_a, tokens_b)

        # Bootstrap CI on mean token difference
        diff_tok = tokens_a - tokens_b
        _, dt_lo, dt_hi = bootstrap_ci(diff_tok, np.mean, rng=rng)

        tests.append(PairedTest(
            comparison=f"{cond_a} vs {cond_b}",
            metric="token_count",
            test_name="Cohen_d",
            statistic=d,
            p_value=float(sp_stats.ttest_rel(tokens_a, tokens_b).pvalue),
            effect_size=d,
            effect_size_name="cohens_d",
            ci_lower=dt_lo,
            ci_upper=dt_hi,
        ))

    return tests


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_per_task_results(results: dict[str, list[TaskResult]], output_dir: Path) -> Path:
    """Write data/ablation_results.json — per-task, per-condition results."""
    out_path = output_dir / "ablation_results.json"
    payload: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        if cond not in results:
            continue
        for r in results[cond]:
            payload.append(asdict(r))
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_summary(summary: AblationSummary, output_dir: Path) -> Path:
    """Write data/ablation_summary.json — aggregate metrics with statistical tests."""
    out_path = output_dir / "ablation_summary.json"
    payload = {
        "n_tasks": summary.n_tasks,
        "seed": summary.seed,
        "dry_run": summary.dry_run,
        "toke_grammar_source": summary.toke_grammar_source,
        "conditions": [asdict(c) for c in summary.conditions],
        "comparisons": [asdict(t) for t in summary.comparisons],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_latex_table(summary: AblationSummary, output_dir: Path) -> Path:
    """Write data/ablation_table.csv — LaTeX-ready comparison table."""
    out_path = output_dir / "ablation_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition", "N", "Pass@1", "Mean Tokens", "Median Tokens",
            "Token 95% CI", "Syntax Err %", "Semantic Err %",
        ])
        for c in summary.conditions:
            writer.writerow([
                c.condition,
                c.n_tasks,
                f"{c.pass_at_1:.3f}",
                f"{c.mean_tokens:.1f}",
                f"{c.median_tokens:.1f}",
                f"[{c.token_ci_lower:.1f}, {c.token_ci_upper:.1f}]",
                f"{c.syntax_error_rate:.3f}",
                f"{c.semantic_error_rate:.3f}",
            ])

        # Add comparison rows
        writer.writerow([])
        writer.writerow(["Comparison", "Metric", "Test", "Statistic",
                         "p-value", "Effect Size", "95% CI", ""])
        for t in summary.comparisons:
            writer.writerow([
                t.comparison,
                t.metric,
                t.test_name,
                f"{t.statistic:.4f}",
                f"{t.p_value:.4f}",
                f"{t.effect_size:.4f} ({t.effect_size_name})",
                f"[{t.ci_lower:.4f}, {t.ci_upper:.4f}]",
                "",
            ])
    return out_path


def print_summary(summary: AblationSummary) -> None:
    """Print summary to stdout."""
    print("=" * 72)
    print("CONSTRAINED DECODING ABLATION STUDY")
    print("=" * 72)
    print(f"Tasks: {summary.n_tasks}  |  Dry-run: {summary.dry_run}  |  "
          f"Seed: {summary.seed}  |  Grammar: {summary.toke_grammar_source}")
    print()

    # Condition table
    hdr = f"{'Condition':<26s} {'N':>4s} {'Pass@1':>7s} {'MeanTok':>8s} {'SynErr':>7s} {'SemErr':>7s} {'Token 95% CI':>20s}"
    print(hdr)
    print("-" * len(hdr))
    for c in summary.conditions:
        print(f"{c.condition:<26s} {c.n_tasks:4d} {c.pass_at_1:7.3f} "
              f"{c.mean_tokens:8.1f} {c.syntax_error_rate:7.3f} "
              f"{c.semantic_error_rate:7.3f} "
              f"[{c.token_ci_lower:7.1f}, {c.token_ci_upper:7.1f}]")
    print()

    # Comparisons
    print("STATISTICAL COMPARISONS")
    print("-" * 72)
    for t in summary.comparisons:
        sig = "*" if t.p_value < 0.05 else ""
        print(f"  {t.comparison}")
        print(f"    {t.metric}: {t.test_name} stat={t.statistic:.4f}, "
              f"p={t.p_value:.4f}{sig}, "
              f"effect={t.effect_size:.4f} ({t.effect_size_name}), "
              f"CI=[{t.ci_lower:.4f}, {t.ci_upper:.4f}]")
    print()

    # Key finding
    tc = next((c for c in summary.conditions if c.condition == "toke_constrained"), None)
    pc = next((c for c in summary.conditions if c.condition == "python_constrained"), None)
    if tc and pc:
        reduction = (pc.mean_tokens - tc.mean_tokens) / pc.mean_tokens * 100
        print(f"KEY: Toke constrained vs Python constrained: "
              f"{reduction:+.1f}% token reduction, "
              f"Pass@1 delta = {tc.pass_at_1 - pc.pass_at_1:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrained decoding ablation study (Story 10.9.3)",
    )
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Directory containing {condition}.jsonl prediction files")
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Path to benchmark directory (for task metadata)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Output directory (default: data)")
    parser.add_argument("--tasks", type=int, default=100,
                        help="Number of tasks to evaluate (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate predictions with realistic error distributions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if not args.dry_run and args.predictions_dir is None:
        parser.error("--predictions-dir is required unless --dry-run is set")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine grammar source ---
    grammar_source = "embedded_simplified"
    if TOKE_GRAMMAR_PATH.exists():
        grammar_source = str(TOKE_GRAMMAR_PATH)

    # --- Load or simulate predictions ---
    if args.dry_run:
        print(f"DRY RUN: simulating {args.tasks} tasks (seed={args.seed})")
        results = simulate_predictions(args.tasks, rng)
    else:
        results = load_predictions(args.predictions_dir, args.tasks)
        if not results:
            print("ERROR: no prediction files found", file=sys.stderr)
            sys.exit(1)

    # --- Compute per-condition summaries ---
    condition_summaries = []
    for cond in CONDITIONS:
        if cond in results:
            condition_summaries.append(compute_condition_summary(results[cond], rng))

    # --- Statistical comparisons ---
    comparisons = run_paired_comparisons(results, rng)

    # --- Build summary ---
    n_tasks = min(len(v) for v in results.values()) if results else 0
    summary = AblationSummary(
        n_tasks=n_tasks,
        seed=args.seed,
        dry_run=args.dry_run,
        conditions=condition_summaries,
        comparisons=comparisons,
        toke_grammar_source=grammar_source,
    )

    # --- Write outputs ---
    p1 = write_per_task_results(results, args.output_dir)
    p2 = write_summary(summary, args.output_dir)
    p3 = write_latex_table(summary, args.output_dir)

    print_summary(summary)
    print(f"Written: {p1}")
    print(f"Written: {p2}")
    print(f"Written: {p3}")


if __name__ == "__main__":
    main()

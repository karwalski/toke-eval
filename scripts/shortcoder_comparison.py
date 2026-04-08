#!/usr/bin/env python3
"""ShortCoder comparison harness for head-to-head token efficiency analysis.

Story 9.4.1: Compares toke vs Python vs ShortCoder-optimized Python on
benchmark tasks, computing token counts and reduction ratios with
bootstrap 95% confidence intervals.

IMPORTANT — ShortCoder proxy disclaimer:
    ShortCoder (2026) is not available as a callable tool.  This script
    uses Python minification via ast.unparse (strip comments, shorten
    variable names, remove docstrings, collapse whitespace) as a *proxy*
    for ShortCoder-optimized Python.  The proxy is a lower bound on
    what ShortCoder achieves — real ShortCoder may produce shorter
    token sequences through learned compression strategies that go
    beyond mechanical minification.  All results should be interpreted
    with this caveat.

Usage:
    python scripts/shortcoder_comparison.py \\
        --benchmark-dir /path/to/benchmark \\
        --corpus-dir /path/to/benchmark/baselines/python \\
        --output-dir results/shortcoder_comparison

Output:
    - per_task.csv          — per-task token counts and ratios
    - summary.txt           — summary table with mean/median/CI
    - flagged_tasks.csv     — tasks where ShortCoder proxy beats toke
    - comparison_results.json — machine-readable results
"""
from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import string
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# ShortCoder proxy: Python minification via AST
# ---------------------------------------------------------------------------
# NOTE: This is a PROXY for ShortCoder, not the real tool.
# It applies mechanical minification:
#   1. Parse Python source to AST
#   2. Strip docstrings and comments
#   3. Rename all local variables to short single/double-letter names
#   4. Unparse back to minimal Python via ast.unparse
#
# Real ShortCoder would likely achieve greater compression through
# learned token-level optimizations beyond simple name shortening.

_SHORT_NAMES = list(string.ascii_lowercase) + [
    a + b for a in string.ascii_lowercase for b in string.ascii_lowercase
]


class _NameShortener(ast.NodeTransformer):
    """Rename local variables and parameters to short names."""

    def __init__(self) -> None:
        super().__init__()
        self._map: dict[str, str] = {}
        self._idx: int = 0
        # Names that must not be renamed (builtins, imports, etc.)
        self._protected: set[str] = set(dir(__builtins__)) | {
            "self", "cls", "super", "print", "range", "len", "int",
            "str", "float", "bool", "list", "dict", "set", "tuple",
            "None", "True", "False", "type", "isinstance", "enumerate",
            "zip", "map", "filter", "sorted", "reversed", "sum", "min",
            "max", "abs", "any", "all", "ord", "chr", "hex", "bin",
            "oct", "input", "open", "math", "reduce", "functools",
            "itertools", "collections", "operator",
        }

    def _short(self, name: str) -> str:
        if name in self._protected or name.startswith("_"):
            return name
        if name not in self._map:
            while self._idx < len(_SHORT_NAMES):
                candidate = _SHORT_NAMES[self._idx]
                self._idx += 1
                if candidate not in self._protected:
                    self._map[name] = candidate
                    break
            else:
                # Exhausted short names; keep original
                self._map[name] = name
        return self._map[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Strip docstrings
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
            node.body = node.body[1:]
        # Do not rename the function itself (it may be called externally)
        for arg in node.args.args:
            arg.arg = self._short(arg.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._short(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._short(node.arg)
        return node


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """Remove docstrings from module/class/function bodies."""
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Module)):
            if (child.body and isinstance(child.body[0], ast.Expr)
                    and isinstance(child.body[0].value, ast.Constant)
                    and isinstance(child.body[0].value.value, str)):
                child.body = child.body[1:]
    return node


def shortcoder_proxy(source: str) -> str:
    """Apply Python minification as a ShortCoder proxy.

    PROXY DISCLAIMER: This is NOT ShortCoder.  It is a mechanical
    minification (ast.unparse + variable renaming) that serves as a
    conservative lower-bound proxy for ShortCoder's learned optimizations.

    Args:
        source: Python source code string.

    Returns:
        Minified Python source string, or original if parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    tree = _strip_docstrings(tree)
    shortener = _NameShortener()
    tree = shortener.visit(tree)
    ast.fix_missing_locations(tree)

    try:
        return ast.unparse(tree)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens_cl100k(text: str) -> int:
    """Count tokens using OpenAI's cl100k_base tokenizer."""
    try:
        import tiktoken
    except ImportError:
        sys.exit(
            "ERROR: tiktoken is required.\n"
            "  pip install tiktoken"
        )
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_tokens_custom_bpe(text: str, bpe_path: Path | None) -> int | None:
    """Count tokens using custom toke BPE tokenizer, if available.

    Returns None if the custom BPE tokenizer is not available.
    """
    if bpe_path is None or not bpe_path.exists():
        return None
    try:
        # Attempt to load custom tokenizer (sentencepiece or similar)
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(bpe_path))
        return len(sp.Encode(text))
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task discovery and loading
# ---------------------------------------------------------------------------

def discover_tasks(
    benchmark_dir: Path,
    corpus_dir: Path,
) -> list[dict[str, Any]]:
    """Find tasks that have both a toke solution and a Python baseline.

    Returns a list of dicts: {task_id, toke_source, python_source}.
    """
    solutions_dir = benchmark_dir / "solutions"
    if not solutions_dir.is_dir():
        sys.exit(f"ERROR: solutions directory not found: {solutions_dir}")

    # Load Python baseline solutions from the solutions module
    python_solutions: dict[str, str] = {}
    solutions_py = corpus_dir / "solutions.py"
    if solutions_py.exists():
        python_solutions = _extract_python_functions(solutions_py)

    tasks: list[dict[str, Any]] = []
    for toke_file in sorted(solutions_dir.glob("task-a-*.toke")):
        task_id = toke_file.stem  # e.g. "task-a-0001"
        toke_source = toke_file.read_text(encoding="utf-8").strip()
        if not toke_source:
            continue

        python_source = python_solutions.get(task_id)
        if python_source is None:
            continue

        tasks.append({
            "task_id": task_id,
            "toke_source": toke_source,
            "python_source": python_source,
        })

    return tasks


def _extract_python_functions(solutions_py: Path) -> dict[str, str]:
    """Extract per-task Python function source from solutions.py.

    Parses the solutions module and extracts each @task("task-a-NNNN")
    decorated function as a standalone source string.
    """
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
        # Check for @task("task-a-NNNN") decorator
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

        # Extract function source lines (excluding decorator)
        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno if node.end_lineno else start + 1
        func_lines = source_lines[start:end]
        func_source = textwrap.dedent("".join(func_lines))
        results[task_id] = func_source.strip()

    return results


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    python_tokens: int
    shortcoder_tokens: int
    toke_cl100k_tokens: int
    toke_bpe_tokens: int | None
    ratio_toke_python: float
    ratio_toke_shortcoder: float
    shortcoder_beats_toke: bool


@dataclass
class CIResult:
    estimate: float
    ci_lower: float
    ci_upper: float


@dataclass
class SummaryStats:
    n_tasks: int
    mean_ratio_toke_python: CIResult
    median_ratio_toke_python: CIResult
    mean_ratio_toke_shortcoder: CIResult
    median_ratio_toke_shortcoder: CIResult
    n_shortcoder_beats_toke: int
    pct_shortcoder_beats_toke: float


# ---------------------------------------------------------------------------
# Bootstrap CIs (BCa, 95%)
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    data: np.ndarray,
    stat_func,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CIResult:
    """Bias-corrected and accelerated bootstrap 95% CI using scipy."""
    if rng is None:
        rng = np.random.default_rng(42)

    if len(data) < 3:
        est = float(stat_func(data))
        return CIResult(estimate=est, ci_lower=est, ci_upper=est)

    result = sp_stats.bootstrap(
        (data,),
        stat_func,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        method="BCa",
        random_state=rng,
    )
    est = float(stat_func(data))
    return CIResult(
        estimate=est,
        ci_lower=float(result.confidence_interval.low),
        ci_upper=float(result.confidence_interval.high),
    )


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def run_comparison(
    benchmark_dir: Path,
    corpus_dir: Path,
    bpe_path: Path | None = None,
    seed: int = 42,
) -> tuple[list[TaskResult], SummaryStats]:
    """Run the full ShortCoder comparison harness.

    Args:
        benchmark_dir: Path to benchmark directory.
        corpus_dir: Path to Python baselines directory (contains solutions.py).
        bpe_path: Optional path to custom BPE model file.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        (per_task_results, summary_stats)
    """
    tasks = discover_tasks(benchmark_dir, corpus_dir)
    if not tasks:
        sys.exit(
            "ERROR: No paired tasks found.  Check that:\n"
            f"  - {benchmark_dir}/solutions/ contains .toke files\n"
            f"  - {corpus_dir}/solutions.py contains @task-decorated functions"
        )

    print(f"Found {len(tasks)} paired tasks.", file=sys.stderr)

    results: list[TaskResult] = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        python_src = task["python_source"]
        toke_src = task["toke_source"]

        # 1. Python original tokens
        py_tokens = count_tokens_cl100k(python_src)

        # 2. ShortCoder proxy (minified Python) tokens
        minified_py = shortcoder_proxy(python_src)
        sc_tokens = count_tokens_cl100k(minified_py)

        # 3. Toke via cl100k
        toke_cl100k = count_tokens_cl100k(toke_src)

        # 4. Toke via custom BPE (if available)
        toke_bpe = count_tokens_custom_bpe(toke_src, bpe_path)

        # Ratios (< 1 means toke is more efficient)
        ratio_toke_py = toke_cl100k / py_tokens if py_tokens > 0 else float("inf")
        ratio_toke_sc = toke_cl100k / sc_tokens if sc_tokens > 0 else float("inf")
        sc_beats_toke = sc_tokens < toke_cl100k

        results.append(TaskResult(
            task_id=task_id,
            python_tokens=py_tokens,
            shortcoder_tokens=sc_tokens,
            toke_cl100k_tokens=toke_cl100k,
            toke_bpe_tokens=toke_bpe,
            ratio_toke_python=ratio_toke_py,
            ratio_toke_shortcoder=ratio_toke_sc,
            shortcoder_beats_toke=sc_beats_toke,
        ))

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(tasks)} tasks...", file=sys.stderr)

    # Summary statistics with bootstrap CIs
    rng = np.random.default_rng(seed)
    ratios_toke_py = np.array([r.ratio_toke_python for r in results])
    ratios_toke_sc = np.array([r.ratio_toke_shortcoder for r in results])

    # scipy.stats.bootstrap expects stat_func(data, axis) signature
    def _mean(x, axis=None):
        return np.mean(x, axis=axis)

    def _median(x, axis=None):
        return np.median(x, axis=axis)

    summary = SummaryStats(
        n_tasks=len(results),
        mean_ratio_toke_python=_bootstrap_ci(ratios_toke_py, _mean, rng=rng),
        median_ratio_toke_python=_bootstrap_ci(ratios_toke_py, _median, rng=rng),
        mean_ratio_toke_shortcoder=_bootstrap_ci(ratios_toke_sc, _mean, rng=rng),
        median_ratio_toke_shortcoder=_bootstrap_ci(ratios_toke_sc, _median, rng=rng),
        n_shortcoder_beats_toke=sum(1 for r in results if r.shortcoder_beats_toke),
        pct_shortcoder_beats_toke=(
            100.0 * sum(1 for r in results if r.shortcoder_beats_toke) / len(results)
        ),
    )

    return results, summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_per_task_csv(results: list[TaskResult], path: Path) -> None:
    """Write per-task CSV with all 4 token counts and ratios."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "python_tokens_cl100k",
            "shortcoder_proxy_tokens_cl100k",
            "toke_tokens_cl100k",
            "toke_tokens_bpe",
            "ratio_toke_div_python",
            "ratio_toke_div_shortcoder",
            "shortcoder_beats_toke",
        ])
        for r in results:
            writer.writerow([
                r.task_id,
                r.python_tokens,
                r.shortcoder_tokens,
                r.toke_cl100k_tokens,
                r.toke_bpe_tokens if r.toke_bpe_tokens is not None else "",
                f"{r.ratio_toke_python:.4f}",
                f"{r.ratio_toke_shortcoder:.4f}",
                r.shortcoder_beats_toke,
            ])


def write_flagged_csv(results: list[TaskResult], path: Path) -> None:
    """Write CSV of tasks where ShortCoder proxy beats toke (flagged)."""
    flagged = [r for r in results if r.shortcoder_beats_toke]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id",
            "shortcoder_proxy_tokens",
            "toke_tokens_cl100k",
            "ratio_toke_div_shortcoder",
            "gap_tokens",
        ])
        for r in flagged:
            writer.writerow([
                r.task_id,
                r.shortcoder_tokens,
                r.toke_cl100k_tokens,
                f"{r.ratio_toke_shortcoder:.4f}",
                r.toke_cl100k_tokens - r.shortcoder_tokens,
            ])


def _ci_str(ci: CIResult) -> str:
    return f"{ci.estimate:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"


def _pct_str(ci: CIResult) -> str:
    """Format ratio as percentage reduction: (1 - ratio) * 100."""
    est = (1 - ci.estimate) * 100
    lo = (1 - ci.ci_upper) * 100  # inverted
    hi = (1 - ci.ci_lower) * 100
    return f"{est:.1f}% [{lo:.1f}%, {hi:.1f}%]"


def write_summary(summary: SummaryStats, path: Path) -> None:
    """Write human-readable summary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 74,
        "SHORTCODER COMPARISON HARNESS — TOKEN EFFICIENCY ANALYSIS",
        "=" * 74,
        "",
        "PROXY DISCLAIMER: ShortCoder results use Python minification",
        "(ast.unparse + variable renaming) as a proxy, NOT the real",
        "ShortCoder tool.  See script docstring for details.",
        "",
        f"Tasks analysed: {summary.n_tasks}",
        "",
        "--- Toke vs Python (original) ---",
        f"  Mean  ratio (toke/python):   {_ci_str(summary.mean_ratio_toke_python)}",
        f"    => mean reduction:         {_pct_str(summary.mean_ratio_toke_python)}",
        f"  Median ratio (toke/python):  {_ci_str(summary.median_ratio_toke_python)}",
        f"    => median reduction:       {_pct_str(summary.median_ratio_toke_python)}",
        "",
        "--- Toke vs ShortCoder Proxy (minified Python) ---",
        f"  Mean  ratio (toke/sc):       {_ci_str(summary.mean_ratio_toke_shortcoder)}",
        f"    => mean reduction:         {_pct_str(summary.mean_ratio_toke_shortcoder)}",
        f"  Median ratio (toke/sc):      {_ci_str(summary.median_ratio_toke_shortcoder)}",
        f"    => median reduction:       {_pct_str(summary.median_ratio_toke_shortcoder)}",
        "",
        "--- Flagged Tasks (ShortCoder proxy beats toke) ---",
        f"  Count: {summary.n_shortcoder_beats_toke} / {summary.n_tasks}"
        f" ({summary.pct_shortcoder_beats_toke:.1f}%)",
        "",
        "95% CIs computed via BCa bootstrap (10,000 resamples).",
        "=" * 74,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(
    results: list[TaskResult],
    summary: SummaryStats,
    path: Path,
) -> None:
    """Write machine-readable JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = {
        "proxy_disclaimer": (
            "ShortCoder results use Python minification (ast.unparse + "
            "variable renaming) as a proxy, NOT the real ShortCoder tool."
        ),
        "summary": asdict(summary),
        "per_task": [asdict(r) for r in results],
    }
    path.write_text(
        json.dumps(obj, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ShortCoder comparison harness: head-to-head token efficiency "
            "analysis of toke vs Python vs ShortCoder-proxy-optimized Python."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            PROXY DISCLAIMER
            ----------------
            ShortCoder (2026) is not available as a callable tool.
            This script uses Python minification via ast.unparse
            (strip comments, shorten variable names, remove docstrings)
            as a proxy.  Results labelled "ShortCoder" are from this
            proxy and should be interpreted accordingly.
        """),
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, required=True,
        help="Path to benchmark directory.",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, required=True,
        help="Path to Python baselines directory (contains solutions.py).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for output files (CSV, summary, JSON).",
    )
    parser.add_argument(
        "--bpe-model", type=Path, default=None,
        help="Optional path to custom BPE model (.model) for toke tokenizer.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap reproducibility (default: 42).",
    )

    args = parser.parse_args(argv)

    if not args.benchmark_dir.is_dir():
        print(f"ERROR: benchmark dir not found: {args.benchmark_dir}",
              file=sys.stderr)
        return 1
    if not args.corpus_dir.is_dir():
        print(f"ERROR: corpus dir not found: {args.corpus_dir}",
              file=sys.stderr)
        return 1

    print("ShortCoder Comparison Harness", file=sys.stderr)
    print("NOTE: Using Python minification as ShortCoder proxy.",
          file=sys.stderr)
    print("", file=sys.stderr)

    results, summary = run_comparison(
        benchmark_dir=args.benchmark_dir,
        corpus_dir=args.corpus_dir,
        bpe_path=args.bpe_model,
        seed=args.seed,
    )

    # Write outputs
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_per_task_csv(results, out / "per_task.csv")
    write_flagged_csv(results, out / "flagged_tasks.csv")
    write_summary(summary, out / "summary.txt")
    write_json(results, summary, out / "comparison_results.json")

    # Print summary to stdout
    summary_text = (out / "summary.txt").read_text(encoding="utf-8")
    print(summary_text)

    print(f"\nOutputs written to: {out}", file=sys.stderr)
    print(f"  per_task.csv            ({len(results)} rows)", file=sys.stderr)
    flagged_count = sum(1 for r in results if r.shortcoder_beats_toke)
    print(f"  flagged_tasks.csv       ({flagged_count} flagged)", file=sys.stderr)
    print(f"  summary.txt", file=sys.stderr)
    print(f"  comparison_results.json", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

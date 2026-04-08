#!/usr/bin/env python3
"""Multi-tokenizer token economy analysis: toke vs Python across tokenizers.

Story 9.4.2: Run token counts across >=4 tokenizers on all benchmark tasks.
Flag tasks where toke is token-longer than Python for ANY tokenizer.

Extends Story 10.1.3 (multi_tokenizer.py) with cross-language comparison
(toke vs Python) and deeper aggregate analysis.

Dependencies:
  Required: tiktoken>=0.5.0
  Optional: transformers (for Qwen and Llama tokenizers)
  Optional: sentencepiece (for toke BPE tokenizer)
  Optional: numpy, scipy (for bootstrap CIs)

Usage:
    python scripts/token_economy.py \\
        --benchmark-dir ../benchmark \\
        --output-dir data

    python scripts/token_economy.py \\
        --benchmark-dir ../benchmark \\
        --output-dir results \\
        --tokenizers cl100k_base,o200k_base \\
        --seed 42
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tokenizer registry — reuses patterns from multi_tokenizer.py
# ---------------------------------------------------------------------------

TIKTOKEN_TOKENIZERS = {"cl100k_base", "o200k_base"}
HF_TOKENIZERS = {
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B",
}
BPE_KEY = "toke_bpe"

ALL_TOKENIZER_KEYS = ["cl100k_base", "o200k_base", "qwen", "llama", "toke_bpe"]

# Display names for summary output
TOKENIZER_DISPLAY = {
    "cl100k_base": "cl100k_base (GPT-4)",
    "o200k_base": "o200k_base (GPT-4o)",
    "qwen": "Qwen2.5-Coder-7B",
    "llama": "Llama-3.1-8B",
    "toke_bpe": "toke BPE (sentencepiece)",
}


@dataclass
class LoadedTokenizer:
    """Wrapper around a loaded tokenizer with a uniform encode API."""
    key: str
    name: str
    encode: Any  # callable(str) -> list[int]


def _load_tiktoken(name: str) -> LoadedTokenizer | None:
    try:
        import tiktoken
    except ImportError:
        print(f"WARNING: tiktoken not installed, skipping {name}", file=sys.stderr)
        return None
    enc = tiktoken.get_encoding(name)
    return LoadedTokenizer(key=name, name=name, encode=enc.encode)


def _load_hf_tokenizer(key: str, model_id: str) -> LoadedTokenizer | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            f"WARNING: transformers not installed, skipping {key} ({model_id})",
            file=sys.stderr,
        )
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        print(f"WARNING: failed to load {model_id}: {exc}", file=sys.stderr)
        return None

    def _encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    return LoadedTokenizer(key=key, name=model_id, encode=_encode)


def _load_bpe_tokenizer(bpe_path: Path | None) -> LoadedTokenizer | None:
    if bpe_path is None:
        # Try default location
        default = Path(__file__).resolve().parent.parent.parent / "toke-model" / "tokenizer" / "toke.model"
        if default.exists():
            bpe_path = default
        else:
            print("WARNING: no BPE model path provided and default not found, skipping toke_bpe", file=sys.stderr)
            return None

    if not bpe_path.exists():
        print(f"WARNING: BPE model not found at {bpe_path}, skipping toke_bpe", file=sys.stderr)
        return None

    try:
        import sentencepiece as spm
    except ImportError:
        print("WARNING: sentencepiece not installed, skipping toke_bpe", file=sys.stderr)
        return None

    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(str(bpe_path))
    except Exception as exc:
        print(f"WARNING: failed to load BPE model: {exc}", file=sys.stderr)
        return None

    def _encode(text: str) -> list[int]:
        return sp.Encode(text)

    return LoadedTokenizer(key=BPE_KEY, name=f"toke_bpe ({bpe_path.name})", encode=_encode)


def load_tokenizers(
    requested: list[str],
    bpe_path: Path | None = None,
) -> list[LoadedTokenizer]:
    """Load requested tokenizers, skipping unavailable ones with warnings."""
    loaded: list[LoadedTokenizer] = []
    for key in requested:
        if key in TIKTOKEN_TOKENIZERS:
            tok = _load_tiktoken(key)
        elif key in HF_TOKENIZERS:
            tok = _load_hf_tokenizer(key, HF_TOKENIZERS[key])
        elif key == BPE_KEY:
            tok = _load_bpe_tokenizer(bpe_path)
        else:
            print(f"WARNING: unknown tokenizer key '{key}', skipping", file=sys.stderr)
            continue
        if tok is not None:
            loaded.append(tok)
    return loaded


# ---------------------------------------------------------------------------
# Task discovery — loads toke solutions + Python baselines
# ---------------------------------------------------------------------------

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
class TaskPair:
    """A benchmark task with both toke and Python source."""
    task_id: str
    toke_source: str
    python_source: str


def discover_tasks(benchmark_dir: Path) -> list[TaskPair]:
    """Find tasks that have both a toke solution and a Python baseline."""
    solutions_dir = benchmark_dir / "solutions"
    baselines_py = benchmark_dir / "baselines" / "python" / "solutions.py"

    if not solutions_dir.is_dir():
        sys.exit(f"ERROR: solutions directory not found: {solutions_dir}")
    if not baselines_py.exists():
        sys.exit(f"ERROR: Python baselines not found: {baselines_py}")

    python_solutions = _extract_python_functions(baselines_py)
    if not python_solutions:
        sys.exit(f"ERROR: no @task functions found in {baselines_py}")

    tasks: list[TaskPair] = []
    for toke_file in sorted(solutions_dir.glob("*.toke")):
        task_id = toke_file.stem
        toke_source = toke_file.read_text(encoding="utf-8").strip()
        if not toke_source:
            continue

        python_source = python_solutions.get(task_id)
        if python_source is None:
            continue

        tasks.append(TaskPair(
            task_id=task_id,
            toke_source=toke_source,
            python_source=python_source,
        ))

    return tasks


# ---------------------------------------------------------------------------
# Per-task tokenization
# ---------------------------------------------------------------------------

@dataclass
class TaskTokenResult:
    task_id: str
    toke_chars: int
    python_chars: int
    toke_counts: dict[str, int]     # tokenizer_key -> token count
    python_counts: dict[str, int]   # tokenizer_key -> token count
    ratios: dict[str, float]        # tokenizer_key -> toke/python ratio
    flagged: bool                   # toke longer than python for ANY tokenizer


def tokenize_task(
    task: TaskPair,
    tokenizers: list[LoadedTokenizer],
) -> TaskTokenResult:
    """Tokenize a single task's toke and Python source with all tokenizers."""
    toke_counts: dict[str, int] = {}
    python_counts: dict[str, int] = {}
    ratios: dict[str, float] = {}
    flagged = False

    for tok in tokenizers:
        tc = len(tok.encode(task.toke_source))
        pc = len(tok.encode(task.python_source))
        toke_counts[tok.key] = tc
        python_counts[tok.key] = pc
        ratio = tc / pc if pc > 0 else float("inf")
        ratios[tok.key] = ratio
        if ratio > 1.0:
            flagged = True

    return TaskTokenResult(
        task_id=task.task_id,
        toke_chars=len(task.toke_source),
        python_chars=len(task.python_source),
        toke_counts=toke_counts,
        python_counts=python_counts,
        ratios=ratios,
        flagged=flagged,
    )


# ---------------------------------------------------------------------------
# Aggregate analysis
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Compute percentile (0-100) from sorted data."""
    s = sorted(data)
    n = len(s)
    if n == 0:
        return 0.0
    k = (p / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _histogram_buckets(data: list[float]) -> dict[str, int]:
    """Create histogram buckets for token ratios."""
    buckets = {
        "<0.50": 0,
        "0.50-0.60": 0,
        "0.60-0.70": 0,
        "0.70-0.80": 0,
        "0.80-0.90": 0,
        "0.90-1.00": 0,
        "1.00-1.10": 0,
        "1.10-1.20": 0,
        ">1.20": 0,
    }
    for v in data:
        if v < 0.50:
            buckets["<0.50"] += 1
        elif v < 0.60:
            buckets["0.50-0.60"] += 1
        elif v < 0.70:
            buckets["0.60-0.70"] += 1
        elif v < 0.80:
            buckets["0.70-0.80"] += 1
        elif v < 0.90:
            buckets["0.80-0.90"] += 1
        elif v < 1.00:
            buckets["0.90-1.00"] += 1
        elif v < 1.10:
            buckets["1.00-1.10"] += 1
        elif v < 1.20:
            buckets["1.10-1.20"] += 1
        else:
            buckets[">1.20"] += 1
    return buckets


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient, or None if insufficient data."""
    if len(xs) < 3 or len(ys) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sx * sy) ** 0.5


@dataclass
class TokenizerStats:
    key: str
    display_name: str
    mean_ratio: float
    median_ratio: float
    p10_ratio: float
    p90_ratio: float
    mean_toke_tokens: float
    mean_python_tokens: float
    total_toke_tokens: int
    total_python_tokens: int
    n_toke_longer: int
    histogram: dict[str, int]


@dataclass
class AggregateSummary:
    n_tasks: int
    n_flagged: int
    pct_flagged: float
    per_tokenizer: list[TokenizerStats]
    correlations: dict[str, float]
    most_favorable_tokenizer: str


def compute_aggregate(
    results: list[TaskTokenResult],
    tokenizers: list[LoadedTokenizer],
) -> AggregateSummary:
    """Compute aggregate statistics across all tasks and tokenizers."""
    n_flagged = sum(1 for r in results if r.flagged)

    per_tok: list[TokenizerStats] = []
    ratio_columns: dict[str, list[float]] = {}

    for tok in tokenizers:
        ratios = [r.ratios[tok.key] for r in results if tok.key in r.ratios]
        toke_tokens = [r.toke_counts[tok.key] for r in results if tok.key in r.toke_counts]
        py_tokens = [r.python_counts[tok.key] for r in results if tok.key in r.python_counts]

        ratio_columns[tok.key] = ratios

        per_tok.append(TokenizerStats(
            key=tok.key,
            display_name=TOKENIZER_DISPLAY.get(tok.key, tok.key),
            mean_ratio=statistics.mean(ratios) if ratios else 0.0,
            median_ratio=statistics.median(ratios) if ratios else 0.0,
            p10_ratio=_percentile(ratios, 10),
            p90_ratio=_percentile(ratios, 90),
            mean_toke_tokens=statistics.mean(toke_tokens) if toke_tokens else 0.0,
            mean_python_tokens=statistics.mean(py_tokens) if py_tokens else 0.0,
            total_toke_tokens=sum(toke_tokens),
            total_python_tokens=sum(py_tokens),
            n_toke_longer=sum(1 for r in ratios if r > 1.0),
            histogram=_histogram_buckets(ratios),
        ))

    # Correlation between tokenizer ratio columns
    keys = list(ratio_columns.keys())
    correlations: dict[str, float] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            corr = _correlation(ratio_columns[keys[i]], ratio_columns[keys[j]])
            if corr is not None:
                correlations[f"{keys[i]} vs {keys[j]}"] = round(corr, 4)

    # Most favorable = lowest mean ratio
    most_favorable = min(per_tok, key=lambda s: s.mean_ratio).key if per_tok else ""

    return AggregateSummary(
        n_tasks=len(results),
        n_flagged=n_flagged,
        pct_flagged=100.0 * n_flagged / len(results) if results else 0.0,
        per_tokenizer=per_tok,
        correlations=correlations,
        most_favorable_tokenizer=most_favorable,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_full_csv(
    results: list[TaskTokenResult],
    tokenizers: list[LoadedTokenizer],
    path: Path,
) -> None:
    """Write per-task, per-tokenizer token counts CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task_id", "toke_chars", "python_chars"]
    for tok in tokenizers:
        fieldnames.append(f"toke_{tok.key}_tokens")
        fieldnames.append(f"python_{tok.key}_tokens")
        fieldnames.append(f"ratio_{tok.key}")
    fieldnames.append("flagged")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row: dict[str, Any] = {
                "task_id": r.task_id,
                "toke_chars": r.toke_chars,
                "python_chars": r.python_chars,
                "flagged": r.flagged,
            }
            for tok in tokenizers:
                row[f"toke_{tok.key}_tokens"] = r.toke_counts.get(tok.key, "")
                row[f"python_{tok.key}_tokens"] = r.python_counts.get(tok.key, "")
                row[f"ratio_{tok.key}"] = (
                    f"{r.ratios[tok.key]:.4f}" if tok.key in r.ratios else ""
                )
            writer.writerow(row)

    print(f"Wrote {len(results)} rows to {path}", file=sys.stderr)


def write_flagged_csv(
    results: list[TaskTokenResult],
    tokenizers: list[LoadedTokenizer],
    path: Path,
) -> None:
    """Write CSV of tasks where toke is token-longer than Python."""
    flagged = [r for r in results if r.flagged]
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["task_id"]
    for tok in tokenizers:
        fieldnames.append(f"toke_{tok.key}_tokens")
        fieldnames.append(f"python_{tok.key}_tokens")
        fieldnames.append(f"ratio_{tok.key}")
    fieldnames.append("worst_tokenizer")
    fieldnames.append("worst_ratio")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in flagged:
            # Find worst tokenizer (highest ratio > 1.0)
            over_one = {k: v for k, v in r.ratios.items() if v > 1.0}
            worst_key = max(over_one, key=over_one.get) if over_one else ""
            worst_ratio = over_one.get(worst_key, 0.0) if worst_key else 0.0

            row: dict[str, Any] = {"task_id": r.task_id}
            for tok in tokenizers:
                row[f"toke_{tok.key}_tokens"] = r.toke_counts.get(tok.key, "")
                row[f"python_{tok.key}_tokens"] = r.python_counts.get(tok.key, "")
                row[f"ratio_{tok.key}"] = (
                    f"{r.ratios[tok.key]:.4f}" if tok.key in r.ratios else ""
                )
            row["worst_tokenizer"] = worst_key
            row["worst_ratio"] = f"{worst_ratio:.4f}" if worst_ratio else ""
            writer.writerow(row)

    print(f"Wrote {len(flagged)} flagged rows to {path}", file=sys.stderr)


def write_summary_json(summary: AggregateSummary, path: Path) -> None:
    """Write aggregate stats as machine-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    obj: dict[str, Any] = {
        "n_tasks": summary.n_tasks,
        "n_flagged": summary.n_flagged,
        "pct_flagged": round(summary.pct_flagged, 2),
        "most_favorable_tokenizer": summary.most_favorable_tokenizer,
        "per_tokenizer": {},
        "correlations": summary.correlations,
    }

    for ts in summary.per_tokenizer:
        obj["per_tokenizer"][ts.key] = {
            "display_name": ts.display_name,
            "mean_ratio": round(ts.mean_ratio, 4),
            "median_ratio": round(ts.median_ratio, 4),
            "p10_ratio": round(ts.p10_ratio, 4),
            "p90_ratio": round(ts.p90_ratio, 4),
            "mean_toke_tokens": round(ts.mean_toke_tokens, 1),
            "mean_python_tokens": round(ts.mean_python_tokens, 1),
            "total_toke_tokens": ts.total_toke_tokens,
            "total_python_tokens": ts.total_python_tokens,
            "n_toke_longer": ts.n_toke_longer,
            "histogram": ts.histogram,
        }

    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary to {path}", file=sys.stderr)


def print_summary_table(summary: AggregateSummary) -> None:
    """Print human-readable summary table to stdout."""
    w = 78
    print("=" * w)
    print("MULTI-TOKENIZER TOKEN ECONOMY ANALYSIS")
    print("=" * w)
    print()
    print(f"Tasks analysed:  {summary.n_tasks}")
    print(f"Tasks flagged:   {summary.n_flagged} / {summary.n_tasks}"
          f" ({summary.pct_flagged:.1f}%)")
    print(f"  (flagged = toke is token-longer than Python for ANY tokenizer)")
    print()

    # Per-tokenizer table
    header = f"{'Tokenizer':<28} {'Mean':>7} {'Median':>7} {'P10':>7} {'P90':>7} {'#Longer':>8}"
    print(header)
    print("-" * len(header))
    for ts in summary.per_tokenizer:
        name = ts.display_name
        if len(name) > 27:
            name = name[:24] + "..."
        print(
            f"{name:<28} {ts.mean_ratio:>7.4f} {ts.median_ratio:>7.4f}"
            f" {ts.p10_ratio:>7.4f} {ts.p90_ratio:>7.4f} {ts.n_toke_longer:>8}"
        )
    print()

    # Most favorable
    fav_key = summary.most_favorable_tokenizer
    fav_name = TOKENIZER_DISPLAY.get(fav_key, fav_key)
    fav_stats = next((t for t in summary.per_tokenizer if t.key == fav_key), None)
    if fav_stats:
        reduction = (1 - fav_stats.mean_ratio) * 100
        print(f"Most favorable tokenizer: {fav_name}")
        print(f"  Mean ratio: {fav_stats.mean_ratio:.4f}"
              f" => {reduction:.1f}% token reduction")
    print()

    # Histogram for first tokenizer
    if summary.per_tokenizer:
        ts0 = summary.per_tokenizer[0]
        print(f"Ratio distribution ({ts0.display_name}):")
        max_count = max(ts0.histogram.values()) if ts0.histogram else 1
        for bucket, count in ts0.histogram.items():
            bar_len = int(40 * count / max_count) if max_count > 0 else 0
            bar = "#" * bar_len
            print(f"  {bucket:>9}: {count:>4} {bar}")
        print()

    # Correlations
    if summary.correlations:
        print("Tokenizer ratio correlations:")
        for pair, corr in summary.correlations.items():
            print(f"  {pair}: {corr:.4f}")
        print()

    print("=" * w)
    print("Ratio < 1.0 = toke is more token-efficient than Python")
    print("Ratio > 1.0 = toke uses MORE tokens than Python (flagged)")
    print("=" * w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_tokenizer_list(value: str) -> list[str]:
    """Parse comma-separated tokenizer keys."""
    return [k.strip() for k in value.split(",") if k.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-tokenizer token economy analysis: compare toke vs Python "
            "token counts across multiple tokenizers on benchmark tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Path to benchmark directory (contains solutions/ and baselines/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for CSV/JSON files (default: data).",
    )
    parser.add_argument(
        "--tokenizers",
        type=parse_tokenizer_list,
        default=ALL_TOKENIZER_KEYS,
        help=(
            "Comma-separated tokenizer keys "
            f"(default: {','.join(ALL_TOKENIZER_KEYS)}). "
            f"Available: {', '.join(ALL_TOKENIZER_KEYS)}"
        ),
    )
    parser.add_argument(
        "--bpe-model",
        type=Path,
        default=None,
        help="Path to custom BPE model (.model) for toke_bpe tokenizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (reserved for future bootstrap CIs, default: 42).",
    )

    args = parser.parse_args(argv)

    # Validate benchmark directory
    if not args.benchmark_dir.is_dir():
        print(f"ERROR: benchmark directory not found: {args.benchmark_dir}",
              file=sys.stderr)
        return 1

    # Discover paired tasks
    tasks = discover_tasks(args.benchmark_dir)
    if not tasks:
        print("ERROR: no paired tasks found (need both .toke and Python baseline)",
              file=sys.stderr)
        return 1
    print(f"Found {len(tasks)} paired tasks (toke + Python).", file=sys.stderr)

    # Load tokenizers
    tokenizers = load_tokenizers(args.tokenizers, bpe_path=args.bpe_model)
    if not tokenizers:
        print("ERROR: no tokenizers could be loaded", file=sys.stderr)
        return 1
    print(
        f"Loaded {len(tokenizers)} tokenizer(s): "
        + ", ".join(t.name for t in tokenizers),
        file=sys.stderr,
    )

    # Tokenize all tasks
    results: list[TaskTokenResult] = []
    for i, task in enumerate(tasks):
        results.append(tokenize_task(task, tokenizers))
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(tasks)} tasks...", file=sys.stderr)

    # Aggregate analysis
    summary = compute_aggregate(results, tokenizers)

    # Output files
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    write_full_csv(results, tokenizers, out / "token_economy_full.csv")
    write_summary_json(summary, out / "token_economy_summary.json")
    write_flagged_csv(results, tokenizers, out / "token_economy_flagged.csv")

    # Summary to stdout
    print_summary_table(summary)

    print(f"\nOutputs written to: {out}", file=sys.stderr)
    print(f"  token_economy_full.csv       ({len(results)} rows)", file=sys.stderr)
    print(f"  token_economy_summary.json   (aggregate stats)", file=sys.stderr)
    n_flagged = sum(1 for r in results if r.flagged)
    print(f"  token_economy_flagged.csv    ({n_flagged} flagged)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

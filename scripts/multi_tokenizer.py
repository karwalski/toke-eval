#!/usr/bin/env python3
"""Multi-tokenizer baseline token counts for toke programs.

Story 10.1.3: Extend token counting from cl100k_base to multiple tokenizers:
  - cl100k_base  (OpenAI GPT-4)
  - o200k_base   (OpenAI GPT-4o)
  - Qwen/Qwen2.5-Coder-7B-Instruct
  - meta-llama/Llama-3.1-8B

Dependencies:
  Required: tiktoken>=0.5.0
  Optional: transformers, torch|sentencepiece (for Qwen and Llama tokenizers)

Usage:
    # All four tokenizers against benchmark solutions
    python scripts/multi_tokenizer.py \\
        --corpus-dir ../benchmark/solutions \\
        --output data/multi_tokenizer_counts.csv

    # Only tiktoken-based tokenizers
    python scripts/multi_tokenizer.py \\
        --corpus-dir ../benchmark/solutions \\
        --tokenizers cl100k_base,o200k_base \\
        --output data/multi_tokenizer_counts.csv

    # Custom directory of .toke or .tk files
    python scripts/multi_tokenizer.py \\
        --corpus-dir /path/to/files \\
        --output results/counts.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tokenizer registry — each entry knows how to load and encode
# ---------------------------------------------------------------------------

TIKTOKEN_TOKENIZERS = {"cl100k_base", "o200k_base"}
HF_TOKENIZERS = {
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B",
}

# Canonical order and CSV column names
TOKENIZER_COLUMNS = {
    "cl100k_base": "cl100k_tokens",
    "o200k_base": "o200k_tokens",
    "qwen": "qwen_tokens",
    "llama": "llama_tokens",
}

ALL_TOKENIZER_KEYS = list(TOKENIZER_COLUMNS.keys())


@dataclass
class LoadedTokenizer:
    """Wrapper around a loaded tokenizer with a uniform encode API."""
    key: str
    name: str
    encode: Any  # callable(str) -> list[int]


def _load_tiktoken(name: str) -> LoadedTokenizer | None:
    """Load a tiktoken encoding by name."""
    try:
        import tiktoken
    except ImportError:
        print(f"WARNING: tiktoken not installed, skipping {name}", file=sys.stderr)
        return None
    enc = tiktoken.get_encoding(name)
    return LoadedTokenizer(key=name, name=name, encode=enc.encode)


def _load_hf_tokenizer(key: str, model_id: str) -> LoadedTokenizer | None:
    """Load a HuggingFace tokenizer via transformers.AutoTokenizer."""
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
        print(
            f"WARNING: failed to load {model_id}: {exc}",
            file=sys.stderr,
        )
        return None

    def _encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    return LoadedTokenizer(key=key, name=model_id, encode=_encode)


def load_tokenizers(requested: list[str]) -> list[LoadedTokenizer]:
    """Load requested tokenizers, skipping unavailable ones with warnings."""
    loaded: list[LoadedTokenizer] = []
    for key in requested:
        if key in TIKTOKEN_TOKENIZERS:
            tok = _load_tiktoken(key)
        elif key in HF_TOKENIZERS:
            tok = _load_hf_tokenizer(key, HF_TOKENIZERS[key])
        else:
            print(f"WARNING: unknown tokenizer key '{key}', skipping", file=sys.stderr)
            continue
        if tok is not None:
            loaded.append(tok)
    return loaded


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def discover_source_files(corpus_dir: Path) -> list[Path]:
    """Find toke source files (.toke or .tk) in corpus_dir, sorted by name."""
    files = sorted(corpus_dir.glob("*.toke"))
    if not files:
        files = sorted(corpus_dir.glob("*.tk"))
    if not files:
        # Try recursive search
        files = sorted(corpus_dir.rglob("*.toke"))
        if not files:
            files = sorted(corpus_dir.rglob("*.tk"))
    return files


# ---------------------------------------------------------------------------
# Core tokenization
# ---------------------------------------------------------------------------

@dataclass
class TaskRow:
    task_id: str
    source_chars: int
    token_counts: dict[str, int]  # tokenizer_key -> count
    char_ratios: dict[str, float]  # tokenizer_key -> chars/tokens


def tokenize_file(
    path: Path, tokenizers: list[LoadedTokenizer]
) -> TaskRow:
    """Tokenize a single source file with all loaded tokenizers."""
    source = path.read_text(encoding="utf-8").strip()
    char_count = len(source)
    task_id = path.stem

    counts: dict[str, int] = {}
    ratios: dict[str, float] = {}

    for tok in tokenizers:
        n_tokens = len(tok.encode(source))
        counts[tok.key] = n_tokens
        ratios[tok.key] = char_count / n_tokens if n_tokens > 0 else 0.0

    return TaskRow(
        task_id=task_id,
        source_chars=char_count,
        token_counts=counts,
        char_ratios=ratios,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(rows: list[TaskRow], tokenizers: list[LoadedTokenizer]) -> None:
    """Print summary statistics (mean, median, p10, p90) per tokenizer."""
    if not rows:
        return

    print("\n=== Summary Statistics ===", file=sys.stderr)
    print(f"Total files: {len(rows)}", file=sys.stderr)

    char_counts = [r.source_chars for r in rows]
    print(f"\nSource characters:", file=sys.stderr)
    print(f"  Mean:   {statistics.mean(char_counts):.1f}", file=sys.stderr)
    print(f"  Median: {statistics.median(char_counts):.1f}", file=sys.stderr)

    for tok in tokenizers:
        counts = [r.token_counts[tok.key] for r in rows]
        ratios = [r.char_ratios[tok.key] for r in rows]
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        p10_idx = max(0, int(n * 0.10) - 1)
        p90_idx = min(n - 1, int(n * 0.90))

        col_name = TOKENIZER_COLUMNS.get(tok.key, tok.key)
        print(f"\n{tok.name} ({col_name}):", file=sys.stderr)
        print(f"  Mean tokens:   {statistics.mean(counts):.1f}", file=sys.stderr)
        print(f"  Median tokens: {statistics.median(counts):.1f}", file=sys.stderr)
        print(f"  P10:           {sorted_counts[p10_idx]}", file=sys.stderr)
        print(f"  P90:           {sorted_counts[p90_idx]}", file=sys.stderr)
        print(f"  Total tokens:  {sum(counts)}", file=sys.stderr)
        print(f"  Mean char/tok: {statistics.mean(ratios):.2f}", file=sys.stderr)


def flag_high_token_tasks(
    rows: list[TaskRow], tokenizers: list[LoadedTokenizer]
) -> None:
    """Flag tasks where one tokenizer produces notably more tokens than others.

    A task is flagged when any tokenizer's count exceeds the minimum count
    across tokenizers by more than 50%.
    """
    if len(tokenizers) < 2:
        return

    flagged: list[tuple[str, str, int, int]] = []
    for row in rows:
        counts = {t.key: row.token_counts[t.key] for t in tokenizers}
        min_count = min(counts.values())
        if min_count == 0:
            continue
        for key, count in counts.items():
            if count > min_count * 1.5:
                flagged.append((row.task_id, key, count, min_count))

    if flagged:
        print(f"\n=== Flagged Tasks (>50% above min tokenizer) ===", file=sys.stderr)
        for task_id, tok_key, count, min_c in flagged[:20]:
            pct = (count - min_c) / min_c * 100
            print(
                f"  {task_id}: {tok_key}={count} vs min={min_c} (+{pct:.0f}%)",
                file=sys.stderr,
            )
        if len(flagged) > 20:
            print(f"  ... and {len(flagged) - 20} more", file=sys.stderr)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(
    rows: list[TaskRow],
    tokenizers: list[LoadedTokenizer],
    output: Path,
) -> None:
    """Write per-task multi-tokenizer CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build fieldnames dynamically based on loaded tokenizers
    fieldnames = ["task_id", "source_chars"]
    ratio_fields = []
    for tok in tokenizers:
        col = TOKENIZER_COLUMNS.get(tok.key, f"{tok.key}_tokens")
        fieldnames.append(col)
        ratio_col = col.replace("_tokens", "_char_ratio")
        ratio_fields.append((tok.key, ratio_col))
    for _, ratio_col in ratio_fields:
        fieldnames.append(ratio_col)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            csv_row: dict[str, Any] = {
                "task_id": row.task_id,
                "source_chars": row.source_chars,
            }
            for tok in tokenizers:
                col = TOKENIZER_COLUMNS.get(tok.key, f"{tok.key}_tokens")
                csv_row[col] = row.token_counts[tok.key]
            for tok_key, ratio_col in ratio_fields:
                csv_row[ratio_col] = f"{row.char_ratios[tok_key]:.2f}"
            writer.writerow(csv_row)

    print(f"\nWrote {len(rows)} rows to {output}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_tokenizer_list(value: str) -> list[str]:
    """Parse comma-separated tokenizer keys."""
    return [k.strip() for k in value.split(",") if k.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-tokenizer baseline token counts for toke programs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing .toke or .tk source files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/multi_tokenizer_counts.csv"),
        help="Output CSV path (default: data/multi_tokenizer_counts.csv)",
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
    args = parser.parse_args(argv)

    # Validate corpus directory
    if not args.corpus_dir.is_dir():
        print(f"ERROR: corpus directory not found: {args.corpus_dir}", file=sys.stderr)
        return 1

    # Discover source files
    source_files = discover_source_files(args.corpus_dir)
    if not source_files:
        print(
            f"ERROR: no .toke or .tk files found in {args.corpus_dir}",
            file=sys.stderr,
        )
        return 1
    print(f"Found {len(source_files)} source files", file=sys.stderr)

    # Load tokenizers
    tokenizers = load_tokenizers(args.tokenizers)
    if not tokenizers:
        print("ERROR: no tokenizers could be loaded", file=sys.stderr)
        return 1
    print(
        f"Loaded {len(tokenizers)} tokenizer(s): "
        + ", ".join(t.name for t in tokenizers),
        file=sys.stderr,
    )

    # Tokenize all files
    rows: list[TaskRow] = []
    for i, path in enumerate(source_files):
        rows.append(tokenize_file(path, tokenizers))
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(source_files)}...", file=sys.stderr)

    # Output
    write_csv(rows, tokenizers, args.output)
    print_summary(rows, tokenizers)
    flag_high_token_tasks(rows, tokenizers)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Token efficiency measurement for toke vs other languages.

Compares token counts across languages using configurable tokenizers.

Usage:
    python -m toke_eval.token_efficiency \\
        --corpus corpus_p2.jsonl \\
        --tokenizer cl100k_base \\
        --output efficiency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LanguageStats:
    language: str
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    total_tokens: int = 0
    program_count: int = 0
    token_counts: list[int] = field(default_factory=list)


@dataclass
class EfficiencyReport:
    timestamp: str = ""
    tokenizer: str = ""
    corpus_entries: int = 0
    languages: dict[str, LanguageStats] = field(default_factory=dict)
    reduction_vs_python: float = 0.0
    reduction_vs_c: float = 0.0
    reduction_vs_java: float = 0.0


def count_tokens(text: str, tokenizer_name: str) -> int:
    """Count tokens using the specified tokenizer."""
    if tokenizer_name == "cl100k_base":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            sys.exit("ERROR: tiktoken required. Install: pip install tiktoken")
    else:
        # Character-based fallback for unknown tokenizers
        return len(text.split())


def evaluate_corpus(corpus_path: Path, tokenizer: str,
                    max_entries: int = 0) -> EfficiencyReport:
    """Evaluate token efficiency across a corpus JSONL file."""
    report = EfficiencyReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tokenizer=tokenizer,
    )

    lang_tokens: dict[str, list[int]] = {
        "toke": [], "python": [], "c": [], "java": []
    }

    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if max_entries and i >= max_entries:
                break

            entry = json.loads(line)
            report.corpus_entries += 1

            # toke source
            tk_src = entry.get("tk_source", "")
            if tk_src:
                lang_tokens["toke"].append(count_tokens(tk_src, tokenizer))

            # reference implementations
            refs = entry.get("references", {})
            for lang in ("python", "c", "java"):
                src = refs.get(f"{lang}_source", "")
                if src:
                    lang_tokens[lang].append(count_tokens(src, tokenizer))

            if report.corpus_entries % 5000 == 0:
                print(f"  [{report.corpus_entries}] processed",
                      file=sys.stderr)

    for lang, counts in lang_tokens.items():
        if counts:
            stats = LanguageStats(
                language=lang,
                mean_tokens=statistics.mean(counts),
                median_tokens=statistics.median(counts),
                total_tokens=sum(counts),
                program_count=len(counts),
            )
            report.languages[lang] = stats

    # Compute reductions
    toke_mean = report.languages.get("toke")
    if toke_mean:
        tm = toke_mean.mean_tokens
        for lang, attr in [("python", "reduction_vs_python"),
                           ("c", "reduction_vs_c"),
                           ("java", "reduction_vs_java")]:
            other = report.languages.get(lang)
            if other and other.mean_tokens > 0:
                setattr(report, attr,
                        1.0 - tm / other.mean_tokens)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Token efficiency measurement for toke")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.corpus.exists():
        sys.exit(f"ERROR: corpus not found: {args.corpus}")

    report = evaluate_corpus(args.corpus, args.tokenizer, args.max_entries)

    print(f"\nToken Efficiency Report ({args.tokenizer})", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    for lang, stats in sorted(report.languages.items()):
        print(f"  {lang:8s}  mean={stats.mean_tokens:6.1f}  "
              f"n={stats.program_count}", file=sys.stderr)
    print(f"\n  Reduction vs Python: {report.reduction_vs_python:.1%}",
          file=sys.stderr)
    print(f"  Reduction vs C:      {report.reduction_vs_c:.1%}",
          file=sys.stderr)
    print(f"  Reduction vs Java:   {report.reduction_vs_java:.1%}",
          file=sys.stderr)

    # Strip raw counts for output (too large)
    for stats in report.languages.values():
        stats.token_counts = []

    output = json.dumps(asdict(report), indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

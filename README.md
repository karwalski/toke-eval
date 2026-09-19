# toke-eval

This repository measures how well AI models write
[toke](https://github.com/karwalski/toke) code. It contains benchmark
tasks, an evaluation harness that compiles and runs generated programs,
and tools for analysing results. It consolidates what was formerly the
standalone `toke-benchmark` repository (now retired and archived).

## About toke

> toke: a compiled language designed for LLM code generation, with a small grammar, one
> canonical form and compiler verification.

toke is a compiled programming language designed for LLM code generation. It has 14
keywords, a 55-character set, a backtrack-free grammar with bounded lookahead, and one
canonical form per construct, chosen by measurement in a 46-pattern catalogue and
reproduced by `tkc --min`. That makes generated code cheap to constrain during decoding,
cheap for a compiler to verify afterwards, and compact to emit. Token efficiency is one
measured property of toke, always reported with its tokenizer and its baseline, not the
whole claim.

*The one-liner and the paragraph above are reproduced word for word from the canonical
description,
[`docs/about/canonical.md`](https://github.com/karwalski/toke/blob/main/docs/about/canonical.md).
Every number published about toke comes from
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
and nowhere else.*

## What's Inside

| Component | Path | Purpose |
|-----------|------|---------|
| **Benchmark tasks** | `benchmark/tasks/` | Task definitions, reference solutions, and baseline implementations in Python/C/Java |
| **Evaluation harness** | `toke_eval/` | Pass@k scorer, token efficiency measurement, and gate report generator |
| **Statistical analysis** | `scripts/statistical_analysis.py` | Bootstrap CIs, Wilcoxon tests, and power analysis for token efficiency claims |
| **Result data** | `data/` | Evaluation outputs (token counts, pass@k results, ablation tables) |
| **Gate card template** | `gate_card_template.md` | Standardised form for recording gate evaluation outcomes |

Held-out test cases used for gate evaluation are **not** included in
this repository. They are stored separately and never committed to any
public repository.

## Reporting rules for anything measured here

Numbers produced by this harness become public only through
[`docs/metrics-baseline.md`](https://github.com/karwalski/toke/blob/main/docs/metrics-baseline.md)
in the toke repository, and they carry the four TEMSpec §6.3 fields — metric type,
tokenizer(s), baseline and sample size — wherever they are quoted.

Two rules the harness exists to enforce:

- **One tokenizer on both sides.** A toke-trained tokenizer (`proxy8k`, `tokenizer_v03`,
  Toke-16K) measures its own training bias when applied to Python, C or Java. Any
  cross-language figure uses the same tokenizer on both sides.
- **Measure the canonical `--min` form.** Measuring readable source understated toke by
  28.2%; `tkc --min` is the basis on both sides of any comparison.

The most recent delivery on this suite is the 60 Gate-1 JSON-CLI tasks re-delivered on
v0.4 ([`docs/gate1-60-v04.md`](docs/gate1-60-v04.md), 2026-09-19): 60/60 `tkc --check`,
60/60 hidden tests, lint 0/0. Those programs are **hand-written, not model-generated** —
27 ids are pure `--migrate` output and 33 were hand-repaired — so the set measures what
the *language* can express, not what a *model* produces, and it may not be quoted as a
model result or as a Pass@1.

## Quick Start

```bash
# Install
pip install -e .

# Run Pass@1 evaluation
python -m toke_eval.pass_at_k \
    --solutions-dir /path/to/solutions/ \
    --tests-dir /path/to/hidden_tests/ \
    --compiler /path/to/toke \
    --output results.json

# Measure token efficiency
python -m toke_eval.token_efficiency \
    --corpus /path/to/corpus_p2.jsonl \
    --tokenizer cl100k_base \
    --output efficiency.json

# Generate gate report
python -m toke_eval.report \
    --pass-at-k results.json \
    --token-efficiency efficiency.json \
    --output gate_report.json

# Run statistical analysis (synthetic data for validation)
python scripts/statistical_analysis.py --generate-mock --n-tasks 1000
```

Requires Python 3.10+.

## Project Structure

```
toke-eval/
  benchmark/
    tasks/           Task YAML definitions and schema
    solutions/       Reference toke solutions
    baselines/       Python/C/Java reference implementations
    harness/         Harness for running models against tasks
    results/         Past gate evaluation results
    tests/           Harness unit tests
  toke_eval/         Python package
    pass_at_k.py     Compile, run, and score generated programs
    token_efficiency.py  Token count comparison across languages
    report.py        Aggregate results into gate decision reports
    generate_token_counts.py  Token count extraction
  scripts/           Standalone analysis and benchmarking scripts
  data/              Evaluation result datasets (CSV, JSON)
  docs/              Design documents (contamination analysis)
```

## Related Repositories

| Repository | Role |
|------------|------|
| [toke](https://github.com/karwalski/toke) | Language specification and reference compiler (`toke`) |
| [toke-corpus](https://github.com/karwalski/toke-corpus) | Training-data generation and curation (the corpora this harness measures against) |
| [toke-model](https://github.com/karwalski/toke-models) | Model training, adapter merging (the models this harness evaluates) |
| [toke-tokenizer](https://github.com/karwalski/toke-tokenizer) | Custom tokenizer used for token-efficiency measurements |
| [toke-mcp](https://github.com/karwalski/toke-mcp) | MCP server providing toke tooling to AI assistants |

## Licence

Apache 2.0. See [LICENSE](LICENSE).

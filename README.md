# toke-eval

This repository measures how well AI models write
[toke](https://github.com/karwalski/toke) code. It contains benchmark
tasks, an evaluation harness that compiles and runs generated programs,
and tools for analysing results. It consolidates what was formerly the
standalone `toke-benchmark` repository.

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

## Quick Start

```bash
# Install
pip install -e .

# Run Pass@1 evaluation
python -m toke_eval.pass_at_k \
    --solutions-dir /path/to/solutions/ \
    --tests-dir /path/to/hidden_tests/ \
    --compiler /path/to/tkc \
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
| [toke](https://github.com/karwalski/toke) | Language specification and reference compiler (`tkc`) |
| [toke-model](https://github.com/karwalski/toke-model) | Model training, corpus generation, adapter merging |
| [toke-mcp](https://github.com/karwalski/toke-mcp) | MCP server providing toke tooling to AI assistants |

## Licence

Apache 2.0. See [LICENSE](LICENSE).

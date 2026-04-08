# toke-eval — evaluation toolkit for toke models

toke-eval provides tools for evaluating toke code generation models:

- **Pass@k scoring** — measure first-pass accuracy on held-out benchmark tasks
- **Token efficiency** — compare token counts across languages and tokenizers
- **Benchmark reporting** — aggregate results into structured JSON reports
- **Safety evaluation** — adversarial prompt testing via LlamaGuard

## Usage

### Pass@1 evaluation

```bash
python -m toke_eval.pass_at_k \
    --solutions-dir /path/to/solutions/ \
    --tests-dir /path/to/hidden_tests/ \
    --compiler /path/to/tkc \
    --output results.json
```

### Token efficiency measurement

```bash
python -m toke_eval.token_efficiency \
    --corpus /path/to/corpus_p2.jsonl \
    --tokenizer cl100k_base \
    --output efficiency.json
```

### Report generation

```bash
python -m toke_eval.report \
    --pass-at-k results.json \
    --token-efficiency efficiency.json \
    --output gate_report.json
```

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Related repositories

| Repository | Role |
|------------|------|
| [toke-eval/benchmark](https://github.com/karwalski/toke-eval) | Task definitions, hidden tests, reference implementations |
| [toke-models](https://github.com/karwalski/toke-models) | Model training and adapter merging |
| [toke](https://github.com/karwalski/toke) | Reference compiler (used by evaluation harness) |

## Licence

Apache 2.0. See [LICENSE](LICENSE).

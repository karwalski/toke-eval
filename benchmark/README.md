# toke-eval/benchmark

Benchmark task definitions and evaluation harness for
[toke](https://github.com/karwalski/toke).

## What is here

- `tasks/` — the task **schema** (`schema.json`) only; there are no task files here
- `hidden_tests/` — the task YAML files the harness actually reads
- `baselines/` — Python, C, and Java reference implementations per task
- `harness/` — retired entry points (see below); the live harness is
  `run_benchmark.py` in this directory

## What is NOT here

Held-out test cases used for gate evaluation are not public.
They are stored separately and never committed to any repository.

## Running a benchmark

    # reference solutions (python / c / toke)
    python run_benchmark.py \
      --solutions-dir baselines/python \
      --tasks-dir hidden_tests/ \
      --language python \
      --output results/baseline.json

    # model inference — strict Pass@1: one sample, one attempt
    python run_benchmark.py \
      --tasks-dir hidden_tests/ \
      --model-endpoint http://localhost:8000/generate \
      --n-samples 1

`--n-samples N` with N > 1 reports **three** distinct numbers: `pass_at_1`
(single-sample success rate), `mean_best_of_n` (an oracle best-of-N figure —
never a Pass@1), and `mean_pass_at_k` (Chen et al. unbiased estimator). See
the `run_benchmark.py` module docstring.

For Pass@k from sampled predictions use `../scripts/pass_at_k.py`, the only
Pass@k implementation the Epic 128 protocol permits
(`toke-model/docs/training-reset-128.md` §6.3).

### Retired entry points

`harness/run.py`, `harness/score.py` and `harness/report.py` were 0-byte files
from the initial subtree import and were never implemented. They now exit
non-zero with a pointer to the live tooling rather than exiting 0 having done
nothing. Story 128.1c; each file's docstring records what it was meant to be.

## Task schema

Tasks are defined as YAML files:

    id: task-a-0042
    phase: A
    description: "Sum all integers in an array"
    input_type: "[i64]"
    output_type: "i64"
    test_inputs:
      - input: [1, 2, 3]
        expected: 6

## Adding tasks

See CONTRIBUTING.md for the task addition process.
Held-out tasks are added by the project maintainer only.

## Licence

Apache 2.0.

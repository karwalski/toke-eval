# toke-eval/benchmark

Benchmark task definitions and evaluation harness for
[toke](https://github.com/karwalski/toke).

## What is here

- `tasks/` — benchmark task definitions used in corpus generation
- `baselines/` — Python, C, and Java reference implementations per task
- `harness/` — evaluation scripts for running models against tasks

## What is NOT here

Held-out test cases used for gate evaluation are not public.
They are stored separately and never committed to any repository.

## Running a benchmark

    python harness/run.py \
      --model /path/to/toke-model \
      --tasks tasks/phase-a/ \
      --out results/

    python harness/score.py \
      --results results/ \
      --baselines baselines/

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

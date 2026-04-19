# Contributing to toke-eval

Thank you for your interest in contributing to the toke evaluation
infrastructure.

## Getting Started

```bash
git clone git@github.com:karwalski/toke-eval.git
cd toke-eval
pip install -e .
```

## How to Contribute

1. **Open an issue** describing the change you'd like to make.
2. **Fork the repository** and create a feature branch.
3. **Submit a pull request** against `main`.

## Adding Benchmark Tasks

- Public tasks (in `benchmark/tasks/`) may be contributed by anyone via PR.
- Held-out test cases are added by the maintainer only and are never
  committed to this repository.
- When adding a task, include reference implementations in all three
  baseline languages (Python, C, Java) in the same PR.
- Task IDs are assigned sequentially within each phase.
- Never change the expected output for an existing task -- raise an
  issue if a reference output is wrong.

### Task Requirements

Every task must have:

- A precise description with no ambiguity
- Typed input and output specification
- At least 20 test inputs covering normal and edge cases
- Reference implementations in Python, C, and Java that all agree on
  all test inputs

### PR Checklist

- [ ] Task YAML is valid against `benchmark/tasks/schema.json`
- [ ] Reference implementations agree on all test inputs
- [ ] `python -m pytest benchmark/tests/` passes

## Code Style

- Python 3.10+ with type hints
- Follow existing patterns in `toke_eval/`
- No new dependencies without discussion in an issue first

## Reporting Bugs

Open a GitHub issue with:

- Steps to reproduce
- Expected and actual behaviour
- Python version and OS

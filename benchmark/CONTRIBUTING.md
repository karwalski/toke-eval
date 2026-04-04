# Contributing to toke-benchmark

## Rules

- Public tasks (in tasks/) may be contributed by anyone via PR
- Held-out test cases are added by the maintainer only and are
  never committed to this repository
- When adding a task, add reference implementations in all three
  baseline languages (Python, C, Java) in the same PR
- Task IDs are assigned sequentially within each phase
- Never change the expected output for an existing task — raise an
  issue if a reference output is wrong

## Task requirements

Every task must have:
- A precise description with no ambiguity
- Typed input and output specification
- At least 20 test inputs covering normal cases and edge cases
- Reference implementations in Python, C, and Java that all
  agree on all test inputs

## PR checklist

- [ ] Task YAML is valid against tasks/schema.json
- [ ] Reference implementations agree on all test inputs
- [ ] `python -m pytest tests/` passes

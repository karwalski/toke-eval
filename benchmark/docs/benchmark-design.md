# toke Benchmark Design

## Task phases

- **Phase A:** D2C algorithmic tasks (data in, computation, value out)
- **Phase B:** Data structure tasks (trees, graphs, linked lists)
- **Phase C:** System interaction tasks (file I/O, HTTP, database)

## Evaluation metrics

- **Pass@1:** Model generates a correct solution on first attempt
- **Token efficiency:** tk tokens vs Python tokens for the same task

## Holdout protocol

Public tasks are in `tasks/`. Held-out test cases used for gate evaluation
are stored separately and never committed to any repository.

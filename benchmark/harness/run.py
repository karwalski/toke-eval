#!/usr/bin/env python3
"""RETIRED ENTRY POINT -- do not use.  See `benchmark/run_benchmark.py`.

Why this file exists at all (story 128.1c)
------------------------------------------
This file, together with `score.py` and `report.py`, sat in the tree as a
**0-byte file** from the initial `benchmark/` subtree import (`3d5aa69`).  It
has never had any content in any commit of any repo.

That is not harmless.  `benchmark/README.md` documented it as the way to run
the benchmark:

    python harness/run.py --model /path/to/toke-model --tasks tasks/phase-a/ \\
        --out results/

Running that command against a 0-byte file **exits 0, prints nothing, and
writes nothing** -- indistinguishable, to a caller or a CI step, from a run
that succeeded and found no work to do.  It is the same failure mode as the
other two 128.1c defects: the harness declines to fail, and silence gets read
as a result.

What it was meant to be
-----------------------
Recovered from the original acceptance criteria, archived at
`archive/tkc/docs/epics_and_stories.md` (Story 2.10.1, "Benchmark evaluation
harness (run.py, score.py, report.py)"):

    - `harness/run.py` accepts a model output directory and a task set,
      executes generated programs against test cases, and records results
    - `harness/score.py` computes Pass@1, Pass@3, Pass@5 metrics;
      per-category breakdowns; token efficiency via cl100k_base
    - `harness/report.py` generates a JSON summary and human-readable report

Why it is not being implemented
-------------------------------
Every one of those acceptance criteria is already met elsewhere, by code that
has tests and recorded runs behind it:

    run     -> `benchmark/run_benchmark.py`  (task discovery, solution
               loading for python/c/toke, model inference, execution against
               `test_inputs`, JSON report)
    Pass@k  -> `scripts/pass_at_k.py`        (Chen et al. 2021 unbiased
               estimator; per `toke-model/docs/training-reset-128.md` §6.3
               this is the ONLY permitted Pass@k implementation)
    tokens  -> `toke_eval/token_efficiency.py`, `toke_eval/generate_token_counts.py`
    report  -> `toke_eval/report.py`

Implementing this file would produce a *fourth* Pass@1 implementation in a
repo that has just spent a story fixing the two that disagreed.  The archived
spec fixes the intent but not the wire formats -- the model-invocation
protocol, the results-file layout, the baseline comparison -- and guessing
those is precisely how a harness ends up emitting a confident wrong number.

So this file now refuses, loudly, and says where to go instead.
"""

from __future__ import annotations

import sys

_MESSAGE = """\
harness/run.py is a retired entry point and was never implemented.

Use instead:

    python benchmark/run_benchmark.py \\
        --solutions-dir <dir> --tasks-dir benchmark/hidden_tests/ \\
        --language toke --output results/run.json

    # model inference (strict Pass@1: one sample, one attempt)
    python benchmark/run_benchmark.py \\
        --tasks-dir benchmark/hidden_tests/ \\
        --model-endpoint <url> --n-samples 1

Note: benchmark/tasks/ holds only schema.json -- there are no task files
there.  Point the harness at benchmark/hidden_tests/.

For Pass@k use scripts/pass_at_k.py, the only permitted implementation
(toke-model/docs/training-reset-128.md §6.3).

See this file's docstring for the full history (story 128.1c).
"""


def main(argv: list[str] | None = None) -> int:
    print(_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

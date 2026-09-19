#!/usr/bin/env python3
"""RETIRED ENTRY POINT -- do not use.  See `scripts/pass_at_k.py`.

A 0-byte file since the initial `benchmark/` subtree import (`3d5aa69`), yet
documented in `benchmark/README.md` as

    python harness/score.py --results results/ --baselines baselines/

which exits 0 and scores nothing.  Story 128.1c; see `run.py` in this
directory for the full reasoning.

Intended scope, from the archived acceptance criteria
(`archive/tkc/docs/epics_and_stories.md`, Story 2.10.1):

    `harness/score.py` computes Pass@1, Pass@3, Pass@5 metrics;
    per-category breakdowns; token efficiency via cl100k_base

All of which now live in:

    Pass@k             -> `scripts/pass_at_k.py` (Chen et al. 2021 unbiased
                          estimator; the ONLY permitted Pass@k implementation
                          per `toke-model/docs/training-reset-128.md` §6.3)
    Pass@1 + execution -> `benchmark/run_benchmark.py`
    token efficiency   -> `toke_eval/token_efficiency.py`

Not reimplemented here: a fifth scorer is how metrics drift apart.
"""

from __future__ import annotations

import sys

_MESSAGE = """\
harness/score.py is a retired entry point and was never implemented.

Use instead:

    # Pass@k (k = 1, 5, 10) from sampled predictions
    python scripts/pass_at_k.py \\
        --predictions-dir <dir> \\
        --benchmark-dir benchmark/hidden_tests/ \\
        --output-dir data/ --k-values 1,5,10

    # strict Pass@1 by execution
    python benchmark/run_benchmark.py --tasks-dir benchmark/hidden_tests/ ...

    # token efficiency
    python -m toke_eval.token_efficiency ...

See this file's docstring for the full history (story 128.1c).
"""


def main(argv: list[str] | None = None) -> int:
    print(_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

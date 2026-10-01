#!/usr/bin/env python3
"""RETIRED ENTRY POINT -- do not use.  See `toke_eval/report.py`.

A 0-byte file since the initial `benchmark/` subtree import (`3d5aa69`).
Unlike `run.py` and `score.py` it is referenced by NO README, no script and no
test -- the only surviving statement of its purpose is one line of archived
acceptance criteria (`archive/tkc/docs/epics_and_stories.md`, Story 2.10.1):

    `harness/report.py` generates a JSON summary and human-readable report

That is the whole specification.  It fixes neither the input format, nor the
summary schema, nor what "human-readable" meant, and no example output of it
has ever existed.  Anything written here would be invention, so nothing is:
per story 128.1c, an undeterminable stub is reported, not filled in with a
plausible stand-in.

Reporting is already covered by `toke_eval/report.py` (aggregate results into
gate decision reports) and by the JSON writers in
`benchmark/run_benchmark.py::report_to_dict` and
`scripts/pass_at_k.py::write_json_report` / `write_csv_summary`.
"""

from __future__ import annotations

import sys

_MESSAGE = """\
harness/report.py is a retired entry point.  It was never implemented and its
intended behaviour is not recoverable from any surviving specification beyond
one line of archived acceptance criteria.

Use instead:

    python -m toke_eval.report ...

See this file's docstring for the full history (story 128.1c).
"""


def main(argv: list[str] | None = None) -> int:
    print(_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

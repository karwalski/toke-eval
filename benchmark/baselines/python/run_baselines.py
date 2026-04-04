#!/usr/bin/env python3
"""Runner that verifies Python baseline solutions against hidden test YAML files.

Usage:
    python3 baselines/python/run_baselines.py [--tests-dir hidden_tests]

Exits with code 0 if all tests pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: pyyaml is required.  Install it with:\n"
        "  pip install pyyaml"
    )

# Ensure the baselines package is importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from solutions import SOLUTIONS  # noqa: E402


def _compare(actual: object, expected: object) -> bool:
    """Deep equality that normalises booleans stored as strings in YAML."""
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        return False
    if isinstance(actual, bool) and not isinstance(expected, bool):
        return False
    return actual == expected


def run(tests_dir: Path, verbose: bool = False) -> bool:
    task_files = sorted(tests_dir.glob("task-a-*.yaml"))
    if not task_files:
        print(f"ERROR: no task YAML files found in {tests_dir}")
        return False

    total_tasks = 0
    passed_tasks = 0
    skipped_tasks = 0
    failed_tasks: list[str] = []

    for tf in task_files:
        with open(tf) as f:
            task = yaml.safe_load(f)

        task_id: str = task["id"]

        if task_id not in SOLUTIONS:
            skipped_tasks += 1
            if verbose:
                print(f"  SKIP  {task_id} (no solution)")
            continue

        total_tasks += 1
        fn = SOLUTIONS[task_id]
        test_cases = task["test_inputs"]
        task_passed = True

        for i, tc in enumerate(test_cases):
            inp = tc["input"]
            expected = tc["expected"]
            try:
                actual = fn(inp)
            except Exception as exc:
                print(f"  FAIL  {task_id} case {i}: raised {exc!r}")
                task_passed = False
                break

            if not _compare(actual, expected):
                print(
                    f"  FAIL  {task_id} case {i}: "
                    f"input={inp!r}  expected={expected!r}  got={actual!r}"
                )
                task_passed = False
                break

        if task_passed:
            passed_tasks += 1
            if verbose:
                print(f"  PASS  {task_id} ({len(test_cases)} cases)")
        else:
            failed_tasks.append(task_id)

    print()
    print("=" * 60)
    print(
        f"Results: {passed_tasks}/{total_tasks} tasks passed, "
        f"{skipped_tasks} skipped, {len(failed_tasks)} failed"
    )
    if failed_tasks:
        print(f"Failed: {', '.join(failed_tasks)}")
    print("=" * 60)

    return len(failed_tasks) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "hidden_tests",
        help="Directory containing task-a-*.yaml files (default: hidden_tests/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    ok = run(args.tests_dir, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

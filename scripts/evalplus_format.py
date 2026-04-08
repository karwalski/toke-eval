#!/usr/bin/env python3
"""Convert benchmark tasks to EvalPlus JSON schema.

Reads task YAML files from toke-eval/benchmark and produces a JSON file
conforming to the EvalPlus schema with these fields per task:

    task_id:            unique identifier (e.g. "toke/task-a-0001")
    prompt:             task description for model input
    entry_point:        function entry point (always "main" for toke)
    test:               test assertions as executable check code
    canonical_solution: reference solution source (if available)

Usage:
    python scripts/evalplus_format.py \
        --tasks-dir ../benchmark/hidden_tests/ \
        --solutions-dir ../benchmark/solutions/ \
        --output data/evalplus_tasks.json

Exit codes:
    0  success
    1  error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml required. Install: pip install pyyaml")


def build_prompt(task: dict[str, Any]) -> str:
    """Build a model prompt from a task definition.

    Constructs a prompt that includes the task description, input/output
    types, and example test cases (first 3 only, to avoid leaking the
    full test suite).
    """
    desc = task.get("description", "")
    input_type = task.get("input_type", "")
    output_type = task.get("output_type", "")

    lines = [
        f"// Task: {desc}",
        f"// Input type:  {input_type}",
        f"// Output type: {output_type}",
        "//",
        "// Examples:",
    ]

    cases = task.get("test_inputs", [])
    for case in cases[:3]:
        inp = json.dumps(case["input"]) if not isinstance(case["input"], str) else case["input"]
        exp = case["expected"]
        lines.append(f"//   input={inp}  expected={exp}")

    lines.append("//")
    lines.append("// Write a toke program that reads input from argv and prints the result.")
    lines.append("")

    return "\n".join(lines)


def build_test_code(task: dict[str, Any]) -> str:
    """Build test assertion code from task test cases.

    Generates a JSON array of {input, expected} pairs that can be used
    by the evaluation harness to verify solutions.
    """
    cases = task.get("test_inputs", [])
    test_entries = []
    for case in cases:
        inp = case["input"]
        exp = case["expected"]
        test_entries.append({"input": inp, "expected": exp})

    return json.dumps(test_entries, indent=2)


def convert_task(
    task_file: Path,
    solutions_dir: Path | None = None,
) -> dict[str, Any]:
    """Convert a single benchmark YAML task to EvalPlus format."""
    with open(task_file) as f:
        task = yaml.safe_load(f)

    task_id_raw = task.get("id", task_file.stem)
    task_id = f"toke/{task_id_raw}"

    # Load canonical solution if available
    canonical = ""
    if solutions_dir is not None:
        sol_path = solutions_dir / f"{task_id_raw}.toke"
        if sol_path.exists():
            canonical = sol_path.read_text()

    return {
        "task_id": task_id,
        "prompt": build_prompt(task),
        "entry_point": "main",
        "test": build_test_code(task),
        "canonical_solution": canonical,
        "metadata": {
            "phase": task.get("phase", ""),
            "category": task.get("category", ""),
            "input_type": task.get("input_type", ""),
            "output_type": task.get("output_type", ""),
            "source": str(task_file.name),
        },
    }


def convert_all(
    tasks_dir: Path,
    solutions_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert all task YAML files in a directory to EvalPlus format."""
    task_files = sorted(tasks_dir.glob("task-*.yaml"))
    if not task_files:
        print("WARNING: no task files found", file=sys.stderr)
        return []

    results = []
    for tf in task_files:
        try:
            entry = convert_task(tf, solutions_dir)
            results.append(entry)
        except Exception as e:
            print(f"WARNING: failed to convert {tf.name}: {e}",
                  file=sys.stderr)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark tasks to EvalPlus JSON schema",
    )
    parser.add_argument(
        "--tasks-dir", type=Path, required=True,
        help="Directory containing task YAML files",
    )
    parser.add_argument(
        "--solutions-dir", type=Path, default=None,
        help="Directory with canonical .toke solutions (optional)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file path (default: stdout)",
    )
    args = parser.parse_args()

    if not args.tasks_dir.is_dir():
        sys.exit(f"ERROR: tasks dir not found: {args.tasks_dir}")

    entries = convert_all(args.tasks_dir, args.solutions_dir)

    output_json = json.dumps(entries, indent=2)
    print(f"Converted {len(entries)} tasks to EvalPlus format",
          file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()

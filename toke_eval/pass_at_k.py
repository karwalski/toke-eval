#!/usr/bin/env python3
"""Pass@k evaluation for toke code generation models.

Compiles solutions with tkc, runs against test cases, and computes pass@k.

Usage:
    python -m toke_eval.pass_at_k \\
        --solutions-dir solutions/ \\
        --tests-dir hidden_tests/ \\
        --compiler ./tkc \\
        --output results.json

Exit codes:
    0  success
    1  error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml required. Install: pip install pyyaml")


def classify_error(error_text: str) -> str:
    """Classify a compile/runtime error into a taxonomy category.

    Categories:
        syntax    — lexer errors (E1xxx)
        parse     — parser/grammar errors (E2xxx)
        name      — name resolution errors (E3xxx)
        type      — type checking errors (E4xxx)
        codegen   — code generation / backend errors (E9xxx)
        runtime   — compiled but failed at runtime (timeout, crash, wrong output)
        logic     — compiled and ran but produced wrong output
        unknown   — error text could not be classified
    """
    import re
    codes = re.findall(r'[EW](\d{4})', error_text)
    if not codes:
        if "timeout" in error_text.lower():
            return "runtime"
        return "unknown"
    # Use the first error code to classify
    code = int(codes[0])
    if 1000 <= code < 2000:
        return "syntax"
    if 2000 <= code < 3000:
        return "parse"
    if 3000 <= code < 4000:
        return "name"
    if 4000 <= code < 5000:
        return "type"
    if 9000 <= code < 10000:
        return "codegen"
    return "unknown"


@dataclass
class TaskResult:
    task_id: str
    compiled: bool = False
    compile_error: str = ""
    error_category: str = ""
    tests_total: int = 0
    tests_passed: int = 0
    pass_at_1: float = 0.0
    runtime_ms: float = 0.0


@dataclass
class ErrorTaxonomy:
    """Breakdown of failures by category."""
    syntax: int = 0
    parse: int = 0
    name: int = 0
    type: int = 0
    codegen: int = 0
    runtime: int = 0
    logic: int = 0
    unknown: int = 0

    def total(self) -> int:
        return (self.syntax + self.parse + self.name + self.type +
                self.codegen + self.runtime + self.logic + self.unknown)


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    compiler: str = ""
    tasks_total: int = 0
    tasks_compiled: int = 0
    tasks_passed: int = 0
    pass_at_1: float = 0.0
    mean_pass_at_1: float = 0.0
    duration_s: float = 0.0
    error_taxonomy: ErrorTaxonomy = field(default_factory=ErrorTaxonomy)
    results: list[TaskResult] = field(default_factory=list)


def compile_toke(source_path: Path, compiler: str, output: Path,
                 timeout: int = 30) -> tuple[bool, str]:
    """Compile a .toke file to a native binary via tkc + clang."""
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
        ll_path = tmp.name

    try:
        # tkc emits LLVM IR to stdout
        result = subprocess.run(
            [compiler, str(source_path)],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout

        Path(ll_path).write_text(result.stdout)

        # clang compiles IR to native
        result = subprocess.run(
            ["clang", "-x", "ir", ll_path, "-o", str(output),
             "-lm"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, result.stderr
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "compilation timeout"
    finally:
        try:
            os.unlink(ll_path)
        except OSError:
            pass


def run_tests(binary: Path, test_file: Path,
              timeout: int = 10) -> tuple[int, int]:
    """Run a compiled binary against test cases. Returns (passed, total)."""
    with open(test_file) as f:
        tests = yaml.safe_load(f)

    if not tests or "test_cases" not in tests:
        return 0, 0

    cases = tests["test_cases"]
    passed = 0
    total = len(cases)

    for case in cases:
        input_val = json.dumps(case["input"]) if not isinstance(case["input"], str) else case["input"]
        expected = str(case["expected"]).strip()

        try:
            result = subprocess.run(
                [str(binary), input_val],
                capture_output=True, text=True, timeout=timeout
            )
            actual = result.stdout.strip()
            if actual == expected:
                passed += 1
        except (subprocess.TimeoutExpired, OSError):
            continue

    return passed, total


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric. n=samples, c=correct, k=attempts."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate(solutions_dir: Path, tests_dir: Path, compiler: str,
             timeout: int = 10) -> BenchmarkReport:
    """Run full pass@1 evaluation."""
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        compiler=compiler,
    )
    start = time.time()

    solution_files = sorted(solutions_dir.glob("*.toke"))
    report.tasks_total = len(solution_files)

    for sol_path in solution_files:
        task_id = sol_path.stem
        test_path = tests_dir / f"{task_id}.yaml"

        result = TaskResult(task_id=task_id)

        if not test_path.exists():
            report.results.append(result)
            continue

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            binary_path = Path(tmp.name)

        try:
            compiled, err = compile_toke(sol_path, compiler, binary_path,
                                         timeout=30)
            result.compiled = compiled
            result.compile_error = err

            if not compiled:
                cat = classify_error(err)
                result.error_category = cat
                setattr(report.error_taxonomy, cat,
                        getattr(report.error_taxonomy, cat) + 1)
            else:
                report.tasks_compiled += 1
                t0 = time.monotonic()
                passed, total = run_tests(binary_path, test_path, timeout)
                result.runtime_ms = (time.monotonic() - t0) * 1000
                result.tests_total = total
                result.tests_passed = passed
                result.pass_at_1 = 1.0 if passed == total and total > 0 else 0.0

                if result.pass_at_1 == 1.0:
                    report.tasks_passed += 1
                elif total > 0:
                    # Compiled but wrong output = logic error
                    result.error_category = "logic"
                    report.error_taxonomy.logic += 1
        finally:
            try:
                os.unlink(binary_path)
            except OSError:
                pass

        report.results.append(result)

        done = len(report.results)
        if done % 100 == 0:
            elapsed = time.time() - start
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = (report.tasks_total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{report.tasks_total}] "
                  f"{rate:.1f} tasks/min, ETA {remaining:.1f}min",
                  file=sys.stderr)

    report.duration_s = time.time() - start
    if report.tasks_compiled > 0:
        report.pass_at_1 = report.tasks_passed / report.tasks_compiled
    report.mean_pass_at_1 = report.pass_at_1

    return report


def main():
    parser = argparse.ArgumentParser(description="Pass@1 evaluation for toke")
    parser.add_argument("--solutions-dir", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="tkc")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    if not args.solutions_dir.is_dir():
        sys.exit(f"ERROR: solutions dir not found: {args.solutions_dir}")
    if not args.tests_dir.is_dir():
        sys.exit(f"ERROR: tests dir not found: {args.tests_dir}")

    report = evaluate(args.solutions_dir, args.tests_dir, args.compiler,
                      args.timeout)

    tax = report.error_taxonomy
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Tasks:     {report.tasks_total}", file=sys.stderr)
    print(f"  Compiled:  {report.tasks_compiled}/{report.tasks_total}", file=sys.stderr)
    print(f"  Pass@1:    {report.tasks_passed}/{report.tasks_compiled}", file=sys.stderr)
    print(f"  Mean:      {report.mean_pass_at_1:.4f}", file=sys.stderr)
    print(f"  Duration:  {report.duration_s:.1f}s", file=sys.stderr)
    if tax.total() > 0:
        print(f"\n  Error Taxonomy ({tax.total()} failures):", file=sys.stderr)
        for cat in ["syntax", "parse", "name", "type", "codegen",
                     "runtime", "logic", "unknown"]:
            count = getattr(tax, cat)
            if count > 0:
                print(f"    {cat:10s}  {count:4d}  "
                      f"({100*count/tax.total():.1f}%)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    output = json.dumps(asdict(report), indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

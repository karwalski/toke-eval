#!/usr/bin/env python3
"""Gate report generator — aggregates pass@k and token efficiency results.

Combines evaluation artifacts into a single gate decision report.

Usage:
    python -m toke_eval.report \\
        --pass-at-k results.json \\
        --token-efficiency efficiency.json \\
        --output gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GateCriterion:
    name: str
    threshold: str
    measured: str
    passed: bool


@dataclass
class GateReport:
    gate: str
    timestamp: str
    verdict: str  # PASS or FAIL
    criteria: list[GateCriterion]
    pass_at_k_source: str = ""
    token_efficiency_source: str = ""


def generate_gate1_report(pass_at_k_path: Path,
                          token_eff_path: Path) -> GateReport:
    """Generate a Gate 1 report from evaluation artifacts."""
    criteria = []

    # Token efficiency
    if token_eff_path and token_eff_path.exists():
        with open(token_eff_path) as f:
            eff = json.load(f)

        reduction = eff.get("reduction_vs_python", 0)
        toke_stats = eff.get("languages", {}).get("toke", {})
        mean_tokens = toke_stats.get("mean_tokens", 0)

        criteria.append(GateCriterion(
            name="Token reduction vs cl100k_base baseline",
            threshold="> 10%",
            measured=f"{reduction:.1%} (toke mean: {mean_tokens:.1f} tokens)",
            passed=reduction > 0.10,
        ))
    else:
        criteria.append(GateCriterion(
            name="Token reduction",
            threshold="> 10%",
            measured="not available",
            passed=False,
        ))

    # Pass@1
    if pass_at_k_path and pass_at_k_path.exists():
        with open(pass_at_k_path) as f:
            pak = json.load(f)

        pass_at_1 = pak.get("pass_at_1", pak.get("mean_pass_at_1", 0))
        compiled = pak.get("tasks_compiled", 0)
        passed = pak.get("tasks_passed", 0)
        total = pak.get("tasks_total", 0)

        criteria.append(GateCriterion(
            name="Pass@1 on held-out tasks",
            threshold=">= 60%",
            measured=f"{pass_at_1:.1%} ({passed}/{compiled} tasks, {total} total)",
            passed=pass_at_1 >= 0.60,
        ))
    else:
        criteria.append(GateCriterion(
            name="Pass@1",
            threshold=">= 60%",
            measured="not available",
            passed=False,
        ))

    all_passed = all(c.passed for c in criteria)

    return GateReport(
        gate="Gate 1",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        verdict="PASS" if all_passed else "FAIL",
        criteria=criteria,
        pass_at_k_source=str(pass_at_k_path) if pass_at_k_path else "",
        token_efficiency_source=str(token_eff_path) if token_eff_path else "",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate gate decision report")
    parser.add_argument("--pass-at-k", type=Path, default=None)
    parser.add_argument("--token-efficiency", type=Path, default=None)
    parser.add_argument("--gate", default="1", choices=["1", "2", "3", "4"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.gate == "1":
        report = generate_gate1_report(args.pass_at_k, args.token_efficiency)
    else:
        sys.exit(f"Gate {args.gate} report not yet implemented")

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Gate:     {report.gate}", file=sys.stderr)
    print(f"  Verdict:  {report.verdict}", file=sys.stderr)
    for c in report.criteria:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name}: {c.measured} (threshold: {c.threshold})",
              file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    output = json.dumps(asdict(report), indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

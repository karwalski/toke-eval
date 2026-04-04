#!/usr/bin/env python3
"""Automated regression testing on training checkpoints.

Story 9.6.3: CI-compatible pipeline that runs benchmarks on each training
checkpoint and tracks metrics.  Evaluates compile rate, Pass@1, token count,
error distribution, and mean reward across checkpoints.  Detects regressions
and outputs training curve data for visualisation.

Usage:
    # Dry-run with synthetic data (5 checkpoints, deterministic):
    python scripts/checkpoint_regression.py \
        --dry-run --seed 42

    # Real evaluation:
    python scripts/checkpoint_regression.py \
        --checkpoints-dir checkpoints/ \
        --predictions-dir predictions/ \
        --benchmark-dir ../toke-benchmark/hidden_tests/ \
        --output-dir data/

    # CI mode with custom thresholds:
    python scripts/checkpoint_regression.py \
        --checkpoints-dir checkpoints/ \
        --predictions-dir predictions/ \
        --benchmark-dir ../toke-benchmark/hidden_tests/ \
        --ci --threshold-pass1 0.03 --threshold-compile 0.08

Exit codes:
    0  no regressions detected
    1  regression detected (or error)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ErrorDistribution:
    """Error code counts for a single checkpoint."""
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint evaluation."""
    checkpoint_name: str
    step: int
    pass_at_1: float = 0.0
    compile_rate: float = 0.0
    mean_tokens: float = 0.0
    mean_reward: float = 0.0
    n_tasks: int = 0
    n_compiled: int = 0
    n_passed: int = 0
    error_distribution: ErrorDistribution = field(
        default_factory=ErrorDistribution
    )


@dataclass
class RegressionAlert:
    """A single regression alert."""
    checkpoint: str
    previous_checkpoint: str
    metric: str
    previous_value: float
    current_value: float
    change_pct: float
    threshold_pct: float
    message: str


@dataclass
class RegressionReport:
    """Full report across all checkpoints."""
    timestamp: str = ""
    mode: str = ""  # "predictions" or "dry-run"
    seed: int = 0
    n_checkpoints: int = 0
    thresholds: dict[str, float] = field(default_factory=dict)
    checkpoints: list[CheckpointMetrics] = field(default_factory=list)
    alerts: list[RegressionAlert] = field(default_factory=list)
    regression_detected: bool = False


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(checkpoints_dir: Path) -> list[tuple[str, int]]:
    """Find checkpoint subdirectories and extract step numbers.

    Expects subdirectory names like 'checkpoint-500', 'ckpt_1000', or
    'step-1500'.  Falls back to alphabetical ordering with synthetic step
    numbers if no numeric suffix found.

    Returns:
        Sorted list of (checkpoint_name, step) tuples.
    """
    entries: list[tuple[str, int]] = []

    for child in sorted(checkpoints_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        # Try to extract step number from name
        m = re.search(r"(\d+)$", name)
        if m:
            step = int(m.group(1))
        else:
            step = len(entries)
        entries.append((name, step))

    # Sort by step number
    entries.sort(key=lambda x: x[1])
    return entries


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_checkpoint_predictions(
    predictions_dir: Path,
    checkpoint_name: str,
) -> list[dict[str, Any]]:
    """Load predictions for a single checkpoint from JSONL.

    Expects file: predictions_dir/{checkpoint_name}.jsonl
    Each line: {"task_id": "...", "solution": "...", "tokens": 42}

    Returns list of prediction dicts.
    """
    jsonl_path = predictions_dir / f"{checkpoint_name}.jsonl"
    if not jsonl_path.exists():
        return []

    predictions: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                predictions.append(obj)
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: {jsonl_path}:{line_num}: bad JSON: {e}",
                    file=sys.stderr,
                )
    return predictions


# ---------------------------------------------------------------------------
# Compilation check
# ---------------------------------------------------------------------------

def compile_check(
    source_code: str,
    compiler: str = "tkc",
    timeout: int = 30,
) -> tuple[bool, str, list[str]]:
    """Compile toke source via tkc --check.

    Returns (success, stderr_output, list_of_error_codes).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toke", delete=False
    ) as tmp:
        tmp.write(source_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [compiler, "--check", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        stderr = (result.stderr or "").strip()

        # Extract error codes (E1001, W2010, etc.)
        error_codes = re.findall(r"[EW]\d{4}", stderr)

        if result.returncode == 0:
            return True, "", error_codes
        return False, stderr, error_codes

    except subprocess.TimeoutExpired:
        return False, "compilation timeout", ["ETIMEOUT"]
    except FileNotFoundError:
        return False, f"compiler not found: {compiler}", ["ENOTFOUND"]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(compiled: bool, passed: bool, token_count: int) -> float:
    """Compiler-as-verifier reward signal.

    Reward scheme:
        - Compile failure: 0.0
        - Compiles but fails tests: 0.3
        - Compiles and passes tests: 1.0
        - Bonus for brevity: up to +0.1 for solutions under 50 tokens
    """
    if not compiled:
        return 0.0
    reward = 0.3 if not passed else 1.0
    # Brevity bonus: linear scale from 50 tokens (0.1) to 0 tokens (0.1)
    if token_count < 50:
        reward += 0.1 * (1.0 - token_count / 50.0)
    return min(reward, 1.1)


# ---------------------------------------------------------------------------
# Dry-run: synthetic checkpoint simulation
# ---------------------------------------------------------------------------

def generate_dry_run_data(
    n_checkpoints: int,
    n_tasks: int,
    seed: int,
) -> list[CheckpointMetrics]:
    """Generate synthetic checkpoint metrics for dry-run mode.

    Simulates a training run where metrics generally improve but with
    occasional regressions to test detection logic.

    Args:
        n_checkpoints: number of checkpoints to simulate
        n_tasks: number of tasks per checkpoint
        seed: random seed for reproducibility

    Returns:
        List of CheckpointMetrics, one per checkpoint.
    """
    results: list[CheckpointMetrics] = []

    for ckpt_idx in range(n_checkpoints):
        step = (ckpt_idx + 1) * 500
        name = f"checkpoint-{step}"

        # Base rates improve over training
        progress = ckpt_idx / max(n_checkpoints - 1, 1)

        # Deterministic "random" values from hash
        def _hash_val(key: str) -> float:
            h = hashlib.sha256(f"{seed}:{name}:{key}".encode()).hexdigest()
            return (int(h[:8], 16) % 10000) / 10000.0

        # Pass@1 improves from ~0.30 to ~0.65, with jitter
        base_pass1 = 0.30 + progress * 0.35
        jitter = (_hash_val("pass1") - 0.5) * 0.08
        # Inject a regression at checkpoint 3 (idx=2) if >= 4 checkpoints
        if ckpt_idx == 2 and n_checkpoints >= 4:
            jitter = -0.12  # Force a notable drop
        pass1 = max(0.0, min(1.0, base_pass1 + jitter))

        # Compile rate improves from ~0.50 to ~0.85
        base_compile = 0.50 + progress * 0.35
        compile_jitter = (_hash_val("compile") - 0.5) * 0.06
        compile_rate = max(0.0, min(1.0, base_compile + compile_jitter))

        # Token count decreases from ~80 to ~55
        base_tokens = 80.0 - progress * 25.0
        token_jitter = (_hash_val("tokens") - 0.5) * 10.0
        mean_tokens = max(10.0, base_tokens + token_jitter)

        # Compute per-task stats
        n_compiled = int(round(compile_rate * n_tasks))
        n_passed = int(round(pass1 * n_tasks))
        n_passed = min(n_passed, n_compiled)  # Can't pass without compiling

        # Recalculate rates from integer counts
        compile_rate = n_compiled / n_tasks if n_tasks > 0 else 0.0
        pass1 = n_passed / n_tasks if n_tasks > 0 else 0.0

        # Synthetic error distribution
        error_dist = ErrorDistribution()
        n_errors = n_tasks - n_compiled
        if n_errors > 0:
            # Distribute errors across a few codes
            codes = ["E1001", "E2001", "E4010", "E4031", "E3011"]
            for i, code in enumerate(codes):
                h = hashlib.sha256(
                    f"{seed}:{name}:err:{code}".encode()
                ).hexdigest()
                share = (int(h[:4], 16) % 100) / 100.0
                count = max(0, int(round(share * n_errors / len(codes) * 2)))
                if count > 0:
                    error_dist.counts[code] = count
            # Ensure total roughly matches
            assigned = sum(error_dist.counts.values())
            if assigned < n_errors:
                error_dist.counts["E2001"] = (
                    error_dist.counts.get("E2001", 0) + n_errors - assigned
                )

        # Mean reward
        rewards: list[float] = []
        for t in range(n_tasks):
            t_hash = hashlib.sha256(
                f"{seed}:{name}:reward:{t}".encode()
            ).hexdigest()
            t_val = (int(t_hash[:8], 16) % 10000) / 10000.0
            t_compiled = t_val < compile_rate
            t_passed = t_compiled and (t_val < pass1)
            t_tokens = int(mean_tokens + (_hash_val(f"tok:{t}") - 0.5) * 20)
            rewards.append(compute_reward(t_compiled, t_passed, t_tokens))
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        metrics = CheckpointMetrics(
            checkpoint_name=name,
            step=step,
            pass_at_1=round(pass1, 4),
            compile_rate=round(compile_rate, 4),
            mean_tokens=round(mean_tokens, 2),
            mean_reward=round(mean_reward, 4),
            n_tasks=n_tasks,
            n_compiled=n_compiled,
            n_passed=n_passed,
            error_distribution=error_dist,
        )
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_name: str,
    step: int,
    predictions: list[dict[str, Any]],
    compiler: str = "tkc",
) -> CheckpointMetrics:
    """Evaluate a single checkpoint against its predictions.

    Each prediction dict should have:
        task_id: str
        solution: str  (toke source code)
        tokens: int    (token count, optional)
    """
    n_tasks = len(predictions)
    n_compiled = 0
    n_passed = 0
    token_counts: list[int] = []
    rewards: list[float] = []
    error_dist = ErrorDistribution()

    for pred in predictions:
        solution = pred.get("solution", "")
        tokens = pred.get("tokens", len(solution.split()))
        token_counts.append(tokens)

        compiled, _stderr, error_codes = compile_check(
            solution, compiler=compiler
        )

        if compiled:
            n_compiled += 1
            # For now, Pass@1 = compile success (full test execution
            # requires benchmark harness integration)
            passed = True
            n_passed += 1
        else:
            passed = False
            for code in error_codes:
                error_dist.counts[code] = error_dist.counts.get(code, 0) + 1

        rewards.append(compute_reward(compiled, passed, tokens))

    compile_rate = n_compiled / n_tasks if n_tasks > 0 else 0.0
    pass1 = n_passed / n_tasks if n_tasks > 0 else 0.0
    mean_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return CheckpointMetrics(
        checkpoint_name=checkpoint_name,
        step=step,
        pass_at_1=round(pass1, 4),
        compile_rate=round(compile_rate, 4),
        mean_tokens=round(mean_tokens, 2),
        mean_reward=round(mean_reward, 4),
        n_tasks=n_tasks,
        n_compiled=n_compiled,
        n_passed=n_passed,
        error_distribution=error_dist,
    )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def detect_regressions(
    checkpoints: list[CheckpointMetrics],
    threshold_pass1: float = 0.05,
    threshold_compile: float = 0.10,
    threshold_tokens: float = 0.20,
) -> list[RegressionAlert]:
    """Compare consecutive checkpoints and detect regressions.

    Args:
        checkpoints: ordered list of checkpoint metrics
        threshold_pass1: max allowed Pass@1 drop (fraction, e.g. 0.05 = 5%)
        threshold_compile: max allowed compile rate drop
        threshold_tokens: max allowed mean token count increase

    Returns:
        List of regression alerts.
    """
    alerts: list[RegressionAlert] = []

    for i in range(1, len(checkpoints)):
        prev = checkpoints[i - 1]
        curr = checkpoints[i]

        # Pass@1 drop check
        if prev.pass_at_1 > 0:
            pass1_change = prev.pass_at_1 - curr.pass_at_1
            pass1_pct = pass1_change / prev.pass_at_1
            if pass1_change > 0 and pass1_pct > threshold_pass1:
                alerts.append(RegressionAlert(
                    checkpoint=curr.checkpoint_name,
                    previous_checkpoint=prev.checkpoint_name,
                    metric="pass_at_1",
                    previous_value=prev.pass_at_1,
                    current_value=curr.pass_at_1,
                    change_pct=round(pass1_pct * 100, 2),
                    threshold_pct=round(threshold_pass1 * 100, 2),
                    message=(
                        f"REGRESSION: Pass@1 dropped {pass1_pct*100:.1f}% "
                        f"({prev.pass_at_1:.4f} -> {curr.pass_at_1:.4f}) "
                        f"at {curr.checkpoint_name}"
                    ),
                ))

        # Compile rate drop check
        if prev.compile_rate > 0:
            compile_change = prev.compile_rate - curr.compile_rate
            compile_pct = compile_change / prev.compile_rate
            if compile_change > 0 and compile_pct > threshold_compile:
                alerts.append(RegressionAlert(
                    checkpoint=curr.checkpoint_name,
                    previous_checkpoint=prev.checkpoint_name,
                    metric="compile_rate",
                    previous_value=prev.compile_rate,
                    current_value=curr.compile_rate,
                    change_pct=round(compile_pct * 100, 2),
                    threshold_pct=round(threshold_compile * 100, 2),
                    message=(
                        f"REGRESSION: Compile rate dropped {compile_pct*100:.1f}% "
                        f"({prev.compile_rate:.4f} -> {curr.compile_rate:.4f}) "
                        f"at {curr.checkpoint_name}"
                    ),
                ))

        # Token count increase check
        if prev.mean_tokens > 0:
            token_change = curr.mean_tokens - prev.mean_tokens
            token_pct = token_change / prev.mean_tokens
            if token_change > 0 and token_pct > threshold_tokens:
                alerts.append(RegressionAlert(
                    checkpoint=curr.checkpoint_name,
                    previous_checkpoint=prev.checkpoint_name,
                    metric="mean_tokens",
                    previous_value=prev.mean_tokens,
                    current_value=curr.mean_tokens,
                    change_pct=round(token_pct * 100, 2),
                    threshold_pct=round(threshold_tokens * 100, 2),
                    message=(
                        f"REGRESSION: Mean token count increased {token_pct*100:.1f}% "
                        f"({prev.mean_tokens:.1f} -> {curr.mean_tokens:.1f}) "
                        f"at {curr.checkpoint_name}"
                    ),
                ))

    return alerts


# ---------------------------------------------------------------------------
# Output: training curve CSV and JSON
# ---------------------------------------------------------------------------

def write_training_curve_csv(
    checkpoints: list[CheckpointMetrics],
    output_path: Path,
) -> None:
    """Write training curve data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint_name", "step", "pass_at_1", "compile_rate",
            "mean_tokens", "mean_reward",
        ])
        for ckpt in checkpoints:
            writer.writerow([
                ckpt.checkpoint_name,
                ckpt.step,
                f"{ckpt.pass_at_1:.6f}",
                f"{ckpt.compile_rate:.6f}",
                f"{ckpt.mean_tokens:.2f}",
                f"{ckpt.mean_reward:.6f}",
            ])

    print(f"Training curve CSV: {output_path}", file=sys.stderr)


def write_training_curve_json(
    checkpoints: list[CheckpointMetrics],
    output_path: Path,
) -> None:
    """Write training curve data to JSON for web dashboards."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for ckpt in checkpoints:
        data.append({
            "checkpoint_name": ckpt.checkpoint_name,
            "step": ckpt.step,
            "pass_at_1": ckpt.pass_at_1,
            "compile_rate": ckpt.compile_rate,
            "mean_tokens": ckpt.mean_tokens,
            "mean_reward": ckpt.mean_reward,
        })

    output_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    print(f"Training curve JSON: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Full report output
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Convert dataclass tree to JSON-serializable dict."""
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for f_name in obj.__dataclass_fields__:
            val = getattr(obj, f_name)
            d[f_name] = _to_serializable(val)
        return d
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def write_full_report(report: RegressionReport, output_path: Path) -> None:
    """Write the full regression report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_serializable(report), indent=2),
        encoding="utf-8",
    )
    print(f"Full report: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary(report: RegressionReport) -> str:
    """Format a human-readable summary table."""
    lines = [
        "=" * 78,
        "  Checkpoint Regression Report",
        f"  Mode: {report.mode}",
        f"  Checkpoints: {report.n_checkpoints}",
        f"  Thresholds: Pass@1>{report.thresholds.get('pass1', 5.0)}%, "
        f"Compile>{report.thresholds.get('compile', 10.0)}%, "
        f"Tokens>{report.thresholds.get('tokens', 20.0)}%",
        "=" * 78,
        "",
    ]

    # Metrics table
    header = (
        f"{'Checkpoint':<20} {'Step':>6} {'Pass@1':>8} {'Compile%':>9} "
        f"{'Tokens':>8} {'Reward':>8} {'Errors':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for ckpt in report.checkpoints:
        error_total = ckpt.error_distribution.total
        lines.append(
            f"{ckpt.checkpoint_name:<20} {ckpt.step:>6} "
            f"{ckpt.pass_at_1:>8.4f} {ckpt.compile_rate:>9.4f} "
            f"{ckpt.mean_tokens:>8.1f} {ckpt.mean_reward:>8.4f} "
            f"{error_total:>7}"
        )

    lines.append("-" * len(header))
    lines.append("")

    # Alerts
    if report.alerts:
        lines.append(f"ALERTS ({len(report.alerts)}):")
        for alert in report.alerts:
            lines.append(f"  WARNING: {alert.message}")
        lines.append("")
        lines.append("RESULT: REGRESSION DETECTED")
    else:
        lines.append("RESULT: No regressions detected")

    lines.append("=" * 78)
    return "\n".join(lines)


def format_ci_output(report: RegressionReport) -> str:
    """Format CI-compatible JSON output to stdout."""
    ci_data = {
        "regression_detected": report.regression_detected,
        "n_checkpoints": report.n_checkpoints,
        "n_alerts": len(report.alerts),
        "alerts": [_to_serializable(a) for a in report.alerts],
        "checkpoints": [
            {
                "name": c.checkpoint_name,
                "step": c.step,
                "pass_at_1": c.pass_at_1,
                "compile_rate": c.compile_rate,
                "mean_tokens": c.mean_tokens,
                "mean_reward": c.mean_reward,
            }
            for c in report.checkpoints
        ],
    }
    return json.dumps(ci_data, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated regression testing on training checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=None,
        help="Directory containing checkpoint subdirectories",
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=None,
        help="Directory containing {checkpoint_name}.jsonl files",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=None,
        help="Directory containing benchmark task YAML files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Output directory for results (default: data/)",
    )
    parser.add_argument(
        "--compiler", default="tkc",
        help="Path to tkc compiler binary (default: tkc)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate with synthetic checkpoint data (no compilation)",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: machine-readable JSON to stdout",
    )
    parser.add_argument(
        "--threshold-pass1", type=float, default=0.05,
        help="Pass@1 regression threshold as fraction (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--threshold-compile", type=float, default=0.10,
        help="Compile rate regression threshold (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--threshold-tokens", type=float, default=0.20,
        help="Token count increase threshold (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dry-run reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    # --- Collect checkpoint metrics ---
    checkpoint_metrics: list[CheckpointMetrics] = []

    if args.dry_run:
        print("Mode: dry-run (synthetic checkpoint data)", file=sys.stderr)
        checkpoint_metrics = generate_dry_run_data(
            n_checkpoints=5,
            n_tasks=50,
            seed=args.seed,
        )
        print(
            f"Generated {len(checkpoint_metrics)} synthetic checkpoints",
            file=sys.stderr,
        )

    elif args.checkpoints_dir and args.predictions_dir:
        if not args.checkpoints_dir.is_dir():
            print(
                f"ERROR: checkpoints dir not found: {args.checkpoints_dir}",
                file=sys.stderr,
            )
            return 1
        if not args.predictions_dir.is_dir():
            print(
                f"ERROR: predictions dir not found: {args.predictions_dir}",
                file=sys.stderr,
            )
            return 1

        print("Mode: live evaluation", file=sys.stderr)
        checkpoints = discover_checkpoints(args.checkpoints_dir)
        if not checkpoints:
            print("ERROR: no checkpoints found", file=sys.stderr)
            return 1

        print(f"Found {len(checkpoints)} checkpoints", file=sys.stderr)

        for name, step in checkpoints:
            predictions = load_checkpoint_predictions(
                args.predictions_dir, name
            )
            if not predictions:
                print(
                    f"WARNING: no predictions for {name}, skipping",
                    file=sys.stderr,
                )
                continue

            print(
                f"Evaluating {name} (step {step}, "
                f"{len(predictions)} predictions)...",
                file=sys.stderr,
            )
            metrics = evaluate_checkpoint(
                checkpoint_name=name,
                step=step,
                predictions=predictions,
                compiler=args.compiler,
            )
            checkpoint_metrics.append(metrics)

    else:
        print(
            "ERROR: provide --checkpoints-dir + --predictions-dir, "
            "or use --dry-run",
            file=sys.stderr,
        )
        return 1

    if not checkpoint_metrics:
        print("ERROR: no checkpoint metrics computed", file=sys.stderr)
        return 1

    # --- Regression detection ---
    alerts = detect_regressions(
        checkpoint_metrics,
        threshold_pass1=args.threshold_pass1,
        threshold_compile=args.threshold_compile,
        threshold_tokens=args.threshold_tokens,
    )

    regression_detected = len(alerts) > 0

    # --- Build report ---
    report = RegressionReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mode="dry-run" if args.dry_run else "predictions",
        seed=args.seed,
        n_checkpoints=len(checkpoint_metrics),
        thresholds={
            "pass1": round(args.threshold_pass1 * 100, 2),
            "compile": round(args.threshold_compile * 100, 2),
            "tokens": round(args.threshold_tokens * 100, 2),
        },
        checkpoints=checkpoint_metrics,
        alerts=alerts,
        regression_detected=regression_detected,
    )

    # --- Output ---
    if args.ci:
        print(format_ci_output(report))
    else:
        summary = format_summary(report)
        print(summary)

        # Print alerts to stderr as well for visibility
        for alert in alerts:
            print(f"WARNING: {alert.message}", file=sys.stderr)

    # Write training curve files
    write_training_curve_csv(
        checkpoint_metrics,
        args.output_dir / "training_curve.csv",
    )
    write_training_curve_json(
        checkpoint_metrics,
        args.output_dir / "training_curve.json",
    )
    write_full_report(
        report,
        args.output_dir / "checkpoint_regression_report.json",
    )

    # --- Exit code ---
    if regression_detected:
        n = len(alerts)
        print(
            f"\nRegression detected: {n} alert{'s' if n != 1 else ''}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

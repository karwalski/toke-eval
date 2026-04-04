#!/usr/bin/env python3
"""Error-code-aware reward shaping for GRPO training.

Story 9.5.2: Compares flat (binary compile/no-compile) reward against shaped
reward that assigns monotonic partial credit based on how far code progresses
through the compiler pipeline.

Severity tiers (highest to lowest severity — earlier failure = lower reward):
  E1xxx  lex errors      — reward 0.10  (fails at tokenisation)
  E2xxx  parse errors    — reward 0.25  (past lexer, fails at parse)
  E3xxx  name errors     — reward 0.40  (past parser, fails at name resolution)
  E4xxx  type errors     — reward 0.55  (past names, fails at type checking)
  E5xxx  semantic errors — reward 0.70  (past types, fails at semantic analysis)
  clean compile          — reward 1.00

Flat baseline:
  compile ok  → 1.0
  compile fail → 0.0

The script runs both reward functions over the same task corpus and outputs a
JSON comparison report with per-tier statistics, reward distributions, and a
recommendation.

Usage::

    # Dry-run with synthetic diagnostics:
    python scripts/error_reward_shaping.py \
        --corpus-path data/predictions.jsonl \
        --output data/shaped_reward_report.json \
        --dry-run --seed 42

    # Real evaluation:
    python scripts/error_reward_shaping.py \
        --corpus-path data/predictions.jsonl \
        --tkc-path ../tkc/build/tkc \
        --output data/shaped_reward_report.json

Exit codes:
    0  success
    1  error
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Error-tier definitions
# ---------------------------------------------------------------------------

# Maps error-code prefix to (tier_name, shaped_reward).
# Monotonic: code that reaches later stages earns higher partial credit.
ERROR_TIERS: dict[str, tuple[str, float]] = {
    "E1": ("lex", 0.10),
    "E2": ("parse", 0.25),
    "E3": ("name", 0.40),
    "E4": ("type", 0.55),
    "E5": ("semantic", 0.70),
}

TIER_ORDER = ["lex", "parse", "name", "type", "semantic", "clean"]
REWARD_CLEAN = 1.0
REWARD_FLAT_PASS = 1.0
REWARD_FLAT_FAIL = 0.0

# ---------------------------------------------------------------------------
# Error code classification
# ---------------------------------------------------------------------------


def classify_error_code(code: str) -> tuple[str, float]:
    """Return (tier_name, shaped_reward) for a diagnostic error code.

    Falls back to the most severe tier (lex) for unrecognised codes.
    """
    if not code:
        return ("lex", ERROR_TIERS["E1"][1])

    for prefix, (tier_name, reward) in ERROR_TIERS.items():
        if code.startswith(prefix):
            return (tier_name, reward)

    # Warning codes (W-prefixed) or unknown — treat as non-error
    if code.startswith("W"):
        return ("clean", REWARD_CLEAN)

    # Unknown error code — treat as lex-level (worst case)
    return ("lex", ERROR_TIERS["E1"][1])


def highest_stage_reached(diagnostics: list[dict]) -> tuple[str, float]:
    """Given a list of diagnostics, find the *latest* pipeline stage reached.

    The latest stage with an error determines the shaped reward — if code
    fails at type checking, it means it passed lex, parse, and name
    resolution, so it gets the type-error reward (higher than lex/parse).

    Returns (tier_name, shaped_reward).  If there are no error diagnostics,
    returns ("clean", 1.0).
    """
    if not diagnostics:
        return ("clean", REWARD_CLEAN)

    # Filter to errors only (ignore warnings)
    errors = [d for d in diagnostics if d.get("severity") == "error"]
    if not errors:
        return ("clean", REWARD_CLEAN)

    best_tier = "lex"
    best_reward = ERROR_TIERS["E1"][1]

    for diag in errors:
        code = diag.get("code", "")
        tier_name, reward = classify_error_code(code)
        if tier_name == "clean":
            continue
        if reward > best_reward:
            best_tier = tier_name
            best_reward = reward

    return (best_tier, best_reward)


# ---------------------------------------------------------------------------
# Compiler interaction
# ---------------------------------------------------------------------------


def run_tkc_check(source: str, tkc_path: str) -> dict:
    """Run ``tkc --check --diag-json`` and return parsed result."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toke", delete=False,
    ) as f:
        f.write(source)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [tkc_path, "--check", "--diag-json", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "returncode": 3,
            "diagnostics": [{"code": "E1000", "severity": "error",
                             "message": str(exc)}],
            "compile_ok": False,
        }
    finally:
        os.unlink(tmp_path)

    diagnostics: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            diagnostics.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return {
        "returncode": result.returncode,
        "diagnostics": diagnostics,
        "compile_ok": result.returncode == 0,
    }


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

# Simulated error codes weighted by realistic frequency
_SIM_ERROR_CODES = [
    # lex errors
    ("E1001", 8), ("E1002", 4), ("E1010", 3),
    # parse errors
    ("E2001", 15), ("E2003", 10), ("E2010", 6), ("E2020", 4),
    # name errors
    ("E3001", 12), ("E3011", 7), ("E3020", 3),
    # type errors
    ("E4001", 10), ("E4010", 8), ("E4020", 5),
    # semantic errors
    ("E5001", 6), ("E5010", 4), ("E5020", 2),
]


def _weighted_choice(items: list[tuple[str, int]], rng: random.Random) -> str:
    codes, weights = zip(*items)
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for code, w in items:
        cumulative += w
        if r <= cumulative:
            return code
    return items[-1][0]


def simulate_check(source: str, rng: random.Random) -> dict:
    """Heuristic-based compilation simulation for dry-run mode."""
    lines = source.strip().splitlines()
    opens = source.count("(") + source.count("{") + source.count("[")
    closes = source.count(")") + source.count("}") + source.count("]")
    balanced = opens == closes

    has_fn = "fn " in source or "fn(" in source
    reasonable_length = 5 <= len(lines) <= 200

    score = 0.0
    if balanced:
        score += 0.35
    if has_fn:
        score += 0.2
    if reasonable_length:
        score += 0.2
    if "lp " in source or "if " in source:
        score += 0.1
    score += rng.uniform(-0.1, 0.1)
    score = max(0.0, min(1.0, score))

    roll = rng.random()
    if roll < score * 0.5:
        # Clean compile
        return {
            "returncode": 0,
            "diagnostics": [],
            "compile_ok": True,
        }
    else:
        # Failure — pick 1-3 error diagnostics
        n_errors = rng.randint(1, 3)
        diags = []
        for _ in range(n_errors):
            code = _weighted_choice(_SIM_ERROR_CODES, rng)
            diags.append({
                "code": code,
                "severity": "error",
                "message": f"simulated {code}",
            })
        # Sometimes add a warning too
        if rng.random() < 0.3:
            diags.append({
                "code": "W1010",
                "severity": "warning",
                "message": "simulated warning",
            })
        return {
            "returncode": 1,
            "diagnostics": diags,
            "compile_ok": False,
        }


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


def compute_flat_reward(compile_result: dict) -> float:
    """Binary flat reward: 1.0 if compiles, 0.0 otherwise."""
    return REWARD_FLAT_PASS if compile_result["compile_ok"] else REWARD_FLAT_FAIL


def compute_shaped_reward(compile_result: dict) -> tuple[float, str]:
    """Shaped reward based on error tier. Returns (reward, tier_name)."""
    if compile_result["compile_ok"]:
        return (REWARD_CLEAN, "clean")
    tier_name, reward = highest_stage_reached(compile_result["diagnostics"])
    return (reward, tier_name)


# ---------------------------------------------------------------------------
# Corpus loading and evaluation
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: Path) -> list[dict]:
    """Load JSONL corpus. Each line: {"task_id": "...", "source": "..."}."""
    tasks: list[dict] = []
    with open(corpus_path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  warning: skipping malformed line {line_no}",
                      file=sys.stderr)
                continue
            tasks.append({
                "task_id": obj.get("task_id", f"task_{line_no}"),
                "source": obj.get("source", ""),
            })
    return tasks


def evaluate_corpus(
    tasks: list[dict],
    tkc_path: str,
    dry_run: bool,
    seed: int | None,
    max_tasks: int | None = None,
) -> list[dict]:
    """Evaluate tasks with both flat and shaped reward functions."""
    rng = random.Random(seed)
    if max_tasks is not None and max_tasks < len(tasks):
        tasks = rng.sample(tasks, max_tasks)

    results: list[dict] = []
    for i, task in enumerate(tasks):
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i + 1}/{len(tasks)} tasks...",
                  file=sys.stderr)

        source = task["source"]

        if dry_run:
            compile_result = simulate_check(source, rng)
        else:
            compile_result = run_tkc_check(source, tkc_path)

        flat = compute_flat_reward(compile_result)
        shaped, tier = compute_shaped_reward(compile_result)

        results.append({
            "task_id": task["task_id"],
            "flat_reward": flat,
            "shaped_reward": shaped,
            "tier": tier,
            "compile_ok": compile_result["compile_ok"],
            "diagnostics": compile_result["diagnostics"],
        })

    return results


# ---------------------------------------------------------------------------
# Statistics and comparison report
# ---------------------------------------------------------------------------


@dataclass
class TierStats:
    tier: str
    count: int = 0
    flat_mean: float = 0.0
    shaped_mean: float = 0.0
    flat_sum: float = 0.0
    shaped_sum: float = 0.0

    def finalize(self) -> None:
        if self.count > 0:
            self.flat_mean = self.flat_sum / self.count
            self.shaped_mean = self.shaped_sum / self.count


def build_report(results: list[dict]) -> dict:
    """Build a comparison report between flat and shaped reward."""
    n = len(results)
    if n == 0:
        return {"error": "no results", "n": 0}

    # Per-tier statistics
    tier_map: dict[str, TierStats] = {}
    for tier_name in TIER_ORDER:
        tier_map[tier_name] = TierStats(tier=tier_name)

    flat_rewards: list[float] = []
    shaped_rewards: list[float] = []

    for r in results:
        tier = r["tier"]
        if tier not in tier_map:
            tier_map[tier] = TierStats(tier=tier)
        ts = tier_map[tier]
        ts.count += 1
        ts.flat_sum += r["flat_reward"]
        ts.shaped_sum += r["shaped_reward"]
        flat_rewards.append(r["flat_reward"])
        shaped_rewards.append(r["shaped_reward"])

    for ts in tier_map.values():
        ts.finalize()

    flat_mean = sum(flat_rewards) / n
    shaped_mean = sum(shaped_rewards) / n

    # Variance for reward signal quality
    flat_var = sum((x - flat_mean) ** 2 for x in flat_rewards) / n
    shaped_var = sum((x - shaped_mean) ** 2 for x in shaped_rewards) / n

    # Reward differentiation: how many distinct reward values does each produce?
    flat_distinct = len(set(flat_rewards))
    shaped_distinct = len(set(round(r, 4) for r in shaped_rewards))

    # Spearman rank correlation between shaped reward and pipeline depth
    # (higher tier index should correlate with higher shaped reward)
    tier_to_rank = {t: i for i, t in enumerate(TIER_ORDER)}

    # Build improvement assessment
    improvement = shaped_var > flat_var and shaped_distinct > flat_distinct
    delta_mean = shaped_mean - flat_mean

    # Per-tier breakdown for report
    tier_breakdown = []
    for tier_name in TIER_ORDER:
        ts = tier_map[tier_name]
        if ts.count > 0:
            tier_breakdown.append({
                "tier": ts.tier,
                "count": ts.count,
                "flat_mean": round(ts.flat_mean, 4),
                "shaped_mean": round(ts.shaped_mean, 4),
                "delta": round(ts.shaped_mean - ts.flat_mean, 4),
            })

    # Reward distribution for each method
    flat_dist = dict(sorted(Counter(round(r, 2) for r in flat_rewards).items()))
    shaped_dist = dict(sorted(
        Counter(round(r, 2) for r in shaped_rewards).items(),
    ))

    report = {
        "n_tasks": n,
        "flat_reward": {
            "mean": round(flat_mean, 4),
            "variance": round(flat_var, 4),
            "distinct_values": flat_distinct,
            "distribution": {str(k): v for k, v in flat_dist.items()},
        },
        "shaped_reward": {
            "mean": round(shaped_mean, 4),
            "variance": round(shaped_var, 4),
            "distinct_values": shaped_distinct,
            "distribution": {str(k): v for k, v in shaped_dist.items()},
        },
        "per_tier": tier_breakdown,
        "delta_mean_reward": round(delta_mean, 4),
        "shaped_has_more_signal": shaped_distinct > flat_distinct,
        "shaped_has_higher_variance": shaped_var > flat_var,
        "recommendation": (
            "shaped reward provides richer gradient signal"
            if improvement
            else "no clear improvement from shaped reward — flat reward sufficient"
        ),
    }

    return report


def print_report_summary(report: dict, file=sys.stdout) -> None:
    """Print human-readable summary of the comparison report."""
    print("=" * 60, file=file)
    print("Error-Code-Aware Reward Shaping — A/B Comparison", file=file)
    print("=" * 60, file=file)
    print(f"Tasks evaluated: {report['n_tasks']}", file=file)
    print(file=file)

    print("--- Flat Reward (binary) ---", file=file)
    fr = report["flat_reward"]
    print(f"  Mean:      {fr['mean']:.4f}", file=file)
    print(f"  Variance:  {fr['variance']:.4f}", file=file)
    print(f"  Distinct:  {fr['distinct_values']}", file=file)
    print(file=file)

    print("--- Shaped Reward (error-tier) ---", file=file)
    sr = report["shaped_reward"]
    print(f"  Mean:      {sr['mean']:.4f}", file=file)
    print(f"  Variance:  {sr['variance']:.4f}", file=file)
    print(f"  Distinct:  {sr['distinct_values']}", file=file)
    print(file=file)

    print("--- Per-Tier Breakdown ---", file=file)
    print(f"  {'Tier':<10s} {'Count':>6s} {'Flat':>8s} {'Shaped':>8s} {'Delta':>8s}",
          file=file)
    for t in report.get("per_tier", []):
        print(
            f"  {t['tier']:<10s} {t['count']:>6d} {t['flat_mean']:>8.4f} "
            f"{t['shaped_mean']:>8.4f} {t['delta']:>+8.4f}",
            file=file,
        )
    print(file=file)

    print(f"Delta mean reward: {report['delta_mean_reward']:+.4f}", file=file)
    print(f"More signal:       {report['shaped_has_more_signal']}", file=file)
    print(f"Higher variance:   {report['shaped_has_higher_variance']}", file=file)
    print(file=file)
    print(f"Recommendation: {report['recommendation']}", file=file)
    print("=" * 60, file=file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Error-code-aware reward shaping: A/B comparison of flat vs "
            "shaped reward for GRPO training."
        ),
    )
    p.add_argument(
        "--corpus-path", type=Path, required=True,
        help="JSONL corpus with {task_id, source} per line.",
    )
    p.add_argument(
        "--tkc-path", type=str, default="tkc",
        help="Path to the tkc compiler binary (default: tkc).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output path for JSON comparison report.",
    )
    p.add_argument(
        "--max-tasks", type=int, default=500,
        help="Maximum tasks to evaluate (default: 500).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Simulate compilation outcomes via heuristics.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for dry-run simulation and task sampling.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.corpus_path.exists():
        print(f"error: corpus file not found: {args.corpus_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus from {args.corpus_path}...", file=sys.stderr)
    tasks = load_corpus(args.corpus_path)
    print(f"Loaded {len(tasks)} tasks.", file=sys.stderr)

    if not tasks:
        print("error: no tasks found in corpus.", file=sys.stderr)
        sys.exit(1)

    # Evaluate with both reward functions
    print(
        f"Evaluating up to {args.max_tasks} tasks "
        f"({'dry-run' if args.dry_run else 'live'})...",
        file=sys.stderr,
    )
    results = evaluate_corpus(
        tasks=tasks,
        tkc_path=args.tkc_path,
        dry_run=args.dry_run,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )

    # Build comparison report
    report = build_report(results)

    # Write JSON report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"\nWrote report to {args.output}\n", file=sys.stderr)

    # Print human-readable summary
    print_report_summary(report)


if __name__ == "__main__":
    main()

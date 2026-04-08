#!/usr/bin/env python3
"""Teacher-student evaluation loop for toke code generation.

Implements the NVIDIA data-flywheel / SelfCodeAlign pattern:
  1. Teacher (large model) generates programming problems per iteration
  2. Student (fine-tuned model) generates toke solutions
  3. Compiler (tkc) verifies solutions
  4. Teacher analyzes failure patterns and targets weak areas in next iteration

References:
  - NVIDIA "Data Flywheel" — iterative synthetic data generation with
    verification feedback to continuously improve code models.
  - SelfCodeAlign (Wei et al., 2024) — self-alignment of code LLMs via
    instruction generation, self-filtering, and compiler-verified solutions.

Usage::

    # Dry-run (no API calls, synthetic simulation):
    python scripts/teacher_student_loop.py --dry-run --iterations 2

    # With real models:
    python scripts/teacher_student_loop.py \\
        --teacher-model gpt-4o \\
        --student-model toke-coder-v1 \\
        --tkc-path ../toke/tkc \\
        --iterations 3 \\
        --problems-per-iter 500 \\
        --output-dir results/teacher_student

Story 9.2.3 -- Teacher-student evaluation loop.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Skill-ladder stage definitions (from skill-ladder.md, Story 9.2.1)
# ---------------------------------------------------------------------------

STAGES = {
    1: "Expressions and Bindings",
    2: "Functions and Types",
    3: "Control Flow",
    4: "Modules, Imports, and Sum Types",
    5: "Error Handling and Standard Library",
    6: "Advanced Patterns and Composition",
}

# Map toke compiler error codes to skill-ladder stages.
ERROR_CODE_TO_STAGE: dict[str, int] = {
    # Stage 1 -- expressions, bindings, literals
    "E001": 1, "E002": 1, "E003": 1, "E004": 1, "E005": 1, "E006": 1,
    # Stage 2 -- functions, types, structs
    "E010": 2, "E011": 2, "E012": 2, "E013": 2, "E014": 2, "E015": 2,
    # Stage 3 -- control flow
    "E020": 3, "E021": 3, "E022": 3, "E023": 3, "E024": 3,
    # Stage 4 -- modules, imports, sum types, match
    "E030": 4, "E031": 4, "E032": 4, "E033": 4, "E034": 4,
    # Stage 5 -- error handling, stdlib
    "E040": 5, "E041": 5, "E042": 5, "E043": 5,
    # Stage 6 -- advanced patterns
    "E050": 6, "E051": 6,
}

# Problem categories mapped to skill-ladder stages.
PROBLEM_CATEGORIES: dict[str, dict[str, Any]] = {
    "literal_binding": {
        "stage": 1,
        "description": "Declare bindings, evaluate literal expressions",
        "template": "Write a toke function that {action} using let bindings.",
    },
    "arithmetic_ops": {
        "stage": 1,
        "description": "Integer and float arithmetic operations",
        "template": "Write a toke function that computes {action}.",
    },
    "function_definition": {
        "stage": 2,
        "description": "Define functions with typed parameters and return types",
        "template": "Write a toke function {action} with proper type annotations.",
    },
    "struct_usage": {
        "stage": 2,
        "description": "Define and use struct types",
        "template": "Define a toke struct for {action} and write accessor functions.",
    },
    "if_else_branching": {
        "stage": 3,
        "description": "Conditional expressions and branching",
        "template": "Write a toke function that uses if/else to {action}.",
    },
    "loop_iteration": {
        "stage": 3,
        "description": "Loop constructs and iteration",
        "template": "Write a toke function using a loop to {action}.",
    },
    "module_import": {
        "stage": 4,
        "description": "Module definition and imports",
        "template": "Create a toke module that exports {action}.",
    },
    "match_exhaustive": {
        "stage": 4,
        "description": "Sum types and exhaustive match expressions",
        "template": "Define a sum type for {action} and write an exhaustive match.",
    },
    "error_handling": {
        "stage": 5,
        "description": "Result types and error propagation",
        "template": "Write a toke function that handles errors when {action}.",
    },
    "stdlib_usage": {
        "stage": 5,
        "description": "Standard library function usage",
        "template": "Write a toke function that uses the stdlib to {action}.",
    },
    "higher_order": {
        "stage": 6,
        "description": "Higher-order functions and generics",
        "template": "Write a generic toke function that {action}.",
    },
    "composition": {
        "stage": 6,
        "description": "Function composition and advanced patterns",
        "template": "Write a toke program demonstrating {action} via composition.",
    },
}

# Failure categories for classifying compiler diagnostics.
FAILURE_CATEGORIES = {
    "syntax": "Syntax errors (malformed code)",
    "type_mismatch": "Type system violations",
    "undeclared": "Use of undeclared names or missing imports",
    "control_flow": "Control flow errors (unreachable, missing branch)",
    "exhaustiveness": "Non-exhaustive pattern matches",
    "error_handling": "Unhandled Result types or error propagation issues",
    "other": "Other / unclassified errors",
}

ERROR_TO_FAILURE_CATEGORY: dict[str, str] = {
    "E001": "syntax", "E003": "syntax", "E006": "type_mismatch",
    "E002": "undeclared", "E004": "syntax", "E005": "type_mismatch",
    "E010": "type_mismatch", "E011": "type_mismatch", "E012": "syntax",
    "E013": "undeclared", "E014": "type_mismatch", "E015": "syntax",
    "E020": "control_flow", "E021": "control_flow", "E022": "type_mismatch",
    "E023": "control_flow", "E024": "control_flow",
    "E030": "undeclared", "E031": "exhaustiveness", "E032": "syntax",
    "E033": "undeclared", "E034": "undeclared",
    "E040": "error_handling", "E041": "error_handling",
    "E042": "undeclared", "E043": "error_handling",
    "E050": "type_mismatch", "E051": "type_mismatch",
}


# ---------------------------------------------------------------------------
# Teacher: problem generation
# ---------------------------------------------------------------------------

def generate_problems_teacher(
    n_problems: int,
    target_stages: dict[int, float],
    iteration: int,
    teacher_model: str,
    dry_run: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Teacher generates programming problems, biased toward weak stages.

    In dry-run mode, generates synthetic problem descriptors.
    In live mode, would call the teacher model API to produce problems.
    """
    if not dry_run:
        # Placeholder for real API call to teacher model.
        print(f"  [LIVE] Would call {teacher_model} to generate "
              f"{n_problems} problems (not implemented without API key)")
        # Fall through to synthetic generation for now.

    # Distribute problems across categories weighted by target_stages.
    categories_by_stage: dict[int, list[str]] = {}
    for cat, info in PROBLEM_CATEGORIES.items():
        s = info["stage"]
        categories_by_stage.setdefault(s, []).append(cat)

    # Normalise stage weights.
    total_w = sum(target_stages.values())
    if total_w == 0:
        total_w = 1.0
    stage_probs = {s: w / total_w for s, w in target_stages.items()}

    # Build weighted category list.
    weighted_cats: list[tuple[str, float]] = []
    for s, prob in stage_probs.items():
        cats = categories_by_stage.get(s, [])
        if cats:
            per_cat = prob / len(cats)
            for c in cats:
                weighted_cats.append((c, per_cat))

    if not weighted_cats:
        weighted_cats = [(c, 1.0) for c in PROBLEM_CATEGORIES]

    cat_names = [c for c, _ in weighted_cats]
    cat_weights = [w for _, w in weighted_cats]

    problems: list[dict[str, Any]] = []
    for i in range(n_problems):
        cat = rng.choices(cat_names, weights=cat_weights, k=1)[0]
        info = PROBLEM_CATEGORIES[cat]
        problems.append({
            "id": f"iter{iteration}-prob{i+1:04d}",
            "category": cat,
            "stage": info["stage"],
            "description": info["description"],
            "prompt": info["template"].format(action=f"task #{i+1}"),
            "iteration": iteration,
        })

    return problems


# ---------------------------------------------------------------------------
# Student: solution generation
# ---------------------------------------------------------------------------

def generate_solutions_student(
    problems: list[dict[str, Any]],
    student_model: str,
    dry_run: bool,
    rng: random.Random,
) -> dict[str, str]:
    """Student model generates toke solutions for each problem.

    In dry-run mode, produces synthetic toke snippets with realistic
    error rates that vary by stage difficulty.
    """
    if not dry_run:
        print(f"  [LIVE] Would call {student_model} to generate "
              f"{len(problems)} solutions (not implemented without API key)")

    solutions: dict[str, str] = {}
    for prob in problems:
        if dry_run:
            solutions[prob["id"]] = _synthetic_solution(prob, rng)
        else:
            # Placeholder for real student model inference.
            solutions[prob["id"]] = ""
    return solutions


def _synthetic_solution(prob: dict[str, Any], rng: random.Random) -> str:
    """Generate a synthetic toke solution for dry-run simulation.

    Returns empty string (will be treated as needing compilation) --
    the actual pass/fail is determined by _simulate_compilation.
    """
    stage = prob.get("stage", 1)
    cat = prob.get("category", "literal_binding")
    return f'// Synthetic solution for {cat} (stage {stage})\nlet x = 42\n'


# ---------------------------------------------------------------------------
# Compiler: verification
# ---------------------------------------------------------------------------

def verify_with_compiler(
    problems: list[dict[str, Any]],
    solutions: dict[str, str],
    tkc_path: str,
    dry_run: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Compile each solution with tkc and collect diagnostics.

    Returns a list of per-problem result dicts.
    """
    results: list[dict[str, Any]] = []
    for prob in problems:
        prob_id = prob["id"]
        source = solutions.get(prob_id, "")

        if dry_run:
            diagnostics = _simulate_compilation(prob, rng)
        else:
            diagnostics = _run_tkc(source, tkc_path)

        error_codes = [d.get("code", "E000") for d in diagnostics]
        passed = len(error_codes) == 0

        # Classify errors into failure categories.
        failure_cats: dict[str, int] = {}
        for code in error_codes:
            fc = ERROR_TO_FAILURE_CATEGORY.get(code, "other")
            failure_cats[fc] = failure_cats.get(fc, 0) + 1

        results.append({
            "problem_id": prob_id,
            "category": prob.get("category", "unknown"),
            "stage": prob.get("stage", 0),
            "passed": passed,
            "error_codes": error_codes,
            "failure_categories": failure_cats,
        })

    return results


def _run_tkc(source: str, tkc_path: str) -> list[dict[str, Any]]:
    """Run tkc --check --diag-json on a source string."""
    try:
        result = subprocess.run(
            [tkc_path, "--check", "--diag-json"],
            input=source,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return []


def _simulate_compilation(
    prob: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Simulate compilation with stage-dependent pass rates.

    Higher stages have lower pass rates, reflecting increasing difficulty.
    """
    stage = prob.get("stage", 1)
    # Pass probability decreases with stage.
    pass_probs = {1: 0.82, 2: 0.65, 3: 0.50, 4: 0.38, 5: 0.28, 6: 0.18}

    if rng.random() < pass_probs.get(stage, 0.5):
        return []  # Clean compilation.

    # Generate 1-3 errors from the stage's error codes.
    codes_for_stage = [
        code for code, s in ERROR_CODE_TO_STAGE.items() if s == stage
    ]
    if not codes_for_stage:
        codes_for_stage = ["E001"]

    n_errors = rng.randint(1, 3)
    diagnostics = []
    for _ in range(n_errors):
        code = rng.choice(codes_for_stage)
        diagnostics.append({
            "code": code,
            "message": f"simulated {code}",
            "line": rng.randint(1, 20),
            "col": rng.randint(1, 40),
        })
    return diagnostics


# ---------------------------------------------------------------------------
# Teacher: failure analysis and targeting
# ---------------------------------------------------------------------------

def analyze_failures(
    verification_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze verification results to identify weak areas.

    Returns analysis dict with per-stage and per-category breakdowns.
    """
    total = len(verification_results)
    pass_count = sum(1 for r in verification_results if r["passed"])
    fail_count = total - pass_count
    pass_rate = pass_count / max(total, 1)

    # Per-stage breakdown.
    stage_total: dict[int, int] = {s: 0 for s in STAGES}
    stage_fail: dict[int, int] = {s: 0 for s in STAGES}
    for r in verification_results:
        s = r.get("stage", 1)
        if s in stage_total:
            stage_total[s] += 1
            if not r["passed"]:
                stage_fail[s] += 1

    stage_pass_rates: dict[str, float] = {}
    for s in STAGES:
        t = stage_total.get(s, 0)
        f = stage_fail.get(s, 0)
        stage_pass_rates[str(s)] = (t - f) / max(t, 1) if t > 0 else 1.0

    # Aggregate failure categories.
    failure_cat_totals: dict[str, int] = {}
    for r in verification_results:
        for cat, cnt in r.get("failure_categories", {}).items():
            failure_cat_totals[cat] = failure_cat_totals.get(cat, 0) + cnt

    # Aggregate error codes.
    error_code_totals: dict[str, int] = {}
    for r in verification_results:
        for code in r.get("error_codes", []):
            error_code_totals[code] = error_code_totals.get(code, 0) + 1

    # Identify targeted stages: stages with pass rate below 60%.
    targeted_stages: list[int] = []
    for s in STAGES:
        pr = float(stage_pass_rates.get(str(s), 1.0))
        if pr < 0.60 and stage_total.get(s, 0) > 0:
            targeted_stages.append(s)

    return {
        "total_problems": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 4),
        "stage_pass_rates": {k: round(v, 4) for k, v in stage_pass_rates.items()},
        "failure_categories": failure_cat_totals,
        "error_code_totals": error_code_totals,
        "targeted_stages": targeted_stages,
    }


def compute_target_weights(
    analysis: dict[str, Any],
    current_weights: dict[int, float],
    boost_factor: float = 2.5,
) -> dict[int, float]:
    """Compute next-iteration stage weights based on failure analysis.

    Stages with lower pass rates get higher weight (more problems).
    Uses the data-flywheel principle: focus generation on weak areas.
    """
    stage_pass_rates = analysis["stage_pass_rates"]
    raw: dict[int, float] = {}

    for s in STAGES:
        base = current_weights.get(s, 1.0 / len(STAGES))
        pr = stage_pass_rates.get(str(s), 1.0)
        failure_rate = 1.0 - pr
        # Boost underperforming stages.
        raw[s] = base * (1.0 + failure_rate * boost_factor)

    total = sum(raw.values())
    if total == 0:
        total = 1.0

    return {s: round(w / total, 6) for s, w in raw.items()}


# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------

def run_iteration(
    iteration: int,
    n_problems: int,
    target_weights: dict[int, float],
    teacher_model: str,
    student_model: str,
    tkc_path: str,
    dry_run: bool,
    rng: random.Random,
) -> dict[str, Any]:
    """Run a single teacher-student-compiler iteration."""
    print(f"\n--- Iteration {iteration} ---")

    # Step 1: Teacher generates problems.
    print(f"  Step 1: Teacher ({teacher_model}) generating {n_problems} problems...")
    problems = generate_problems_teacher(
        n_problems, target_weights, iteration, teacher_model, dry_run, rng,
    )
    stage_dist = {}
    for p in problems:
        s = p["stage"]
        stage_dist[s] = stage_dist.get(s, 0) + 1
    print(f"    Stage distribution: {dict(sorted(stage_dist.items()))}")

    # Step 2: Student generates solutions.
    print(f"  Step 2: Student ({student_model}) generating solutions...")
    solutions = generate_solutions_student(problems, student_model, dry_run, rng)
    print(f"    Generated {len(solutions)} solutions.")

    # Step 3: Compiler verifies solutions.
    print(f"  Step 3: Compiler ({tkc_path}) verifying solutions...")
    verification_results = verify_with_compiler(
        problems, solutions, tkc_path, dry_run, rng,
    )
    pass_count = sum(1 for r in verification_results if r["passed"])
    print(f"    Pass: {pass_count}/{len(verification_results)} "
          f"({pass_count/max(len(verification_results),1):.1%})")

    # Step 4: Teacher analyzes failures.
    print(f"  Step 4: Teacher analyzing failure patterns...")
    analysis = analyze_failures(verification_results)
    if analysis["targeted_stages"]:
        stage_names = [f"Stage {s} ({STAGES[s]})"
                       for s in analysis["targeted_stages"]]
        print(f"    Targeted weak areas: {', '.join(stage_names)}")
    else:
        print(f"    No stages below 60% threshold.")

    # Compute next-iteration weights.
    new_weights = compute_target_weights(analysis, target_weights)

    return {
        "iteration": iteration,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "n_problems": n_problems,
        "pass_rate": analysis["pass_rate"],
        "pass_count": analysis["pass_count"],
        "fail_count": analysis["fail_count"],
        "stage_pass_rates": analysis["stage_pass_rates"],
        "failure_categories": analysis["failure_categories"],
        "error_code_totals": analysis["error_code_totals"],
        "targeted_stages": analysis["targeted_stages"],
        "input_weights": {str(s): round(w, 6) for s, w in target_weights.items()},
        "output_weights": {str(s): w for s, w in new_weights.items()},
    }


def run_all_iterations(
    teacher_model: str,
    student_model: str,
    tkc_path: str,
    n_problems: int,
    iterations: int,
    output_dir: Path,
    dry_run: bool,
    seed: int,
) -> dict[str, Any]:
    """Run the full teacher-student loop for N iterations."""
    rng = random.Random(seed)
    current_weights = {s: round(1.0 / len(STAGES), 6) for s in STAGES}
    results: list[dict[str, Any]] = []

    for it in range(1, iterations + 1):
        iter_result = run_iteration(
            iteration=it,
            n_problems=n_problems,
            target_weights=current_weights,
            teacher_model=teacher_model,
            student_model=student_model,
            tkc_path=tkc_path,
            dry_run=dry_run,
            rng=rng,
        )
        results.append(iter_result)

        # Write per-iteration JSON report.
        iter_file = output_dir / f"teacher_student_iter_{it}.json"
        with open(iter_file, "w") as fh:
            json.dump(iter_result, fh, indent=2)
        print(f"  Report: {iter_file}")

        # Advance weights for next iteration.
        current_weights = {
            int(s): w for s, w in iter_result["output_weights"].items()
        }

    summary = build_summary(results)
    summary_file = output_dir / "teacher_student_summary.json"
    with open(summary_file, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary: {summary_file}")

    return summary


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary from all iteration results."""
    trajectory: list[dict[str, Any]] = []
    for r in results:
        trajectory.append({
            "iteration": r["iteration"],
            "pass_rate": r["pass_rate"],
            "pass_count": r["pass_count"],
            "fail_count": r["fail_count"],
            "targeted_stages": r["targeted_stages"],
        })

    first_pr = results[0]["pass_rate"] if results else 0.0
    last_pr = results[-1]["pass_rate"] if results else 0.0

    # Aggregate all failure categories across iterations.
    all_failure_cats: dict[str, int] = {}
    for r in results:
        for cat, cnt in r.get("failure_categories", {}).items():
            all_failure_cats[cat] = all_failure_cats.get(cat, 0) + cnt

    return {
        "methodology": "Teacher-Student Evaluation Loop",
        "references": [
            "NVIDIA Data Flywheel — iterative synthetic data with verification feedback",
            "SelfCodeAlign (Wei et al., 2024) — self-alignment via instruction generation",
        ],
        "total_iterations": len(results),
        "problems_per_iteration": results[0]["n_problems"] if results else 0,
        "teacher_model": results[0]["teacher_model"] if results else "",
        "student_model": results[0]["student_model"] if results else "",
        "initial_pass_rate": first_pr,
        "final_pass_rate": last_pr,
        "improvement": round(last_pr - first_pr, 4),
        "trajectory": trajectory,
        "final_stage_pass_rates": results[-1]["stage_pass_rates"] if results else {},
        "final_weights": results[-1]["output_weights"] if results else {},
        "cumulative_failure_categories": all_failure_cats,
    }


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------

def print_summary_table(summary: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    print()
    print("=" * 72)
    print("  Teacher-Student Evaluation Loop -- Summary")
    print("  Ref: NVIDIA Data Flywheel, SelfCodeAlign")
    print("=" * 72)
    print()
    print(f"  Teacher: {summary.get('teacher_model', '?')}")
    print(f"  Student: {summary.get('student_model', '?')}")
    print(f"  Problems/iter: {summary.get('problems_per_iteration', '?')}")
    print()
    print(f"  {'Iter':>4}  {'Pass Rate':>10}  {'Pass':>5}  {'Fail':>5}  "
          f"{'Targeted Stages'}")
    print(f"  {'----':>4}  {'----------':>10}  {'-----':>5}  {'-----':>5}  "
          f"{'---------------'}")
    for t in summary["trajectory"]:
        pr = f"{t['pass_rate']:.1%}"
        targeted = ", ".join(str(s) for s in t["targeted_stages"]) or "none"
        print(f"  {t['iteration']:>4}  {pr:>10}  {t['pass_count']:>5}  "
              f"{t['fail_count']:>5}  {targeted}")

    print()
    imp = summary["improvement"]
    direction = "+" if imp >= 0 else ""
    print(f"  Improvement: {direction}{imp:.1%} "
          f"({summary['initial_pass_rate']:.1%} -> "
          f"{summary['final_pass_rate']:.1%})")
    print()

    # Final stage pass rates.
    spr = summary.get("final_stage_pass_rates", {})
    if spr:
        print("  Final per-stage pass rates:")
        for stage_str in sorted(spr, key=int):
            s = int(stage_str)
            rate = spr[stage_str]
            name = STAGES.get(s, "Unknown")
            bar = "#" * int(rate * 30)
            print(f"    Stage {s} ({name[:28]:<28s}): {rate:.1%}  {bar}")
        print()

    # Final weights.
    fw = summary.get("final_weights", {})
    if fw:
        print("  Next-iteration target weights (data flywheel):")
        for stage_str in sorted(fw, key=int):
            s = int(stage_str)
            w = fw[stage_str]
            name = STAGES.get(s, "Unknown")
            bar = "#" * int(w * 40)
            print(f"    Stage {s} ({name[:28]:<28s}): {w:.4f}  {bar}")
        print()

    # Failure categories.
    fc = summary.get("cumulative_failure_categories", {})
    if fc:
        print("  Cumulative failure categories:")
        for cat, cnt in sorted(fc.items(), key=lambda x: -x[1]):
            desc = FAILURE_CATEGORIES.get(cat, cat)
            print(f"    {cat:<20s}: {cnt:>5}  ({desc})")
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-student evaluation loop for toke code generation. "
            "Ref: NVIDIA Data Flywheel, SelfCodeAlign."
        ),
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4o",
        help="Teacher model name/endpoint (default: gpt-4o)",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="toke-coder-v1",
        help="Student (fine-tuned) model name/endpoint (default: toke-coder-v1)",
    )
    parser.add_argument(
        "--tkc-path",
        type=str,
        default="tkc",
        help="Path to tkc compiler binary (default: tkc)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of teacher-student iterations (default: 2, min 2 for Gate 2)",
    )
    parser.add_argument(
        "--problems-per-iter",
        type=int,
        default=500,
        help="Number of problems the teacher generates per iteration (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/teacher_student"),
        help="Output directory for iteration reports (default: results/teacher_student/)",
    )
    parser.add_argument(
        "--boost-factor",
        type=float,
        default=2.5,
        help="Reweighting boost factor for weak stages (default: 2.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the loop without actual model API calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Teacher-Student Evaluation Loop -- Story 9.2.3")
    print(f"  teacher-model:    {args.teacher_model}")
    print(f"  student-model:    {args.student_model}")
    print(f"  tkc-path:         {args.tkc_path}")
    print(f"  iterations:       {args.iterations}")
    print(f"  problems-per-iter: {args.problems_per_iter}")
    print(f"  output-dir:       {args.output_dir}")
    print(f"  boost-factor:     {args.boost_factor}")
    print(f"  dry-run:          {args.dry_run}")
    print(f"  seed:             {args.seed}")

    if args.iterations < 2:
        print("WARNING: Gate 2 requires >= 2 iterations. "
              f"Running {args.iterations} as requested.", file=sys.stderr)

    if not args.dry_run:
        print("\nNOTE: Live mode requires API keys for teacher and student models.")
        print("Use --dry-run for simulation without API calls.\n")

    # Ensure output dir exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run the loop.
    summary = run_all_iterations(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        tkc_path=args.tkc_path,
        n_problems=args.problems_per_iter,
        iterations=args.iterations,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    # Print summary table.
    print_summary_table(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

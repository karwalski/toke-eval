#!/usr/bin/env python3
"""Statistical analysis of token efficiency with confidence intervals.

Story 10.1.6: Computes bootstrap CIs, Wilcoxon signed-rank tests,
effect sizes, and power analysis for toke vs Python token counts.

Supports two CSV formats:
  1. Paired format (one row per task):
     task_id, category, toke_tokens, python_tokens [, ...]
  2. Long format (toke-eval gate1_token_counts.csv):
     task_id, category, tokenizer, language, token_count, char_count, pass1

When paired Python data is unavailable, use --generate-mock to create
synthetic paired data for methodology validation.

Usage:
    python scripts/statistical_analysis.py data/gate1_token_counts.csv
    python scripts/statistical_analysis.py --generate-mock --n-tasks 1000
    python scripts/statistical_analysis.py data/paired_counts.csv --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class CIResult:
    """A point estimate with a bootstrap confidence interval."""
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    method: str = "BCa"


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    n: int


@dataclass
class CategoryStats:
    """Per-category summary."""
    category: str
    n: int
    median_ratio: float
    ci_median: CIResult
    mean_ratio: float
    ci_mean: CIResult


@dataclass
class AnalysisResults:
    """Top-level analysis output."""
    n_tasks: int = 0
    overall_median_ratio: CIResult | None = None
    overall_trimmed_mean_ratio: CIResult | None = None
    wilcoxon_test: TestResult | None = None
    power_analysis: dict[str, Any] = field(default_factory=dict)
    category_results: list[CategoryStats] = field(default_factory=list)
    data_source: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bootstrap CI (BCa method)
# ---------------------------------------------------------------------------

def _bca_ci(
    data: np.ndarray,
    stat_func,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CIResult:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    data : 1-D array
    stat_func : callable(array) -> scalar
    n_resamples : number of bootstrap resamples
    alpha : significance level (0.05 -> 95 % CI)
    rng : numpy random generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(data)
    theta_hat = stat_func(data)

    # Bootstrap distribution
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_stats = np.array([stat_func(data[idx]) for idx in boot_indices])

    # --- Bias correction ---
    # Proportion of bootstrap estimates below the observed statistic
    z0 = sp_stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # --- Acceleration (jackknife) ---
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats[i] = stat_func(jack_sample)

    jack_mean = jackknife_stats.mean()
    diff = jack_mean - jackknife_stats
    a_hat = np.sum(diff ** 3) / (6.0 * (np.sum(diff ** 2) ** 1.5))

    # --- Adjusted percentiles ---
    z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
    z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def _adjusted_quantile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a_hat * num
        if abs(denom) < 1e-12:
            return 0.5  # fallback
        adj_z = z0 + num / denom
        return sp_stats.norm.cdf(adj_z)

    q_lo = _adjusted_quantile(z_alpha_lo)
    q_hi = _adjusted_quantile(z_alpha_hi)

    # Clip quantiles to [0, 1] for safety
    q_lo = np.clip(q_lo, 1 / n_resamples, 1 - 1 / n_resamples)
    q_hi = np.clip(q_hi, 1 / n_resamples, 1 - 1 / n_resamples)

    ci_lower = float(np.quantile(boot_stats, q_lo))
    ci_upper = float(np.quantile(boot_stats, q_hi))

    return CIResult(
        estimate=float(theta_hat),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=1 - alpha,
        method="BCa",
    )


# ---------------------------------------------------------------------------
# Trimmed mean helper
# ---------------------------------------------------------------------------

def trimmed_mean(data: np.ndarray, proportiontocut: float = 0.10) -> float:
    """Trimmed mean using scipy."""
    return float(sp_stats.trim_mean(data, proportiontocut))


# ---------------------------------------------------------------------------
# Effect size: rank-biserial correlation for Wilcoxon
# ---------------------------------------------------------------------------

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation for paired samples.

    r = 1 - (2T) / (n(n+1)/2)
    where T is the smaller of the Wilcoxon signed-rank sums.
    """
    diffs = x - y
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0

    abs_diffs = np.abs(diffs)
    ranks = sp_stats.rankdata(abs_diffs)

    r_plus = np.sum(ranks[diffs > 0])
    r_minus = np.sum(ranks[diffs < 0])

    t_val = min(r_plus, r_minus)
    r = 1.0 - (2.0 * t_val) / (n * (n + 1) / 2.0)
    # Sign convention: positive means x < y (toke is smaller)
    if r_plus < r_minus:
        r = -r
    return float(r)


# ---------------------------------------------------------------------------
# Power analysis (normal approximation for Wilcoxon)
# ---------------------------------------------------------------------------

def wilcoxon_power_analysis(
    effect_size_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate minimum sample size for a Wilcoxon signed-rank test.

    Uses the asymptotic relative efficiency (ARE) adjustment:
    Wilcoxon requires n_wilcoxon ~= n_ttest / 0.955 for normal data,
    but ~= n_ttest / (3/pi) for heavy-tailed data.

    We use the conservative ARE = 0.955 (normal case).
    """
    if abs(effect_size_d) < 1e-10:
        return 999_999  # effectively infinite

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    # Paired t-test sample size
    n_ttest = ((z_alpha + z_beta) / effect_size_d) ** 2

    # Adjust for Wilcoxon ARE (worst case for normal data)
    are = 0.955
    n_wilcoxon = int(np.ceil(n_ttest / are))

    return max(n_wilcoxon, 6)  # Wilcoxon needs at least 6


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def generate_mock_data(
    n_tasks: int = 1000,
    seed: int = 42,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic paired token count data for methodology testing.

    Simulates toke achieving ~25-35% token reduction vs Python across
    categories with varying difficulty (matching real corpus properties).
    """
    rng = np.random.default_rng(seed)

    if categories is None:
        categories = ["arithmetic", "string", "list", "control_flow", "io"]

    # Category-specific parameters (python_mean, python_std, reduction_pct)
    cat_params = {
        "arithmetic":   (80, 30, 0.30),
        "string":       (110, 40, 0.28),
        "list":         (100, 35, 0.32),
        "control_flow": (130, 50, 0.25),
        "io":           (90, 25, 0.27),
    }

    rows = []
    tasks_per_cat = n_tasks // len(categories)
    remainder = n_tasks - tasks_per_cat * len(categories)

    task_num = 0
    for i, cat in enumerate(categories):
        n_cat = tasks_per_cat + (1 if i < remainder else 0)
        params = cat_params.get(cat, (100, 35, 0.28))
        py_mean, py_std, reduction = params

        py_tokens = rng.normal(py_mean, py_std, n_cat).astype(int)
        py_tokens = np.clip(py_tokens, 10, py_mean * 5)

        # toke tokens: apply reduction with noise
        reduction_per_task = rng.normal(reduction, 0.08, n_cat)
        reduction_per_task = np.clip(reduction_per_task, 0.0, 0.70)
        toke_tokens = (py_tokens * (1 - reduction_per_task)).astype(int)
        toke_tokens = np.clip(toke_tokens, 5, None)

        for j in range(n_cat):
            task_num += 1
            rows.append({
                "task_id": f"task-a-{task_num:04d}",
                "category": cat,
                "toke_tokens": int(toke_tokens[j]),
                "python_tokens": int(py_tokens[j]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paired_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and return DataFrame with columns:
    task_id, category, toke_tokens, python_tokens.

    Handles both paired format and long (gate1) format.
    """
    df = pd.read_csv(csv_path)

    # Check if this is already paired format
    if "toke_tokens" in df.columns and "python_tokens" in df.columns:
        required = ["task_id", "category", "toke_tokens", "python_tokens"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df[required].copy()

    # Long format: pivot language rows into paired columns
    if "language" in df.columns and "token_count" in df.columns:
        languages = df["language"].unique()

        if "toke" in languages and "python" in languages:
            toke_df = (
                df[df["language"] == "toke"][["task_id", "category", "token_count"]]
                .rename(columns={"token_count": "toke_tokens"})
            )
            py_df = (
                df[df["language"] == "python"][["task_id", "token_count"]]
                .rename(columns={"token_count": "python_tokens"})
            )
            merged = toke_df.merge(py_df, on="task_id", how="inner")
            if len(merged) == 0:
                raise ValueError(
                    "Long format detected but no matching task_ids "
                    "between toke and python rows."
                )
            return merged

        if "toke" in languages and len(languages) == 1:
            raise ValueError(
                "CSV contains only toke data (no Python baseline). "
                "Use --generate-mock to create synthetic paired data "
                "for methodology validation, or provide a CSV with "
                "both toke and python rows."
            )

    # Try alternative column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "toke" in cl and "token" in cl:
            col_map["toke_tokens"] = col
        elif ("python" in cl or "py" in cl) and "token" in cl:
            col_map["python_tokens"] = col

    if len(col_map) == 2:
        rename = {v: k for k, v in col_map.items()}
        df = df.rename(columns=rename)
        if "task_id" not in df.columns:
            df["task_id"] = [f"task-{i:04d}" for i in range(len(df))]
        if "category" not in df.columns:
            df["category"] = "unknown"
        return df[["task_id", "category", "toke_tokens", "python_tokens"]].copy()

    raise ValueError(
        f"Cannot parse CSV. Expected either paired columns "
        f"(toke_tokens, python_tokens) or long format "
        f"(language, token_count). Found columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> AnalysisResults:
    """Run the full statistical analysis on paired token count data."""
    results = AnalysisResults(n_tasks=len(df))
    rng = np.random.default_rng(seed)

    toke = df["toke_tokens"].to_numpy(dtype=float)
    python = df["python_tokens"].to_numpy(dtype=float)

    # Token reduction ratio: toke / python (< 1 means toke is more efficient)
    ratios = toke / python

    # --- Overall statistics with BCa CIs ---
    results.overall_median_ratio = _bca_ci(
        ratios, np.median, n_resamples=n_resamples, alpha=alpha, rng=rng,
    )

    results.overall_trimmed_mean_ratio = _bca_ci(
        ratios,
        lambda x: trimmed_mean(x, 0.10),
        n_resamples=n_resamples,
        alpha=alpha,
        rng=rng,
    )

    # --- Paired Wilcoxon signed-rank test ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value = sp_stats.wilcoxon(toke, python, alternative="two-sided")

    r_rb = rank_biserial_r(toke, python)

    results.wilcoxon_test = TestResult(
        test_name="Wilcoxon signed-rank (two-sided)",
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=r_rb,
        effect_size_name="rank-biserial r",
        n=len(df),
    )

    # --- Power analysis ---
    # Use Cohen's d from the paired differences for power calculation
    diffs = python - toke
    cohens_d = float(np.mean(diffs) / np.std(diffs, ddof=1))

    min_n = wilcoxon_power_analysis(
        effect_size_d=abs(cohens_d),
        alpha=alpha,
        power=0.80,
    )

    results.power_analysis = {
        "observed_cohens_d": round(cohens_d, 4),
        "target_power": 0.80,
        "alpha": alpha,
        "min_sample_size_80pct_power": min_n,
        "current_n": len(df),
        "adequately_powered": len(df) >= min_n,
    }

    # --- Per-category stratification ---
    categories = sorted(df["category"].unique())
    for cat in categories:
        cat_df = df[df["category"] == cat]
        if len(cat_df) < 6:
            results.warnings.append(
                f"Category '{cat}' has only {len(cat_df)} tasks "
                f"(skipping CI computation, need >= 6)."
            )
            continue

        cat_ratios = (
            cat_df["toke_tokens"].to_numpy(dtype=float)
            / cat_df["python_tokens"].to_numpy(dtype=float)
        )

        cat_ci_median = _bca_ci(
            cat_ratios, np.median, n_resamples=n_resamples, alpha=alpha, rng=rng,
        )
        cat_ci_mean = _bca_ci(
            cat_ratios,
            lambda x: trimmed_mean(x, 0.10),
            n_resamples=n_resamples,
            alpha=alpha,
            rng=rng,
        )

        results.category_results.append(CategoryStats(
            category=cat,
            n=len(cat_df),
            median_ratio=float(np.median(cat_ratios)),
            ci_median=cat_ci_median,
            mean_ratio=float(trimmed_mean(cat_ratios, 0.10)),
            ci_mean=cat_ci_mean,
        ))

    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _ci_str(ci: CIResult) -> str:
    return f"{ci.estimate:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"


def _pct_str(ci: CIResult) -> str:
    """Format ratio as percentage reduction: (1 - ratio) * 100."""
    est = (1 - ci.estimate) * 100
    lo = (1 - ci.ci_upper) * 100  # note: inverted
    hi = (1 - ci.ci_lower) * 100
    return f"{est:.1f}% [{lo:.1f}%, {hi:.1f}%]"


def format_stdout(r: AnalysisResults) -> str:
    """Format results as a readable summary table for stdout."""
    lines = [
        "=" * 70,
        "TOKEN EFFICIENCY STATISTICAL ANALYSIS",
        "=" * 70,
        f"Data source: {r.data_source}",
        f"Tasks analysed: {r.n_tasks}",
        "",
        "--- Overall Token Reduction (toke / python ratio) ---",
        f"  Median ratio:       {_ci_str(r.overall_median_ratio)}",
        f"    => reduction:     {_pct_str(r.overall_median_ratio)}",
        f"  Trimmed mean (10%): {_ci_str(r.overall_trimmed_mean_ratio)}",
        f"    => reduction:     {_pct_str(r.overall_trimmed_mean_ratio)}",
        f"  CI method:          {r.overall_median_ratio.method} bootstrap "
        f"({r.overall_median_ratio.ci_level:.0%} level)",
        "",
        "--- Paired Wilcoxon Signed-Rank Test ---",
        f"  H0: no difference between toke and python token counts",
        f"  Test statistic:     {r.wilcoxon_test.statistic:.1f}",
        f"  p-value:            {r.wilcoxon_test.p_value:.2e}",
        f"  Effect size ({r.wilcoxon_test.effect_size_name}): "
        f"{r.wilcoxon_test.effect_size:.4f}",
        f"  n (pairs):          {r.wilcoxon_test.n}",
        "",
        "--- Power Analysis ---",
        f"  Observed Cohen's d: {r.power_analysis['observed_cohens_d']:.4f}",
        f"  Min n for 80% power: {r.power_analysis['min_sample_size_80pct_power']}",
        f"  Current n:          {r.power_analysis['current_n']}",
        f"  Adequately powered: {'YES' if r.power_analysis['adequately_powered'] else 'NO'}",
    ]

    if r.category_results:
        lines.extend([
            "",
            "--- Per-Category Stratification ---",
            f"{'Category':<15} {'n':>5}  {'Median ratio':>25}  {'Trimmed mean':>25}",
            "-" * 75,
        ])
        for cat in r.category_results:
            lines.append(
                f"{cat.category:<15} {cat.n:>5}  "
                f"{_ci_str(cat.ci_median):>25}  "
                f"{_ci_str(cat.ci_mean):>25}"
            )

    if r.warnings:
        lines.extend(["", "--- Warnings ---"])
        for w in r.warnings:
            lines.append(f"  * {w}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_json(r: AnalysisResults) -> str:
    """Serialize results to JSON."""
    def _convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    return json.dumps(asdict(r), indent=2, default=_convert)


def format_markdown(r: AnalysisResults) -> str:
    """Format results as a Markdown summary for inclusion in reports."""
    lines = [
        "## Token Efficiency Statistical Analysis",
        "",
        f"**Tasks analysed:** {r.n_tasks}  ",
        f"**Data source:** {r.data_source}",
        "",
        "### Overall Token Reduction",
        "",
        "| Metric | Ratio (toke/python) | 95% CI | Reduction |",
        "|--------|--------------------:|-------:|----------:|",
    ]

    med = r.overall_median_ratio
    lines.append(
        f"| Median | {med.estimate:.4f} | "
        f"[{med.ci_lower:.4f}, {med.ci_upper:.4f}] | "
        f"{_pct_str(med)} |"
    )
    tm = r.overall_trimmed_mean_ratio
    lines.append(
        f"| Trimmed mean (10%) | {tm.estimate:.4f} | "
        f"[{tm.ci_lower:.4f}, {tm.ci_upper:.4f}] | "
        f"{_pct_str(tm)} |"
    )

    lines.extend([
        "",
        f"CI method: {med.method} bootstrap ({med.ci_level:.0%} level)",
        "",
        "### Hypothesis Test",
        "",
        f"**Wilcoxon signed-rank test** (two-sided, n={r.wilcoxon_test.n})  ",
        f"H0: no difference between toke and python token counts  ",
        f"Statistic: {r.wilcoxon_test.statistic:.1f}  ",
        f"p-value: {r.wilcoxon_test.p_value:.2e}  ",
        f"Effect size ({r.wilcoxon_test.effect_size_name}): "
        f"{r.wilcoxon_test.effect_size:.4f}",
        "",
        "### Power Analysis",
        "",
        f"| Parameter | Value |",
        f"|-----------|------:|",
        f"| Observed Cohen's d | {r.power_analysis['observed_cohens_d']:.4f} |",
        f"| Min n for 80% power | "
        f"{r.power_analysis['min_sample_size_80pct_power']} |",
        f"| Current n | {r.power_analysis['current_n']} |",
        f"| Adequately powered | "
        f"{'Yes' if r.power_analysis['adequately_powered'] else 'No'} |",
    ])

    if r.category_results:
        lines.extend([
            "",
            "### Per-Category Results",
            "",
            "| Category | n | Median ratio | 95% CI | Trimmed mean | 95% CI |",
            "|----------|--:|-------------:|-------:|-------------:|-------:|",
        ])
        for cat in r.category_results:
            lines.append(
                f"| {cat.category} | {cat.n} | "
                f"{cat.median_ratio:.4f} | "
                f"[{cat.ci_median.ci_lower:.4f}, {cat.ci_median.ci_upper:.4f}] | "
                f"{cat.mean_ratio:.4f} | "
                f"[{cat.ci_mean.ci_lower:.4f}, {cat.ci_mean.ci_upper:.4f}] |"
            )

    if r.warnings:
        lines.extend(["", "### Warnings", ""])
        for w in r.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of token efficiency with CIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv", nargs="?", type=Path, default=None,
        help="Path to paired token counts CSV.",
    )
    parser.add_argument(
        "--generate-mock", action="store_true",
        help="Generate synthetic paired data instead of loading CSV.",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=1000,
        help="Number of tasks for mock data (default: 1000).",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05 for 95%% CI).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for JSON and Markdown output files.",
    )

    args = parser.parse_args(argv)

    if args.generate_mock:
        print("Generating mock paired data...", file=sys.stderr)
        df = generate_mock_data(n_tasks=args.n_tasks, seed=args.seed)
        data_source = f"mock (n={args.n_tasks}, seed={args.seed})"
    elif args.csv is not None:
        if not args.csv.exists():
            print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
            return 1
        try:
            df = load_paired_data(args.csv)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(
                "\nTip: Use --generate-mock to run analysis on synthetic data.",
                file=sys.stderr,
            )
            return 1
        data_source = str(args.csv)
    else:
        parser.print_help()
        return 1

    print(f"Loaded {len(df)} paired tasks.", file=sys.stderr)
    print(f"Running bootstrap ({args.n_resamples} resamples)...", file=sys.stderr)

    results = run_analysis(
        df,
        n_resamples=args.n_resamples,
        alpha=args.alpha,
        seed=args.seed,
    )
    results.data_source = data_source

    # --- Stdout summary ---
    print(format_stdout(results))

    # --- File outputs ---
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = args.output_dir / "statistical_analysis.json"
        json_path.write_text(format_json(results), encoding="utf-8")
        print(f"\nJSON results: {json_path}", file=sys.stderr)

        md_path = args.output_dir / "statistical_analysis.md"
        md_path.write_text(format_markdown(results), encoding="utf-8")
        print(f"Markdown summary: {md_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

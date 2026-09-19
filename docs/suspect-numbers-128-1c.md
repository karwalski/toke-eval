# Numbers produced by the broken evaluation harness (story 128.1c)

**Status:** open. **Date:** 2026-09-19. **Scope:** every published or recorded
figure that passed through `toke-eval/benchmark/run_benchmark.py`,
`toke-eval/toke_eval/pass_at_k.py` or `toke-eval/scripts/pass_at_k.py` before
the 128.1c repairs.

This ledger does **not** replace any number. BENCH-128 is being rebuilt under
128.1a and no re-run is possible or permitted until it is frozen (128.1b, G0).
The purpose here is to say precisely which figures may not be carried forward,
and why each one is wrong — so that a number cannot be re-quoted on the grounds
that nobody wrote down what was wrong with it.

Every claim below is verified against the artefact named. Where a figure has no
surviving machine-readable artefact, that is stated.

---

## 1. Gate 1 — `Pass@1 = 63.7%` — INFLATED, and it fails its own gate

**The defect is the denominator, not best-of-N.**

`load_toke_solutions()` dropped every `.toke` solution that failed to compile
(`continue`, no record kept). A dropped solution never became a `TaskResult`,
so it left the denominator entirely. The figure published as Pass@1 is
therefore Pass@1 *given that the solution compiled* — a different and strictly
more generous quantity.

| Quantity | Value |
|---|---|
| Solutions generated | 1,000 (`benchmark/solutions/*.toke`) |
| Solutions that compiled | 923 |
| Solutions that passed every hidden test | 588 |
| **Published** "Pass@1" = 588/923 | **63.7%** |
| **Actual Pass@1** = 588/1000 | **58.8%** |

**Gate 1's own declared threshold was `pass_at_1_minimum: 0.60`**
(`toke-model/huggingface/eval_results.json`). 0.588 is below it. The
artefact simultaneously records `"benchmark_size": 1000` and
`"pass_at_1": 0.637`, which is the whole error visible in one file.

`eval_results.json` also records `"n_samples": 1`, and `run_inference_mlx.py`
draws a single sample per task at `--temp 0.2` with no sampling loop. The
best-of-N path was therefore **not** used for this figure — see §2.

**Where this number is published or recorded:**

| Location | Form |
|---|---|
| `toke-eval/benchmark/results/gate1_v5_1000.json` | `total_pass_at_1: 588`, `mean_pass_at_1: 0.6371`, `tasks_evaluated: 923` |
| `toke-model/huggingface/eval_results.json` | `"pass_at_1": 0.637`, `"gate_result": "PASS"` |
| `toke-model/huggingface/model-card-toke-coder-7b-UNPUBLISHED.md:45-47,129-130` | `model-index value: 63.7`; "Pass@1 \| **63.7%** (588/923)" — never uploaded |
| `toke-model/README.md:5` | "Gate 1 (2026-04-03): 63.7% compile Pass@1" — also mislabelled: 63.7% is functional, the compile rate is 92.3% |
| `toke-website/content/roadmap.json:28,31` | "63.7% Pass@1" |
| `toke-website/templates/index.tkt`, `roadmap.tkt` | 63.7% |
| commit `a9bde66` message | "Gate 1 final results: 63.7% Pass@1 (588/923 tasks)" |

**Verdict:** UNSAFE as published. The corrected figure, 58.8%, is recoverable
by arithmetic from artefacts already on disk and does **not** require a re-run.
It fails the Gate-1 threshold it was declared to pass.

**Also unsafe, same defect, unpublished:**
`gate1_v2.json` (219/329 = 66.6%), `gate1_pass_at_1.json` (153/183 = 83.6%),
and the archived `gate1_v3.json` / `gate1_v4_1000.json` (both 312/435 = 71.7%).
Each is passed/compiled over a generated set of unrecorded size.

---

## 2. Any figure produced with `--n-samples > 1` — INFLATED, none found published

`score_model_pass_at_1()` kept the best of N samples and `break`-ed on the
first perfect one, then wrote the result to a field named `pass_at_1`. That is
best-of-N: an oracle metric that assumes a selector that knows which sample
passes the hidden tests. It is strictly ≥ Pass@1 and rises with N.

**No recorded artefact is provably from this path.** The four `gate1_*.json`
files all carry `"language": "toke"`, which only the `--solutions-dir` path
emits; the model path hardcodes `"toke-model"`. No file with
`"language": "toke-model"` exists in either repo.

**But the artefacts cannot rule it out by themselves.** None of the recorded
reports contains an `n_samples` field — the pre-128.1c `report_to_dict()` did
not write one. Attribution above rests on the `language` field and on
`eval_results.json`'s separate `"n_samples": 1`. Any *future* figure is now
safe by construction: `n_samples` and `metric_note` are written into every
report, and `mean_best_of_n` is a separate field that is `None` at N=1.

**Verdict:** the path was live and inflating for the whole period; no published
figure is attributable to it. Treat any figure that surfaces later without an
`n_samples` record as unsafe.

---

## 3. `data/pass_at_k_results.json` and `data/pass_at_k_summary.csv` — SYNTHETIC

Both files are `--dry-run` output: **randomly generated pass/fail outcomes
from `seed=42`, no compilation, no execution, no model.**

- `data/pass_at_k_results.json` records `"mode": "dry-run"` and `"seed": 42`.
- **`data/pass_at_k_summary.csv` carries no marker of any kind.** Its header is
  `temperature,n_tasks,pass_at_1,pass_at_5,pass_at_10` and its rows
  (`0.8,1000,0.445800,0.838254,0.925000`) are indistinguishable from a real
  evaluation of 1,000 tasks.

**Verdict:** UNSAFE. Not results at all. The CSV is the more dangerous of the
two because it is the one shaped like something you would paste into a table.
The `1000` in its `n_tasks` column matches the real Gate-1 benchmark size,
which makes the coincidence worse.

Same failure mode, same directory, outside this story's harness scope but filed
below: `data/ablation_summary.json` (`"dry_run": true`) with its unmarked
derivative `data/ablation_table.csv`; `data/checkpoint_regression_report.json`
(`"mode": "dry-run"`) with unmarked `data/training_curve.csv` / `.json`.

---

## 4. `toke_eval/pass_at_k.py` — schema mismatch — NO number reached publication

The module read `tests["test_cases"]`; every benchmark file in both repos uses
`test_inputs`. It returned `(0, 0)`, so every task scored 0 with
`tests_total == 0`, and the run still printed a headline Pass@1. Verified
empirically against an unmodified copy of the original module and a real
benchmark file: `run_tests(...) -> (0, 0)`.

Its report shape (`error_taxonomy` + `tasks_compiled`) appears in **no** result
file in `toke-eval/data/`, `benchmark/results/`, anywhere in `toke-model`, or
in the gate-2 archive.

**Verdict:** the defect was total — the module could never have scored anything
above zero — but it produced no recorded or published number. Its second
defect, `pass_at_1 = tasks_passed / tasks_compiled`, is the same denominator
error as §1 and is now fixed in both places.

---

## 5. Gate 2 — `100% compile` / `55.6% functional` — NOT from these harnesses

Traced to `gate2_pipeline.py`, named as the evaluator in
`archive/toke-legacy-20260819/docs/spec/gate2-decision.md:5`. That script
survives only as `archive/toke-corpus-phase-eras-20260819/infra/run_gate2_pipeline.sh`.

**No machine-readable artefact for 272/489 exists anywhere.** Both figures are
recoverable only from prose in an archived decision document. The compile
figure uses a denominator of 500 and the functional figure a denominator of
489; the document does not reconcile them.

**Verdict:** outside 128.1c's harnesses, so not re-derivable here — but equally
not carryable forward, for a different reason: there is nothing behind them to
check. Separately, `training-reset-128.md` K11 records that Gate 2 was declared
PASS against the loosest of three competing criteria, one of which had no
numeric floor at all.

Published at: `toke-website/templates/status.tkt:54-60,82`,
`toke-website/content/roadmap.json:42-46`,
`toke-website/sites/tokelang.dev/llms.txt:85`, `toke-model/README.md:5`,
`toke-model/huggingface/README.md:24-32,103`, `toke/docs/metrics-baseline.md`.

---

## 6. The live Hugging Face model card — `Functional Pass@1 = 8`, `verified: true`

Per `toke/docs/registry-descriptions.md:374`, the **live** card's `model-index`
publishes `Functional Pass@1 = 8` with `verified: true`. That is the superseded
pre-`io.readln`-fix Gate-2 number (`gate2-decision.md:70`, "4/50 | 8%"),
machine-readable, consumed by leaderboards, and asserted as verified.

The local `toke-model/huggingface/README.md` is a prepared rewrite (132.4) that
has not been uploaded; publishing is an open owner action (132.24). **Nothing
in this story publishes anything.**

**Verdict:** UNSAFE and live. The highest-priority item in this ledger, because
it is the only one a third party is currently consuming.

---

## 7. Contaminated benchmark sets (cross-reference to 128.16)

`toke-model/benchmark/tasks_v2.jsonl` is 272 of 272 families corpus-derived,
19 of them in the tokenizer holdout; `tasks.jsonl` is 160 of 161, 7 in the
holdout (`training-reset-128.md:215-218`).

No result file measured on `tasks_v2.jsonl` exists on disk. The two
`eval_summary.json` files in `archive/toke-model-gate2-era-20260819/eval-results/`
name `"tasks_file": "benchmark/tasks.jsonl"` and both scored
`"pass_at_1": 0.0` — contaminated and worthless, but never published.

**Verdict:** handled by 128.16. Recorded here so the two ledgers agree.

---

## 8. Not affected

`toke-eval/docs/gate1-60-v04.md` (60/60) went through `run_benchmark.py` on the
`--language toke --solutions-dir` path with hand-written, not model-generated,
programs and no compile failures to drop. Both the README and the website
already carry the correct guard that it may not be quoted as a model result or
as a Pass@1. It stands.

---

## Follow-up stories requested

These are **requested, not filed** — `toke/docs/progress.md` is owner-edited.

1. **Correct the Gate-1 figure to 58.8% everywhere it appears, and re-open the
   Gate-1 verdict.** 588/1000 is below the declared `pass_at_1_minimum: 0.60`.
   Touches `toke-model/README.md`, `toke-model/huggingface/eval_results.json`,
   the unpublished card, `toke-website/content/roadmap.json` and three `.tkt`
   templates. Website files are owned by another lane. **P0.**
2. **Quarantine the synthetic `data/` derivatives.** `pass_at_k_summary.csv`,
   `ablation_table.csv`, `training_curve.csv`/`.json` carry no dry-run marker.
   Either stamp every dry-run output with an unremovable marker or refuse to
   write a CSV in dry-run mode at all. **P1.**
3. **Fix the live Hugging Face `model-index`** (`Functional Pass@1 = 8`,
   `verified: true`). Owner action; relates to 132.24 and 128.16. **P0.**
4. **`toke-model/README.md:5` calls 63.7% a *compile* Pass@1.** It is
   functional; the compile rate was 92.3%. **P2.**
5. **Backfill `n_samples` provenance** into the four archived `gate1_*.json`
   files, or mark them unattributable. **P2.**
6. **`toke-model` 0-byte files outside this story's scope:**
   `tests/test_eval.py`, `tests/test_prepare_data.py` (0 bytes since
   `f835382`, "initial repo scaffold" — two named test modules that collect
   zero tests and pass), and `finetune/configs/32b.yaml` plus its duplicate
   `finetune/configs/configs/32b.yaml` (0 bytes, referenced by nothing). The
   test files are the same disease as `harness/*.py`: a green suite that
   asserts nothing. **P2.**
7. **`archive/toke-benchmark/hidden_tests/` contains 79 unparseable YAML
   files** (`task-e-*`), found when the repaired harness refused to score
   them. Archive-only; record or delete. **P3.**

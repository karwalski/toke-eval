#!/usr/bin/env python3
"""Positive attribution check for the Gate-1 result files (story 128.22).

Question: did any recorded ``gate1_*.json`` come through the inflated
best-of-N model-sampling path of the pre-128.1c harness, rather than through
the ``--solutions-dir`` path that scores one pre-generated solution per task?

The pre-128.1c ``report_to_dict()`` wrote no ``n_samples`` field, so the
artefacts alone never recorded which path produced them. This checker proves
the attribution from git objects instead of inferring it. For every Gate-1
result file it finds the commit that introduced it and checks, **at that
commit**:

  A1  the recorded ``language`` is ``"toke"``;
  A2  the harness at that commit cannot write ``"toke"`` from the model path:
      ``score_model_pass_at_1()`` returns ``generate_report(..., "toke-model",
      ...)`` with a literal, and ``"toke-model"`` is not a key of
      ``LANGUAGE_LOADERS`` (the ``--language`` choices of the solutions path);
  A3  the solutions committed alongside hold exactly one ``<task_id>.toke``
      per task, no sample-indexed files, and ``inference_stats.json`` agrees
      that one solution was generated per task — so even the generation step
      drew a single sample;
  A4  every scored task id has a committed solution file, ``tasks_evaluated``
      equals the number of task records, and no task record carries a
      best-of-N field;
  A5  for a file still present at ``HEAD``, its ``language`` and ``tasks``
      are unchanged since introduction, so the attribution applies to the live
      artefact and not only to its first version.

Nothing here reads a ``hidden_tests/`` directory: it reads result files,
solution *file names*, ``inference_stats.json`` and harness source only.

Exit status: 0 if every file is attributed to the solutions path, 1 if any
check fails, 2 on a usage or git error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULT_NAMES = (
    "gate1_pass_at_1.json",
    "gate1_v2.json",
    "gate1_v3.json",
    "gate1_v4_1000.json",
    "gate1_v5_1000.json",
)
# The benchmark tree was subtree-merged into toke-eval under benchmark/;
# before the merge the same files lived at the repository root.
PREFIXES = ("", "benchmark/")
SOLUTION_RE = re.compile(r"^task-a-\d{4}\.toke$")
BEST_OF_N_KEYS = {"n_samples", "samples", "best_of_n", "mean_best_of_n", "sample_idx"}


class CheckError(RuntimeError):
    """A git object the check needs could not be read."""


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def introducing_commit(repo: Path, name: str) -> tuple[str, str]:
    """Return (commit, prefix) of the first commit that added the file."""
    for prefix in PREFIXES:
        out = git(repo, "log", "--all", "--full-history", "--diff-filter=A",
                  "--format=%H",
                  "--", f"{prefix}results/{name}").split()
        if out:
            return out[-1], prefix
    raise CheckError(f"no commit introduces results/{name}")


def show(repo: Path, commit: str, path: str) -> str:
    return git(repo, "show", f"{commit}:{path}")


def model_path_literal(source: str) -> str | None:
    """Language literal the model path passes to generate_report, if any."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "score_model_pass_at_1":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id == "generate_report" and len(sub.args) >= 2
                        and isinstance(sub.args[1], ast.Constant)):
                    return str(sub.args[1].value)
    return None


def loader_languages(source: str) -> set[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "LANGUAGE_LOADERS"
                for t in node.targets) and isinstance(node.value, ast.Dict)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return set()


def check_file(repo: Path, name: str) -> dict:
    commit, prefix = introducing_commit(repo, name)
    checks: dict[str, dict] = {}
    report = json.loads(show(repo, commit, f"{prefix}results/{name}"))

    lang = report.get("language")
    checks["A1"] = {"language": lang, "ok": lang == "toke"}

    harness = show(repo, commit, f"{prefix}run_benchmark.py")
    literal = model_path_literal(harness)
    loaders = loader_languages(harness)
    checks["A2"] = {
        "model_path_language": literal,
        "solutions_path_languages": sorted(loaders),
        "ok": literal is not None and literal != "toke"
        and "toke" in loaders and literal not in loaders,
    }
    # The result may have been produced by an earlier harness revision than
    # the one committed beside it, so every revision reachable from the
    # introducing commit must satisfy the same property.
    revs = git(repo, "log", "--format=%H", commit, "--",
               f"{prefix}run_benchmark.py").split()
    bad_revs = []
    for rev in revs:
        src = show(repo, rev, f"{prefix}run_benchmark.py")
        lit = model_path_literal(src)
        if lit is None and "score_model_pass_at_1" not in src:
            continue  # no model path at all in this revision
        if lit in (None, "toke") or lit in loader_languages(src):
            bad_revs.append(rev[:12])
    checks["A2"]["harness_revisions_checked"] = len(revs)
    checks["A2"]["revisions_failing"] = bad_revs
    checks["A2"]["ok"] = checks["A2"]["ok"] and bool(revs) and not bad_revs

    files = git(repo, "ls-tree", "--name-only", commit, f"{prefix}solutions/").split()
    toke_files = [Path(f).name for f in files if f.endswith(".toke")]
    odd = [f for f in toke_files if not SOLUTION_RE.match(f)]
    solution_ids = {f[: -len(".toke")] for f in toke_files if SOLUTION_RE.match(f)}
    stats = json.loads(show(repo, commit, f"{prefix}solutions/inference_stats.json"))
    checks["A3"] = {
        "solution_files": len(toke_files),
        "non_task_named_files": odd[:5],
        "inference_stats": {k: stats.get(k) for k in ("total", "generated")},
        "ok": not odd and len(solution_ids) == len(toke_files)
        and stats.get("generated") == stats.get("total") == len(solution_ids),
    }

    tasks = report.get("tasks", [])
    task_ids = [t.get("task_id") for t in tasks]
    missing = sorted(set(task_ids) - solution_ids)
    extra_keys = sorted({k for t in tasks for k in t} & BEST_OF_N_KEYS)
    checks["A4"] = {
        "tasks": len(tasks),
        "tasks_evaluated": report.get("tasks_evaluated"),
        "tasks_without_solution_file": missing[:5],
        "best_of_n_keys": extra_keys,
        "ok": not missing and report.get("tasks_evaluated") == len(tasks)
        and len(set(task_ids)) == len(task_ids) and not extra_keys,
    }

    live_path = next((p for p in (f"benchmark/results/{name}", f"results/{name}")
                      if (repo / p).is_file()), None)
    if live_path is None:
        checks["A5"] = {"live": False, "ok": True}
    else:
        live = json.loads((repo / live_path).read_text(encoding="utf-8"))
        same = (live.get("language") == report.get("language")
                and live.get("tasks") == report.get("tasks"))
        checks["A5"] = {"live": live_path, "unchanged": same, "ok": same}

    ok = all(c["ok"] for c in checks.values())
    return {"file": name, "introduced_in": commit[:12], "verdict":
            "SOLUTIONS_PATH" if ok else "NOT_ATTRIBUTED", "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=Path, default=REPO,
                        help="toke-eval checkout (default: this repository)")
    parser.add_argument("--json", action="store_true", help="print full JSON")
    args = parser.parse_args(argv)
    try:
        results = [check_file(args.repo, n) for n in RESULT_NAMES]
    except (CheckError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            failed = [k for k, c in r["checks"].items() if not c["ok"]]
            tail = f" (failed: {', '.join(failed)})" if failed else ""
            print(f"{r['verdict']:<15} {r['file']:<22} introduced {r['introduced_in']}{tail}")
    bad = [r for r in results if r["verdict"] != "SOLUTIONS_PATH"]
    print(f"gate1 attribution: {len(results) - len(bad)}/{len(results)} "
          f"attributed to the solutions path", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

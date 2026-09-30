#!/usr/bin/env python3
"""BENCH-128 disjointness checker, checks D1-D8 (story 128.1b).

Normative definition: ``toke-model/docs/training-reset-128.md`` section 5.3.
This script makes the claim "BENCH-128 is disjoint from everything a model
trains or is tuned on" falsifiable. It exits non-zero unless **every** check
reports ``PASS``.

Verdicts:

``PASS``      the check ran on non-empty, comparable operands and found nothing.
``FAIL``      the check found an overlap (or its guard tripped).
``VACUOUS``   the operands could not intersect by construction: one is empty,
              or no id in one is drawn from the same namespace as any id in the
              other. A vacuous check is **never** a pass (D6).
``NOT_RUN``   an optional input the check needs was not supplied (for example
              no lane training export exists yet). Also never a pass.
``ERROR``     an input is missing, unreadable or inconsistent (D7), or the
              canonicaliser failed.

Checks:

D1  id disjointness: bench ids vs tokenizer-holdout ids, and vs the 131.20
    baseline sample ids.
D2  family disjointness: bench families vs tokenizer-holdout families
    (``_BASE`` regex, same as the corpus).
D3  family integrity: bench families vs every supplied lane training export.
D4  content disjointness across namespaces: sha256 of the masked ``tkc --min``
    form of each bench reference vs every line of the tokenizer holdout and
    training files, and vs every published solution in ``benchmark/solutions/``
    of a task that is not itself in the bench.
D5  near-duplicates: MinHash (128 permutations, 32x4 LSH bands) over character
    5-gram shingles of the masked ``--min`` text, confirmed by exact Jaccard
    >= 0.80, vs every holdout and training line; plus TF-IDF cosine >= 0.85 over
    task descriptions via ``toke-corpus/registry/firewall.py``.
D6  vacuity guard over D1 and D2.
D7  input presence: every named file exists, parses, and matches the SHA its
    manifest records for it.
D8  determinism: the report is computed twice from fresh reads and must be
    byte-identical.

This checker never opens a ``hidden_tests/`` path: a bench reference that
points into one is an ``ERROR``.

Bench manifest shape (``bench128/manifest.json``, section 5.2)::

    {"tasks": [{"task_id": "...", "stratum": "S1"|"S2", "base": "...",
                "reference": "<path relative to the workspace root>"}, ...],
     ...header fields...}

``bench128/prompts.jsonl`` holds ``{"task_id", "prompt"}`` per line.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve()
EVAL_ROOT = HERE.parent.parent
WORKSPACE = Path(os.environ.get("TOKE_WORKSPACE", EVAL_ROOT.parent))

_BASE = re.compile(r"^([A-Z]-[A-Z]+-\d+)v\d+$")  # same rule as the corpus
NAMESPACES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("corpus-variant", re.compile(r"^[A-Z]-[A-Z]+-\d+v\d+$")),
    ("corpus-family", re.compile(r"^[A-Z]-[A-Z]+-\d+$")),
    ("library", re.compile(r"^[A-Z]{2,5}-\d+$")),
    ("procedural", re.compile(r"^task-[a-z]-\d+$")),
    ("integer", re.compile(r"^\d+$")),
    ("external", re.compile(r"^[A-Za-z]+/\d+$")),
)
# A corpus variant id and its family id live in one namespace for the purpose
# of "could these ever collide": D1 compares variants, D2 compares families.
NAMESPACE_GROUP = {"corpus-variant": "corpus", "corpus-family": "corpus"}

STUB_PREFIX = "m=harness;i=io:std.io;"  # toke-corpus curate_tokenizer_set.py
SHINGLE = 5
N_PERM = 128
BANDS, ROWS = 32, 4
JACCARD_MIN = 0.80
COSINE_MIN = 0.85
_P = (1 << 31) - 1

PASS, FAIL, VACUOUS, NOT_RUN, ERROR = "PASS", "FAIL", "VACUOUS", "NOT_RUN", "ERROR"

Canon = Callable[[str], str]


class InputError(Exception):
    """D7: a named input is missing, unparseable or inconsistent."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def base_of(task_id: str) -> str:
    m = _BASE.match(task_id)
    return m.group(1) if m else task_id


def namespace(i: str) -> str:
    for name, rx in NAMESPACES:
        if rx.match(i):
            return NAMESPACE_GROUP.get(name, name)
    return "other"


def must_exist(p: Path, what: str) -> Path:
    if not p.is_file():
        raise InputError(f"{what}: {p} does not exist")
    if "hidden_tests" in p.parts:
        raise InputError(f"{what}: {p} is under hidden_tests/ (AGENTS.md section 5)")
    return p


def load_json(p: Path, what: str) -> object:
    must_exist(p, what)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"{what}: {p} does not parse: {exc}") from exc


def read_lines(p: Path, what: str) -> list[str]:
    must_exist(p, what)
    return p.read_text(encoding="utf-8").split("\n")[:-1] if p.stat().st_size else []


def read_id_file(p: Path, what: str) -> list[str]:
    """First whitespace field of every non-comment line. Missing is an error."""
    ids = []
    for line in read_lines(p, what):
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s.split()[0])
    if not ids:
        raise InputError(f"{what}: {p} holds no ids")
    return ids


def read_jsonl(p: Path, what: str) -> list[dict]:
    rows = []
    for n, line in enumerate(read_lines(p, what), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputError(f"{what}: {p}:{n} does not parse: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# canonical form: pinned `tkc --min` + escape-aware string mask
# ---------------------------------------------------------------------------

def load_mask(toke_repo: Path) -> Callable[[str], str]:
    path = must_exist(toke_repo / "scripts" / "patterns" / "mask_strings.py",
                      "mask_strings.py (toke repo)")
    spec = importlib.util.spec_from_file_location("mask_strings", path)
    if spec is None or spec.loader is None:
        raise InputError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mask_strings  # type: ignore[no-any-return]


def make_tkc_canon(tkc: Path, mask: Callable[[str], str], timeout: float = 30.0) -> Canon:
    must_exist(tkc, "tkc binary")

    def canon(src: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".tk", delete=False,
                                         encoding="utf-8") as f:
            f.write(src)
            path = Path(f.name)
        try:
            proc = subprocess.run([str(tkc), "--min", str(path)], capture_output=True,
                                  text=True, timeout=timeout, check=False)
        finally:
            path.unlink()
        out = proc.stdout.rstrip("\n")
        if proc.returncode != 0 or not out.strip() or "\n" in out:
            raise RuntimeError(f"tkc --min failed (exit {proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[:200]}")
        return mask(out)

    return canon


def reference_source(path: Path) -> str:
    """A reference is a toke source file or a corpus record carrying tk_source."""
    if path.suffix == ".json":
        rec = load_json(path, "bench reference record")
        if not isinstance(rec, dict) or not isinstance(rec.get("tk_source"), str):
            raise InputError(f"bench reference {path} has no tk_source")
        return rec["tk_source"]
    must_exist(path, "bench reference")
    return path.read_text(encoding="utf-8")


def line_variants(masked: str) -> set[str]:
    """The forms a program can take as a tokenizer line (stub may be stripped)."""
    out = {masked}
    if masked.startswith(STUB_PREFIX):
        out.add(masked[len(STUB_PREFIX):])
    return out


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------

def _perms() -> list[tuple[int, int]]:
    rng = random.Random(128)
    return [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(N_PERM)]


PERMS = _perms()


def shingles(text: str) -> set[int]:
    grams = {text[i:i + SHINGLE] for i in range(max(1, len(text) - SHINGLE + 1))}
    return {int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(),
                           "big") % _P for g in grams}


_PERM_A = None
_PERM_B = None


def signature(sh: set[int]) -> tuple[int, ...]:
    """MinHash signature; vectorised (a*x+b mod p, p = 2^31-1 keeps it in uint64)."""
    import numpy as np  # local: only D5 needs it

    global _PERM_A, _PERM_B
    if _PERM_A is None:
        _PERM_A = np.array([a for a, _ in PERMS], dtype=np.uint64)[:, None]
        _PERM_B = np.array([b for _, b in PERMS], dtype=np.uint64)[:, None]
    x = np.fromiter(sorted(sh), dtype=np.uint64, count=len(sh))[None, :]
    return tuple(int(v) for v in ((_PERM_A * x + _PERM_B) % np.uint64(_P)).min(axis=1))


def bands(sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    return [(i, sig[i * ROWS:(i + 1) * ROWS]) for i in range(BANDS)]


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    bench_dir: Path
    labels: list[str]
    corpus_repo: Path
    tokenizer_repo: Path
    solutions_dir: Path
    workspace: Path
    exports: list[Path] = field(default_factory=list)
    train_descriptions: Path | None = None
    tkc_sha256: str | None = None
    firewall_repo: Path | None = None  # where registry/firewall.py lives; default corpus_repo


def id_intersection(name: str, bench: set[str], other: set[str]) -> dict:
    b_ns = sorted({namespace(i) for i in bench})
    o_ns = sorted({namespace(i) for i in other})
    shared = sorted(set(b_ns) & set(o_ns))
    comparable = sorted(i for i in bench if namespace(i) in shared)
    hit = sorted(bench & other)
    if hit:
        verdict = FAIL
    elif not bench or not other or not shared:
        verdict = VACUOUS
    else:
        verdict = PASS
    return {"name": name, "bench_n": len(bench), "other_n": len(other),
            "bench_namespaces": b_ns, "other_namespaces": o_ns,
            "shared_namespaces": shared, "bench_n_comparable": len(comparable),
            "intersection_n": len(hit), "intersection_sample": hit[:20],
            "verdict": verdict}


def combine(verdicts: list[str]) -> str:
    for v in (ERROR, FAIL, VACUOUS, NOT_RUN):
        if v in verdicts:
            return v
    return PASS if verdicts else NOT_RUN


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def run_checks(cfg: Config, canon: Canon | None) -> dict:
    report: dict = {"schema": "bench128_disjoint v1 (story 128.1b)",
                    "protocol": "toke-model/docs/training-reset-128.md section 5.3",
                    "inputs": {}, "checks": {}}
    inputs = report["inputs"]
    checks = report["checks"]

    # ---- D7: every named input exists, parses and matches its recorded SHA
    d7_errors: list[str] = []
    bench: list[dict] = []
    labels: dict[str, dict] = {}
    try:
        man_path = cfg.bench_dir / "manifest.json"
        man = load_json(man_path, "bench manifest")
        inputs["bench_manifest"] = {"path": "bench128/manifest.json",
                                    "sha256": sha256_file(man_path)}
        if not isinstance(man, dict) or not isinstance(man.get("tasks"), list) \
                or not man["tasks"]:
            raise InputError("bench manifest: no tasks[]")
        bench = man["tasks"]
        for t in bench:
            if not isinstance(t, dict) or not isinstance(t.get("task_id"), str):
                raise InputError("bench manifest: a task has no task_id")
        prompts_path = cfg.bench_dir / "prompts.jsonl"
        prompts = read_jsonl(prompts_path, "bench prompts")
        inputs["bench_prompts"] = {"path": "bench128/prompts.jsonl",
                                   "sha256": sha256_file(prompts_path)}
    except InputError as exc:
        d7_errors.append(str(exc))
        prompts = []

    for label in cfg.labels:
        try:
            data = cfg.corpus_repo / "data"
            mp = data / f"tokenizer_manifest_{label}.json"
            tm = load_json(mp, f"tokenizer manifest {label}")
            if not isinstance(tm, dict) or not isinstance(tm.get("records"), list):
                raise InputError(f"tokenizer manifest {label}: no records[]")
            txt = {}
            for kind in ("holdout", "training"):
                p = data / f"tokenizer_{kind}_{label}.txt"
                must_exist(p, f"tokenizer {kind} {label}")
                sha = sha256_file(p)
                want = (tm.get("outputs") or {}).get(p.name, {}).get("sha256")
                if want is None:
                    raise InputError(f"tokenizer manifest {label} records no sha256 "
                                     f"for {p.name}")
                if want != sha:
                    raise InputError(f"{p.name}: sha256 {sha[:12]} does not match the "
                                     f"manifest's {want[:12]}")
                txt[kind] = read_lines(p, f"tokenizer {kind} {label}")
                inputs[p.name] = {"sha256": sha, "lines": len(txt[kind])}
            inputs[mp.name] = {"sha256": sha256_file(mp)}
            labels[label] = {"manifest": tm, "txt": txt,
                             "tkc_sha256": (tm.get("tkc") or {}).get("binary_sha256")}
        except InputError as exc:
            d7_errors.append(str(exc))

    try:
        bp = cfg.tokenizer_repo / "data" / "baseline_sample_ids_v04.txt"
        baseline_ids = set(read_id_file(bp, "baseline sample ids"))
        inputs[bp.name] = {"sha256": sha256_file(bp), "ids": len(baseline_ids)}
    except InputError as exc:
        d7_errors.append(str(exc))
        baseline_ids = set()

    exports: dict[str, set[str]] = {}
    for ep in cfg.exports:
        try:
            fams: set[str] = set()
            for row in read_jsonl(ep, "training export"):
                tid = row.get("base") or row.get("task_id")
                if not isinstance(tid, str):
                    raise InputError(f"training export {ep}: a row has no task_id/base")
                fams.add(base_of(tid))
            exports[ep.name] = fams
            inputs[ep.name] = {"sha256": sha256_file(ep), "families": len(fams)}
        except InputError as exc:
            d7_errors.append(str(exc))

    train_desc: list[dict] = []
    if cfg.train_descriptions is not None:
        try:
            train_desc = read_jsonl(cfg.train_descriptions, "training descriptions")
            inputs[cfg.train_descriptions.name] = {
                "sha256": sha256_file(cfg.train_descriptions), "rows": len(train_desc)}
        except InputError as exc:
            d7_errors.append(str(exc))

    if not cfg.solutions_dir.is_dir():
        d7_errors.append(f"solutions dir {cfg.solutions_dir} does not exist")
    for label, lab in labels.items():
        if cfg.tkc_sha256 is not None and lab["tkc_sha256"] != cfg.tkc_sha256:
            d7_errors.append(
                f"tkc pin: the binary in use ({cfg.tkc_sha256[:12]}) is not the one "
                f"tokenizer manifest {label} was built with "
                f"({str(lab['tkc_sha256'])[:12]}); --min forms would not be comparable")
    if cfg.tkc_sha256 is not None:
        inputs["tkc_sha256"] = cfg.tkc_sha256

    checks["D7"] = {"errors": d7_errors, "verdict": ERROR if d7_errors else PASS}
    if d7_errors:
        for d in ("D1", "D2", "D3", "D4", "D5", "D6"):
            checks[d] = {"verdict": NOT_RUN, "reason": "D7 failed"}
        return report

    bench_ids = {t["task_id"] for t in bench}
    bench_fams = {t.get("base") or base_of(t["task_id"]) for t in bench}

    # ---- D1 / D2 (with D6 vacuity classification built in)
    d1, d2 = [], []
    for label, lab in labels.items():
        recs = lab["manifest"]["records"]
        hold_ids = {r["task_id"] for r in recs if r.get("split") == "holdout"}
        hold_fams = {r.get("base") or base_of(r["task_id"])
                     for r in recs if r.get("split") == "holdout"}
        d1.append(id_intersection(f"bench_ids x tokholdout_ids[{label}]",
                                  bench_ids, hold_ids))
        d2.append(id_intersection(f"bench_families x tokholdout_families[{label}]",
                                  bench_fams, hold_fams))
    d1.append(id_intersection("bench_ids x baseline_ids", bench_ids, baseline_ids))
    checks["D1"] = {"comparisons": d1, "verdict": combine([c["verdict"] for c in d1])}
    checks["D2"] = {"comparisons": d2, "verdict": combine([c["verdict"] for c in d2])}

    # ---- D6: vacuity guard over D1 and D2
    vac = [c["name"] for c in d1 + d2 if c["verdict"] == VACUOUS]
    checks["D6"] = {"vacuous": vac, "verdict": FAIL if vac else PASS,
                    "note": "a vacuous D1/D2 comparison is reported, never passed; "
                            "procedural ids (task-a-NNNN) cannot meet corpus ids, "
                            "so for them D4/D5 carry the evidence"}

    # ---- D3: family integrity against every lane training export
    if not exports:
        checks["D3"] = {"verdict": NOT_RUN,
                        "reason": "no lane training export supplied (--export); "
                                  "the 131.29 export does not exist yet"}
    else:
        d3 = [id_intersection(f"bench_families x export[{n}]", bench_fams, fams)
              for n, fams in sorted(exports.items())]
        checks["D3"] = {"comparisons": d3, "verdict": combine([c["verdict"] for c in d3])}

    # ---- D4 / D5 need the canonical text of every bench reference
    if canon is None:
        for d in ("D4", "D5"):
            checks[d] = {"verdict": NOT_RUN, "reason": "no canonicaliser (tkc/mask)"}
        return report

    canon_err: list[str] = []
    bench_text: dict[str, str] = {}
    for t in sorted(bench, key=lambda t: t["task_id"]):
        ref = t.get("reference")
        if not isinstance(ref, str):
            canon_err.append(f"{t['task_id']}: no reference")
            continue
        try:
            p = (cfg.workspace / ref).resolve()
            if "hidden_tests" in p.parts:
                raise InputError(f"reference {ref} is under hidden_tests/")
            bench_text[t["task_id"]] = canon(reference_source(p))
        except (InputError, RuntimeError, OSError) as exc:
            canon_err.append(f"{t['task_id']}: {exc}")

    sol_hashes: dict[str, str] = {}
    sol_files = sorted(cfg.solutions_dir.glob("*.toke"))

    def _canon_sol(p: Path) -> tuple[str, str | None, str | None]:
        try:
            return p.stem, sha256_bytes(canon(p.read_text(encoding="utf-8"))
                                        .encode("utf-8")), None
        except (RuntimeError, OSError) as exc:
            return p.stem, None, str(exc)

    sol_skipped = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for stem, h, err in pool.map(_canon_sol, sol_files):
            if h is None:
                sol_skipped.append(f"{stem}: {err}")
            else:
                sol_hashes[stem] = h

    # D4 exact content
    bench_hash = {tid: {sha256_bytes(v.encode("utf-8")) for v in line_variants(txt)}
                  for tid, txt in bench_text.items()}
    d4_cmp = []
    for label, lab in labels.items():
        for kind, lines in sorted(lab["txt"].items()):
            line_h = {sha256_bytes(line.encode("utf-8")) for line in lines}
            hits = sorted(tid for tid, hs in bench_hash.items() if hs & line_h)
            d4_cmp.append({"name": f"bench_min x tokenizer_{kind}[{label}]",
                           "bench_n": len(bench_hash), "other_n": len(line_h),
                           "intersection_n": len(hits), "intersection_sample": hits[:20],
                           "verdict": FAIL if hits else
                           (VACUOUS if not bench_hash or not line_h else PASS)})
    # Only solutions of tasks outside the bench can leak it: two bench tasks
    # sharing a solution is a duplicate inside the bench, not contamination.
    outside = {h for s, h in sol_hashes.items() if s not in bench_ids}
    sol_hits = sorted(tid for tid, hs in bench_hash.items() if hs & outside)
    d4_cmp.append({"name": "bench_min x benchmark/solutions (tasks not in the bench)",
                   "bench_n": len(bench_hash), "other_n": len(outside),
                   "solutions_not_canonicalisable": len(sol_skipped),
                   "intersection_n": len(sol_hits), "intersection_sample": sol_hits[:20],
                   "verdict": FAIL if sol_hits else
                   (VACUOUS if not bench_hash or not outside else PASS)})
    v4 = [c["verdict"] for c in d4_cmp]
    if canon_err:
        v4.append(ERROR)
    checks["D4"] = {"comparisons": d4_cmp, "bench_not_canonicalisable": canon_err[:50],
                    "bench_not_canonicalisable_n": len(canon_err),
                    "verdict": combine(v4)}

    # D5a MinHash near-duplicates over code
    bench_sh = {tid: shingles(txt) for tid, txt in bench_text.items()}
    index: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for tid in sorted(bench_sh):
        for b in bands(signature(bench_sh[tid])):
            index.setdefault(b, []).append(tid)
    d5_cmp = []
    for label, lab in labels.items():
        for kind, lines in sorted(lab["txt"].items()):
            pairs = []
            for n, line in enumerate(lines):
                sh = shingles(line)
                cands = {tid for b in bands(signature(sh)) for tid in index.get(b, ())}
                for tid in sorted(cands):
                    j = jaccard(bench_sh[tid], sh)
                    if j >= JACCARD_MIN:
                        pairs.append({"task_id": tid, "line": n + 1, "jaccard": round(j, 4)})
            d5_cmp.append({"name": f"minhash bench x tokenizer_{kind}[{label}]",
                           "bench_n": len(bench_sh), "other_n": len(lines),
                           "pairs_n": len(pairs), "pairs_sample": pairs[:20],
                           "verdict": FAIL if pairs else
                           (VACUOUS if not bench_sh or not lines else PASS)})

    # D5b TF-IDF over descriptions
    if cfg.train_descriptions is None:
        d5_cmp.append({"name": "tfidf bench prompts x training descriptions",
                       "verdict": NOT_RUN,
                       "reason": "no --train-descriptions supplied"})
    else:
        d5_cmp.append(tfidf_check(cfg, prompts, train_desc))
    checks["D5"] = {"comparisons": d5_cmp, "shingle": f"char {SHINGLE}-gram",
                    "permutations": N_PERM, "lsh": f"{BANDS}x{ROWS}",
                    "jaccard_min": JACCARD_MIN, "cosine_min": COSINE_MIN,
                    "verdict": combine([c["verdict"] for c in d5_cmp])}
    return report


def tfidf_check(cfg: Config, prompts: list[dict], train_desc: list[dict]) -> dict:
    sys.path.insert(0, str(cfg.firewall_repo or cfg.corpus_repo))
    try:
        firewall = importlib.import_module("registry.firewall")
    except ImportError as exc:
        return {"name": "tfidf bench prompts x training descriptions",
                "verdict": ERROR, "reason": f"cannot import firewall: {exc}"}
    finally:
        sys.path.pop(0)
    ptext = [(p.get("task_id"), p.get("prompt")) for p in prompts]
    ptext = [(t, s) for t, s in ptext if isinstance(t, str) and isinstance(s, str)]
    fw = firewall.ContaminationFirewall([], similarity_threshold=COSINE_MIN)
    fw.add_evaluation_descriptions([s for _, s in ptext])
    pairs = []
    for row in train_desc:
        desc = row.get("description")
        if not isinstance(desc, str):
            continue
        _, score, match = fw.check_similarity(desc)
        if score >= COSINE_MIN and match is not None:
            tid = ptext[int(match.split("-", 1)[1])][0]
            pairs.append({"task_id": tid, "train_id": row.get("task_id"),
                          "cosine": round(score, 4)})
    pairs.sort(key=lambda p: (p["task_id"], str(p["train_id"])))
    return {"name": "tfidf bench prompts x training descriptions",
            "bench_n": len(ptext), "other_n": len(train_desc),
            "pairs_n": len(pairs), "pairs_sample": pairs[:20],
            "verdict": FAIL if pairs else
            (VACUOUS if not ptext or not train_desc else PASS)}


def render(report: dict) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def evaluate(cfg: Config, canon: Canon | None) -> tuple[dict, bytes]:
    """Run D1-D7 twice from fresh reads; D8 is their byte equality."""
    first = render(run_checks(cfg, canon))
    second_report = run_checks(cfg, canon)
    second = render(second_report)
    second_report["checks"]["D8"] = {
        "verdict": PASS if first == second else FAIL,
        "sha256_run1": sha256_bytes(first), "sha256_run2": sha256_bytes(second)}
    verdicts = {k: v["verdict"] for k, v in sorted(second_report["checks"].items())}
    second_report["verdicts"] = verdicts
    second_report["overall"] = PASS if all(v == PASS for v in verdicts.values()) else FAIL
    return second_report, render(second_report)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench-dir", type=Path, default=EVAL_ROOT / "benchmark" / "bench128")
    ap.add_argument("--label", action="append", dest="labels",
                    help="tokenizer manifest label; repeat for pre131 and v04 "
                         "(default: pre131)")
    ap.add_argument("--workspace", type=Path, default=WORKSPACE,
                    help="root that bench references are relative to")
    ap.add_argument("--corpus-repo", type=Path, default=WORKSPACE / "toke-corpus")
    ap.add_argument("--tokenizer-repo", type=Path, default=WORKSPACE / "toke-tokenizer")
    ap.add_argument("--toke-repo", type=Path, default=WORKSPACE / "toke")
    ap.add_argument("--tkc", type=Path, default=None,
                    help="the pinned tkc binary (default: <toke-repo>/toke)")
    ap.add_argument("--solutions-dir", type=Path,
                    default=EVAL_ROOT / "benchmark" / "solutions")
    ap.add_argument("--export", type=Path, action="append", default=[],
                    help="a lane training export (JSONL with task_id or base)")
    ap.add_argument("--train-descriptions", type=Path, default=None,
                    help="JSONL {task_id, description} of training task descriptions")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default: <bench-dir>/bench128_disjoint.json)")
    args = ap.parse_args(argv)

    tkc = args.tkc or args.toke_repo / "toke"
    canon: Canon | None = None
    tkc_sha = None
    try:
        canon = make_tkc_canon(tkc, load_mask(args.toke_repo))
        tkc_sha = sha256_file(tkc)
    except InputError as exc:
        print(f"ERROR (D7): {exc}", file=sys.stderr)

    cfg = Config(bench_dir=args.bench_dir, labels=args.labels or ["pre131"],
                 corpus_repo=args.corpus_repo, tokenizer_repo=args.tokenizer_repo,
                 solutions_dir=args.solutions_dir, workspace=args.workspace,
                 exports=args.export, train_descriptions=args.train_descriptions,
                 tkc_sha256=tkc_sha)
    report, blob = evaluate(cfg, canon)
    if canon is None:
        report["checks"]["D7"]["verdict"] = ERROR
        report["checks"]["D7"]["errors"].append("canonicaliser unavailable (tkc/mask)")
        report["verdicts"]["D7"] = ERROR
        report["overall"] = FAIL
        blob = render(report)

    out = args.out or args.bench_dir / "bench128_disjoint.json"
    if args.bench_dir.is_dir():
        out.write_bytes(blob)
    for k, v in report["verdicts"].items():
        print(f"{k}  {v}")
    print(f"bench128 disjointness: {report['overall']}", file=sys.stderr)
    return 0 if report["overall"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())

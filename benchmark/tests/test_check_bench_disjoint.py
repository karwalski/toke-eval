"""Tests for scripts/check_bench_disjoint.py (story 128.1b).

Every fixture is synthetic: no BENCH-128 data, no tkc, no toke repo. The
canonicaliser is injected (newline-joining stands in for ``tkc --min``), so the
checks are exercised on their own logic. The TF-IDF half of D5 uses the real
``toke-corpus/registry/firewall.py`` when a sibling checkout is present.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import string
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_bench_disjoint.py"
_spec = importlib.util.spec_from_file_location("check_bench_disjoint", SCRIPT)
assert _spec is not None and _spec.loader is not None
cbd = importlib.util.module_from_spec(_spec)
sys.modules["check_bench_disjoint"] = cbd  # dataclasses resolve the module by name
_spec.loader.exec_module(cbd)

FIREWALL_REPO = cbd.WORKSPACE / "toke-corpus"


def canon(src: str) -> str:
    return ";".join(line.strip() for line in src.splitlines() if line.strip())


def program(seed: int) -> str:
    rng = random.Random(seed)
    body = "".join(rng.choice(string.ascii_lowercase + "=;(){}+*<>")
                   for _ in range(300))
    return f"m=p{seed};\nf=main():i64{{\n{body}\n}}\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build(tmp_path: Path, *, bench: list[dict] | None = None,
          holdout: dict[str, str] | None = None,
          training: dict[str, str] | None = None,
          baseline: list[str] | None = None,
          solutions: dict[str, str] | None = None,
          export: list[str] | None = None,
          descriptions: list[dict] | None = None,
          prompts: dict[str, str] | None = None,
          corrupt_holdout_sha: bool = False,
          omit_baseline: bool = False) -> cbd.Config:
    ws = tmp_path / "ws"
    data = ws / "toke-corpus" / "data"
    data.mkdir(parents=True)
    holdout = holdout if holdout is not None else {
        "A-ARR-0001v1": program(1), "A-ARR-0001v2": program(2)}
    training = training if training is not None else {
        "A-STR-0007v1": program(7), "A-STR-0008v3": program(8)}
    records, outputs = [], {}
    for split, recs in (("holdout", holdout), ("training", training)):
        name = f"tokenizer_{split}_x.txt"
        text = "".join(canon(src) + "\n" for src in recs.values())
        (data / name).write_text(text)
        outputs[name] = {"sha256": _sha(text.encode())}
        for tid in recs:
            records.append({"task_id": tid, "base": cbd.base_of(tid),
                            "split": "holdout" if split == "holdout" else "train"})
    if corrupt_holdout_sha:
        outputs["tokenizer_holdout_x.txt"]["sha256"] = "0" * 64
    (data / "tokenizer_manifest_x.json").write_text(json.dumps(
        {"records": records, "outputs": outputs, "tkc": {"binary_sha256": None}}))

    tok = ws / "toke-tokenizer" / "data"
    tok.mkdir(parents=True)
    if not omit_baseline:
        ids = baseline if baseline is not None else ["A-MTH-0100v1", "A-MTH-0101v2"]
        (tok / "baseline_sample_ids_v04.txt").write_text(
            "# header\n" + "".join(f"{i}\tA-MTH\t1\tsha\n" for i in ids))

    sol = ws / "toke-eval" / "benchmark" / "solutions"
    sol.mkdir(parents=True)
    solutions = solutions if solutions is not None else {
        "task-a-0501": program(501), "task-a-0001": program(9001)}
    for tid, src in solutions.items():
        (sol / f"{tid}.toke").write_text(src)

    refs = ws / "refs"
    refs.mkdir()
    if bench is None:
        bench = [{"task_id": "A-CND-0050v4", "stratum": "S2", "src": program(50)},
                 {"task_id": "task-a-0501", "stratum": "S1", "src": program(501),
                  "reference": "toke-eval/benchmark/solutions/task-a-0501.toke"}]
    tasks = []
    for t in bench:
        t = dict(t)
        src = t.pop("src", None)
        if "reference" not in t and src is not None:
            (refs / f"{t['task_id']}.tk").write_text(src)
            t["reference"] = f"refs/{t['task_id']}.tk"
        tasks.append(t)
    bdir = ws / "toke-eval" / "benchmark" / "bench128"
    bdir.mkdir(parents=True)
    (bdir / "manifest.json").write_text(json.dumps({"tasks": tasks}))
    prompts = prompts if prompts is not None else {
        t["task_id"]: f"task {t['task_id']}: compute something distinct" for t in tasks}
    (bdir / "prompts.jsonl").write_text("".join(
        json.dumps({"task_id": k, "prompt": v}) + "\n" for k, v in prompts.items()))

    exports = []
    if export is not None:
        ep = ws / "export.jsonl"
        ep.write_text("".join(json.dumps({"task_id": i}) + "\n" for i in export))
        exports.append(ep)
    desc_path = None
    if descriptions is not None:
        desc_path = ws / "descriptions.jsonl"
        desc_path.write_text("".join(json.dumps(d) + "\n" for d in descriptions))

    return cbd.Config(bench_dir=bdir, labels=["x"], corpus_repo=ws / "toke-corpus",
                      tokenizer_repo=ws / "toke-tokenizer", solutions_dir=sol,
                      workspace=ws, exports=exports, train_descriptions=desc_path,
                      firewall_repo=FIREWALL_REPO)


def run(cfg: cbd.Config) -> dict:
    report, _ = cbd.evaluate(cfg, canon)
    return report


def clean_config(tmp_path: Path) -> cbd.Config:
    return build(tmp_path, export=["A-STR-0007v1", "A-STR-0008v3"],
                 descriptions=[{"task_id": "A-STR-0007v1",
                                "description": "reverse the words of a sentence"}])


needs_firewall = pytest.mark.skipif(
    not (FIREWALL_REPO / "registry" / "firewall.py").is_file(),
    reason="toke-corpus sibling checkout not present")


@needs_firewall
def test_clean_bench_passes_every_check(tmp_path: Path) -> None:
    r = run(clean_config(tmp_path))
    assert r["verdicts"] == {f"D{i}": "PASS" for i in range(1, 9)}, r["verdicts"]
    assert r["overall"] == "PASS"


def test_procedural_only_bench_is_vacuous_not_pass(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[{"task_id": "task-a-0501", "stratum": "S1",
                                  "reference": "toke-eval/benchmark/solutions/task-a-0501.toke"}],
                export=["A-STR-0007v1"])
    r = run(cfg)
    assert r["checks"]["D1"]["verdict"] == "VACUOUS"
    assert r["checks"]["D2"]["verdict"] == "VACUOUS"
    assert r["checks"]["D6"]["verdict"] == "FAIL"
    assert r["overall"] == "FAIL"


def test_id_overlap_with_holdout_fails_d1(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[{"task_id": "A-ARR-0001v1", "src": program(77)}])
    r = run(cfg)
    assert r["checks"]["D1"]["verdict"] == "FAIL"


def test_id_overlap_with_baseline_fails_d1(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[{"task_id": "A-MTH-0100v1", "src": program(77)}])
    r = run(cfg)
    assert r["checks"]["D1"]["verdict"] == "FAIL"


def test_family_overlap_fails_d2_even_with_distinct_ids(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[{"task_id": "A-ARR-0001v9", "src": program(77)}])
    r = run(cfg)
    assert r["checks"]["D1"]["verdict"] == "PASS"
    assert r["checks"]["D2"]["verdict"] == "FAIL"


def test_family_in_training_export_fails_d3(tmp_path: Path) -> None:
    cfg = build(tmp_path, export=["A-CND-0050v1"])
    assert run(cfg)["checks"]["D3"]["verdict"] == "FAIL"


def test_no_export_is_not_run_and_blocks_overall(tmp_path: Path) -> None:
    r = run(build(tmp_path))
    assert r["checks"]["D3"]["verdict"] == "NOT_RUN"
    assert r["overall"] == "FAIL"


def test_content_leak_across_namespaces_fails_d4(tmp_path: Path) -> None:
    # A procedural id can never meet a corpus id, but its text can.
    cfg = build(tmp_path, bench=[
        {"task_id": "A-CND-0050v4", "src": program(50)},
        {"task_id": "task-a-0777", "src": program(7)}], export=[])
    r = run(cfg)
    names = {c["name"]: c for c in r["checks"]["D4"]["comparisons"]}
    assert names["bench_min x tokenizer_training[x]"]["verdict"] == "FAIL"
    assert names["bench_min x tokenizer_training[x]"]["intersection_sample"] == ["task-a-0777"]


def test_solution_of_task_outside_bench_fails_d4(tmp_path: Path) -> None:
    cfg = build(tmp_path, solutions={"task-a-0501": program(501),
                                     "task-a-0030": program(501)})
    comps = run(cfg)["checks"]["D4"]["comparisons"]
    assert comps[-1]["verdict"] == "FAIL"


def test_duplicate_inside_bench_is_not_a_leak(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[
        {"task_id": "A-CND-0050v4", "src": program(50)},
        {"task_id": "task-a-0501", "reference": "toke-eval/benchmark/solutions/task-a-0501.toke"},
        {"task_id": "task-a-0809", "reference": "toke-eval/benchmark/solutions/task-a-0809.toke"}],
        solutions={"task-a-0501": program(501), "task-a-0809": program(501),
                   "task-a-0001": program(9001)})
    comps = run(cfg)["checks"]["D4"]["comparisons"]
    assert comps[-1]["verdict"] == "PASS"


def test_near_duplicate_fails_d5_not_d4(tmp_path: Path) -> None:
    near = program(8).replace("m=p8;", "m=q8;")
    cfg = build(tmp_path, bench=[{"task_id": "A-CND-0050v4", "src": near}])
    r = run(cfg)
    assert r["checks"]["D4"]["verdict"] != "FAIL"
    mh = [c for c in r["checks"]["D5"]["comparisons"] if c["name"].startswith("minhash")]
    assert any(c["verdict"] == "FAIL" for c in mh)


@needs_firewall
def test_description_near_duplicate_fails_d5(tmp_path: Path) -> None:
    cfg = build(tmp_path, export=["A-STR-0007v1"],
                prompts={"A-CND-0050v4": "sort a list of integers in ascending order",
                         "task-a-0501": "count vowels in a string"},
                descriptions=[{"task_id": "A-STR-0007v1",
                               "description": "sort a list of integers in ascending order"}])
    r = run(cfg)
    tf = next(c for c in r["checks"]["D5"]["comparisons"] if c["name"].startswith("tfidf"))
    assert tf["verdict"] == "FAIL"


def test_missing_input_is_an_error_not_an_empty_set(tmp_path: Path) -> None:
    r = run(build(tmp_path, omit_baseline=True))
    assert r["checks"]["D7"]["verdict"] == "ERROR"
    assert r["checks"]["D1"]["verdict"] == "NOT_RUN"
    assert r["overall"] == "FAIL"


def test_input_sha_mismatch_is_an_error(tmp_path: Path) -> None:
    r = run(build(tmp_path, corrupt_holdout_sha=True))
    assert r["checks"]["D7"]["verdict"] == "ERROR"


def test_tkc_pin_mismatch_is_an_error(tmp_path: Path) -> None:
    cfg = build(tmp_path)
    cfg.tkc_sha256 = "f" * 64
    assert run(cfg)["checks"]["D7"]["verdict"] == "ERROR"


def test_hidden_tests_reference_is_refused(tmp_path: Path) -> None:
    cfg = build(tmp_path, bench=[{"task_id": "A-CND-0050v4",
                                  "reference": "toke-eval/benchmark/hidden_tests/x.yaml"}])
    r = run(cfg)
    assert r["checks"]["D4"]["verdict"] == "ERROR"
    assert "hidden_tests" in r["checks"]["D4"]["bench_not_canonicalisable"][0]


def test_report_is_deterministic(tmp_path: Path) -> None:
    cfg = clean_config(tmp_path)
    _, a = cbd.evaluate(cfg, canon)
    _, b = cbd.evaluate(cfg, canon)
    assert a == b
    assert json.loads(a)["checks"]["D8"]["verdict"] == "PASS"


def test_cli_without_bench_exits_nonzero(tmp_path: Path) -> None:
    rc = cbd.main(["--bench-dir", str(tmp_path / "absent"), "--workspace", str(tmp_path),
                   "--toke-repo", str(tmp_path / "no-toke")])
    assert rc == 1

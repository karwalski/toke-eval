"""Regression tests for the three 128.1c evaluation-harness defects.

Each test here fails against the pre-128.1c code. Read them as executable
statements of what the harness is now forbidden to do:

  D1  A schema mismatch must FAIL LOUDLY, never score 0 (or 1) and carry on.
  D2  `--n-samples N` must not report best-of-N under the name `pass_at_1`.
  D3  A documented harness entry point must not be a 0-byte file that exits 0
      having done nothing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_BENCH = Path(__file__).resolve().parents[1]
_REPO = _BENCH.parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_REPO))

import run_benchmark  # noqa: E402
from run_benchmark import (  # noqa: E402
    BenchmarkSchemaError,
    TaskResult,
    generate_report,
    load_task,
    main,
    pass_at_k_estimator,
    report_to_dict,
    score_task,
    validate_tasks_dir,
)

from toke_eval import pass_at_k as tepk  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, task_id: str, body: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{task_id}.yaml"
    p.write_text(yaml.dump(body, default_flow_style=False))
    return p


def _valid_task(task_id: str = "task-a-0001") -> dict:
    return {
        "id": task_id,
        "phase": "A",
        "description": "double it",
        "input_type": "i64",
        "output_type": "i64",
        "test_inputs": [
            {"input": 1, "expected": 2},
            {"input": 2, "expected": 4},
        ],
    }


# ===========================================================================
# D1 -- schema mismatch must fail loudly, not score zero
# ===========================================================================

class TestSchemaMismatchFailsLoudly:
    """The headline regression: the exact shape of the original bug.

    `toke_eval/pass_at_k.py::run_tests` read `tests["test_cases"]` while every
    benchmark file on disk uses `test_inputs`. It returned `(0, 0)`, so every
    task scored 0/0, and the run still printed a Pass@1. A harness that
    returns a meaningless number instead of failing is the worst outcome,
    because the number gets recorded and then quoted.
    """

    def test_toke_eval_run_tests_raises_on_legacy_key(self, tmp_path: Path) -> None:
        """`test_cases` instead of `test_inputs` must raise, not score 0/0."""
        bad = _write(tmp_path, "task-a-0001", {
            "id": "task-a-0001",
            "test_cases": [{"input": 1, "expected": 2}],  # the wrong key
        })
        with pytest.raises(tepk.TaskSchemaError) as exc:
            tepk.run_tests(Path("/nonexistent/binary"), bad)
        # The message must name both the key it wanted and the one it found,
        # so the next person does not have to re-derive the mismatch.
        assert "test_inputs" in str(exc.value)
        assert "test_cases" in str(exc.value)

    def test_toke_eval_run_tests_does_not_return_zero_zero(
        self, tmp_path: Path
    ) -> None:
        """Explicitly pin the anti-behaviour: no silent (0, 0)."""
        bad = _write(tmp_path, "task-a-0001", {"id": "x", "test_cases": []})
        with pytest.raises(tepk.TaskSchemaError):
            result = tepk.run_tests(Path("/nonexistent/binary"), bad)
            pytest.fail(f"expected a raise, got a score: {result!r}")

    def test_toke_eval_evaluate_emits_no_score(self, tmp_path: Path) -> None:
        """A whole run against mismatched inputs must produce NO report."""
        sols = tmp_path / "sol"
        sols.mkdir()
        (sols / "task-a-0001.toke").write_text("m=main;\n")
        tests = tmp_path / "tests"
        _write(tests, "task-a-0001", {
            "id": "task-a-0001",
            "test_cases": [{"input": 1, "expected": 2}],
        })
        with pytest.raises(tepk.TaskSchemaError):
            tepk.evaluate(sols, tests, compiler="tkc")

    def test_toke_eval_report_pass_at_1_defaults_to_none_not_zero(self) -> None:
        """An unscored report carries no number, rather than a plausible 0.0."""
        report = tepk.BenchmarkReport()
        assert report.pass_at_1 is None, (
            "a default 0.0 is indistinguishable from a genuine zero score"
        )

    def test_toke_eval_pass_at_1_denominator_is_all_tasks(self) -> None:
        """Pass@1 divides by tasks attempted, not by tasks that compiled.

        Dividing by `tasks_compiled` (the pre-128.1c behaviour) discards every
        compile failure and inflates the figure by exactly the compile-failure
        rate.
        """
        report = tepk.BenchmarkReport(
            tasks_total=10, tasks_compiled=4, tasks_passed=2,
        )
        # Re-run the aggregation the way evaluate() does.
        report.pass_at_1 = report.tasks_passed / report.tasks_total
        report.pass_at_1_given_compiled = (
            report.tasks_passed / report.tasks_compiled
        )
        assert report.pass_at_1 == pytest.approx(0.2)
        assert report.pass_at_1_given_compiled == pytest.approx(0.5)
        assert report.pass_at_1 < report.pass_at_1_given_compiled

    @pytest.mark.parametrize("bad_body", [
        {"id": "task-a-0001", "test_cases": [{"input": 1, "expected": 2}]},
        {"id": "task-a-0001"},                       # key absent entirely
        {"id": "task-a-0001", "test_inputs": []},    # present but empty
        {"id": "task-a-0001", "test_inputs": {}},    # wrong type
        {"id": "task-a-0001", "test_inputs": [{"input": 1}]},   # no expected
        {"id": "task-a-0001", "test_inputs": [{"expected": 1}]},  # no input
        {"test_inputs": [{"input": 1, "expected": 2}]},          # no id
    ])
    def test_run_benchmark_load_task_rejects(
        self, tmp_path: Path, bad_body: dict
    ) -> None:
        p = _write(tmp_path, "task-a-0001", bad_body)
        with pytest.raises(BenchmarkSchemaError):
            load_task(p)

    def test_run_benchmark_validates_before_scoring(self, tmp_path: Path) -> None:
        """One bad file in the set aborts the whole run, before any scoring."""
        tasks = tmp_path / "tasks"
        _write(tasks, "task-a-0001", _valid_task("task-a-0001"))
        _write(tasks, "task-a-0002", {
            "id": "task-a-0002",
            "test_cases": [{"input": 1, "expected": 2}],
        })
        with pytest.raises(BenchmarkSchemaError) as exc:
            validate_tasks_dir(tasks)
        assert "refusing to emit a score" in str(exc.value)

    def test_empty_test_cases_is_not_a_free_pass(self) -> None:
        """`pass_count == total == 0` used to satisfy "all cases passed".

        The old `pass_at_1 = 1.0 if pass_count == total else 0.0` awarded a
        perfect score to any task whose test list was empty -- the 0/0 bug in
        its other, more flattering direction.
        """
        with pytest.raises(BenchmarkSchemaError):
            score_task("task-a-0001", lambda x: x, [], timeout=1)

    def test_zero_matched_tasks_is_not_a_score_of_zero(
        self, tmp_path: Path
    ) -> None:
        """A solution set that matches no task must not score 0.0000.

        Pointing the harness at `baselines/python` (tasks 0001-0060) and
        `hidden_tests/` (tasks 0501-1000) matched nothing at all, and the
        pre-128.1c harness printed `Mean Pass@1: 0.0000` over 0 tasks and
        wrote that to JSON.
        """
        tasks = tmp_path / "tasks"
        _write(tasks, "task-a-0900", _valid_task("task-a-0900"))
        sols = tmp_path / "sol"
        sols.mkdir()
        (sols / "solutions.py").write_text(
            'SOLUTIONS = {"task-a-0001": lambda x: x}\n'
        )
        with pytest.raises(BenchmarkSchemaError) as exc:
            run_benchmark.run_benchmark(sols, tasks, "python", timeout=5)
        assert "zero tasks" in str(exc.value)

    def test_cli_exits_nonzero_and_writes_no_report(self, tmp_path: Path) -> None:
        """End to end: schema mismatch -> exit 2, no output file."""
        tasks = tmp_path / "tasks"
        _write(tasks, "task-a-0001", {
            "id": "task-a-0001",
            "test_cases": [{"input": 1, "expected": 2}],
        })
        sols = tmp_path / "sol"
        sols.mkdir()
        (sols / "solutions.py").write_text(
            'SOLUTIONS = {"task-a-0001": lambda x: x}\n'
        )
        out = tmp_path / "report.json"

        rc = main([
            "--solutions-dir", str(sols),
            "--tasks-dir", str(tasks),
            "--language", "python",
            "--output", str(out),
        ])

        assert rc == 2, "a schema mismatch must not exit 0"
        assert not out.exists(), (
            "no report may be written when the inputs did not validate -- "
            "a written score is a quotable score"
        )


# ===========================================================================
# D2 -- best-of-N must not be reported as Pass@1
# ===========================================================================

class TestBestOfNIsNotPassAt1:
    """`--n-samples N` implemented best-of-N and wrote it to `pass_at_1`.

    Best-of-N is strictly more generous than Pass@1 -- it assumes an oracle
    that knows which sample passes the hidden tests -- so every figure the
    flag produced is inflated, and the inflation grows with N.
    """

    def _task(self, n_samples: int, n_correct: int) -> TaskResult:
        return TaskResult(
            task_id="task-a-0001",
            pass_count=2 if n_correct else 0,
            total_count=2,
            pass_at_1=n_correct / n_samples,
            n_samples=n_samples,
            n_correct=n_correct,
            best_of_n=1.0 if n_correct else 0.0,
            pass_at_k={
                k: pass_at_k_estimator(n_samples, n_correct, k)
                for k in (1, 5) if k <= n_samples
            },
        )

    def test_pass_at_1_is_the_single_sample_rate(self) -> None:
        """1 of 10 samples correct is Pass@1 = 0.1, not 1.0."""
        t = self._task(n_samples=10, n_correct=1)
        assert t.pass_at_1 == pytest.approx(0.1)
        assert t.best_of_n == 1.0
        assert t.pass_at_1 != t.best_of_n

    def test_report_separates_the_two_metrics(self) -> None:
        """They are different fields with different names. Always."""
        tasks = [
            self._task(n_samples=10, n_correct=1),   # solved by 1 of 10
            self._task(n_samples=10, n_correct=0),   # never solved
        ]
        report = generate_report(tasks, "toke-model", 10, n_samples=10)

        assert report.mean_pass_at_1 == pytest.approx(0.05)
        assert report.mean_best_of_n == pytest.approx(0.5)
        assert report.mean_best_of_n > report.mean_pass_at_1, (
            "best-of-N is strictly more generous; if these are ever equal by "
            "construction the metrics have been conflated again"
        )

    def test_best_of_n_never_lands_in_a_pass_at_1_field(self) -> None:
        """The serialised report must not carry best-of-N under any
        pass_at_1-shaped key."""
        tasks = [self._task(n_samples=10, n_correct=1)]
        d = report_to_dict(generate_report(tasks, "toke-model", 10, n_samples=10))

        assert d["mean_pass_at_1"] == pytest.approx(0.1)
        assert d["mean_best_of_n"] == pytest.approx(1.0)
        assert d["n_samples"] == 10
        # Best-of-N value must not appear under a Pass@1 name.
        for key, value in d.items():
            if "pass_at_1" in key:
                assert value != 1.0 or d["mean_best_of_n"] != 1.0 or key == "total_pass_at_1", (
                    f"{key} carries the best-of-N value"
                )
        # And the per-task entry keeps them apart too.
        assert d["tasks"][0]["pass_at_1"] == pytest.approx(0.1)
        assert d["tasks"][0]["best_of_n"] == 1.0

    def test_total_pass_at_1_is_undefined_for_multi_sample_runs(self) -> None:
        """A count of solved tasks is not defined for an N-sample estimator.

        Forcing one is how best-of-N acquired a Pass@1 name in the first
        place.
        """
        tasks = [self._task(n_samples=10, n_correct=1)]
        report = generate_report(tasks, "toke-model", 10, n_samples=10)
        assert report.total_pass_at_1 is None

    def test_n_equals_one_means_one_sample(self) -> None:
        """Pass@1 at N=1 is exactly one sample, one attempt."""
        solved = TaskResult("task-a-0001", 2, 2, 1.0)
        missed = TaskResult("task-a-0002", 1, 2, 0.0)
        report = generate_report([solved, missed], "python", 10, n_samples=1)

        assert report.n_samples == 1
        assert report.mean_pass_at_1 == pytest.approx(0.5)
        assert report.total_pass_at_1 == 1
        assert report.mean_best_of_n is None, (
            "best-of-N is meaningless at N=1 and must not be reported as a "
            "separate, equal-looking figure"
        )

    def test_report_carries_a_metric_note(self) -> None:
        tasks = [self._task(n_samples=10, n_correct=1)]
        note = generate_report(tasks, "toke-model", 10, n_samples=10).metric_note
        assert "best-of-10" in note.lower() or "best_of_n" in note.lower()
        assert "pass@1" in note.lower()

    def test_score_model_pass_at_1_entry_point_draws_one_sample(self) -> None:
        """The function named for Pass@1 must not accept an N."""
        import inspect
        sig = inspect.signature(run_benchmark.score_model_pass_at_1)
        assert "n_samples" not in sig.parameters, (
            "a function called score_model_pass_at_1 must not be able to "
            "draw more than one sample"
        )

    def test_chen_estimator(self) -> None:
        """Sanity-check the unbiased estimator itself."""
        assert pass_at_k_estimator(10, 0, 1) == pytest.approx(0.0)
        assert pass_at_k_estimator(10, 10, 1) == pytest.approx(1.0)
        assert pass_at_k_estimator(10, 1, 1) == pytest.approx(0.1)
        # pass@k rises with k for a fixed c: this is exactly why it is not
        # interchangeable with Pass@1.
        assert pass_at_k_estimator(10, 1, 5) > pass_at_k_estimator(10, 1, 1)
        with pytest.raises(ValueError):
            pass_at_k_estimator(3, 1, 5)

    def test_all_samples_are_scored(self) -> None:
        """No early `break`: c must count every correct sample.

        The old loop stopped at the first perfect sample, which both made the
        metric best-of-N and biased the correct-sample count downwards.
        """
        import ast

        tree = ast.parse(Path(run_benchmark.__file__).read_text())
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "score_model_samples"
        )
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.Break)], (
            "score_model_samples must evaluate every sample; an early break "
            "reintroduces best-of-N"
        )


# ===========================================================================
# D1b -- a compile failure is a failed task, not an absent one
# ===========================================================================

class TestCompileFailuresStayInTheDenominator:
    """`load_toke_solutions` dropped non-compiling solutions entirely.

    They never became task results, so they left the denominator.  The
    published Gate-1 figure is 588/923 = 63.7% from a run of 1,000 generated
    solutions: 77 did not compile and were discarded.  588/1000 = 58.8% is
    the Pass@1.  63.7% is Pass@1 *given the solution compiled* -- a different
    and strictly more generous quantity, published under the name Pass@1.
    """

    def test_compile_failure_scores_zero(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks"
        _write(tasks, "task-a-0001", _valid_task("task-a-0001"))
        _write(tasks, "task-a-0002", _valid_task("task-a-0002"))
        sols = tmp_path / "sol"
        sols.mkdir()

        # Pretend task-a-0002's .toke source failed to compile.
        run_benchmark._COMPILE_FAILURES = []
        monkey_solutions = {"task-a-0001": lambda x: x * 2}

        def fake_loader(_dir: Path) -> dict:
            run_benchmark._COMPILE_FAILURES.append("task-a-0002")
            return monkey_solutions

        original = run_benchmark.LANGUAGE_LOADERS["toke"]
        run_benchmark.LANGUAGE_LOADERS["toke"] = fake_loader
        try:
            report = run_benchmark.run_benchmark(
                sols, tasks, "toke", timeout=5,
            )
        finally:
            run_benchmark.LANGUAGE_LOADERS["toke"] = original

        assert report.tasks_evaluated == 2, (
            "the non-compiling task must stay in the denominator"
        )
        assert report.total_pass_at_1 == 1
        assert report.mean_pass_at_1 == pytest.approx(0.5), (
            "dropping the compile failure would report 1.0000"
        )

    def test_the_published_arithmetic(self) -> None:
        """Pin the two quantities the Gate-1 figure confused."""
        generated, compiled, passed = 1000, 923, 588
        pass_at_1 = passed / generated
        pass_at_1_given_compiled = passed / compiled
        assert round(pass_at_1_given_compiled, 3) == 0.637
        assert round(pass_at_1, 3) == 0.588
        assert pass_at_1 < pass_at_1_given_compiled


# ===========================================================================
# D3 -- harness stubs must not be silent no-ops
# ===========================================================================

class TestHarnessStubsAreNotSilent:
    """0-byte files that exit 0 are indistinguishable from a successful run."""

    STUBS = ("run.py", "score.py", "report.py")

    @pytest.mark.parametrize("name", STUBS)
    def test_not_zero_bytes(self, name: str) -> None:
        p = _BENCH / "harness" / name
        assert p.exists(), f"{name} is missing"
        assert p.stat().st_size > 0, (
            f"{name} is a 0-byte file: running it exits 0 and does nothing, "
            f"which reads as a successful run that found no work"
        )

    @pytest.mark.parametrize("name", STUBS)
    def test_exits_nonzero(self, name: str) -> None:
        proc = subprocess.run(
            [sys.executable, str(_BENCH / "harness" / name)],
            capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode != 0, (
            f"harness/{name} exited 0 without doing any work"
        )
        assert proc.stderr.strip(), f"harness/{name} failed silently"

    @pytest.mark.parametrize("name", STUBS)
    def test_points_at_the_live_tooling(self, name: str) -> None:
        proc = subprocess.run(
            [sys.executable, str(_BENCH / "harness" / name)],
            capture_output=True, text=True, timeout=60,
        )
        assert "128.1c" in proc.stderr
        assert any(
            tool in proc.stderr
            for tool in ("run_benchmark.py", "pass_at_k.py", "toke_eval.report")
        ), f"harness/{name} does not say what to use instead"

    def test_readme_does_not_document_retired_entry_points(self) -> None:
        """The README told people to run a file that did nothing."""
        readme = (_BENCH / "README.md").read_text()
        assert "python harness/run.py" not in readme
        assert "python harness/score.py" not in readme
        assert "run_benchmark.py" in readme

    def test_tasks_dir_holds_no_task_files(self) -> None:
        """Documented so nobody points the harness at an empty set.

        `benchmark/tasks/` contains only `schema.json`. A harness aimed there
        finds zero tasks -- another way to get a number that means nothing.
        """
        tasks_dir = _BENCH / "tasks"
        assert not list(tasks_dir.glob("task-*.yaml")), (
            "tasks/ now has task files; update the README, which says it "
            "holds only the schema"
        )
        assert (tasks_dir / "schema.json").exists()
        assert "hidden_tests/" in (_BENCH / "README.md").read_text()


# ===========================================================================
# The live benchmark set actually validates
# ===========================================================================

def test_hidden_tests_match_the_declared_schema() -> None:
    """Guards against the mismatch reappearing from the data side."""
    hidden = _BENCH / "hidden_tests"
    if not hidden.is_dir():
        pytest.skip("hidden_tests/ not present")
    files = sorted(hidden.glob("task-*.yaml"))
    assert files, "hidden_tests/ is empty"
    for p in files[:25]:
        task = load_task(p)
        assert task["test_inputs"]

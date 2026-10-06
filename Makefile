# toke-eval CI entry point. `make ci` is what CI runs.
PY ?= python3
# Tokenizer manifest labels the disjointness checker compares against (128.1, section 5.3).
BENCH_LABELS ?= pre131
# Lane training exports (D3) and training task descriptions (D5); unset -> NOT_RUN -> non-zero.
BENCH_EXPORTS ?=
BENCH_DESCRIPTIONS ?=

.PHONY: ci test lint check-bench-disjoint check-gate1-attribution check-c-baselines

ci: lint test check-bench-disjoint check-c-baselines

test:
	cd benchmark && $(PY) -m pytest -q tests

# 128.27: all of scripts/; 128.33: all of benchmark/ (tests included) and toke_eval/.
# Any hidden_tests/ directory is excluded so lint never opens one (AGENTS.md section 5).
lint:
	$(PY) -m ruff check --extend-exclude hidden_tests scripts/ benchmark/ toke_eval/

# 128.1b: D1-D8. Runs for real once BENCH-128 is frozen (128.1a); until then the
# checker's own fixtures run under `test`, and this target says plainly that it did not run.
check-bench-disjoint:
	@if [ -f benchmark/bench128/manifest.json ]; then \
		$(PY) scripts/check_bench_disjoint.py $(foreach l,$(BENCH_LABELS),--label $(l)) \
			$(foreach e,$(BENCH_EXPORTS),--export $(e)) \
			$(if $(BENCH_DESCRIPTIONS),--train-descriptions $(BENCH_DESCRIPTIONS)); \
	else \
		echo "check-bench-disjoint: NOT RUN - benchmark/bench128/manifest.json does not exist (BENCH-128 not frozen, 128.1a). This is not a PASS."; \
	fi

# 128.34: the C baselines must still compile. This is the one check of the retired
# benchmark/.github/workflows/benchmark.yml that `make test` did not already cover.
# Built into a temp dir so no binary lands in the tree.
# 128.35: also run the smoke test, which asserts each output.
check-c-baselines:
	@d=$$(mktemp -d) && $(MAKE) --no-print-directory -s -C benchmark/baselines/c test TARGET=$$d/solutions \
		>$$d/log 2>&1 \
		|| { cat $$d/log; rm -rf $$d; exit 1; }; \
	rm -rf $$d && echo "check-c-baselines: OK - benchmark/baselines/c/solutions.c compiles and its smoke test passes"

# 128.22: needs full git history (not a shallow clone).
check-gate1-attribution:
	$(PY) scripts/check_gate1_attribution.py

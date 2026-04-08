# Contamination Analysis and Holdout Governance

**Story:** 10.6.6
**Date:** 2026-04-04
**Decision class:** D5 = A+D (strict separation of proprietary vs open-weight outputs)

This document defines the methodology for preventing data contamination between
the toke training corpus and the evaluation benchmark, and establishes the
governance processes that maintain this separation over time.

---

## 1. Data Separation Architecture

### 1.1 Repository isolation

Training and evaluation data live in entirely separate git repositories with no
code-level cross-references:

| Purpose | Repository | Contents |
|---------|-----------|----------|
| Training data | `toke-model/corpus` | 46,754 validated toke programs |
| Evaluation data | `toke-eval/benchmark` | 500+ held-out benchmark tasks |
| Evaluation harness | `toke-eval` | Pass@k scorer, gate cards, analysis tooling |

This repository boundary is the first line of defence. No import, submodule, or
symlink connects the corpus and benchmark repositories.

### 1.2 Task ID namespaces

Training and evaluation tasks occupy disjoint ID namespaces:

| Dataset | ID pattern | Examples |
|---------|-----------|----------|
| Training corpus | `{Stage}-{Category}-{Seq}` | `A-MTH-0001`, `B-MFN-0042`, `C-BND-0100` |
| Evaluation benchmark | `task-a-{NNNN}` | `task-a-0001`, `task-a-0016`, `task-a-0029` |

The prefix structure makes accidental cross-contamination syntactically
detectable. Any task ID matching `task-a-*` appearing in corpus output is an
immediate contamination signal.

### 1.3 Temporal ordering

The benchmark task set was designed and frozen before the final training corpus
was assembled:

1. Benchmark tasks generated and tagged (`toke-eval` tag `v0.1-gate1`)
2. Holdout task ID list extracted and committed to corpus config
3. Training corpus generated with holdout enforcement active

Git commit timestamps across both repositories provide an auditable record of
this ordering.

---

## 2. Contamination Detection Methods

### 2.1 Exact match (SHA-256 of normalised source)

The strongest contamination signal. For each benchmark task solution and each
corpus entry:

1. Normalise the toke source: strip comments, collapse whitespace, remove blank
   lines
2. Compute `SHA-256(normalised_source)`
3. Check for hash collisions between the benchmark solution set and the corpus

**Status:** Method defined; can be run using existing `hashlib.sha256` in
`toke-model/corpus/store/writer.py` (line 124). Full cross-set comparison not yet
automated.

**TODO:** Build a script that loads all corpus entry `tk_source` fields, all
benchmark solutions, normalises both, and reports any SHA-256 collisions.

### 2.2 Near-duplicate detection (edit distance)

Catches trivially modified copies (renamed variables, reordered statements):

1. Normalise source as above
2. Compute Levenshtein edit distance between each benchmark solution and each
   corpus entry
3. Flag pairs where `edit_distance / max(len_a, len_b) < threshold`

Recommended threshold: 0.15 (i.e., fewer than 15% of characters differ).

**Status:** Method defined. Not yet implemented.

**TODO:** Implement using `rapidfuzz` or `python-Levenshtein`. Given the corpus
size (46,754) x benchmark size (500+), use locality-sensitive hashing (MinHash)
to avoid O(n*m) brute-force comparison.

### 2.3 Semantic similarity (embedding cosine similarity)

Detects functionally equivalent programs with different surface syntax:

1. Generate embeddings for each normalised source using a code embedding model
   (e.g., `text-embedding-3-small` or a locally-run model)
2. Compute pairwise cosine similarity between benchmark and corpus embeddings
3. Flag pairs above a similarity threshold

**Status:** Not yet implemented. This is the most important contamination check
for Gate 2 credibility.

**TODO:**
- Select embedding model (prefer local inference on Apple Silicon for cost and
  privacy; `nomic-embed-text` via MLX is a candidate)
- Determine rejection threshold (see Open Questions, section 6)
- Build indexed similarity search using FAISS or Annoy for efficient lookup

### 2.4 N-gram overlap analysis

Statistical contamination signal across the full dataset:

1. Extract character-level or token-level n-grams (n = 5, 7, 10) from each
   source file
2. Build an n-gram frequency set for the corpus and for the benchmark
3. Compute Jaccard similarity of n-gram sets per benchmark task vs the full
   corpus
4. Flag tasks with Jaccard similarity above threshold

This method is less precise than semantic similarity but is fast, deterministic,
and requires no external model.

**Status:** Not yet implemented.

**TODO:** Implement as a standalone script in `toke-eval/`. Use token-level
n-grams (cl100k_base tokenisation is already available in the corpus pipeline).

---

## 3. Holdout Governance

### 3.1 QualityScorer enforcement (Story 10.7.4)

The `QualityScorer` class (`toke-model/corpus/validate/quality.py`) enforces holdout
isolation at scoring time:

- Constructor **requires** a non-empty `holdout_task_ids: set[str]` parameter
- Raises `ValueError` if the set is empty or missing
- Any task whose `task_id` appears in the holdout set receives a hard reject
  (`accepted = False`) regardless of quality score

Relevant code path:
```
QualityScorer.__init__() -> validates holdout_task_ids is non-empty set
QualityScorer.score()    -> checks task_id in self.holdout_task_ids
```

### 3.2 CorpusWriter enforcement (dual layer)

The `CorpusWriter` class (`toke-model/corpus/store/writer.py`) provides a second,
independent enforcement layer:

- Constructor **requires** a non-empty `holdout_task_ids` set
- `write()` method raises `ValueError` with message `HOLDOUT VIOLATION` if the
  entry's `task_id` is in the holdout set
- This is a last-line defence: even if the scorer is bypassed or misconfigured,
  the writer refuses to persist contaminated entries

### 3.3 Pipeline startup enforcement

The main pipeline (`toke-model/corpus/main.py`) enforces holdout configuration at
startup:

1. Reads `holdout.task_ids` list and/or `holdout.file` path from `config.yaml`
2. Merges both sources into a single `holdout_task_ids` set
3. Aborts with `sys.exit(1)` if the resulting set is empty
4. Passes the set to both `QualityScorer` and `CorpusWriter`

The pipeline **cannot start** without an explicit holdout configuration.

### 3.4 Holdout configuration format

In `toke-model/corpus/config.yaml`:

```yaml
holdout:
  task_ids:
    - "task-a-0001"
    - "task-a-0002"
    # ... all benchmark task IDs
  file: "/path/to/holdout_ids.txt"  # optional, one ID per line
```

Both `task_ids` and `file` sources are merged. The file format supports comments
(lines starting with `#`) and blank lines.

### 3.5 Process for adding new benchmark tasks

When new tasks are added to `toke-eval/benchmark`:

1. Generate new task with `task-a-NNNN` ID in `toke-eval/benchmark`
2. Add the task ID to the holdout configuration (`config.yaml` or holdout file)
3. Verify the holdout set size matches the benchmark task count
4. Re-run contamination detection (section 2) against the existing corpus
5. Commit both the new task and the updated holdout config before any new
   corpus generation runs

**Rule:** No corpus generation run may begin until the holdout set is confirmed
to include all current benchmark task IDs.

---

## 4. Hash Commitments

### 4.1 Per-task hash

For each benchmark task, compute:

```
SHA-256(canonical_yaml(task_file))
```

where `canonical_yaml` is the YAML file with keys sorted and consistent
formatting. This hash covers the task specification, all test case inputs, and
all expected outputs.

### 4.2 Merkle tree commitment

Build a Merkle tree over the sorted per-task hashes:

1. Sort all per-task SHA-256 hashes lexicographically
2. Concatenate adjacent pairs and hash: `SHA-256(hash_i || hash_{i+1})`
3. Repeat until a single root hash remains
4. Publish the root hash in gate decision documents

This structure allows:
- Efficient proof that a specific task was included in the committed set
  (log(n) hashes needed)
- Detection of any task addition, removal, or modification after commitment

### 4.3 Commitment protocol

| Step | When | Action |
|------|------|--------|
| 1 | Benchmark freeze | Compute Merkle root of all task hashes |
| 2 | Before training | Publish root hash in gate evaluation plan |
| 3 | After evaluation | Reveal individual task hashes on request |
| 4 | Verification | Third party recomputes Merkle root from revealed hashes |

### 4.4 Existing hash infrastructure

The reproducibility package (`toke/spec/docs/gate1-reproducibility.md`) already
defines hash commitments for:

- `gate1_v5_1000.json` (results file)
- Hidden test set (tar of task YAMLs)
- Solutions directory (tar of `.toke` files)

The Merkle tree extends this by enabling per-task verification without
revealing the entire test set.

**TODO:** Implement Merkle tree computation script. Publish root hash for the
current benchmark set in the Gate 2 evaluation plan.

---

## 5. Audit Trail

### 5.1 Git history proves temporal ordering

The separation claim is verifiable through git commit history:

1. `toke-eval` tag `v0.1-gate1` timestamps when the benchmark was frozen
2. `toke-model` commit history shows when holdout enforcement was added
   (Story 10.7.4)
3. Training data generation commits in `toke-model` postdate the holdout
   enforcement commit

Any third party with repository access can verify these timestamps
independently.

### 5.2 CI checks for cross-contamination

**TODO:** Implement the following CI checks:

- **Namespace check:** Scan all corpus entries for task IDs matching
  `task-a-*` pattern. Fail the build if any are found.
- **Holdout completeness check:** Compare the holdout set in `config.yaml`
  against the full list of task IDs in `toke-eval/benchmark`. Fail if any benchmark
  task ID is missing from the holdout set.
- **Hash verification:** Recompute the Merkle root of the current benchmark
  set and compare against the committed value. Fail on mismatch.

### 5.3 Process for third-party verification

A reviewer can independently verify contamination isolation:

1. **Repository structure:** Confirm `toke-model/corpus` and `toke-eval/benchmark` are
   separate repositories with no cross-references
2. **Namespace disjointness:** Confirm no `task-a-*` IDs exist in corpus
   output: `grep -r "task-a-" toke-model/corpus/`
3. **Holdout enforcement:** Read `QualityScorer` and `CorpusWriter` source to
   confirm hard-reject logic
4. **Temporal ordering:** Check git log dates across repositories
5. **Hash commitment:** Recompute Merkle root from benchmark tasks, compare
   to published value
6. **Contamination scan:** Run exact-match and near-duplicate detection
   (section 2) and review results

### 5.4 Audit artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Holdout config | `toke-model/corpus/config.yaml` | Lists blocked task IDs |
| QualityScorer | `toke-model/corpus/validate/quality.py` | Scoring-time enforcement |
| CorpusWriter | `toke-model/corpus/store/writer.py` | Write-time enforcement |
| Pipeline main | `toke-model/corpus/main.py` | Startup enforcement |
| Benchmark tag | `toke-eval` tag `v0.1-gate1` | Frozen benchmark state |
| Reproducibility package | `toke/spec/docs/gate1-reproducibility.md` | Hash commitments |
| This document | `toke-eval/docs/contamination-analysis.md` | Methodology |

---

## 6. Open Questions

### 6.1 Semantic similarity rejection threshold

What cosine similarity score between a corpus entry and a benchmark task
constitutes contamination?

- Too low (e.g., 0.7): many false positives from programs solving similar
  algorithmic problems with different approaches
- Too high (e.g., 0.95): misses paraphrased copies

**Proposed approach:** Calibrate the threshold empirically. Generate a small set
of known-clean and known-contaminated pairs, measure the similarity
distribution, and set the threshold at the point that maximises separation.
Report the ROC curve in the Gate 2 evaluation.

### 6.2 Public dataset overlap (APPS, MBPP, HumanEval)

Some benchmark tasks may be semantically similar to problems in public coding
benchmarks (APPS, MBPP, HumanEval). This is a distinct concern from
corpus-benchmark contamination:

- The toke benchmark tests toke-language-specific constructs, not general
  algorithms, which limits overlap
- However, algorithmic cores may be similar (e.g., "sort an array")

**Proposed approach:**
1. Compare benchmark task descriptions against APPS/MBPP/HumanEval problem
   statements using embedding similarity
2. For any high-similarity matches, document them explicitly
3. Assess whether the toke-specific aspects (syntax, type system, error
   handling) make the tasks sufficiently distinct
4. Report findings in the Gate 2 evaluation; do not automatically reject
   tasks that share algorithmic structure with public benchmarks, but
   disclose them

### 6.3 Corpus-benchmark generation pipeline overlap

Both the training corpus and benchmark tasks were generated using the same
infrastructure (LLM-based generation, differential testing). This shared
pipeline could theoretically produce similar outputs even without direct data
leakage.

**Mitigation:** The task specifications (prompts, categories, difficulty
parameters) are different between corpus and benchmark generation. Document
these differences explicitly in the Gate 2 evaluation.

---

## References

- Story 10.7.4: Holdout enforcement in corpus builder
- `toke/spec/docs/gate1-reproducibility.md`: Gate 1 contamination section
- `toke-model/corpus/validate/quality.py`: QualityScorer with holdout isolation
- `toke-model/corpus/store/writer.py`: CorpusWriter with holdout enforcement
- `toke-model/corpus/main.py`: Pipeline startup holdout validation
- Chen et al., "Evaluating Large Language Models Trained on Code" (2021):
  Pass@k methodology and contamination discussion

# Gate Evaluation Card — Template

**Gate:** (e.g. Gate 1, Gate 2)
**Date:** YYYY-MM-DD
**Verdict:** PASS / FAIL

---

## Model

| Field | Value |
|-------|-------|
| Base model | (e.g. Qwen/Qwen2.5-Coder-7B-Instruct) |
| Adapter | (e.g. QLoRA rank 64, alpha 128) |
| Weights hash (SHA-256) | |
| Training script version | |
| Training config hash | |

## Tokenizer

| Field | Value |
|-------|-------|
| Tokenizer type | (e.g. SentencePiece BPE 8K vocab) |
| Tokenizer artifact hash | |
| Vocabulary size | |

## Decoding Parameters

| Field | Value |
|-------|-------|
| Temperature | |
| Top-p | |
| Max tokens | |
| Seed | |
| Number of samples (n) | |

## Benchmark

| Field | Value |
|-------|-------|
| Benchmark version | |
| Hidden test set hash (SHA-256) | |
| Number of tasks | |
| Task categories | |

## Compiler

| Field | Value |
|-------|-------|
| Compiler version (tkc) | |
| Compiler binary hash | |
| Compiler flags | |
| LLVM version | |

## Hardware & Environment

| Field | Value |
|-------|-------|
| Hardware | |
| OS | |
| Python version | |
| MLX version (if applicable) | |
| CUDA version (if applicable) | |

## Results

| Metric | Value |
|--------|-------|
| Pass@1 | |
| Compile rate | |
| Tasks passed | |
| Tasks compiled | |
| Token reduction (cl100k_base) | |
| Token reduction (custom BPE) | |
| Duration | |

## Data Hashes

| Artifact | SHA-256 |
|----------|---------|
| Training corpus | |
| Evaluation corpus | |
| Holdout task IDs | |
| Training config | |

## Notes

(Any additional context, anomalies, or observations)

---

*This card must be completed for every gate evaluation run. Incomplete cards invalidate the gate result.*

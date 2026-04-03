#!/usr/bin/env python3
"""Generate toke solutions using mlx-lm with LoRA adapter on Apple Silicon.

Loads Qwen 2.5 Coder 7B (4-bit quantized) + toke LoRA adapter,
generates solutions for each benchmark task, saves as .toke files.

Usage:
    python run_inference_mlx.py
    python run_inference_mlx.py --tasks-dir hidden_tests/ --output-dir solutions/
    python run_inference_mlx.py --base-model Qwen/Qwen2.5-Coder-7B-Instruct --adapter ../toke-models/results/train-results-20260402-055104/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from mlx_lm import load, generate

SYSTEM_PROMPT = """You are a toke programming language expert. Write correct, idiomatic toke code.

toke syntax reference:
- M=name; module declaration (must be first)
- I=alias:module.path; imports
- F=name(param:Type):ReturnType{ body }; function declaration
- let x=expr; immutable binding, mut x=expr; mutable binding
- <expr; returns a value
- Types: i64, f64, Str, bool, u8, void
- Arrays: [i64] type, [1;2;3] literal, arr[i] indexing, arr.len length
- Structs: T=Point{x:i64;y:i64};
- if(cond){...}el{...};
- lp(let i=0;i<n;i=i+1){...};
- Every statement ends with ;

Write a COMPLETE program with F=main that reads JSON from the first command-line argument, computes the result, and prints JSON to stdout.

Example structure for a program that sums a list of integers:
M=sum;I=j:std.json;I=s:std.str;F=main():void{let input=j.parse(s.argv(1));let arr=input as [i64];let r=mut.0;lp(let i=0;i<arr.len;i=i+1){r=r+arr[i];};j.print(r);};

Write ONLY toke source code. No markdown fences, no explanation."""


def build_prompt(task: dict) -> str:
    desc = task["description"]
    input_type = task.get("input_type", "")
    output_type = task.get("output_type", "")
    examples = task.get("test_inputs", [])[:2]

    msg = f"Write a complete toke program with F=main that: {desc}"
    msg += f"\n\nInput JSON type: {input_type}\nOutput JSON type: {output_type}"
    msg += "\n\nThe program must: read JSON from argv[1] using I=j:std.json;I=s:std.str; and j.parse(s.argv(1)), compute the answer, and print the result with j.print(result)."
    if examples:
        msg += "\n\nExample I/O:"
        for ex in examples:
            msg += f"\n  {json.dumps(ex['input'])} → {json.dumps(ex['expected'])}"
    return msg


def extract_toke_source(text: str) -> str:
    # Strip special tokens
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            lines = part.strip().split("\n")
            if lines and lines[0].strip().lower() in ("toke", "tk", ""):
                return "\n".join(lines[1:]).strip()
            return part.strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                        help="Base model (HF name or local path)")
    parser.add_argument("--adapter", type=Path,
                        default=Path("../toke-models/results/train-results-20260402-055104/adapter-mlx"),
                        help="LoRA adapter directory (MLX format)")
    parser.add_argument("--tasks-dir", type=Path, default=Path("hidden_tests/"),
                        help="Directory with task-*.yaml files")
    parser.add_argument("--output-dir", type=Path, default=Path("solutions/"),
                        help="Output directory for .toke files")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N tasks (0=all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(args.tasks_dir.glob("task-*.yaml"))
    if not task_files:
        print(f"ERROR: no task-*.yaml files in {args.tasks_dir}", file=sys.stderr)
        return 1

    if args.limit > 0:
        task_files = task_files[:args.limit]

    print(f"Tasks: {len(task_files)}")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output_dir}")

    # Check if adapter exists and has MLX-compatible weights
    adapter_path = str(args.adapter) if args.adapter.exists() else None
    if adapter_path:
        print(f"Loading model + adapter...")
    else:
        print(f"WARNING: adapter not found at {args.adapter}, using base model only")
        print(f"Loading base model...")

    model, tokenizer = load(
        args.base_model,
        adapter_path=adapter_path,
    )
    print("Model loaded.")

    stats = {"total": 0, "generated": 0, "empty": 0, "errors": 0}
    start_time = time.time()

    for i, tf in enumerate(task_files):
        with open(tf) as f:
            task = yaml.safe_load(f)

        task_id = task["id"]
        stats["total"] += 1

        # Skip tasks that already have a solution file
        out_path = args.output_dir / f"{task_id}.toke"
        if out_path.exists():
            stats["generated"] += 1
            continue

        try:
            user_msg = build_prompt(task)

            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}\n\n"

            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )

            source = extract_toke_source(response)

            if not source:
                stats["empty"] += 1
                continue

            out_path.write_text(source)
            stats["generated"] += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                remaining = len(task_files) - i - 1
                eta = remaining / (rate / 60) if rate > 0 else 0
                print(f"  [{i+1}/{len(task_files)}] {rate:.1f} tasks/min, ETA {eta/60:.1f}min")

        except Exception as exc:
            print(f"  ERROR {task_id}: {exc}", file=sys.stderr)
            stats["errors"] += 1

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Generated: {stats['generated']}/{stats['total']}")
    print(f"  Empty: {stats['empty']}, Errors: {stats['errors']}")

    with open(args.output_dir / "inference_stats.json", "w") as f:
        json.dump({**stats, "elapsed_seconds": elapsed}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

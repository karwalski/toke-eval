#!/usr/bin/env bash
# Runner that verifies C baseline solutions against hidden test YAML files.
#
# Usage:
#   ./run_baselines.sh [--tests-dir hidden_tests] [-v|--verbose]
#
# Requires: yq (or python3 + pyyaml as fallback) for YAML parsing.
# Exits with code 0 if all tests pass, 1 otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$SCRIPT_DIR/solutions"

TESTS_DIR="$REPO_ROOT/hidden_tests"
VERBOSE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests-dir)
            TESTS_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Build if needed
if [[ ! -x "$BINARY" ]] || [[ "$SCRIPT_DIR/solutions.c" -nt "$BINARY" ]]; then
    echo "Building C solutions..."
    make -C "$SCRIPT_DIR" -s
fi

if [[ ! -d "$TESTS_DIR" ]]; then
    echo "ERROR: tests directory not found: $TESTS_DIR" >&2
    exit 1
fi

# We use a small Python helper to extract test data from YAML since
# pure-bash YAML parsing is fragile.  This keeps the dependency light
# (just python3 + pyyaml, which the Python baselines already require).
EXTRACT_HELPER=$(cat <<'PYEOF'
import json, sys, yaml

tf = sys.argv[1]
with open(tf) as f:
    task = yaml.safe_load(f)

task_id = task["id"]
cases = task.get("test_inputs", [])

# Output: one line per test case
# FORMAT: task_id\tjson_input\tjson_expected
for tc in cases:
    inp = json.dumps(tc["input"], separators=(",", ":"))
    exp = json.dumps(tc["expected"], separators=(",", ":"))
    print(f"{task_id}\t{inp}\t{exp}")
PYEOF
)

total_tasks=0
passed_tasks=0
skipped_tasks=0
failed_tasks=()

for tf in "$TESTS_DIR"/task-a-*.yaml; do
    [[ -f "$tf" ]] || continue

    # Extract test cases via Python
    cases_output=$(python3 -c "$EXTRACT_HELPER" "$tf" 2>/dev/null) || {
        echo "  WARN  Could not parse $tf" >&2
        continue
    }

    if [[ -z "$cases_output" ]]; then
        continue
    fi

    # Read task_id from first line
    task_id=$(echo "$cases_output" | head -1 | cut -f1)

    # Check if the C binary supports this task
    if ! "$BINARY" "$task_id" '0' >/dev/null 2>&1 && \
       ! "$BINARY" "$task_id" '[0]' >/dev/null 2>&1 && \
       ! "$BINARY" "$task_id" '[[0]]' >/dev/null 2>&1; then
        ((skipped_tasks++)) || true
        [[ $VERBOSE -eq 1 ]] && echo "  SKIP  $task_id (no C solution)"
        continue
    fi

    ((total_tasks++)) || true
    task_passed=1
    case_num=0

    while IFS=$'\t' read -r _tid json_input json_expected; do
        # Run the C solution
        actual=$("$BINARY" "$task_id" "$json_input" 2>/dev/null) || {
            echo "  FAIL  $task_id case $case_num: crashed"
            task_passed=0
            break
        }

        # Trim whitespace
        actual=$(echo "$actual" | tr -d '[:space:]')
        json_expected=$(echo "$json_expected" | tr -d '[:space:]')

        if [[ "$actual" != "$json_expected" ]]; then
            echo "  FAIL  $task_id case $case_num: input=$json_input  expected=$json_expected  got=$actual"
            task_passed=0
            break
        fi

        ((case_num++)) || true
    done <<< "$cases_output"

    if [[ $task_passed -eq 1 ]]; then
        ((passed_tasks++)) || true
        [[ $VERBOSE -eq 1 ]] && echo "  PASS  $task_id ($case_num cases)"
    else
        failed_tasks+=("$task_id")
    fi
done

echo ""
echo "============================================================"
echo "Results: $passed_tasks/$total_tasks tasks passed, $skipped_tasks skipped, ${#failed_tasks[@]} failed"
if [[ ${#failed_tasks[@]} -gt 0 ]]; then
    echo "Failed: ${failed_tasks[*]}"
fi
echo "============================================================"

[[ ${#failed_tasks[@]} -eq 0 ]]

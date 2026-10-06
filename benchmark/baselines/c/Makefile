# Makefile for C baseline solutions
#
# Usage:
#   make          -- build solutions binary
#   make clean    -- remove build artifacts
#   make test     -- smoke test: run a few tasks and assert their output

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

TARGET  = solutions
SRC     = solutions.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Each case is "<task-id>|<input>|<expected stdout>". The expected values
# follow from each task's definition in solutions.c (sum, is_prime, ...);
# they are not taken from hidden_tests/, which nothing here reads (128.35).
SMOKE_CASES = \
	'task-a-0001|[1,2,3]|6' \
	'task-a-0010|7|true' \
	'task-a-0011|10|55' \
	'task-a-0027|5|25' \
	'task-a-0042|[3,1,2]|[1, 2, 3]' \
	'task-a-0044|[1,2,3]|[3, 2, 1]' \
	'task-a-0038|[[1,2],[3,4]]|[4, 6]' \
	'task-a-0052|[[1,2,3],10]|[11, 12, 13]' \
	'task-a-0054|[[1,2,3,4,5],2]|[3, 4, 5, 1, 2]' \
	'task-a-0057|[[1,2,3,4,5],2]|[[1, 2], [3, 4], [5]]'

test: $(TARGET)
	@echo "=== Smoke tests ==="
	@fail=0; for c in $(SMOKE_CASES); do \
		id=$${c%%|*}; rest=$${c#*|}; in=$${rest%%|*}; want=$${rest#*|}; \
		got=$$($(abspath $(TARGET)) "$$id" "$$in") || { printf 'FAIL %s %s: exit %s\n' "$$id" "$$in" "$$?"; fail=1; continue; }; \
		if [ "$$got" = "$$want" ]; then printf 'ok   %s %s => %s\n' "$$id" "$$in" "$$got"; \
		else printf 'FAIL %s %s: got [%s], want [%s]\n' "$$id" "$$in" "$$got" "$$want"; fail=1; fi; \
	done; \
	if [ $$fail -ne 0 ]; then echo "=== Smoke tests FAILED ==="; exit 1; fi; \
	echo "=== All smoke tests passed ==="

# Makefile for C baseline solutions
#
# Usage:
#   make          -- build solutions binary
#   make clean    -- remove build artifacts
#   make test     -- quick smoke test

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

test: $(TARGET)
	@echo "=== Smoke tests ==="
	@echo -n "task-a-0001 [1,2,3] => "; ./$(TARGET) task-a-0001 '[1,2,3]'
	@echo -n "task-a-0010 7        => "; ./$(TARGET) task-a-0010 '7'
	@echo -n "task-a-0011 10       => "; ./$(TARGET) task-a-0011 '10'
	@echo -n "task-a-0027 5        => "; ./$(TARGET) task-a-0027 '5'
	@echo -n "task-a-0042 [3,1,2]  => "; ./$(TARGET) task-a-0042 '[3,1,2]'
	@echo -n "task-a-0044 [1,2,3]  => "; ./$(TARGET) task-a-0044 '[1,2,3]'
	@echo -n "task-a-0038 [[1,2],[3,4]] => "; ./$(TARGET) task-a-0038 '[[1,2],[3,4]]'
	@echo -n "task-a-0054 [[1,2,3,4,5],2] => "; ./$(TARGET) task-a-0054 '[[1,2,3,4,5],2]'
	@echo -n "task-a-0057 [[1,2,3,4,5],2] => "; ./$(TARGET) task-a-0057 '[[1,2,3,4,5],2]'
	@echo "=== All smoke tests passed ==="

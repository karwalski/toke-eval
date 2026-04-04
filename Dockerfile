# Dockerfile for sandboxed toke evaluation
# Outline only — actual Docker build is a separate story.
#
# Build:
#   docker build -t toke-eval .
#
# Run:
#   docker run --rm -v $(pwd)/results:/app/results toke-eval \
#       --tasks-dir /app/tasks \
#       --solutions-dir /app/solutions \
#       --compiler /app/tkc \
#       --output /app/results/evalplus_results.json

FROM python:3.11-slim AS base

# Install clang for compiling LLVM IR to native binaries
RUN apt-get update && \
    apt-get install -y --no-install-recommends clang && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir . scipy

# Copy evaluation scripts
COPY scripts/ scripts/
COPY toke_eval/ toke_eval/

# Copy tkc compiler binary (must be Linux amd64/arm64)
# COPY tkc /app/tkc
# RUN chmod +x /app/tkc

# Copy benchmark tasks and solutions
# COPY tasks/ /app/tasks/
# COPY solutions/ /app/solutions/

# Security: run as non-root user
RUN useradd -m evaluator
USER evaluator

# Default entrypoint
ENTRYPOINT ["python", "scripts/evalplus_harness.py"]

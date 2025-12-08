# Entangle Benchmark Environment
# 
# This Dockerfile creates a reproducible environment for running Entangle benchmarks
# with all competitor databases installed.
#
# Build (from project root):
#   docker build -t entangle-bench -f benchmarks/Dockerfile .
#
# Or from benchmarks directory:
#   docker build -t entangle-bench -f Dockerfile ..
#
# Run all benchmarks:
#   docker run --rm -v $(pwd)/results:/app/results entangle-bench all --output /app/results
#
# Run specific benchmark:
#   docker run --rm entangle-bench write --iterations 10000
#
# Interactive shell:
#   docker run --rm -it entangle-bench bash
#
# _Requirements: 12.6_

FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy the entire project (build context should be project root)
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benchmarks ./benchmarks

# Build the benchmark suite
RUN cd benchmarks && cargo build --release

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy the built binary
COPY --from=builder /app/target/release/entangle_bench /app/entangle_bench
COPY --from=builder /app/benchmarks/README.md /app/README.md

# Create results directory
RUN mkdir -p /app/results

# Set environment variables for reproducibility
ENV RUST_BACKTRACE=1
ENV ENTANGLE_BENCH_SEED=0xDEADBEEFCAFEBABE

# Default command shows help
ENTRYPOINT ["/app/entangle_bench"]
CMD ["--help"]

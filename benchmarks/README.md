# Syna Benchmarks

Performance benchmark suite comparing Syna against other embedded databases.

## Quick Start

```bash
cd benchmarks

# Run all benchmarks
cargo run --release -- all --output results

# Run specific benchmarks
cargo run --release -- write --iterations 10000 --sizes 64,1024,65536
cargo run --release -- read --iterations 10000 --threads 1,4,8
cargo run --release -- mixed --workload A --operations 100000
cargo run --release -- storage --entries 100000
cargo run --release -- vector --num-vectors 100000 --dimensions 768
cargo run --release -- faiss --quick  # FAISS vs HNSW comparison
```

## Benchmark Types

### Write Benchmarks

Tests sequential write throughput and latency:
- Value sizes: 64B, 1KB, 64KB, 1MB
- Sync modes: sync_on_write enabled/disabled
- Metrics: ops/sec, p50/p95/p99 latency

### Read Benchmarks

Tests random read performance:
- Point reads (single key lookup)
- History/tensor extraction
- Concurrent reads (1, 4, 8, 16 threads)

### Mixed Workloads (YCSB)

Standard YCSB workloads:
- **A**: 50% read, 50% update (update heavy)
- **B**: 95% read, 5% update (read mostly)
- **C**: 100% read (read only)
- **D**: 95% read, 5% insert (read latest)
- **F**: 50% read-modify-write
- **timeseries**: 90% append, 10% read

### Storage Efficiency

Measures disk usage:
- Bytes per entry
- Compression ratios (none, LZ4, delta, both)
- Compaction effectiveness

### Vector Benchmarks

Tests vector store performance:
- Insert throughput (vectors/sec)
- Search latency (brute force and HNSW)
- Recall@k accuracy

### FAISS vs HNSW Comparison

Compares SynaDB's native HNSW index against FAISS indexes:

```bash
# Quick comparison (10K vectors)
cargo run --release -- faiss --quick

# Custom configuration
cargo run --release -- faiss --dimensions 768 --num-vectors 100000 --k 10

# Full benchmark (100K and 1M vectors)
cargo run --release -- faiss --full

# With FAISS enabled (requires FAISS library)
cargo run --release --features faiss -- faiss --quick
```

**Metrics compared:**
- Insert throughput (vectors/sec)
- Search latency p50/p99 (ms)
- Memory usage (MB)
- Recall@10 accuracy

**Example output:**
```
┌─────────────────────┬───────────────┬──────────────┬──────────────┬────────────┬────────────┐
│ Index               │ Insert (v/s)  │ Search p50   │ Search p99   │ Memory MB  │ Recall@10  │
├─────────────────────┼───────────────┼──────────────┼──────────────┼────────────┼────────────┤
│ HNSW                │         50000 │       0.50ms │       1.20ms │       80.0 │      95.0% │
│ FAISS-Flat          │        100000 │      10.00ms │      15.00ms │       60.0 │     100.0% │
│ FAISS-IVF1024,Flat  │         80000 │       1.00ms │       2.00ms │       65.0 │      92.0% │
└─────────────────────┴───────────────┴──────────────┴──────────────┴────────────┴────────────┘
```

## Compared Databases

| Database | Type | Notes |
|----------|------|-------|
| Syna | Log-structured KV | Our database |
| SQLite | Relational | Industry standard embedded DB |
| DuckDB | Columnar OLAP | Analytical workloads |
| LevelDB | LSM-tree KV | Google's embedded KV |
| RocksDB | LSM-tree KV | Facebook's LevelDB fork |

## Output Formats

### JSON Report (`results/report.json`)

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "syna_version": "0.1.0",
    "system": {
      "os": "Linux 6.1",
      "cpu": "AMD Ryzen 9 5900X",
      "cores": 12,
      "ram_gb": 64
    }
  },
  "results": [...]
}
```

### Markdown Report (`results/report.md`)

Human-readable summary with tables and analysis.

### Charts (`results/*.png`)

Visual comparisons of throughput and latency.

## Reproducibility

The benchmark suite includes comprehensive reproducibility infrastructure to ensure consistent, verifiable results.

### Deterministic Random Seeds

All benchmarks use deterministic random number generation with configurable seeds:

```rust
use syna_benchmarks::reproducibility::DeterministicRng;

// Same seed always produces same sequence
let mut rng = DeterministicRng::new(42);
let value = rng.gen_bytes(1024);  // Reproducible test data
```

Default seed: `0xDEADBEEF_CAFEBABE`

### Configurable Warmup/Measurement Phases

```rust
use syna_benchmarks::reproducibility::{PhaseConfig, PhaseRunner};

let config = PhaseConfig {
    warmup_iterations: 1000,
    measurement_iterations: 10000,
    min_warmup_duration: Duration::from_secs(1),
    discard_outliers: true,
    outlier_percentile: 0.99,
};

let mut runner = PhaseRunner::new(config, seed);
runner.run_warmup(|rng| { /* warmup operation */ });
let latencies = runner.run_measurement(|rng| { /* measured operation */ });
```

### OS Cache Flushing

For cold-start benchmarks, the suite can flush OS page caches:

```rust
use syna_benchmarks::reproducibility::cache;

// Flush cache for a specific file (best-effort)
cache::try_flush_file_cache(&db_path);

// Flush system-wide cache (requires root on Linux)
cache::try_flush_system_cache();
```

Platform support:
- **Linux**: `posix_fadvise` for files, `/proc/sys/vm/drop_caches` for system
- **macOS**: `fcntl F_NOCACHE` for files, `purge` command for system
- **Windows**: `FILE_FLAG_NO_BUFFERING` (limited effectiveness)

### Statistical Analysis

Run multiple iterations and analyze variance:

```rust
use syna_benchmarks::reproducibility::{run_multiple_iterations, StatisticalAnalysis};

let results = run_multiple_iterations(5, seed, |run_seed| {
    run_benchmark(run_seed)
});

results.print_summary();
// Output:
// Throughput (ops/sec): n=5, mean=150000, stddev=2500, cv=1.7%
// ✓ Results are consistent (CV < 10%)
```

### Docker Environment

For maximum reproducibility, use the Docker environment:

```bash
# Build the benchmark image
docker build -t Syna-bench .

# Run all benchmarks
docker run --rm -v $(pwd)/results:/app/results Syna-bench all --output /app/results

# Run specific benchmark
docker run --rm Syna-bench write --iterations 10000

# Interactive shell
docker run --rm -it Syna-bench bash
```

Using Docker Compose:

```bash
# Run all benchmarks
docker-compose up benchmark-all

# Run write benchmarks only
docker-compose up benchmark-write

# Run read benchmarks only
docker-compose up benchmark-read
```

### Manual Reproducibility Checklist

If not using Docker:

1. **System preparation**:
   - Close other applications
   - Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
   - Use a dedicated SSD with consistent free space

2. **Run multiple iterations**:
   ```bash
   for i in {1..5}; do
     cargo run --release -- all --output results/run_$i
   done
   ```

3. **Check consistency**:
   - Coefficient of variation (CV) should be < 10%
   - If CV > 10%, investigate system interference

## Interpreting Results

### Throughput

Higher is better. Measured in operations per second.

| Range | Assessment |
|-------|------------|
| > 100K ops/sec | Excellent |
| 50K - 100K ops/sec | Good |
| 10K - 50K ops/sec | Acceptable |
| < 10K ops/sec | Investigate |

### Latency

Lower is better. We report:
- **p50**: Median latency (50th percentile) - typical case
- **p95**: 95th percentile (tail latency) - occasional slow operations
- **p99**: 99th percentile (worst case) - rare outliers

| Metric | Good | Acceptable | Investigate |
|--------|------|------------|-------------|
| p50 | < 10 μs | < 100 μs | > 100 μs |
| p95 | < 50 μs | < 500 μs | > 500 μs |
| p99 | < 100 μs | < 1 ms | > 1 ms |

### Storage Efficiency

Lower is better. Measured in bytes per logical entry.

| Compression | Expected Ratio | Best For |
|-------------|----------------|----------|
| None | 1.0x | Baseline |
| LZ4 | 1.5-3x | Large values, text |
| Delta | 5-100x | Time-series floats |
| Both | 10-200x | Gradual float sequences |

### Anomaly Flags

The benchmark suite automatically flags anomalies:

| Flag | Meaning | Action |
|------|---------|--------|
| ⚠️ UnexpectedlyLow | Performance below expected | Check system load |
| ⚠️ UnexpectedlyHigh | Performance above expected | Verify cold start |
| ⚠️ HighVariance | Results inconsistent | Run more iterations |
| ❌ ZeroValue | No results recorded | Check benchmark setup |

## Configuration

Edit `benchmarks/src/config.rs` to customize:
- Warmup iterations
- Measurement iterations
- Value sizes
- Thread counts
- Random seeds

### Default Configuration

```rust
BenchmarkConfig {
    warmup_iterations: 1000,
    measurement_iterations: 10000,
    value_size_bytes: 1024,
    thread_count: 1,
    sync_on_write: false,
}
```

## Requirements Coverage

This benchmark suite implements the following requirements:

### Requirement 7: Write Performance (7.1-7.6)
- Sequential write throughput ✅
- Write latency distribution ✅
- Various value sizes ✅
- Sync vs async comparison ✅
- Competitor comparison ✅
- Memory usage tracking ✅

### Requirement 8: Read Performance (8.1-8.6)
- Point read throughput ✅
- Read latency distribution ✅
- Hot/cold read testing ✅
- History/tensor extraction ✅
- Competitor comparison ✅
- Concurrent read scaling ✅

### Requirement 9: Mixed Workloads (9.1-9.6)
- YCSB-A (50/50) ✅
- YCSB-B (95/5) ✅
- YCSB-C (100% read) ✅
- YCSB-F (read-modify-write) ✅
- Time-series workload ✅
- Separate read/write metrics ✅

### Requirement 10: Storage Efficiency (10.1-10.6)
- Bytes per entry ✅
- Compression comparison ✅
- Compaction testing ✅
- Competitor comparison ✅
- Time-series patterns ✅
- Compression ratios ✅

### Requirement 11: Reporting (11.1-11.6)
- JSON report ✅
- Markdown summary ✅
- Charts (PNG) ✅
- System specifications ✅
- Version/config info ✅
- Anomaly detection ✅

### Requirement 12: Reproducibility (12.1-12.6)
- Deterministic seeds ✅
- Configurable phases ✅
- Cache flushing ✅
- Multiple iterations ✅
- Configuration logging ✅
- Docker environment ✅


# Syna Benchmark Report

## System Information

- **Date**: 2025-12-06T09:36:49.565783+00:00
- **Syna Version**: 0.1.0
- **Benchmark Version**: 1.0.0
- **OS**: Windows 11 (26200)
- **CPU**: Intel(R) Core(TM) i9-14900KF
- **Cores**: 32
- **RAM**: 63.7 GB
- **Disk Type**: Unknown

## Benchmark Configuration

- **Warmup Iterations**: 1000
- **Measurement Iterations**: 10000
- **Value Sizes**: [64, 1024, 65536] bytes
- **Thread Counts**: [1, 4, 8]
- **Databases Tested**: Syna

## Summary

- **Total Benchmarks**: 10
- **Total Duration**: 1.7 seconds
- **Best Write Throughput**: 139346 ops/sec (Syna - sequential_write)
- **Best Read Throughput**: 134725 ops/sec (Syna - random_read)
- **Lowest Latency (p50)**: 3.2 μs (Syna - ycsb_c)

## Write Performance

| Database | Value Size | Throughput (ops/s) | p50 (μs) | p95 (μs) | p99 (μs) | Disk (MB) |
|----------|------------|-------------------|----------|----------|----------|----------|
| Syna | 64 B | 139346 | 5.6 | 10.7 | 16.9 | 1.06 |
| Syna | 1024 B | 98269 | 6.8 | 16.5 | 62.7 | 11.13 |
| Syna | 65536 B | 11475 | 71.9 | 159.4 | 238.4 | 687.89 |

## Read Performance

| Database | Benchmark | Threads | Throughput (ops/s) | p50 (μs) | p95 (μs) | p99 (μs) |
|----------|-----------|---------|-------------------|----------|----------|----------|
| Syna | random_read | 1 | 134725 | 6.2 | 12.0 | 18.0 |
| Syna | random_read | 4 | 106489 | 6.9 | 14.3 | 28.2 |
| Syna | random_read | 8 | 95341 | 8.1 | 15.8 | 39.3 |

## Mixed Workloads

| Database | Workload | Throughput (ops/s) | p50 (μs) | p95 (μs) | Duration (s) |
|----------|----------|-------------------|----------|----------|-------------|
| Syna | YCSB-a | 97405 | 7.3 | 22.9 | 0.10 |
| Syna | YCSB-b | 111487 | 8.5 | 23.2 | 0.09 |
| Syna | YCSB-c | 121197 | 3.2 | 6.8 | 0.08 |

## Storage Efficiency

| Database | Benchmark | Disk (MB) | Duration (s) |
|----------|-----------|-----------|-------------|
| Syna | storage_efficiency | 0.34 | 0.06 |

## ⚠️ Anomalies Detected

The following anomalies were detected during benchmarking:

| Severity | Benchmark | Database | Issue | Suggestion |
|----------|-----------|----------|-------|------------|
| ❌ | storage_efficiency | Syna | Throughput is zero or negative | Check if the benchmark ran correctly. Consider re-running. |

---

*Report generated at 2025-12-06T09:36:49.565783+00:00 by Syna Benchmark Suite v1.0.0*


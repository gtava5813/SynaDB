//! DuckDB write benchmark implementation.
//!
//! This module provides write benchmarks for DuckDB to compare against Entangle.
//! Uses duckdb crate for DuckDB access.
//!
//! Note: This module requires the `duckdb` feature to be enabled.
//! Build with: cargo build --features duckdb
//!
//! _Requirements: 7.5_

#[cfg(feature = "duckdb")]
use duckdb::{Connection, params};
#[cfg(feature = "duckdb")]
use crate::calculate_percentiles;
#[cfg(feature = "duckdb")]
use std::time::Instant;
#[cfg(feature = "duckdb")]
use tempfile::tempdir;

use crate::{BenchmarkConfig, BenchmarkResult};

/// Run DuckDB sequential write benchmark
///
/// Tests DuckDB write performance with configurable value sizes.
/// Matches Entangle test parameters for fair comparison.
///
/// _Requirements: 7.5_
#[cfg(feature = "duckdb")]
pub fn run_duckdb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.duckdb");
    
    let conn = Connection::open(&db_path).expect("Failed to create DuckDB database");
    
    // Create table for key-value storage
    conn.execute(
        "CREATE TABLE kv (key VARCHAR PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Generate test data with deterministic content
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup phase
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            params![key, value.as_slice()],
        ).ok();
    }
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for i in 0..config.measurement_iterations {
        let key = format!("bench/{}", i);
        let op_start = Instant::now();
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            params![key, value.as_slice()],
        ).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "duckdb".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run DuckDB batch write benchmark (using transactions)
///
/// Tests DuckDB performance when batching writes in transactions.
#[cfg(feature = "duckdb")]
pub fn run_duckdb_batch_write(config: &BenchmarkConfig, batch_size: usize) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.duckdb");
    
    let conn = Connection::open(&db_path).expect("Failed to create DuckDB database");
    
    conn.execute(
        "CREATE TABLE kv (key VARCHAR PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup with batched writes
    conn.execute("BEGIN TRANSACTION", []).ok();
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            params![key, value.as_slice()],
        ).ok();
    }
    conn.execute("COMMIT", []).ok();
    
    // Measurement with batched writes
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    let num_batches = (config.measurement_iterations + batch_size - 1) / batch_size;
    
    for batch_idx in 0..num_batches {
        let batch_start = Instant::now();
        conn.execute("BEGIN TRANSACTION", []).ok();
        
        let start_i = batch_idx * batch_size;
        let end_i = std::cmp::min(start_i + batch_size, config.measurement_iterations);
        
        for i in start_i..end_i {
            let key = format!("bench/{}", i);
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                params![key, value.as_slice()],
            ).ok();
        }
        
        conn.execute("COMMIT", []).ok();
        
        // Record latency per operation in batch
        let batch_duration = batch_start.elapsed();
        let per_op_duration = batch_duration / (end_i - start_i) as u32;
        for _ in start_i..end_i {
            latencies.push(per_op_duration);
        }
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: format!("batch_write_{}", batch_size),
        database: "duckdb".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Stub function when duckdb feature is not enabled
#[cfg(not(feature = "duckdb"))]
pub fn run_duckdb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("DuckDB benchmarks require the 'duckdb' feature. Build with: cargo build --features duckdb");
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "duckdb".to_string(),
        config: config.clone(),
        throughput_ops_sec: 0.0,
        latency_p50_us: 0.0,
        latency_p95_us: 0.0,
        latency_p99_us: 0.0,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: 0.0,
    }
}

/// Stub function when duckdb feature is not enabled
#[cfg(not(feature = "duckdb"))]
pub fn run_duckdb_batch_write(config: &BenchmarkConfig, _batch_size: usize) -> BenchmarkResult {
    run_duckdb_write(config)
}

/// Generate deterministic test value
fn generate_test_value(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bytes = vec![0u8; size];
    rng.fill(&mut bytes[..]);
    bytes
}

/// Print DuckDB benchmark result
pub fn print_result(result: &BenchmarkResult) {
    println!("\nDuckDB - {} ({} bytes):", 
        result.benchmark,
        result.config.value_size_bytes
    );
    println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", result.latency_p50_us);
    println!("  Latency p95: {:.1} μs", result.latency_p95_us);
    println!("  Latency p99: {:.1} μs", result.latency_p99_us);
    println!("  Disk usage: {:.2} MB", result.disk_mb);
    println!("  Duration: {:.2} s", result.duration_secs);
}

/// Check if DuckDB feature is enabled
pub fn is_available() -> bool {
    cfg!(feature = "duckdb")
}

#[cfg(all(test, feature = "duckdb"))]
mod tests {
    use super::*;
    
    #[test]
    fn test_duckdb_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_duckdb_write(&config);
        
        assert_eq!(result.database, "duckdb");
        assert_eq!(result.benchmark, "sequential_write");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_duckdb_batch_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_duckdb_batch_write(&config, 50);
        
        assert_eq!(result.database, "duckdb");
        assert!(result.benchmark.contains("batch_write"));
        assert!(result.throughput_ops_sec > 0.0);
    }
}


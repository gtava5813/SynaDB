//! LevelDB write benchmark implementation.
//!
//! This module provides write benchmarks for LevelDB to compare against Entangle.
//! Uses rusty-leveldb crate for LevelDB access.
//!
//! Note: This module requires the `leveldb` feature to be enabled.
//! Build with: cargo build --features leveldb
//!
//! _Requirements: 7.5_

#[cfg(feature = "leveldb")]
use rusty_leveldb::{DB, Options, WriteBatch};

use crate::{BenchmarkConfig, BenchmarkResult};

#[cfg(feature = "leveldb")]
use crate::calculate_percentiles;
#[cfg(feature = "leveldb")]
use std::time::Instant;
#[cfg(feature = "leveldb")]
use tempfile::tempdir;

/// Run LevelDB sequential write benchmark
///
/// Tests LevelDB write performance with configurable value sizes.
/// Matches Entangle test parameters for fair comparison.
///
/// _Requirements: 7.5_
#[cfg(feature = "leveldb")]
pub fn run_leveldb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.leveldb");
    
    let mut opts = Options::default();
    opts.create_if_missing = true;
    
    // Configure sync based on config
    // Note: LevelDB sync is per-write, controlled via WriteOptions
    
    let mut db = DB::open(db_path.to_str().unwrap(), opts)
        .expect("Failed to create LevelDB database");
    
    // Generate test data with deterministic content
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup phase
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        db.put(key.as_bytes(), &value).ok();
    }
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for i in 0..config.measurement_iterations {
        let key = format!("bench/{}", i);
        let op_start = Instant::now();
        db.put(key.as_bytes(), &value).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    // Calculate disk usage (LevelDB uses multiple files)
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "leveldb".to_string(),
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

/// Run LevelDB batch write benchmark
///
/// Tests LevelDB performance when batching writes.
#[cfg(feature = "leveldb")]
pub fn run_leveldb_batch_write(config: &BenchmarkConfig, batch_size: usize) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.leveldb");
    
    let mut opts = Options::default();
    opts.create_if_missing = true;
    
    let mut db = DB::open(db_path.to_str().unwrap(), opts)
        .expect("Failed to create LevelDB database");
    
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup with batched writes
    {
        let mut batch = WriteBatch::new();
        for i in 0..config.warmup_iterations {
            let key = format!("warmup/{}", i);
            batch.put(key.as_bytes(), &value);
        }
        db.write(batch, false).ok();
    }
    
    // Measurement with batched writes
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    let num_batches = (config.measurement_iterations + batch_size - 1) / batch_size;
    
    for batch_idx in 0..num_batches {
        let batch_start = Instant::now();
        let mut batch = WriteBatch::new();
        
        let start_i = batch_idx * batch_size;
        let end_i = std::cmp::min(start_i + batch_size, config.measurement_iterations);
        
        for i in start_i..end_i {
            let key = format!("bench/{}", i);
            batch.put(key.as_bytes(), &value);
        }
        
        db.write(batch, false).ok();
        
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
    
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: format!("batch_write_{}", batch_size),
        database: "leveldb".to_string(),
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

/// Stub function when leveldb feature is not enabled
#[cfg(not(feature = "leveldb"))]
pub fn run_leveldb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("LevelDB benchmarks require the 'leveldb' feature. Build with: cargo build --features leveldb");
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "leveldb".to_string(),
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

/// Stub function when leveldb feature is not enabled
#[cfg(not(feature = "leveldb"))]
pub fn run_leveldb_batch_write(config: &BenchmarkConfig, _batch_size: usize) -> BenchmarkResult {
    run_leveldb_write(config)
}

/// Calculate total size of a directory
#[cfg(feature = "leveldb")]
fn calculate_dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    total += metadata.len();
                } else if metadata.is_dir() {
                    total += calculate_dir_size(&entry.path());
                }
            }
        }
    }
    total
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

/// Print LevelDB benchmark result
pub fn print_result(result: &BenchmarkResult) {
    println!("\nLevelDB - {} ({} bytes):", 
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

/// Check if LevelDB feature is enabled
pub fn is_available() -> bool {
    cfg!(feature = "leveldb")
}

#[cfg(all(test, feature = "leveldb"))]
mod tests {
    use super::*;
    
    #[test]
    fn test_leveldb_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_leveldb_write(&config);
        
        assert_eq!(result.database, "leveldb");
        assert_eq!(result.benchmark, "sequential_write");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_leveldb_batch_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_leveldb_batch_write(&config, 50);
        
        assert_eq!(result.database, "leveldb");
        assert!(result.benchmark.contains("batch_write"));
        assert!(result.throughput_ops_sec > 0.0);
    }
}


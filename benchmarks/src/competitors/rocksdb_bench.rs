//! RocksDB write benchmark implementation.
//!
//! This module provides write benchmarks for RocksDB to compare against Entangle.
//! Uses rust-rocksdb crate for RocksDB access.
//!
//! Note: This module requires the `rocksdb` feature to be enabled.
//! Build with: cargo build --features rocksdb
//!
//! RocksDB requires a C++ toolchain to build. On Windows, you may need Visual Studio
//! with C++ support. On Linux/macOS, you need clang or gcc.
//!
//! _Requirements: 7.5_

#[cfg(feature = "rocksdb")]
use rocksdb::{DB, Options, WriteBatch, WriteOptions};

use crate::{BenchmarkConfig, BenchmarkResult};

#[cfg(feature = "rocksdb")]
use crate::calculate_percentiles;
#[cfg(feature = "rocksdb")]
use std::time::Instant;
#[cfg(feature = "rocksdb")]
use tempfile::tempdir;

/// Run RocksDB sequential write benchmark
///
/// Tests RocksDB write performance with configurable value sizes.
/// Matches Entangle test parameters for fair comparison.
///
/// _Requirements: 7.5_
#[cfg(feature = "rocksdb")]
pub fn run_rocksdb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.rocksdb");
    
    let mut opts = Options::default();
    opts.create_if_missing(true);
    
    // Configure for fair comparison
    opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB write buffer
    opts.set_max_write_buffer_number(3);
    
    let db = DB::open(&opts, &db_path).expect("Failed to create RocksDB database");
    
    // Configure write options based on sync setting
    let mut write_opts = WriteOptions::default();
    write_opts.set_sync(config.sync_on_write);
    
    // Generate test data with deterministic content
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup phase
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        db.put_opt(key.as_bytes(), &value, &write_opts).ok();
    }
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for i in 0..config.measurement_iterations {
        let key = format!("bench/{}", i);
        let op_start = Instant::now();
        db.put_opt(key.as_bytes(), &value, &write_opts).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    // Calculate disk usage (RocksDB uses multiple files)
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "rocksdb".to_string(),
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

/// Run RocksDB batch write benchmark
///
/// Tests RocksDB performance when batching writes using WriteBatch.
#[cfg(feature = "rocksdb")]
pub fn run_rocksdb_batch_write(config: &BenchmarkConfig, batch_size: usize) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.rocksdb");
    
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    opts.set_max_write_buffer_number(3);
    
    let db = DB::open(&opts, &db_path).expect("Failed to create RocksDB database");
    
    let mut write_opts = WriteOptions::default();
    write_opts.set_sync(config.sync_on_write);
    
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup with batched writes
    {
        let mut batch = WriteBatch::default();
        for i in 0..config.warmup_iterations {
            let key = format!("warmup/{}", i);
            batch.put(key.as_bytes(), &value);
        }
        db.write_opt(batch, &write_opts).ok();
    }
    
    // Measurement with batched writes
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    let num_batches = (config.measurement_iterations + batch_size - 1) / batch_size;
    
    for batch_idx in 0..num_batches {
        let batch_start = Instant::now();
        let mut batch = WriteBatch::default();
        
        let start_i = batch_idx * batch_size;
        let end_i = std::cmp::min(start_i + batch_size, config.measurement_iterations);
        
        for i in start_i..end_i {
            let key = format!("bench/{}", i);
            batch.put(key.as_bytes(), &value);
        }
        
        db.write_opt(batch, &write_opts).ok();
        
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
        database: "rocksdb".to_string(),
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

/// Stub function when rocksdb feature is not enabled
#[cfg(not(feature = "rocksdb"))]
pub fn run_rocksdb_write(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("RocksDB benchmarks require the 'rocksdb' feature. Build with: cargo build --features rocksdb");
    eprintln!("Note: RocksDB requires a C++ toolchain to build.");
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "rocksdb".to_string(),
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

/// Stub function when rocksdb feature is not enabled
#[cfg(not(feature = "rocksdb"))]
pub fn run_rocksdb_batch_write(config: &BenchmarkConfig, _batch_size: usize) -> BenchmarkResult {
    run_rocksdb_write(config)
}

/// Calculate total size of a directory
#[cfg(feature = "rocksdb")]
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

/// Print RocksDB benchmark result
pub fn print_result(result: &BenchmarkResult) {
    println!("\nRocksDB - {} ({} bytes):", 
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

/// Check if RocksDB feature is enabled
pub fn is_available() -> bool {
    cfg!(feature = "rocksdb")
}

#[cfg(all(test, feature = "rocksdb"))]
mod tests {
    use super::*;
    
    #[test]
    fn test_rocksdb_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_rocksdb_write(&config);
        
        assert_eq!(result.database, "rocksdb");
        assert_eq!(result.benchmark, "sequential_write");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_rocksdb_batch_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_rocksdb_batch_write(&config, 50);
        
        assert_eq!(result.database, "rocksdb");
        assert!(result.benchmark.contains("batch_write"));
        assert!(result.throughput_ops_sec > 0.0);
    }
}


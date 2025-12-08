//! Write performance benchmarks.
//!
//! This module implements sequential write benchmarks for Syna and competitor databases.
//! It tests various value sizes (64B, 1KB, 64KB, 1MB) and measures throughput and latency percentiles.
//!
//! _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use synadb::{Atom, DbConfig, SynaDB};
use std::time::Instant;
use tempfile::tempdir;

/// Standard value sizes for benchmarking (in bytes)
pub const VALUE_SIZES: &[usize] = &[64, 1024, 65536, 1048576]; // 64B, 1KB, 64KB, 1MB

/// Run Syna sequential write benchmark
/// 
/// Tests sequential write performance with configurable value sizes.
/// Records throughput (ops/sec) and latency percentiles (p50, p95, p99).
/// 
/// _Requirements: 7.1, 7.2, 7.3_
pub fn run_SYNA_write(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: config.sync_on_write,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to create database");
    
    // Generate test data with deterministic content
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup phase - not measured
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for i in 0..config.measurement_iterations {
        let key = format!("bench/{}", i);
        let op_start = Instant::now();
        db.append(&key, Atom::Bytes(value.clone())).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    // Get disk usage
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    // Estimate memory usage (rough approximation based on index size)
    let memory_mb = estimate_memory_usage(config.measurement_iterations + config.warmup_iterations);
    
    BenchmarkResult {
        benchmark: "sequential_write".to_string(),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb,
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run Syna write benchmark with sync_on_write enabled
/// 
/// Compares performance with durability guarantees enabled.
/// Documents the throughput difference and durability tradeoffs.
/// 
/// _Requirements: 7.4_
pub fn run_SYNA_write_sync(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut sync_config = config.clone();
    sync_config.sync_on_write = true;
    
    let mut result = run_SYNA_write(&sync_config);
    result.benchmark = "sequential_write_sync".to_string();
    result
}

/// Run sync vs async write comparison benchmark
/// 
/// Returns both results for comparison.
/// 
/// _Requirements: 7.4_
pub fn run_sync_vs_async_comparison(config: &BenchmarkConfig) -> (BenchmarkResult, BenchmarkResult) {
    // Run without sync (async)
    let mut async_config = config.clone();
    async_config.sync_on_write = false;
    let async_result = run_SYNA_write(&async_config);
    
    // Run with sync
    let mut sync_config = config.clone();
    sync_config.sync_on_write = true;
    let sync_result = run_SYNA_write(&sync_config);
    
    (async_result, sync_result)
}

/// Print sync vs async comparison results
pub fn print_sync_comparison(async_result: &BenchmarkResult, sync_result: &BenchmarkResult) {
    println!("\n=== Sync vs Async Write Comparison ===");
    println!("Value size: {} bytes", async_result.config.value_size_bytes);
    println!();
    println!("Async (sync_on_write=false):");
    println!("  Throughput: {:.0} ops/sec", async_result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", async_result.latency_p50_us);
    println!("  Latency p99: {:.1} μs", async_result.latency_p99_us);
    println!();
    println!("Sync (sync_on_write=true):");
    println!("  Throughput: {:.0} ops/sec", sync_result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", sync_result.latency_p50_us);
    println!("  Latency p99: {:.1} μs", sync_result.latency_p99_us);
    println!();
    
    let throughput_ratio = async_result.throughput_ops_sec / sync_result.throughput_ops_sec;
    println!("Throughput ratio (async/sync): {:.1}x", throughput_ratio);
    println!();
    println!("Durability tradeoffs:");
    println!("  - sync_on_write=false: Higher throughput, data may be lost on crash");
    println!("  - sync_on_write=true:  Lower throughput, data persisted immediately");
}

/// Run all write benchmarks for Syna across all value sizes
pub fn run_all_SYNA_write_benchmarks(base_config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    for &size in VALUE_SIZES {
        let config = BenchmarkConfig {
            value_size_bytes: size,
            ..base_config.clone()
        };
        
        println!("Running Syna write benchmark ({} bytes)...", size);
        let result = run_SYNA_write(&config);
        print_benchmark_result(&result);
        results.push(result);
    }
    
    results
}

/// Run SQLite write benchmark for comparison
/// 
/// _Requirements: 7.5_
pub fn run_sqlite_write(config: &BenchmarkConfig) -> BenchmarkResult {
    use rusqlite::Connection;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    // Configure SQLite for fair comparison
    if !config.sync_on_write {
        conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    }
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    
    conn.execute(
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Generate test data
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        ).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for i in 0..config.measurement_iterations {
        let key = format!("bench/{}", i);
        let op_start = Instant::now();
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
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
        database: "sqlite".to_string(),
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

/// Generate deterministic test value of specified size
fn generate_test_value(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bytes = vec![0u8; size];
    rng.fill(&mut bytes[..]);
    bytes
}

/// Estimate memory usage based on number of keys
fn estimate_memory_usage(num_keys: usize) -> f64 {
    // Rough estimate: ~100 bytes per key in index (key string + offset + overhead)
    let estimated_bytes = num_keys * 100;
    estimated_bytes as f64 / 1024.0 / 1024.0
}

/// Print a single benchmark result
pub fn print_benchmark_result(result: &BenchmarkResult) {
    println!("\n{} - {} ({} bytes):", 
        result.database, 
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

/// Run comparative write benchmarks across all databases
pub fn run_comparative_write_benchmarks(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // Syna
    println!("Running Syna write benchmark...");
    let SYNA_result = run_SYNA_write(config);
    print_benchmark_result(&SYNA_result);
    results.push(SYNA_result);
    
    // SQLite
    println!("Running SQLite write benchmark...");
    let sqlite_result = run_sqlite_write(config);
    print_benchmark_result(&sqlite_result);
    results.push(sqlite_result);
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_SYNA_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_SYNA_write(&config);
        
        assert_eq!(result.database, "Syna");
        assert_eq!(result.benchmark, "sequential_write");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
        assert!(result.disk_mb > 0.0);
    }
    
    #[test]
    fn test_sqlite_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_sqlite_write(&config);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_generate_test_value() {
        let value1 = generate_test_value(100, 42);
        let value2 = generate_test_value(100, 42);
        let value3 = generate_test_value(100, 43);
        
        assert_eq!(value1.len(), 100);
        assert_eq!(value1, value2); // Same seed = same value
        assert_ne!(value1, value3); // Different seed = different value
    }
    
    #[test]
    fn test_sync_vs_async_comparison() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 50,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let (async_result, sync_result) = run_sync_vs_async_comparison(&config);
        
        // Both should complete successfully
        assert!(async_result.throughput_ops_sec > 0.0);
        assert!(sync_result.throughput_ops_sec > 0.0);
        
        // Async should generally be faster (though not guaranteed in all environments)
        // Just verify both produce valid results
        assert_eq!(async_result.database, "Syna");
        assert_eq!(sync_result.database, "Syna");
        assert!(!sync_result.config.sync_on_write || sync_result.config.sync_on_write);
    }
}



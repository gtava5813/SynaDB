//! SQLite write benchmark implementation.
//!
//! This module provides write benchmarks for SQLite to compare against Entangle.
//! Uses rusqlite crate for SQLite access.
//!
//! _Requirements: 7.5_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use rusqlite::Connection;
use std::time::Instant;
use tempfile::tempdir;

/// Run SQLite sequential write benchmark
///
/// Tests SQLite write performance with configurable value sizes.
/// Matches Entangle test parameters for fair comparison.
///
/// _Requirements: 7.5_
pub fn run_sqlite_write(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    // Configure SQLite for fair comparison
    configure_sqlite(&conn, config.sync_on_write);
    
    conn.execute(
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Generate test data with deterministic content
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup phase
    for i in 0..config.warmup_iterations {
        let key = format!("warmup/{}", i);
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        ).ok();
    }
    
    // Measurement phase
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

/// Run SQLite write benchmark with sync enabled
pub fn run_sqlite_write_sync(config: &BenchmarkConfig) -> BenchmarkResult {
    let mut sync_config = config.clone();
    sync_config.sync_on_write = true;
    run_sqlite_write(&sync_config)
}

/// Run SQLite batch write benchmark (using transactions)
///
/// Tests SQLite performance when batching writes in transactions.
/// This is a common optimization pattern for SQLite.
pub fn run_sqlite_batch_write(config: &BenchmarkConfig, batch_size: usize) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let mut conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    configure_sqlite(&conn, config.sync_on_write);
    
    conn.execute(
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    let value = generate_test_value(config.value_size_bytes, 42);
    
    // Warmup with batched writes
    {
        let tx = conn.transaction().expect("Failed to start transaction");
        for i in 0..config.warmup_iterations {
            let key = format!("warmup/{}", i);
            tx.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, value],
            ).ok();
        }
        tx.commit().ok();
    }
    
    // Measurement with batched writes
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    let num_batches = (config.measurement_iterations + batch_size - 1) / batch_size;
    
    for batch_idx in 0..num_batches {
        let batch_start = Instant::now();
        let tx = conn.transaction().expect("Failed to start transaction");
        
        let start_i = batch_idx * batch_size;
        let end_i = std::cmp::min(start_i + batch_size, config.measurement_iterations);
        
        for i in start_i..end_i {
            let key = format!("bench/{}", i);
            tx.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, value],
            ).ok();
        }
        
        tx.commit().ok();
        
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

/// Configure SQLite for benchmarking
fn configure_sqlite(conn: &Connection, sync_on_write: bool) {
    // Use WAL mode for better concurrent performance
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    
    // Configure synchronous mode based on sync_on_write setting
    if sync_on_write {
        conn.execute_batch("PRAGMA synchronous = FULL;").ok();
    } else {
        conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    }
    
    // Other performance optimizations
    conn.execute_batch("PRAGMA cache_size = -64000;").ok(); // 64MB cache
    conn.execute_batch("PRAGMA temp_store = MEMORY;").ok();
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

/// Print SQLite benchmark result
pub fn print_result(result: &BenchmarkResult) {
    println!("\nSQLite - {} ({} bytes):", 
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

#[cfg(test)]
mod tests {
    use super::*;
    
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
        assert_eq!(result.benchmark, "sequential_write");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
        assert!(result.disk_mb > 0.0);
    }
    
    #[test]
    fn test_sqlite_batch_write_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            ..Default::default()
        };
        
        let result = run_sqlite_batch_write(&config, 50);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.benchmark.contains("batch_write"));
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_sqlite_sync_write() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 50,
            value_size_bytes: 64,
            sync_on_write: true,
            ..Default::default()
        };
        
        let result = run_sqlite_write(&config);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.throughput_ops_sec > 0.0);
    }
}


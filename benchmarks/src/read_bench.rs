//! Read performance benchmarks.
//!
//! This module implements read benchmarks for Syna and competitor databases.
//! It tests point reads (hot and cold), history/tensor extraction, and concurrent reads.
//!
//! _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use synadb::{Atom, DbConfig, SynaDB};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;
use tempfile::tempdir;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// History lengths to test for tensor extraction benchmarks
pub const HISTORY_LENGTHS: &[usize] = &[10, 100, 1000, 10000];

/// Thread counts to test for concurrent read benchmarks
pub const THREAD_COUNTS: &[usize] = &[1, 4, 8, 16];

/// Run Syna random point read benchmark
///
/// Tests random key point read performance.
/// Records throughput (ops/sec) and latency percentiles (p50, p95, p99).
///
/// _Requirements: 8.1, 8.2, 8.3_
pub fn run_SYNA_read(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config.clone())
        .expect("Failed to create database");
    
    // Pre-populate database with test data
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = config.measurement_iterations;
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }
    
    drop(db);
    
    // Reopen for reads (simulates cold start)
    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to reopen database");
    
    // Warmup phase
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..config.warmup_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        db.get(&key).ok();
    }
    
    // Measurement phase - random reads
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        db.get(&key).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "random_read".to_string(),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: estimate_memory_usage(num_keys),
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run Syna hot read benchmark (repeated keys - cached)
///
/// Tests read performance when accessing the same keys repeatedly.
/// This simulates cache-friendly access patterns.
///
/// _Requirements: 8.3_
pub fn run_SYNA_hot_read(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config.clone())
        .expect("Failed to create database");
    
    // Pre-populate with a small set of keys (hot set)
    let value = generate_test_value(config.value_size_bytes, 42);
    let hot_set_size = 100; // Small set for cache-friendly access
    
    for i in 0..hot_set_size {
        let key = format!("hot/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }
    
    drop(db);
    
    // Reopen for reads
    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to reopen database");
    
    // Warmup - access all hot keys to warm cache
    for i in 0..hot_set_size {
        let key = format!("hot/{}", i);
        db.get(&key).ok();
    }
    
    // Measurement - repeatedly access hot keys
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("hot/{}", rng.gen_range(0..hot_set_size));
        let op_start = Instant::now();
        db.get(&key).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "hot_read".to_string(),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: estimate_memory_usage(hot_set_size),
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run Syna cold read benchmark (uncached random keys)
///
/// Tests read performance with cache-unfriendly access patterns.
/// Uses a large key space to minimize cache hits.
///
/// _Requirements: 8.3_
pub fn run_SYNA_cold_read(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config.clone())
        .expect("Failed to create database");
    
    // Pre-populate with a large set of keys (cold set)
    let value = generate_test_value(config.value_size_bytes, 42);
    let cold_set_size = config.measurement_iterations * 10; // Large set for cache misses
    
    for i in 0..cold_set_size {
        let key = format!("cold/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }
    
    drop(db);
    
    // Reopen for reads (cold start)
    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to reopen database");
    
    // No warmup for cold reads - we want to measure cold access
    
    // Measurement - random access across large key space
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("cold/{}", rng.gen_range(0..cold_set_size));
        let op_start = Instant::now();
        db.get(&key).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "cold_read".to_string(),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: estimate_memory_usage(cold_set_size),
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}


/// Run Syna history/tensor read benchmark
///
/// Tests history extraction performance with various history lengths.
/// Measures tensor extraction overhead for ML workloads.
///
/// _Requirements: 8.4_
pub fn run_SYNA_history_read(config: &BenchmarkConfig, history_length: usize) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config.clone())
        .expect("Failed to create database");
    
    // Pre-populate with history data (simulating sensor readings)
    let num_keys = 100; // 100 different sensors
    for key_idx in 0..num_keys {
        let key = format!("sensor/{}", key_idx);
        for i in 0..history_length {
            // Simulate gradual changes (realistic sensor data)
            let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
            db.append(&key, Atom::Float(value)).ok();
        }
    }
    
    drop(db);
    
    // Reopen for reads
    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to reopen database");
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..config.warmup_iterations.min(num_keys) {
        let key = format!("sensor/{}", rng.gen_range(0..num_keys));
        db.get_history_floats(&key).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("sensor/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        let history = db.get_history_floats(&key).ok();
        latencies.push(op_start.elapsed());
        
        // Verify we got the expected history length
        if let Some(h) = history {
            debug_assert_eq!(h.len(), history_length);
        }
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    // Calculate values per second (for tensor extraction metric)
    let values_per_sec = throughput * history_length as f64;
    
    BenchmarkResult {
        benchmark: format!("history_read_{}", history_length),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: estimate_memory_usage(num_keys * history_length),
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run Syna concurrent read benchmark
///
/// Tests multi-threaded read performance.
/// Measures scaling efficiency across 1, 4, 8, 16 threads.
///
/// _Requirements: 8.6_
pub fn run_SYNA_concurrent_read(config: &BenchmarkConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");
    
    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, db_config.clone())
        .expect("Failed to create database");
    
    // Pre-populate database
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = 10000; // Shared key space
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }
    
    drop(db);
    
    // Reopen for concurrent reads
    let db = Arc::new(parking_lot::Mutex::new(
        SynaDB::with_config(&db_path, db_config)
            .expect("Failed to reopen database")
    ));
    
    let thread_count = config.thread_count;
    let ops_per_thread = config.measurement_iterations / thread_count;
    let barrier = Arc::new(Barrier::new(thread_count));
    
    let mut handles = Vec::new();
    let all_latencies = Arc::new(parking_lot::Mutex::new(Vec::new()));
    
    let start = Instant::now();
    
    for thread_id in 0..thread_count {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let all_latencies = Arc::clone(&all_latencies);
        
        let handle = thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(42 + thread_id as u64);
            let mut thread_latencies = Vec::with_capacity(ops_per_thread);
            
            // Synchronize thread start
            barrier.wait();
            
            for _ in 0..ops_per_thread {
                let key = format!("key/{}", rng.gen_range(0..num_keys));
                let op_start = Instant::now();
                {
                    let mut db = db.lock();
                    db.get(&key).ok();
                }
                thread_latencies.push(op_start.elapsed());
            }
            
            // Collect latencies
            all_latencies.lock().extend(thread_latencies);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    let total_duration = start.elapsed();
    let total_ops = ops_per_thread * thread_count;
    let throughput = total_ops as f64 / total_duration.as_secs_f64();
    
    let latencies = Arc::try_unwrap(all_latencies)
        .expect("Failed to unwrap latencies")
        .into_inner();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: format!("concurrent_read_{}t", thread_count),
        database: "Syna".to_string(),
        config: config.clone(),
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: estimate_memory_usage(num_keys),
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run all Syna read benchmarks
pub fn run_all_SYNA_read_benchmarks(base_config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // Point reads
    println!("Running Syna random read benchmark...");
    let result = run_SYNA_read(base_config);
    print_benchmark_result(&result);
    results.push(result);
    
    // Hot reads
    println!("Running Syna hot read benchmark...");
    let result = run_SYNA_hot_read(base_config);
    print_benchmark_result(&result);
    results.push(result);
    
    // Cold reads
    println!("Running Syna cold read benchmark...");
    let result = run_SYNA_cold_read(base_config);
    print_benchmark_result(&result);
    results.push(result);
    
    // History reads with various lengths
    for &history_length in HISTORY_LENGTHS {
        println!("Running Syna history read benchmark (length={})...", history_length);
        let result = run_SYNA_history_read(base_config, history_length);
        print_benchmark_result(&result);
        results.push(result);
    }
    
    // Concurrent reads with various thread counts
    for &thread_count in THREAD_COUNTS {
        println!("Running Syna concurrent read benchmark ({} threads)...", thread_count);
        let config = BenchmarkConfig {
            thread_count,
            ..base_config.clone()
        };
        let result = run_SYNA_concurrent_read(&config);
        print_benchmark_result(&result);
        results.push(result);
    }
    
    results
}


// ============================================================================
// Competitor Read Benchmarks
// ============================================================================

/// Run SQLite random point read benchmark
///
/// _Requirements: 8.5_
pub fn run_sqlite_read(config: &BenchmarkConfig) -> BenchmarkResult {
    use rusqlite::Connection;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    // Configure SQLite
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    conn.execute_batch("PRAGMA cache_size = -64000;").ok();
    
    conn.execute(
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Pre-populate
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = config.measurement_iterations;
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        ).ok();
    }
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut stmt = conn.prepare("SELECT value FROM kv WHERE key = ?1")
        .expect("Failed to prepare statement");
    
    for _ in 0..config.warmup_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

/// Run SQLite hot read benchmark
///
/// _Requirements: 8.5_
pub fn run_sqlite_hot_read(config: &BenchmarkConfig) -> BenchmarkResult {
    use rusqlite::Connection;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    conn.execute_batch("PRAGMA cache_size = -64000;").ok();
    
    conn.execute(
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Pre-populate with small hot set
    let value = generate_test_value(config.value_size_bytes, 42);
    let hot_set_size = 100;
    
    for i in 0..hot_set_size {
        let key = format!("hot/{}", i);
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        ).ok();
    }
    
    // Warmup all hot keys
    let mut stmt = conn.prepare("SELECT value FROM kv WHERE key = ?1")
        .expect("Failed to prepare statement");
    
    for i in 0..hot_set_size {
        let key = format!("hot/{}", i);
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("hot/{}", rng.gen_range(0..hot_set_size));
        let op_start = Instant::now();
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "hot_read".to_string(),
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

/// Run SQLite history read benchmark (simulated with multiple rows)
///
/// _Requirements: 8.5_
pub fn run_sqlite_history_read(config: &BenchmarkConfig, history_length: usize) -> BenchmarkResult {
    use rusqlite::Connection;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    conn.execute_batch("PRAGMA cache_size = -64000;").ok();
    
    conn.execute(
        "CREATE TABLE readings (key TEXT, timestamp INTEGER, value REAL, PRIMARY KEY (key, timestamp))",
        [],
    ).expect("Failed to create table");
    conn.execute("CREATE INDEX idx_key ON readings(key)", []).ok();
    
    // Pre-populate with history data
    let num_keys = 100;
    for key_idx in 0..num_keys {
        let key = format!("sensor/{}", key_idx);
        for i in 0..history_length {
            let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
            conn.execute(
                "INSERT INTO readings (key, timestamp, value) VALUES (?1, ?2, ?3)",
                rusqlite::params![key, i as i64, value],
            ).ok();
        }
    }
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut stmt = conn.prepare("SELECT value FROM readings WHERE key = ?1 ORDER BY timestamp")
        .expect("Failed to prepare statement");
    
    for _ in 0..config.warmup_iterations.min(num_keys) {
        let key = format!("sensor/{}", rng.gen_range(0..num_keys));
        let _: Vec<f64> = stmt.query_map([&key], |row| row.get(0))
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("sensor/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        let _: Vec<f64> = stmt.query_map([&key], |row| row.get(0))
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: format!("history_read_{}", history_length),
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


// ============================================================================
// DuckDB Read Benchmarks (feature-gated)
// ============================================================================

/// Run DuckDB random point read benchmark
///
/// _Requirements: 8.5_
#[cfg(feature = "duckdb")]
pub fn run_duckdb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    use duckdb::{Connection, params};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.duckdb");
    
    let conn = Connection::open(&db_path).expect("Failed to create DuckDB database");
    
    conn.execute(
        "CREATE TABLE kv (key VARCHAR PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");
    
    // Pre-populate
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = config.measurement_iterations;
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)",
            params![key, value.as_slice()],
        ).ok();
    }
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut stmt = conn.prepare("SELECT value FROM kv WHERE key = ?")
        .expect("Failed to prepare statement");
    
    for _ in 0..config.warmup_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        let _: Option<Vec<u8>> = stmt.query_row([&key], |row| row.get(0)).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

#[cfg(not(feature = "duckdb"))]
pub fn run_duckdb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("DuckDB benchmarks require the 'duckdb' feature. Build with: cargo build --features duckdb");
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

// ============================================================================
// LevelDB Read Benchmarks (feature-gated)
// ============================================================================

/// Run LevelDB random point read benchmark
///
/// _Requirements: 8.5_
#[cfg(feature = "leveldb")]
pub fn run_leveldb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    use rusty_leveldb::{DB, Options};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.leveldb");
    
    let mut opts = Options::default();
    opts.create_if_missing = true;
    
    let mut db = DB::open(db_path.to_str().unwrap(), opts)
        .expect("Failed to create LevelDB database");
    
    // Pre-populate
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = config.measurement_iterations;
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        db.put(key.as_bytes(), &value).ok();
    }
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..config.warmup_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        db.get(key.as_bytes()).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        db.get(key.as_bytes()).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

#[cfg(not(feature = "leveldb"))]
pub fn run_leveldb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("LevelDB benchmarks require the 'leveldb' feature. Build with: cargo build --features leveldb");
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

// ============================================================================
// RocksDB Read Benchmarks (feature-gated)
// ============================================================================

/// Run RocksDB random point read benchmark
///
/// _Requirements: 8.5_
#[cfg(feature = "rocksdb")]
pub fn run_rocksdb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    use rocksdb::{DB, Options};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.rocksdb");
    
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    
    let db = DB::open(&opts, &db_path).expect("Failed to create RocksDB database");
    
    // Pre-populate
    let value = generate_test_value(config.value_size_bytes, 42);
    let num_keys = config.measurement_iterations;
    
    for i in 0..num_keys {
        let key = format!("key/{}", i);
        db.put(key.as_bytes(), &value).ok();
    }
    
    // Warmup
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..config.warmup_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        db.get(key.as_bytes()).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.measurement_iterations);
    let start = Instant::now();
    
    for _ in 0..config.measurement_iterations {
        let key = format!("key/{}", rng.gen_range(0..num_keys));
        let op_start = Instant::now();
        db.get(key.as_bytes()).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.measurement_iterations as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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

#[cfg(not(feature = "rocksdb"))]
pub fn run_rocksdb_read(config: &BenchmarkConfig) -> BenchmarkResult {
    eprintln!("RocksDB benchmarks require the 'rocksdb' feature. Build with: cargo build --features rocksdb");
    BenchmarkResult {
        benchmark: "random_read".to_string(),
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


// ============================================================================
// Comparative Benchmarks
// ============================================================================

/// Run comparative read benchmarks across all databases
///
/// _Requirements: 8.5_
pub fn run_comparative_read_benchmarks(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // Syna
    println!("Running Syna read benchmark...");
    let SYNA_result = run_SYNA_read(config);
    print_benchmark_result(&SYNA_result);
    results.push(SYNA_result);
    
    // SQLite
    println!("Running SQLite read benchmark...");
    let sqlite_result = run_sqlite_read(config);
    print_benchmark_result(&sqlite_result);
    results.push(sqlite_result);
    
    // DuckDB (if available)
    println!("Running DuckDB read benchmark...");
    let duckdb_result = run_duckdb_read(config);
    if duckdb_result.throughput_ops_sec > 0.0 {
        print_benchmark_result(&duckdb_result);
    }
    results.push(duckdb_result);
    
    // LevelDB (if available)
    println!("Running LevelDB read benchmark...");
    let leveldb_result = run_leveldb_read(config);
    if leveldb_result.throughput_ops_sec > 0.0 {
        print_benchmark_result(&leveldb_result);
    }
    results.push(leveldb_result);
    
    // RocksDB (if available)
    println!("Running RocksDB read benchmark...");
    let rocksdb_result = run_rocksdb_read(config);
    if rocksdb_result.throughput_ops_sec > 0.0 {
        print_benchmark_result(&rocksdb_result);
    }
    results.push(rocksdb_result);
    
    results
}

/// Run comparative history read benchmarks
pub fn run_comparative_history_benchmarks(config: &BenchmarkConfig, history_length: usize) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    println!("Running Syna history read benchmark (length={})...", history_length);
    let SYNA_result = run_SYNA_history_read(config, history_length);
    print_benchmark_result(&SYNA_result);
    results.push(SYNA_result);
    
    println!("Running SQLite history read benchmark (length={})...", history_length);
    let sqlite_result = run_sqlite_history_read(config, history_length);
    print_benchmark_result(&sqlite_result);
    results.push(sqlite_result);
    
    results
}

/// Print comparison summary
pub fn print_read_comparison(results: &[BenchmarkResult]) {
    println!("\n=== Read Benchmark Comparison ===\n");
    println!("{:<12} {:>15} {:>12} {:>12} {:>12}",
        "Database", "Throughput", "p50 (μs)", "p95 (μs)", "p99 (μs)");
    println!("{}", "-".repeat(65));
    
    for result in results {
        if result.throughput_ops_sec > 0.0 {
            println!("{:<12} {:>12.0} ops/s {:>12.1} {:>12.1} {:>12.1}",
                result.database,
                result.throughput_ops_sec,
                result.latency_p50_us,
                result.latency_p95_us,
                result.latency_p99_us
            );
        }
    }
    println!();
}

// ============================================================================
// Utility Functions
// ============================================================================

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

/// Calculate total size of a directory
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

/// Print a single benchmark result
pub fn print_benchmark_result(result: &BenchmarkResult) {
    println!("\n{} - {}:", result.database, result.benchmark);
    println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", result.latency_p50_us);
    println!("  Latency p95: {:.1} μs", result.latency_p95_us);
    println!("  Latency p99: {:.1} μs", result.latency_p99_us);
    if result.disk_mb > 0.0 {
        println!("  Disk usage: {:.2} MB", result.disk_mb);
    }
    println!("  Duration: {:.2} s", result.duration_secs);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            thread_count: 1,
            ..Default::default()
        }
    }
    
    #[test]
    fn test_SYNA_read_benchmark() {
        let config = test_config();
        let result = run_SYNA_read(&config);
        
        assert_eq!(result.database, "Syna");
        assert_eq!(result.benchmark, "random_read");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_SYNA_hot_read_benchmark() {
        let config = test_config();
        let result = run_SYNA_hot_read(&config);
        
        assert_eq!(result.database, "Syna");
        assert_eq!(result.benchmark, "hot_read");
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_SYNA_cold_read_benchmark() {
        let config = test_config();
        let result = run_SYNA_cold_read(&config);
        
        assert_eq!(result.database, "Syna");
        assert_eq!(result.benchmark, "cold_read");
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_SYNA_history_read_benchmark() {
        let config = test_config();
        let result = run_SYNA_history_read(&config, 100);
        
        assert_eq!(result.database, "Syna");
        assert!(result.benchmark.contains("history_read"));
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_SYNA_concurrent_read_benchmark() {
        let config = BenchmarkConfig {
            warmup_iterations: 10,
            measurement_iterations: 100,
            value_size_bytes: 64,
            thread_count: 4,
            ..Default::default()
        };
        
        let result = run_SYNA_concurrent_read(&config);
        
        assert_eq!(result.database, "Syna");
        assert!(result.benchmark.contains("concurrent_read"));
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_sqlite_read_benchmark() {
        let config = test_config();
        let result = run_sqlite_read(&config);
        
        assert_eq!(result.database, "sqlite");
        assert_eq!(result.benchmark, "random_read");
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_sqlite_hot_read_benchmark() {
        let config = test_config();
        let result = run_sqlite_hot_read(&config);
        
        assert_eq!(result.database, "sqlite");
        assert_eq!(result.benchmark, "hot_read");
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_sqlite_history_read_benchmark() {
        let config = test_config();
        let result = run_sqlite_history_read(&config, 100);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.benchmark.contains("history_read"));
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
}



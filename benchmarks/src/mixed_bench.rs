//! Mixed workload benchmarks (YCSB-style).
//!
//! This module implements YCSB (Yahoo Cloud Serving Benchmark) workloads
//! and time-series specific workloads for Syna and competitor databases.
//!
//! YCSB Workloads:
//! - A: 50% read, 50% update (update heavy)
//! - B: 95% read, 5% update (read mostly)
//! - C: 100% read (read only)
//! - F: 50% read, 50% read-modify-write
//! - Timeseries: 90% append, 5% point read, 5% range read
//!
//! _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use synadb::{Atom, DbConfig, SynaDB};
use std::time::{Duration, Instant};
use tempfile::tempdir;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// YCSB workload definitions
#[derive(Debug, Clone, Copy)]
pub struct YcsbWorkload {
    pub name: &'static str,
    pub read_proportion: f64,
    pub update_proportion: f64,
    pub insert_proportion: f64,
    pub read_modify_write_proportion: f64,
}

impl YcsbWorkload {
    /// YCSB-A: 50% read, 50% update (update heavy)
    /// _Requirements: 9.1_
    pub fn a() -> Self {
        Self {
            name: "A",
            read_proportion: 0.5,
            update_proportion: 0.5,
            insert_proportion: 0.0,
            read_modify_write_proportion: 0.0,
        }
    }

    /// YCSB-B: 95% read, 5% update (read mostly)
    /// _Requirements: 9.2_
    pub fn b() -> Self {
        Self {
            name: "B",
            read_proportion: 0.95,
            update_proportion: 0.05,
            insert_proportion: 0.0,
            read_modify_write_proportion: 0.0,
        }
    }

    /// YCSB-C: 100% read (read only)
    /// _Requirements: 9.3_
    pub fn c() -> Self {
        Self {
            name: "C",
            read_proportion: 1.0,
            update_proportion: 0.0,
            insert_proportion: 0.0,
            read_modify_write_proportion: 0.0,
        }
    }

    /// YCSB-D: 95% read, 5% insert (read latest)
    pub fn d() -> Self {
        Self {
            name: "D",
            read_proportion: 0.95,
            update_proportion: 0.0,
            insert_proportion: 0.05,
            read_modify_write_proportion: 0.0,
        }
    }

    /// YCSB-F: 50% read, 50% read-modify-write
    /// _Requirements: 9.4_
    pub fn f() -> Self {
        Self {
            name: "F",
            read_proportion: 0.5,
            update_proportion: 0.0,
            insert_proportion: 0.0,
            read_modify_write_proportion: 0.5,
        }
    }

    /// Time-series workload: 90% append, 5% point read, 5% range read
    /// _Requirements: 9.5_
    pub fn timeseries() -> Self {
        Self {
            name: "timeseries",
            read_proportion: 0.05,
            update_proportion: 0.0,
            insert_proportion: 0.90,
            read_modify_write_proportion: 0.05, // Used for range reads
        }
    }
}


/// Result of a mixed workload benchmark with separate read/write metrics
#[derive(Debug, Clone)]
pub struct MixedWorkloadResult {
    pub workload: String,
    pub database: String,
    pub total_ops: usize,
    pub total_duration_secs: f64,
    pub total_throughput_ops_sec: f64,
    pub read_ops: usize,
    pub read_throughput_ops_sec: f64,
    pub read_latency_p50_us: f64,
    pub read_latency_p95_us: f64,
    pub read_latency_p99_us: f64,
    pub write_ops: usize,
    pub write_throughput_ops_sec: f64,
    pub write_latency_p50_us: f64,
    pub write_latency_p95_us: f64,
    pub write_latency_p99_us: f64,
    pub disk_mb: f64,
}

impl MixedWorkloadResult {
    /// Convert to standard BenchmarkResult (uses combined metrics)
    pub fn to_benchmark_result(&self) -> BenchmarkResult {
        BenchmarkResult {
            benchmark: format!("ycsb_{}", self.workload.to_lowercase()),
            database: self.database.clone(),
            config: BenchmarkConfig {
                measurement_iterations: self.total_ops,
                ..Default::default()
            },
            throughput_ops_sec: self.total_throughput_ops_sec,
            latency_p50_us: (self.read_latency_p50_us + self.write_latency_p50_us) / 2.0,
            latency_p95_us: (self.read_latency_p95_us + self.write_latency_p95_us) / 2.0,
            latency_p99_us: (self.read_latency_p99_us + self.write_latency_p99_us) / 2.0,
            memory_mb: 0.0,
            disk_mb: self.disk_mb,
            duration_secs: self.total_duration_secs,
        }
    }
}


/// Run YCSB-style benchmark on Syna
///
/// Implements standard YCSB workloads with separate read/write metrics.
///
/// _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_
pub fn run_ycsb(workload_name: &str, operations: usize) -> BenchmarkResult {
    run_ycsb_detailed(workload_name, operations).to_benchmark_result()
}

/// Run YCSB-style benchmark with detailed metrics
pub fn run_ycsb_detailed(workload_name: &str, operations: usize) -> MixedWorkloadResult {
    let workload = match workload_name.to_uppercase().as_str() {
        "A" => YcsbWorkload::a(),
        "B" => YcsbWorkload::b(),
        "C" => YcsbWorkload::c(),
        "D" => YcsbWorkload::d(),
        "F" => YcsbWorkload::f(),
        "TIMESERIES" => YcsbWorkload::timeseries(),
        _ => YcsbWorkload::a(),
    };

    run_SYNA_ycsb(&workload, operations)
}

/// Run YCSB workload on Syna database
fn run_SYNA_ycsb(workload: &YcsbWorkload, operations: usize) -> MixedWorkloadResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");

    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to create database");

    // Pre-populate with initial data
    let initial_records = 10000;
    let value = generate_test_value(1024, 42);

    for i in 0..initial_records {
        let key = format!("record/{}", i);
        db.append(&key, Atom::Bytes(value.clone())).ok();
    }

    // Run workload
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();
    let mut insert_counter = initial_records;

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let cumulative_read = workload.read_proportion;
        let cumulative_update = cumulative_read + workload.update_proportion;
        let cumulative_insert = cumulative_update + workload.insert_proportion;

        if roll < cumulative_read {
            // Read operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.get(&key).ok();
            read_latencies.push(op_start.elapsed());
        } else if roll < cumulative_update {
            // Update operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.append(&key, Atom::Bytes(value.clone())).ok();
            write_latencies.push(op_start.elapsed());
        } else if roll < cumulative_insert {
            // Insert operation
            let key = format!("record/{}", insert_counter);
            insert_counter += 1;
            let op_start = Instant::now();
            db.append(&key, Atom::Bytes(value.clone())).ok();
            write_latencies.push(op_start.elapsed());
        } else {
            // Read-modify-write operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            if let Ok(Some(Atom::Bytes(data))) = db.get(&key) {
                // Modify: increment first byte
                let mut modified = data;
                if !modified.is_empty() {
                    modified[0] = modified[0].wrapping_add(1);
                }
                db.append(&key, Atom::Bytes(modified)).ok();
            }
            write_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    // Calculate read metrics
    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    // Calculate write metrics
    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "Syna".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}


/// Run time-series workload benchmark on Syna
///
/// Simulates realistic IoT ingestion pattern:
/// - 90% appends (new sensor readings)
/// - 5% point reads (latest value)
/// - 5% range reads (history extraction)
///
/// _Requirements: 9.5_
pub fn run_timeseries_workload(operations: usize) -> MixedWorkloadResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.db");

    let db_config = DbConfig {
        enable_compression: false,
        enable_delta: true, // Enable delta for time-series
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, db_config)
        .expect("Failed to create database");

    // Simulate 10 IoT sensors
    let num_sensors = 10;
    let mut sensor_counters = vec![0usize; num_sensors];

    // Pre-populate with some initial readings
    for sensor_id in 0..num_sensors {
        for i in 0..100 {
            let key = format!("sensor/{}", sensor_id);
            let value = 20.0 + (i as f64 * 0.1).sin() * 5.0;
            db.append(&key, Atom::Float(value)).ok();
            sensor_counters[sensor_id] = i + 1;
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let sensor_id = rng.gen_range(0..num_sensors);
        let key = format!("sensor/{}", sensor_id);

        if roll < 0.90 {
            // Append operation (90%)
            let reading_num = sensor_counters[sensor_id];
            let value = 20.0 + (reading_num as f64 * 0.1).sin() * 5.0;
            let op_start = Instant::now();
            db.append(&key, Atom::Float(value)).ok();
            write_latencies.push(op_start.elapsed());
            sensor_counters[sensor_id] += 1;
        } else if roll < 0.95 {
            // Point read operation (5%)
            let op_start = Instant::now();
            db.get(&key).ok();
            read_latencies.push(op_start.elapsed());
        } else {
            // Range read / history extraction (5%)
            let op_start = Instant::now();
            db.get_history_floats(&key).ok();
            read_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    // Calculate read metrics
    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    // Calculate write metrics
    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    MixedWorkloadResult {
        workload: "timeseries".to_string(),
        database: "Syna".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}


// ============================================================================
// Competitor Mixed Workload Benchmarks
// ============================================================================

/// Run YCSB workload on SQLite
///
/// _Requirements: 9.6_
pub fn run_sqlite_ycsb(workload: &YcsbWorkload, operations: usize) -> MixedWorkloadResult {
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

    // Pre-populate with initial data
    let initial_records = 10000;
    let value = generate_test_value(1024, 42);

    for i in 0..initial_records {
        let key = format!("record/{}", i);
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        ).ok();
    }

    // Prepare statements
    let mut read_stmt = conn.prepare("SELECT value FROM kv WHERE key = ?1")
        .expect("Failed to prepare read statement");

    // Run workload
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();
    let mut insert_counter = initial_records;

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let cumulative_read = workload.read_proportion;
        let cumulative_update = cumulative_read + workload.update_proportion;
        let cumulative_insert = cumulative_update + workload.insert_proportion;

        if roll < cumulative_read {
            // Read operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            let _: Option<Vec<u8>> = read_stmt.query_row([&key], |row| row.get(0)).ok();
            read_latencies.push(op_start.elapsed());
        } else if roll < cumulative_update {
            // Update operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, value],
            ).ok();
            write_latencies.push(op_start.elapsed());
        } else if roll < cumulative_insert {
            // Insert operation
            let key = format!("record/{}", insert_counter);
            insert_counter += 1;
            let op_start = Instant::now();
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, value],
            ).ok();
            write_latencies.push(op_start.elapsed());
        } else {
            // Read-modify-write operation
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            if let Ok(data) = read_stmt.query_row([&key], |row| row.get::<_, Vec<u8>>(0)) {
                let mut modified = data;
                if !modified.is_empty() {
                    modified[0] = modified[0].wrapping_add(1);
                }
                conn.execute(
                    "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
                    rusqlite::params![key, modified],
                ).ok();
            }
            write_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    // Calculate metrics
    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "sqlite".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}


/// Run YCSB workload on DuckDB (feature-gated)
///
/// _Requirements: 9.6_
#[cfg(feature = "duckdb")]
pub fn run_duckdb_ycsb(workload: &YcsbWorkload, operations: usize) -> MixedWorkloadResult {
    use duckdb::{Connection, params};

    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.duckdb");

    let conn = Connection::open(&db_path).expect("Failed to create DuckDB database");

    conn.execute(
        "CREATE TABLE kv (key VARCHAR PRIMARY KEY, value BLOB)",
        [],
    ).expect("Failed to create table");

    // Pre-populate with initial data
    let initial_records = 10000;
    let value = generate_test_value(1024, 42);

    for i in 0..initial_records {
        let key = format!("record/{}", i);
        conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?)",
            params![key, value.as_slice()],
        ).ok();
    }

    // Run workload
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();
    let mut insert_counter = initial_records;

    let mut read_stmt = conn.prepare("SELECT value FROM kv WHERE key = ?")
        .expect("Failed to prepare read statement");

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let cumulative_read = workload.read_proportion;
        let cumulative_update = cumulative_read + workload.update_proportion;
        let cumulative_insert = cumulative_update + workload.insert_proportion;

        if roll < cumulative_read {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            let _: Option<Vec<u8>> = read_stmt.query_row([&key], |row| row.get(0)).ok();
            read_latencies.push(op_start.elapsed());
        } else if roll < cumulative_update {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                params![key, value.as_slice()],
            ).ok();
            write_latencies.push(op_start.elapsed());
        } else if roll < cumulative_insert {
            let key = format!("record/{}", insert_counter);
            insert_counter += 1;
            let op_start = Instant::now();
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?)",
                params![key, value.as_slice()],
            ).ok();
            write_latencies.push(op_start.elapsed());
        } else {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            if let Ok(data) = read_stmt.query_row([&key], |row| row.get::<_, Vec<u8>>(0)) {
                let mut modified = data;
                if !modified.is_empty() {
                    modified[0] = modified[0].wrapping_add(1);
                }
                conn.execute(
                    "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                    params![key, modified.as_slice()],
                ).ok();
            }
            write_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "duckdb".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}

#[cfg(not(feature = "duckdb"))]
pub fn run_duckdb_ycsb(workload: &YcsbWorkload, _operations: usize) -> MixedWorkloadResult {
    eprintln!("DuckDB benchmarks require the 'duckdb' feature.");
    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "duckdb".to_string(),
        total_ops: 0,
        total_duration_secs: 0.0,
        total_throughput_ops_sec: 0.0,
        read_ops: 0,
        read_throughput_ops_sec: 0.0,
        read_latency_p50_us: 0.0,
        read_latency_p95_us: 0.0,
        read_latency_p99_us: 0.0,
        write_ops: 0,
        write_throughput_ops_sec: 0.0,
        write_latency_p50_us: 0.0,
        write_latency_p95_us: 0.0,
        write_latency_p99_us: 0.0,
        disk_mb: 0.0,
    }
}


/// Run YCSB workload on LevelDB (feature-gated)
///
/// _Requirements: 9.6_
#[cfg(feature = "leveldb")]
pub fn run_leveldb_ycsb(workload: &YcsbWorkload, operations: usize) -> MixedWorkloadResult {
    use rusty_leveldb::{DB, Options};

    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.leveldb");

    let mut opts = Options::default();
    opts.create_if_missing = true;

    let mut db = DB::open(db_path.to_str().unwrap(), opts)
        .expect("Failed to create LevelDB database");

    // Pre-populate with initial data
    let initial_records = 10000;
    let value = generate_test_value(1024, 42);

    for i in 0..initial_records {
        let key = format!("record/{}", i);
        db.put(key.as_bytes(), &value).ok();
    }

    // Run workload
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();
    let mut insert_counter = initial_records;

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let cumulative_read = workload.read_proportion;
        let cumulative_update = cumulative_read + workload.update_proportion;
        let cumulative_insert = cumulative_update + workload.insert_proportion;

        if roll < cumulative_read {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.get(key.as_bytes()).ok();
            read_latencies.push(op_start.elapsed());
        } else if roll < cumulative_update {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.put(key.as_bytes(), &value).ok();
            write_latencies.push(op_start.elapsed());
        } else if roll < cumulative_insert {
            let key = format!("record/{}", insert_counter);
            insert_counter += 1;
            let op_start = Instant::now();
            db.put(key.as_bytes(), &value).ok();
            write_latencies.push(op_start.elapsed());
        } else {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            if let Some(data) = db.get(key.as_bytes()) {
                let mut modified = data;
                if !modified.is_empty() {
                    modified[0] = modified[0].wrapping_add(1);
                }
                db.put(key.as_bytes(), &modified).ok();
            }
            write_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    // LevelDB uses a directory, calculate total size
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;

    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "leveldb".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}

#[cfg(not(feature = "leveldb"))]
pub fn run_leveldb_ycsb(workload: &YcsbWorkload, _operations: usize) -> MixedWorkloadResult {
    eprintln!("LevelDB benchmarks require the 'leveldb' feature.");
    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "leveldb".to_string(),
        total_ops: 0,
        total_duration_secs: 0.0,
        total_throughput_ops_sec: 0.0,
        read_ops: 0,
        read_throughput_ops_sec: 0.0,
        read_latency_p50_us: 0.0,
        read_latency_p95_us: 0.0,
        read_latency_p99_us: 0.0,
        write_ops: 0,
        write_throughput_ops_sec: 0.0,
        write_latency_p50_us: 0.0,
        write_latency_p95_us: 0.0,
        write_latency_p99_us: 0.0,
        disk_mb: 0.0,
    }
}


/// Run YCSB workload on RocksDB (feature-gated)
///
/// _Requirements: 9.6_
#[cfg(feature = "rocksdb")]
pub fn run_rocksdb_ycsb(workload: &YcsbWorkload, operations: usize) -> MixedWorkloadResult {
    use rocksdb::{DB, Options};

    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.rocksdb");

    let mut opts = Options::default();
    opts.create_if_missing(true);

    let db = DB::open(&opts, &db_path).expect("Failed to create RocksDB database");

    // Pre-populate with initial data
    let initial_records = 10000;
    let value = generate_test_value(1024, 42);

    for i in 0..initial_records {
        let key = format!("record/{}", i);
        db.put(key.as_bytes(), &value).ok();
    }

    // Run workload
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut read_latencies = Vec::new();
    let mut write_latencies = Vec::new();
    let mut insert_counter = initial_records;

    let start = Instant::now();

    for _ in 0..operations {
        let roll: f64 = rng.gen();
        let cumulative_read = workload.read_proportion;
        let cumulative_update = cumulative_read + workload.update_proportion;
        let cumulative_insert = cumulative_update + workload.insert_proportion;

        if roll < cumulative_read {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.get(key.as_bytes()).ok();
            read_latencies.push(op_start.elapsed());
        } else if roll < cumulative_update {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            db.put(key.as_bytes(), &value).ok();
            write_latencies.push(op_start.elapsed());
        } else if roll < cumulative_insert {
            let key = format!("record/{}", insert_counter);
            insert_counter += 1;
            let op_start = Instant::now();
            db.put(key.as_bytes(), &value).ok();
            write_latencies.push(op_start.elapsed());
        } else {
            let key = format!("record/{}", rng.gen_range(0..initial_records));
            let op_start = Instant::now();
            if let Ok(Some(data)) = db.get(key.as_bytes()) {
                let mut modified = data;
                if !modified.is_empty() {
                    modified[0] = modified[0].wrapping_add(1);
                }
                db.put(key.as_bytes(), &modified).ok();
            }
            write_latencies.push(op_start.elapsed());
        }
    }

    let total_duration = start.elapsed();
    let total_throughput = operations as f64 / total_duration.as_secs_f64();

    let read_ops = read_latencies.len();
    let read_duration: Duration = read_latencies.iter().sum();
    let read_throughput = if read_duration.as_secs_f64() > 0.0 {
        read_ops as f64 / read_duration.as_secs_f64()
    } else {
        0.0
    };
    let (read_p50, read_p95, read_p99) = calculate_percentiles(read_latencies);

    let write_ops = write_latencies.len();
    let write_duration: Duration = write_latencies.iter().sum();
    let write_throughput = if write_duration.as_secs_f64() > 0.0 {
        write_ops as f64 / write_duration.as_secs_f64()
    } else {
        0.0
    };
    let (write_p50, write_p95, write_p99) = calculate_percentiles(write_latencies);

    // RocksDB uses a directory, calculate total size
    let disk_mb = calculate_dir_size(&db_path) as f64 / 1024.0 / 1024.0;

    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "rocksdb".to_string(),
        total_ops: operations,
        total_duration_secs: total_duration.as_secs_f64(),
        total_throughput_ops_sec: total_throughput,
        read_ops,
        read_throughput_ops_sec: read_throughput,
        read_latency_p50_us: read_p50,
        read_latency_p95_us: read_p95,
        read_latency_p99_us: read_p99,
        write_ops,
        write_throughput_ops_sec: write_throughput,
        write_latency_p50_us: write_p50,
        write_latency_p95_us: write_p95,
        write_latency_p99_us: write_p99,
        disk_mb,
    }
}

#[cfg(not(feature = "rocksdb"))]
pub fn run_rocksdb_ycsb(workload: &YcsbWorkload, _operations: usize) -> MixedWorkloadResult {
    eprintln!("RocksDB benchmarks require the 'rocksdb' feature.");
    MixedWorkloadResult {
        workload: workload.name.to_string(),
        database: "rocksdb".to_string(),
        total_ops: 0,
        total_duration_secs: 0.0,
        total_throughput_ops_sec: 0.0,
        read_ops: 0,
        read_throughput_ops_sec: 0.0,
        read_latency_p50_us: 0.0,
        read_latency_p95_us: 0.0,
        read_latency_p99_us: 0.0,
        write_ops: 0,
        write_throughput_ops_sec: 0.0,
        write_latency_p50_us: 0.0,
        write_latency_p95_us: 0.0,
        write_latency_p99_us: 0.0,
        disk_mb: 0.0,
    }
}


// ============================================================================
// Utility Functions
// ============================================================================

/// Generate deterministic test value
fn generate_test_value(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bytes = vec![0u8; size];
    rng.fill(&mut bytes[..]);
    bytes
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

/// Print mixed workload result with detailed metrics
pub fn print_mixed_result(result: &MixedWorkloadResult) {
    println!("\n=== {} - YCSB-{} ===", result.database, result.workload);
    println!("Total operations: {}", result.total_ops);
    println!("Total duration: {:.2} s", result.total_duration_secs);
    println!("Total throughput: {:.0} ops/sec", result.total_throughput_ops_sec);
    println!();
    println!("Read operations: {} ({:.1}%)", 
        result.read_ops, 
        result.read_ops as f64 / result.total_ops as f64 * 100.0
    );
    println!("  Throughput: {:.0} ops/sec", result.read_throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", result.read_latency_p50_us);
    println!("  Latency p95: {:.1} μs", result.read_latency_p95_us);
    println!("  Latency p99: {:.1} μs", result.read_latency_p99_us);
    println!();
    println!("Write operations: {} ({:.1}%)", 
        result.write_ops, 
        result.write_ops as f64 / result.total_ops as f64 * 100.0
    );
    println!("  Throughput: {:.0} ops/sec", result.write_throughput_ops_sec);
    println!("  Latency p50: {:.1} μs", result.write_latency_p50_us);
    println!("  Latency p95: {:.1} μs", result.write_latency_p95_us);
    println!("  Latency p99: {:.1} μs", result.write_latency_p99_us);
    println!();
    println!("Disk usage: {:.2} MB", result.disk_mb);
}

/// Run all YCSB workloads on Syna
pub fn run_all_SYNA_ycsb(operations: usize) -> Vec<MixedWorkloadResult> {
    let mut results = Vec::new();

    for workload_name in ["A", "B", "C", "F"] {
        println!("Running Syna YCSB-{} benchmark...", workload_name);
        let result = run_ycsb_detailed(workload_name, operations);
        print_mixed_result(&result);
        results.push(result);
    }

    // Time-series workload
    println!("Running Syna time-series benchmark...");
    let result = run_timeseries_workload(operations);
    print_mixed_result(&result);
    results.push(result);

    results
}

/// Run comparative YCSB benchmarks across all databases
///
/// _Requirements: 9.6_
pub fn run_comparative_ycsb(workload_name: &str, operations: usize) -> Vec<MixedWorkloadResult> {
    let workload = match workload_name.to_uppercase().as_str() {
        "A" => YcsbWorkload::a(),
        "B" => YcsbWorkload::b(),
        "C" => YcsbWorkload::c(),
        "D" => YcsbWorkload::d(),
        "F" => YcsbWorkload::f(),
        _ => YcsbWorkload::a(),
    };

    let mut results = Vec::new();

    // Syna
    println!("Running Syna YCSB-{} benchmark...", workload_name);
    let result = run_SYNA_ycsb(&workload, operations);
    print_mixed_result(&result);
    results.push(result);

    // SQLite
    println!("Running SQLite YCSB-{} benchmark...", workload_name);
    let result = run_sqlite_ycsb(&workload, operations);
    print_mixed_result(&result);
    results.push(result);

    // DuckDB (if available)
    println!("Running DuckDB YCSB-{} benchmark...", workload_name);
    let result = run_duckdb_ycsb(&workload, operations);
    if result.total_ops > 0 {
        print_mixed_result(&result);
    }
    results.push(result);

    // LevelDB (if available)
    println!("Running LevelDB YCSB-{} benchmark...", workload_name);
    let result = run_leveldb_ycsb(&workload, operations);
    if result.total_ops > 0 {
        print_mixed_result(&result);
    }
    results.push(result);

    // RocksDB (if available)
    println!("Running RocksDB YCSB-{} benchmark...", workload_name);
    let result = run_rocksdb_ycsb(&workload, operations);
    if result.total_ops > 0 {
        print_mixed_result(&result);
    }
    results.push(result);

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycsb_a_workload() {
        let result = run_ycsb_detailed("A", 1000);
        
        assert_eq!(result.workload, "A");
        assert_eq!(result.database, "Syna");
        assert_eq!(result.total_ops, 1000);
        assert!(result.total_throughput_ops_sec > 0.0);
        // YCSB-A is 50/50 read/update
        let read_ratio = result.read_ops as f64 / result.total_ops as f64;
        assert!(read_ratio > 0.4 && read_ratio < 0.6);
    }

    #[test]
    fn test_ycsb_b_workload() {
        let result = run_ycsb_detailed("B", 1000);
        
        assert_eq!(result.workload, "B");
        // YCSB-B is 95/5 read/update
        let read_ratio = result.read_ops as f64 / result.total_ops as f64;
        assert!(read_ratio > 0.9);
    }

    #[test]
    fn test_ycsb_c_workload() {
        let result = run_ycsb_detailed("C", 1000);
        
        assert_eq!(result.workload, "C");
        // YCSB-C is 100% read
        assert_eq!(result.read_ops, result.total_ops);
        assert_eq!(result.write_ops, 0);
    }

    #[test]
    fn test_ycsb_f_workload() {
        let result = run_ycsb_detailed("F", 1000);
        
        assert_eq!(result.workload, "F");
        // YCSB-F is 50% read, 50% read-modify-write
        let read_ratio = result.read_ops as f64 / result.total_ops as f64;
        assert!(read_ratio > 0.4 && read_ratio < 0.6);
    }

    #[test]
    fn test_timeseries_workload() {
        let result = run_timeseries_workload(1000);
        
        assert_eq!(result.workload, "timeseries");
        assert_eq!(result.database, "Syna");
        // Time-series is 90% append, 10% read
        let write_ratio = result.write_ops as f64 / result.total_ops as f64;
        assert!(write_ratio > 0.85);
    }

    #[test]
    fn test_sqlite_ycsb() {
        let workload = YcsbWorkload::a();
        let result = run_sqlite_ycsb(&workload, 500);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.total_throughput_ops_sec > 0.0);
    }
}



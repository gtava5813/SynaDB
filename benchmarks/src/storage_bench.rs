//! Storage efficiency benchmarks.
//!
//! This module implements storage efficiency benchmarks for Syna and competitor databases.
//! It measures bytes per entry, compression ratios, and compaction efficiency.
//!
//! _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

use crate::{BenchmarkConfig, BenchmarkResult};
use synadb::{Atom, DbConfig, SynaDB};
use std::time::Instant;
use tempfile::tempdir;

/// Standard value sizes for storage benchmarks (in bytes)
pub const VALUE_SIZES: &[usize] = &[64, 256, 1024, 4096];

/// Result of a storage efficiency benchmark
#[derive(Debug, Clone)]
pub struct StorageResult {
    pub database: String,
    pub config_name: String,
    pub entries: usize,
    pub value_size: usize,
    pub disk_bytes: u64,
    pub bytes_per_entry: f64,
    pub compression_ratio: f64,
    pub duration_secs: f64,
}

impl StorageResult {
    pub fn print(&self) {
        println!("  {}/{}: {:.2} MB ({:.1} bytes/entry, {:.2}x compression)",
            self.database,
            self.config_name,
            self.disk_bytes as f64 / 1024.0 / 1024.0,
            self.bytes_per_entry,
            self.compression_ratio
        );
    }
}

/// Run storage efficiency benchmark for Syna
///
/// Measures bytes stored per logical entry for different value sizes.
///
/// _Requirements: 10.1_
pub fn run_storage_bench(entries: usize) -> BenchmarkResult {
    println!("\n=== Storage Efficiency Benchmark ===");
    println!("Entries: {}", entries);
    
    let results = run_SYNA_storage_all_configs(entries, 64);
    
    // Return result for no_compression as baseline
    let baseline = &results[0];
    
    BenchmarkResult {
        benchmark: "storage_efficiency".to_string(),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            measurement_iterations: entries,
            value_size_bytes: 64,
            ..Default::default()
        },
        throughput_ops_sec: 0.0,
        latency_p50_us: 0.0,
        latency_p95_us: 0.0,
        latency_p99_us: 0.0,
        memory_mb: 0.0,
        disk_mb: baseline.disk_bytes as f64 / 1024.0 / 1024.0,
        duration_secs: baseline.duration_secs,
    }
}

/// Run Syna storage benchmark with all compression configurations
///
/// Tests uncompressed, LZ4 only, delta only, and both compression modes.
///
/// _Requirements: 10.1, 10.2_
pub fn run_SYNA_storage_all_configs(entries: usize, value_size: usize) -> Vec<StorageResult> {
    let dir = tempdir().expect("Failed to create temp dir");
    
    let configs = [
        ("no_compression", DbConfig {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: false,
        }),
        ("lz4_only", DbConfig {
            enable_compression: true,
            enable_delta: false,
            sync_on_write: false,
        }),
        ("delta_only", DbConfig {
            enable_compression: false,
            enable_delta: true,
            sync_on_write: false,
        }),
        ("both", DbConfig {
            enable_compression: true,
            enable_delta: true,
            sync_on_write: false,
        }),
    ];
    
    let mut results = Vec::new();
    let mut baseline_bytes = 0u64;
    
    for (name, config) in configs {
        let db_path = dir.path().join(format!("{}.db", name));
        let start = Instant::now();
        
        let mut db = SynaDB::with_config(&db_path, config)
            .expect("Failed to create database");
        
        // Write time-series data (simulating sensor readings)
        for i in 0..entries {
            let key = format!("sensor/{}", i % 100); // 100 different sensors
            // Simulate gradual changes (good for delta compression)
            let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
            db.append(&key, Atom::Float(value)).ok();
        }
        
        drop(db);
        
        let duration = start.elapsed();
        let disk_bytes = std::fs::metadata(&db_path)
            .map(|m| m.len())
            .unwrap_or(0);
        
        if name == "no_compression" {
            baseline_bytes = disk_bytes;
        }
        
        let bytes_per_entry = disk_bytes as f64 / entries as f64;
        let compression_ratio = if disk_bytes > 0 {
            baseline_bytes as f64 / disk_bytes as f64
        } else {
            1.0
        };
        
        let result = StorageResult {
            database: "Syna".to_string(),
            config_name: name.to_string(),
            entries,
            value_size,
            disk_bytes,
            bytes_per_entry,
            compression_ratio,
            duration_secs: duration.as_secs_f64(),
        };
        
        result.print();
        results.push(result);
    }
    
    results
}

/// Run storage benchmark with different value sizes
///
/// Tests storage efficiency across various value sizes.
///
/// _Requirements: 10.1_
pub fn run_storage_by_value_size(entries: usize) -> Vec<StorageResult> {
    println!("\n=== Storage by Value Size ===");
    
    let dir = tempdir().expect("Failed to create temp dir");
    let mut results = Vec::new();
    
    for &size in VALUE_SIZES {
        let db_path = dir.path().join(format!("size_{}.db", size));
        let start = Instant::now();
        
        let config = DbConfig {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: false,
        };
        
        let mut db = SynaDB::with_config(&db_path, config)
            .expect("Failed to create database");
        
        // Generate test data
        let value = generate_test_bytes(size, 42);
        
        for i in 0..entries {
            let key = format!("key/{}", i);
            db.append(&key, Atom::Bytes(value.clone())).ok();
        }
        
        drop(db);
        
        let duration = start.elapsed();
        let disk_bytes = std::fs::metadata(&db_path)
            .map(|m| m.len())
            .unwrap_or(0);
        
        // Calculate theoretical minimum (key + value + header overhead)
        let avg_key_len = format!("key/{}", entries / 2).len();
        let theoretical_min = (avg_key_len + size + 15) * entries; // 15 bytes for LogHeader
        
        let result = StorageResult {
            database: "Syna".to_string(),
            config_name: format!("{}B_values", size),
            entries,
            value_size: size,
            disk_bytes,
            bytes_per_entry: disk_bytes as f64 / entries as f64,
            compression_ratio: theoretical_min as f64 / disk_bytes as f64,
            duration_secs: duration.as_secs_f64(),
        };
        
        println!("  {}B values: {:.2} MB ({:.1} bytes/entry)",
            size,
            disk_bytes as f64 / 1024.0 / 1024.0,
            result.bytes_per_entry
        );
        
        results.push(result);
    }
    
    results
}


// ============================================================================
// Compression Comparison Benchmarks
// ============================================================================

/// Result of a compression comparison benchmark
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub config_name: String,
    pub entries: usize,
    pub uncompressed_bytes: u64,
    pub compressed_bytes: u64,
    pub compression_ratio: f64,
    pub write_throughput: f64,
    pub data_pattern: String,
}

impl CompressionResult {
    pub fn print(&self) {
        println!("  {}: {:.2} MB -> {:.2} MB ({:.2}x ratio, {:.0} ops/sec)",
            self.config_name,
            self.uncompressed_bytes as f64 / 1024.0 / 1024.0,
            self.compressed_bytes as f64 / 1024.0 / 1024.0,
            self.compression_ratio,
            self.write_throughput
        );
    }
}

/// Run compression comparison benchmark
///
/// Compares uncompressed, LZ4, delta, and combined compression modes
/// with time-series data patterns.
///
/// _Requirements: 10.2, 10.5, 10.6_
pub fn run_compression_comparison(entries: usize) -> Vec<CompressionResult> {
    println!("\n=== Compression Comparison ===");
    println!("Entries: {}", entries);
    
    let mut results = Vec::new();
    
    // Test with different data patterns
    let patterns = [
        ("time_series_gradual", DataPattern::TimeSeriesGradual),
        ("time_series_spiky", DataPattern::TimeSeriesSpiky),
        ("random_floats", DataPattern::RandomFloats),
        ("text_repetitive", DataPattern::TextRepetitive),
        ("binary_random", DataPattern::BinaryRandom),
    ];
    
    for (pattern_name, pattern) in patterns {
        println!("\nPattern: {}", pattern_name);
        let pattern_results = run_compression_for_pattern(entries, pattern, pattern_name);
        results.extend(pattern_results);
    }
    
    results
}

/// Data patterns for compression testing
#[derive(Clone, Copy)]
pub enum DataPattern {
    /// Gradual changes (good for delta compression)
    TimeSeriesGradual,
    /// Spiky changes (less compressible)
    TimeSeriesSpiky,
    /// Random float values
    RandomFloats,
    /// Repetitive text (good for LZ4)
    TextRepetitive,
    /// Random binary data (hard to compress)
    BinaryRandom,
}

/// Run compression benchmark for a specific data pattern
fn run_compression_for_pattern(entries: usize, pattern: DataPattern, pattern_name: &str) -> Vec<CompressionResult> {
    let dir = tempdir().expect("Failed to create temp dir");
    let mut results = Vec::new();
    
    let configs = [
        ("no_compression", false, false),
        ("lz4_only", true, false),
        ("delta_only", false, true),
        ("both", true, true),
    ];
    
    let mut uncompressed_size = 0u64;
    
    for (name, enable_compression, enable_delta) in configs {
        let db_path = dir.path().join(format!("{}_{}.db", pattern_name, name));
        let start = Instant::now();
        
        let config = DbConfig {
            enable_compression,
            enable_delta,
            sync_on_write: false,
        };
        
        let mut db = SynaDB::with_config(&db_path, config)
            .expect("Failed to create database");
        
        // Write data based on pattern
        write_pattern_data(&mut db, entries, pattern);
        
        drop(db);
        
        let duration = start.elapsed();
        let disk_bytes = std::fs::metadata(&db_path)
            .map(|m| m.len())
            .unwrap_or(0);
        
        if name == "no_compression" {
            uncompressed_size = disk_bytes;
        }
        
        let compression_ratio = if disk_bytes > 0 {
            uncompressed_size as f64 / disk_bytes as f64
        } else {
            1.0
        };
        
        let result = CompressionResult {
            config_name: name.to_string(),
            entries,
            uncompressed_bytes: uncompressed_size,
            compressed_bytes: disk_bytes,
            compression_ratio,
            write_throughput: entries as f64 / duration.as_secs_f64(),
            data_pattern: pattern_name.to_string(),
        };
        
        result.print();
        results.push(result);
    }
    
    results
}

/// Write data to database based on pattern
fn write_pattern_data(db: &mut SynaDB, entries: usize, pattern: DataPattern) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    match pattern {
        DataPattern::TimeSeriesGradual => {
            // Gradual changes - excellent for delta compression
            for i in 0..entries {
                let key = format!("sensor/{}", i % 100);
                let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
                db.append(&key, Atom::Float(value)).ok();
            }
        }
        DataPattern::TimeSeriesSpiky => {
            // Spiky changes - less compressible
            for i in 0..entries {
                let key = format!("sensor/{}", i % 100);
                let base = 20.0 + (i as f64 * 0.01).sin() * 5.0;
                let spike = if rng.gen_bool(0.1) { rng.gen_range(-50.0..50.0) } else { 0.0 };
                db.append(&key, Atom::Float(base + spike)).ok();
            }
        }
        DataPattern::RandomFloats => {
            // Random floats - hard to compress
            for i in 0..entries {
                let key = format!("data/{}", i);
                let value: f64 = rng.gen_range(-1000.0..1000.0);
                db.append(&key, Atom::Float(value)).ok();
            }
        }
        DataPattern::TextRepetitive => {
            // Repetitive text - good for LZ4
            let templates = [
                "Temperature reading at location A: normal",
                "Temperature reading at location B: normal",
                "Humidity sensor status: OK",
                "Pressure sensor status: OK",
            ];
            for i in 0..entries {
                let key = format!("log/{}", i);
                let text = format!("{} - timestamp {}", templates[i % templates.len()], i);
                db.append(&key, Atom::Text(text)).ok();
            }
        }
        DataPattern::BinaryRandom => {
            // Random binary - hard to compress
            for i in 0..entries {
                let key = format!("blob/{}", i);
                let mut bytes = vec![0u8; 64];
                rng.fill(&mut bytes[..]);
                db.append(&key, Atom::Bytes(bytes)).ok();
            }
        }
    }
}


// ============================================================================
// Compaction Benchmarks
// ============================================================================

/// Result of a compaction benchmark
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub entries_written: usize,
    pub entries_deleted: usize,
    pub size_before_compaction: u64,
    pub size_after_compaction: u64,
    pub space_reclaimed: u64,
    pub reclamation_ratio: f64,
    pub compaction_duration_secs: f64,
}

impl CompactionResult {
    pub fn print(&self) {
        println!("  Before: {:.2} MB", self.size_before_compaction as f64 / 1024.0 / 1024.0);
        println!("  After:  {:.2} MB", self.size_after_compaction as f64 / 1024.0 / 1024.0);
        println!("  Reclaimed: {:.2} MB ({:.1}%)",
            self.space_reclaimed as f64 / 1024.0 / 1024.0,
            self.reclamation_ratio * 100.0
        );
        println!("  Compaction time: {:.3} s", self.compaction_duration_secs);
    }
}

/// Run compaction benchmark
///
/// Measures space reclamation after deletes and times the compaction operation.
///
/// _Requirements: 10.3_
pub fn run_compaction_benchmark(entries: usize, delete_ratio: f64) -> CompactionResult {
    println!("\n=== Compaction Benchmark ===");
    println!("Entries: {}, Delete ratio: {:.0}%", entries, delete_ratio * 100.0);
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("compaction_test.db");
    
    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };
    
    let mut db = SynaDB::with_config(&db_path, config.clone())
        .expect("Failed to create database");
    
    // Write initial data
    for i in 0..entries {
        let key = format!("key/{}", i);
        let value = format!("value_{}_with_some_padding_to_make_it_larger", i);
        db.append(&key, Atom::Text(value)).ok();
    }
    
    // Delete a portion of entries
    let entries_to_delete = (entries as f64 * delete_ratio) as usize;
    for i in 0..entries_to_delete {
        let key = format!("key/{}", i);
        db.delete(&key).ok();
    }
    
    drop(db);
    
    let size_before = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);
    
    // Reopen and compact
    let mut db = SynaDB::with_config(&db_path, config.clone())
        .expect("Failed to reopen database");
    
    let compact_start = Instant::now();
    db.compact().expect("Compaction failed");
    let compact_duration = compact_start.elapsed();
    
    drop(db);
    
    let size_after = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);
    
    let space_reclaimed = size_before.saturating_sub(size_after);
    let reclamation_ratio = if size_before > 0 {
        space_reclaimed as f64 / size_before as f64
    } else {
        0.0
    };
    
    let result = CompactionResult {
        entries_written: entries,
        entries_deleted: entries_to_delete,
        size_before_compaction: size_before,
        size_after_compaction: size_after,
        space_reclaimed,
        reclamation_ratio,
        compaction_duration_secs: compact_duration.as_secs_f64(),
    };
    
    result.print();
    result
}

/// Run compaction benchmark with various delete ratios
pub fn run_compaction_sweep(entries: usize) -> Vec<CompactionResult> {
    let ratios = [0.1, 0.25, 0.5, 0.75, 0.9];
    let mut results = Vec::new();
    
    for ratio in ratios {
        let result = run_compaction_benchmark(entries, ratio);
        results.push(result);
    }
    
    results
}


// ============================================================================
// Competitor Storage Benchmarks
// ============================================================================

/// Run SQLite storage benchmark
///
/// Measures storage efficiency for SQLite with equivalent data.
///
/// _Requirements: 10.4_
pub fn run_sqlite_storage_bench(entries: usize) -> StorageResult {
    use rusqlite::Connection;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.sqlite");
    
    let start = Instant::now();
    let conn = Connection::open(&db_path).expect("Failed to create SQLite database");
    
    // Configure for fair comparison
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    conn.execute_batch("PRAGMA synchronous = OFF;").ok();
    
    conn.execute(
        "CREATE TABLE readings (key TEXT, value REAL, timestamp INTEGER)",
        [],
    ).expect("Failed to create table");
    
    // Write same data pattern as Syna
    for i in 0..entries {
        let key = format!("sensor/{}", i % 100);
        let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
        conn.execute(
            "INSERT INTO readings (key, value, timestamp) VALUES (?1, ?2, ?3)",
            rusqlite::params![key, value, i as i64],
        ).ok();
    }
    
    drop(conn);
    
    let duration = start.elapsed();
    
    // SQLite may have WAL file, count total
    let disk_bytes = calculate_sqlite_size(&db_path);
    
    StorageResult {
        database: "sqlite".to_string(),
        config_name: "default".to_string(),
        entries,
        value_size: 8, // f64
        disk_bytes,
        bytes_per_entry: disk_bytes as f64 / entries as f64,
        compression_ratio: 1.0,
        duration_secs: duration.as_secs_f64(),
    }
}

/// Calculate total SQLite database size including WAL
fn calculate_sqlite_size(db_path: &std::path::Path) -> u64 {
    let mut total = std::fs::metadata(db_path)
        .map(|m| m.len())
        .unwrap_or(0);
    
    // Check for WAL file
    let wal_path = db_path.with_extension("sqlite-wal");
    if let Ok(meta) = std::fs::metadata(&wal_path) {
        total += meta.len();
    }
    
    // Check for SHM file
    let shm_path = db_path.with_extension("sqlite-shm");
    if let Ok(meta) = std::fs::metadata(&shm_path) {
        total += meta.len();
    }
    
    total
}

/// Run DuckDB storage benchmark (if feature enabled)
///
/// _Requirements: 10.4_
#[cfg(feature = "duckdb")]
pub fn run_duckdb_storage_bench(entries: usize) -> StorageResult {
    use duckdb::{Connection, params};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.duckdb");
    
    let start = Instant::now();
    let conn = Connection::open(&db_path).expect("Failed to create DuckDB database");
    
    conn.execute(
        "CREATE TABLE readings (key VARCHAR, value DOUBLE, timestamp BIGINT)",
        [],
    ).expect("Failed to create table");
    
    for i in 0..entries {
        let key = format!("sensor/{}", i % 100);
        let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
        conn.execute(
            "INSERT INTO readings (key, value, timestamp) VALUES (?, ?, ?)",
            params![key, value, i as i64],
        ).ok();
    }
    
    drop(conn);
    
    let duration = start.elapsed();
    let disk_bytes = std::fs::metadata(&db_path)
        .map(|m| m.len())
        .unwrap_or(0);
    
    StorageResult {
        database: "duckdb".to_string(),
        config_name: "default".to_string(),
        entries,
        value_size: 8,
        disk_bytes,
        bytes_per_entry: disk_bytes as f64 / entries as f64,
        compression_ratio: 1.0,
        duration_secs: duration.as_secs_f64(),
    }
}

#[cfg(not(feature = "duckdb"))]
pub fn run_duckdb_storage_bench(entries: usize) -> StorageResult {
    eprintln!("DuckDB storage benchmark requires 'duckdb' feature");
    StorageResult {
        database: "duckdb".to_string(),
        config_name: "unavailable".to_string(),
        entries,
        value_size: 0,
        disk_bytes: 0,
        bytes_per_entry: 0.0,
        compression_ratio: 0.0,
        duration_secs: 0.0,
    }
}

/// Run LevelDB storage benchmark (if feature enabled)
///
/// _Requirements: 10.4_
#[cfg(feature = "leveldb")]
pub fn run_leveldb_storage_bench(entries: usize) -> StorageResult {
    use rusty_leveldb::{DB, Options};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.leveldb");
    
    let start = Instant::now();
    
    let mut opts = Options::default();
    opts.create_if_missing = true;
    
    let mut db = DB::open(db_path.to_str().unwrap(), opts)
        .expect("Failed to create LevelDB database");
    
    for i in 0..entries {
        let key = format!("sensor/{}", i % 100);
        let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
        db.put(key.as_bytes(), &value.to_le_bytes()).ok();
    }
    
    drop(db);
    
    let duration = start.elapsed();
    let disk_bytes = calculate_dir_size(&db_path);
    
    StorageResult {
        database: "leveldb".to_string(),
        config_name: "default".to_string(),
        entries,
        value_size: 8,
        disk_bytes,
        bytes_per_entry: disk_bytes as f64 / entries as f64,
        compression_ratio: 1.0,
        duration_secs: duration.as_secs_f64(),
    }
}

#[cfg(not(feature = "leveldb"))]
pub fn run_leveldb_storage_bench(entries: usize) -> StorageResult {
    eprintln!("LevelDB storage benchmark requires 'leveldb' feature");
    StorageResult {
        database: "leveldb".to_string(),
        config_name: "unavailable".to_string(),
        entries,
        value_size: 0,
        disk_bytes: 0,
        bytes_per_entry: 0.0,
        compression_ratio: 0.0,
        duration_secs: 0.0,
    }
}

/// Run RocksDB storage benchmark (if feature enabled)
///
/// _Requirements: 10.4_
#[cfg(feature = "rocksdb")]
pub fn run_rocksdb_storage_bench(entries: usize) -> StorageResult {
    use rocksdb::{DB, Options};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("bench.rocksdb");
    
    let start = Instant::now();
    
    let mut opts = Options::default();
    opts.create_if_missing(true);
    
    let db = DB::open(&opts, &db_path).expect("Failed to create RocksDB database");
    
    for i in 0..entries {
        let key = format!("sensor/{}", i % 100);
        let value = 20.0 + (i as f64 * 0.01).sin() * 5.0;
        db.put(key.as_bytes(), &value.to_le_bytes()).ok();
    }
    
    drop(db);
    
    let duration = start.elapsed();
    let disk_bytes = calculate_dir_size(&db_path);
    
    StorageResult {
        database: "rocksdb".to_string(),
        config_name: "default".to_string(),
        entries,
        value_size: 8,
        disk_bytes,
        bytes_per_entry: disk_bytes as f64 / entries as f64,
        compression_ratio: 1.0,
        duration_secs: duration.as_secs_f64(),
    }
}

#[cfg(not(feature = "rocksdb"))]
pub fn run_rocksdb_storage_bench(entries: usize) -> StorageResult {
    eprintln!("RocksDB storage benchmark requires 'rocksdb' feature");
    StorageResult {
        database: "rocksdb".to_string(),
        config_name: "unavailable".to_string(),
        entries,
        value_size: 0,
        disk_bytes: 0,
        bytes_per_entry: 0.0,
        compression_ratio: 0.0,
        duration_secs: 0.0,
    }
}

/// Calculate total size of a directory (for LevelDB/RocksDB)
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


// ============================================================================
// Comparative Storage Benchmarks
// ============================================================================

/// Run comparative storage benchmarks across all databases
///
/// _Requirements: 10.4_
pub fn run_comparative_storage(entries: usize) -> Vec<StorageResult> {
    println!("\n=== Comparative Storage Benchmark ===");
    println!("Entries: {}", entries);
    
    let mut results = Vec::new();
    
    // Syna (all configs)
    println!("\nSyna:");
    let SYNA_results = run_SYNA_storage_all_configs(entries, 8);
    results.extend(SYNA_results);
    
    // SQLite
    println!("\nSQLite:");
    let sqlite_result = run_sqlite_storage_bench(entries);
    sqlite_result.print();
    results.push(sqlite_result);
    
    // DuckDB (if available)
    if cfg!(feature = "duckdb") {
        println!("\nDuckDB:");
        let duckdb_result = run_duckdb_storage_bench(entries);
        duckdb_result.print();
        results.push(duckdb_result);
    }
    
    // LevelDB (if available)
    if cfg!(feature = "leveldb") {
        println!("\nLevelDB:");
        let leveldb_result = run_leveldb_storage_bench(entries);
        leveldb_result.print();
        results.push(leveldb_result);
    }
    
    // RocksDB (if available)
    if cfg!(feature = "rocksdb") {
        println!("\nRocksDB:");
        let rocksdb_result = run_rocksdb_storage_bench(entries);
        rocksdb_result.print();
        results.push(rocksdb_result);
    }
    
    results
}

/// Print storage comparison summary
pub fn print_storage_summary(results: &[StorageResult]) {
    println!("\n=== Storage Summary ===");
    println!("{:<20} {:<15} {:>12} {:>15}",
        "Database", "Config", "Size (MB)", "Bytes/Entry");
    println!("{}", "-".repeat(65));
    
    for result in results {
        if result.disk_bytes > 0 {
            println!("{:<20} {:<15} {:>12.2} {:>15.1}",
                result.database,
                result.config_name,
                result.disk_bytes as f64 / 1024.0 / 1024.0,
                result.bytes_per_entry
            );
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate deterministic test bytes
fn generate_test_bytes(size: usize, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bytes = vec![0u8; size];
    rng.fill(&mut bytes[..]);
    bytes
}

/// Run full storage benchmark suite
pub fn run_full_storage_suite(entries: usize) {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Syna STORAGE BENCHMARK SUITE                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    // 1. Storage by value size
    let size_results = run_storage_by_value_size(entries);
    
    // 2. Compression comparison
    let compression_results = run_compression_comparison(entries);
    
    // 3. Compaction benchmark
    let compaction_result = run_compaction_benchmark(entries, 0.5);
    
    // 4. Comparative storage
    let comparative_results = run_comparative_storage(entries);
    
    // Print summary
    print_storage_summary(&comparative_results);
    
    println!("\n=== Compression Effectiveness ===");
    for result in &compression_results {
        if result.config_name == "both" {
            println!("  {}: {:.2}x compression ratio",
                result.data_pattern,
                result.compression_ratio
            );
        }
    }
    
    println!("\n=== Compaction Effectiveness ===");
    println!("  Space reclaimed: {:.1}%", compaction_result.reclamation_ratio * 100.0);
    println!("  Compaction time: {:.3}s", compaction_result.compaction_duration_secs);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_storage_bench() {
        let result = run_storage_bench(100);
        assert_eq!(result.database, "Syna");
        assert!(result.disk_mb > 0.0);
    }
    
    #[test]
    fn test_storage_by_value_size() {
        let results = run_storage_by_value_size(100);
        assert!(!results.is_empty());
        
        // Larger values should result in larger storage
        for i in 1..results.len() {
            assert!(results[i].bytes_per_entry >= results[i-1].bytes_per_entry * 0.5);
        }
    }
    
    #[test]
    fn test_compression_comparison() {
        let results = run_compression_comparison(100);
        assert!(!results.is_empty());
        
        // Find time_series_gradual results
        let gradual_results: Vec<_> = results.iter()
            .filter(|r| r.data_pattern == "time_series_gradual")
            .collect();
        
        // Delta compression should help with gradual time series
        let no_comp = gradual_results.iter().find(|r| r.config_name == "no_compression");
        let delta = gradual_results.iter().find(|r| r.config_name == "delta_only");
        
        if let (Some(nc), Some(d)) = (no_comp, delta) {
            // Delta should provide some compression for gradual data
            assert!(d.compression_ratio >= 0.9, "Delta compression should help with gradual data");
        }
    }
    
    #[test]
    fn test_compaction_benchmark() {
        let result = run_compaction_benchmark(100, 0.5);
        
        assert_eq!(result.entries_written, 100);
        assert_eq!(result.entries_deleted, 50);
        assert!(result.size_before_compaction > 0);
        // After compaction, size should be smaller or equal
        assert!(result.size_after_compaction <= result.size_before_compaction);
    }
    
    #[test]
    fn test_sqlite_storage_bench() {
        let result = run_sqlite_storage_bench(100);
        
        assert_eq!(result.database, "sqlite");
        assert!(result.disk_bytes > 0);
        assert!(result.bytes_per_entry > 0.0);
    }
    
    #[test]
    fn test_comparative_storage() {
        let results = run_comparative_storage(100);
        
        // Should have at least Syna and SQLite results
        assert!(results.len() >= 5); // 4 Syna configs + 1 SQLite
        
        // Verify Syna results exist
        let SYNA_count = results.iter()
            .filter(|r| r.database == "Syna")
            .count();
        assert_eq!(SYNA_count, 4);
        
        // Verify SQLite result exists
        let sqlite_count = results.iter()
            .filter(|r| r.database == "sqlite")
            .count();
        assert_eq!(sqlite_count, 1);
    }
}



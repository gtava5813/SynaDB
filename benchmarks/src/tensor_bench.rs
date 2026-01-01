//! Tensor engine performance benchmarks.
//!
//! This module implements benchmarks for batch tensor operations.
//! It tests:
//! - Batch tensor load throughput (target: 1 GB/s from SSD)
//! - Tensor put/get round-trip performance
//! - Pattern matching performance
//!
//! _Requirements: 9.3 (Performance Guarantees)_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use synadb::tensor::{DType, TensorEngine};
use synadb::SynaDB;
use std::time::Instant;
use tempfile::tempdir;

/// Generate deterministic tensor data
fn generate_tensor_data(num_elements: usize, dtype: DType, seed: u64) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let element_size = dtype.size();
    let mut data = vec![0u8; num_elements * element_size];
    
    match dtype {
        DType::Float64 => {
            for i in 0..num_elements {
                let value: f64 = rng.gen::<f64>() * 100.0;
                let bytes = value.to_le_bytes();
                data[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
            }
        }
        DType::Float32 => {
            for i in 0..num_elements {
                let value: f32 = rng.gen::<f32>() * 100.0;
                let bytes = value.to_le_bytes();
                data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
        }
        DType::Int64 => {
            for i in 0..num_elements {
                let value: i64 = rng.gen::<i64>() % 1_000_000;
                let bytes = value.to_le_bytes();
                data[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
            }
        }
        DType::Int32 => {
            for i in 0..num_elements {
                let value: i32 = rng.gen::<i32>() % 1_000_000;
                let bytes = value.to_le_bytes();
                data[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
        }
    }
    
    data
}

/// Configuration for tensor benchmarks
#[derive(Debug, Clone)]
pub struct TensorBenchConfig {
    /// Number of elements in the tensor
    pub num_elements: usize,
    /// Data type for tensor elements
    pub dtype: DType,
    /// Number of iterations for measurement
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for TensorBenchConfig {
    fn default() -> Self {
        Self {
            num_elements: 100_000,
            dtype: DType::Float64,
            iterations: 10,
            warmup_iterations: 2,
        }
    }
}

/// Run tensor put benchmark
///
/// Tests batch tensor write performance.
pub fn run_tensor_put_benchmark(config: &TensorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("tensor.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    // Generate tensor data
    let data = generate_tensor_data(config.num_elements, config.dtype, 42);
    let shape = vec![config.num_elements];
    let data_size_bytes = data.len();
    
    // Warmup
    for i in 0..config.warmup_iterations {
        let prefix = format!("warmup_{}/", i);
        engine.put_tensor(&prefix, &data, &shape, config.dtype).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.iterations);
    let start = Instant::now();
    
    for i in 0..config.iterations {
        let prefix = format!("tensor_{}/", i);
        let op_start = Instant::now();
        engine.put_tensor(&prefix, &data, &shape, config.dtype).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let total_bytes = data_size_bytes * config.iterations;
    let throughput_bytes_sec = total_bytes as f64 / total_duration.as_secs_f64();
    let throughput_mb_sec = throughput_bytes_sec / 1024.0 / 1024.0;
    let throughput_gb_sec = throughput_mb_sec / 1024.0;
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    // Get disk usage
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    println!("  Data throughput: {:.2} MB/s ({:.3} GB/s)", throughput_mb_sec, throughput_gb_sec);
    
    BenchmarkResult {
        benchmark: format!("tensor_put_{}_{}", config.dtype.name(), config.num_elements),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.iterations,
            value_size_bytes: data_size_bytes,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput_mb_sec, // Using MB/s as "ops/sec" for tensor benchmarks
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run tensor get benchmark
///
/// Tests batch tensor read performance.
/// Target: 1 GB/s throughput from SSD.
///
/// _Requirements: 9.3_
pub fn run_tensor_get_benchmark(config: &TensorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("tensor.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    // Generate and store tensor data
    let data = generate_tensor_data(config.num_elements, config.dtype, 42);
    let shape = vec![config.num_elements];
    let data_size_bytes = data.len();
    
    // Store the tensor
    engine.put_tensor("data/", &data, &shape, config.dtype)
        .expect("Failed to store tensor");
    
    // Warmup reads
    for _ in 0..config.warmup_iterations {
        engine.get_tensor("data/*", config.dtype).ok();
    }
    
    // Measurement
    let mut latencies = Vec::with_capacity(config.iterations);
    let start = Instant::now();
    
    for _ in 0..config.iterations {
        let op_start = Instant::now();
        let _result = engine.get_tensor("data/*", config.dtype);
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let total_bytes = data_size_bytes * config.iterations;
    let throughput_bytes_sec = total_bytes as f64 / total_duration.as_secs_f64();
    let throughput_mb_sec = throughput_bytes_sec / 1024.0 / 1024.0;
    let throughput_gb_sec = throughput_mb_sec / 1024.0;
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    println!("  Data throughput: {:.2} MB/s ({:.3} GB/s)", throughput_mb_sec, throughput_gb_sec);
    
    BenchmarkResult {
        benchmark: format!("tensor_get_{}_{}", config.dtype.name(), config.num_elements),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.iterations,
            value_size_bytes: data_size_bytes,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput_mb_sec, // Using MB/s as "ops/sec" for tensor benchmarks
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run tensor round-trip benchmark
///
/// Tests put + get round-trip performance.
pub fn run_tensor_roundtrip_benchmark(config: &TensorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("tensor.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    // Generate tensor data
    let data = generate_tensor_data(config.num_elements, config.dtype, 42);
    let shape = vec![config.num_elements];
    let data_size_bytes = data.len();
    
    // Measurement: put + get
    let mut latencies = Vec::with_capacity(config.iterations);
    let start = Instant::now();
    
    for i in 0..config.iterations {
        let prefix = format!("rt_{}/", i);
        let pattern = format!("rt_{}/*", i);
        
        let op_start = Instant::now();
        engine.put_tensor(&prefix, &data, &shape, config.dtype).ok();
        let _result = engine.get_tensor(&pattern, config.dtype);
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let total_bytes = data_size_bytes * config.iterations * 2; // put + get
    let throughput_bytes_sec = total_bytes as f64 / total_duration.as_secs_f64();
    let throughput_mb_sec = throughput_bytes_sec / 1024.0 / 1024.0;
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    println!("  Round-trip throughput: {:.2} MB/s", throughput_mb_sec);
    
    BenchmarkResult {
        benchmark: format!("tensor_roundtrip_{}_{}", config.dtype.name(), config.num_elements),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.iterations,
            value_size_bytes: data_size_bytes,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput_mb_sec,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run all tensor benchmarks and print results
///
/// This includes:
/// 1. Per-element storage (original, slow for large tensors)
/// 2. Blob storage (single entry, shows raw throughput potential)
/// 3. Chunked storage (new optimized API, target: 1 GB/s)
pub fn run_all_tensor_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    println!("\n=== Tensor Benchmarks ===");
    println!("Testing three storage modes:");
    println!("  1. Per-element: Each element as separate key (slow, O(n) writes)");
    println!("  2. Blob: Entire tensor as single entry (fast, limited flexibility)");
    println!("  3. Chunked: Fixed-size chunks with metadata (fast, flexible)\n");
    
    // Use small element counts - per-element storage is slow by design
    let element_counts = [100usize, 500];
    
    println!("--- Per-Element Storage (legacy) ---");
    
    for &num_elements in &element_counts {
        let config = TensorBenchConfig {
            num_elements,
            dtype: DType::Float64,
            iterations: 2,
            warmup_iterations: 0,
        };
        
        let data_size_kb = (num_elements * 8) as f64 / 1024.0;
        println!("\nPut+Get {} elements ({:.1} KB)...", num_elements, data_size_kb);
        
        let put_result = run_tensor_put_benchmark(&config);
        let elements_per_sec = num_elements as f64 / (put_result.latency_p50_us / 1_000_000.0);
        println!("  Put: {:.0} elements/sec, {:.1}ms latency", elements_per_sec, put_result.latency_p50_us / 1000.0);
        results.push(put_result);
        
        let get_result = run_tensor_get_benchmark(&config);
        let get_elements_per_sec = num_elements as f64 / (get_result.latency_p50_us / 1_000_000.0);
        println!("  Get: {:.0} elements/sec, {:.1}ms latency", get_elements_per_sec, get_result.latency_p50_us / 1000.0);
        results.push(get_result);
    }
    
    // Run blob benchmark to show potential throughput
    println!("\n--- Blob Storage (raw throughput baseline) ---");
    println!("Stores entire tensor as single Atom::Bytes entry.\n");
    let blob_result = run_tensor_blob_benchmark();
    results.push(blob_result);
    
    // Run chunked benchmark - the new optimized API
    println!("\n--- Chunked Storage (optimized API) ---");
    println!("Stores tensor as 1MB chunks with metadata.");
    println!("Target: 1 GB/s throughput\n");
    let chunked_result = run_tensor_chunked_benchmark();
    results.push(chunked_result);
    
    results
}

/// Print tensor benchmark result
fn print_tensor_result(result: &BenchmarkResult) {
    println!("  Throughput: {:.2} MB/s", result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs ({:.2} ms)", result.latency_p50_us, result.latency_p50_us / 1000.0);
    println!("  Latency p95: {:.1} μs ({:.2} ms)", result.latency_p95_us, result.latency_p95_us / 1000.0);
    println!("  Latency p99: {:.1} μs ({:.2} ms)", result.latency_p99_us, result.latency_p99_us / 1000.0);
    if result.disk_mb > 0.0 {
        println!("  Disk usage: {:.2} MB", result.disk_mb);
    }
}

/// Run blob-based tensor benchmark to show potential throughput.
///
/// This stores the entire tensor as a single Atom::Bytes entry,
/// demonstrating the throughput achievable with blob storage.
pub fn run_tensor_blob_benchmark() -> BenchmarkResult {
    use synadb::Atom;
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("blob.db");
    
    let mut db = SynaDB::new(&db_path).expect("Failed to create database");
    
    // Generate 10 MB of tensor data
    let num_elements = 1_250_000; // 10 MB of f64 data
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let data_size_mb = data.len() as f64 / 1024.0 / 1024.0;
    
    println!("Storing {:.1} MB as single blob...", data_size_mb);
    
    // Warmup
    db.append("warmup", Atom::Bytes(data.clone())).ok();
    
    // Measurement: store as single blob
    let iterations = 5;
    let mut latencies = Vec::with_capacity(iterations);
    let start = Instant::now();
    
    for i in 0..iterations {
        let key = format!("tensor_blob_{}", i);
        let op_start = Instant::now();
        db.append(&key, Atom::Bytes(data.clone())).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let total_bytes = data.len() * iterations;
    let throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / total_duration.as_secs_f64();
    let throughput_gb_sec = throughput_mb_sec / 1024.0;
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    println!("  Blob throughput: {:.2} MB/s ({:.3} GB/s)", throughput_mb_sec, throughput_gb_sec);
    println!("  Latency p50: {:.1} ms", p50 / 1000.0);
    
    if throughput_gb_sec >= 0.5 {
        println!("  ✓ Blob storage shows good throughput potential");
    }
    
    BenchmarkResult {
        benchmark: "tensor_blob_10mb".to_string(),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: iterations,
            value_size_bytes: data.len(),
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput_mb_sec,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run chunked tensor storage benchmark.
///
/// This tests the new chunked storage API which stores tensors as
/// multiple fixed-size chunks for high throughput.
///
/// Target: 1 GB/s throughput
///
/// _Requirements: 9.3_
pub fn run_tensor_chunked_benchmark() -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("chunked.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    // Generate 10 MB of tensor data
    let num_elements = 1_250_000; // 10 MB of f64 data
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let shape = vec![num_elements];
    let data_size_mb = data.len() as f64 / 1024.0 / 1024.0;
    
    println!("Storing {:.1} MB as chunked tensor (1MB chunks)...", data_size_mb);
    
    // Warmup
    engine.put_tensor_chunked("warmup", &data, &shape, DType::Float64).ok();
    
    // Measurement: store as chunked tensor
    let iterations = 5;
    let mut put_latencies = Vec::with_capacity(iterations);
    let put_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        engine.put_tensor_chunked(&name, &data, &shape, DType::Float64).ok();
        put_latencies.push(op_start.elapsed());
    }
    
    let put_duration = put_start.elapsed();
    let put_total_bytes = data.len() * iterations;
    let put_throughput_mb_sec = (put_total_bytes as f64 / 1024.0 / 1024.0) / put_duration.as_secs_f64();
    let put_throughput_gb_sec = put_throughput_mb_sec / 1024.0;
    
    let (put_p50, put_p95, put_p99) = calculate_percentiles(put_latencies);
    
    println!("  PUT throughput: {:.2} MB/s ({:.3} GB/s)", put_throughput_mb_sec, put_throughput_gb_sec);
    println!("  PUT latency p50: {:.1} ms", put_p50 / 1000.0);
    
    // Measurement: load chunked tensor
    let mut get_latencies = Vec::with_capacity(iterations);
    let get_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        let _result = engine.get_tensor_chunked(&name);
        get_latencies.push(op_start.elapsed());
    }
    
    let get_duration = get_start.elapsed();
    let get_total_bytes = data.len() * iterations;
    let get_throughput_mb_sec = (get_total_bytes as f64 / 1024.0 / 1024.0) / get_duration.as_secs_f64();
    let get_throughput_gb_sec = get_throughput_mb_sec / 1024.0;
    
    let (get_p50, get_p95, get_p99) = calculate_percentiles(get_latencies);
    
    println!("  GET throughput: {:.2} MB/s ({:.3} GB/s)", get_throughput_mb_sec, get_throughput_gb_sec);
    println!("  GET latency p50: {:.1} ms", get_p50 / 1000.0);
    
    // Check if we hit the 1 GB/s target
    if put_throughput_gb_sec >= 1.0 {
        println!("  ✓ PUT meets 1 GB/s target!");
    } else if put_throughput_gb_sec >= 0.5 {
        println!("  ~ PUT approaching 1 GB/s target ({:.1}%)", put_throughput_gb_sec * 100.0);
    }
    
    if get_throughput_gb_sec >= 1.0 {
        println!("  ✓ GET meets 1 GB/s target!");
    } else if get_throughput_gb_sec >= 0.5 {
        println!("  ~ GET approaching 1 GB/s target ({:.1}%)", get_throughput_gb_sec * 100.0);
    }
    
    // Return PUT results (primary metric for this benchmark)
    BenchmarkResult {
        benchmark: "tensor_chunked_10mb".to_string(),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: iterations,
            value_size_bytes: data.len(),
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: put_throughput_mb_sec,
        latency_p50_us: put_p50,
        latency_p95_us: put_p95,
        latency_p99_us: put_p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: put_duration.as_secs_f64(),
    }
}

/// Run chunked tensor benchmark with varying sizes.
///
/// Tests throughput at different tensor sizes to find optimal performance.
pub fn run_tensor_chunked_size_sweep() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    println!("\n--- Chunked Storage Size Sweep ---");
    
    // Test different sizes: 1MB, 10MB, 50MB, 100MB
    let sizes_mb = [1, 10, 50, 100];
    
    for &size_mb in &sizes_mb {
        let dir = tempdir().expect("Failed to create temp dir");
        let db_path = dir.path().join("chunked.db");
        
        let db = SynaDB::new(&db_path).expect("Failed to create database");
        let mut engine = TensorEngine::new(db);
        
        let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
        let data = generate_tensor_data(num_elements, DType::Float64, 42);
        let shape = vec![num_elements];
        
        println!("\nTesting {} MB tensor...", size_mb);
        
        // Warmup
        engine.put_tensor_chunked("warmup", &data, &shape, DType::Float64).ok();
        
        // Measure PUT
        let iterations = 3;
        let mut latencies = Vec::with_capacity(iterations);
        let start = Instant::now();
        
        for i in 0..iterations {
            let name = format!("tensor_{}", i);
            let op_start = Instant::now();
            engine.put_tensor_chunked(&name, &data, &shape, DType::Float64).ok();
            latencies.push(op_start.elapsed());
        }
        
        let duration = start.elapsed();
        let total_bytes = data.len() * iterations;
        let throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
        let throughput_gb_sec = throughput_mb_sec / 1024.0;
        
        let (p50, p95, p99) = calculate_percentiles(latencies);
        
        println!("  Throughput: {:.2} MB/s ({:.3} GB/s)", throughput_mb_sec, throughput_gb_sec);
        
        results.push(BenchmarkResult {
            benchmark: format!("tensor_chunked_{}mb", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig {
                warmup_iterations: 1,
                measurement_iterations: iterations,
                value_size_bytes: data.len(),
                thread_count: 1,
                sync_on_write: false,
            },
            throughput_ops_sec: throughput_mb_sec,
            latency_p50_us: p50,
            latency_p95_us: p95,
            latency_p99_us: p99,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: duration.as_secs_f64(),
        });
    }
    
    results
}

/// Run quick tensor benchmark for CI/testing
#[allow(dead_code)]
pub fn run_quick_tensor_benchmark() -> BenchmarkResult {
    let config = TensorBenchConfig {
        num_elements: 1_000,
        dtype: DType::Float64,
        iterations: 3,
        warmup_iterations: 1,
    };
    
    run_tensor_get_benchmark(&config)
}

// =============================================================================
// 2 GB/s Throughput Validation Benchmark Suite
// =============================================================================

/// Result of a throughput benchmark for a specific method and size.
#[derive(Debug, Clone)]
pub struct ThroughputResult {
    /// Storage method name
    pub method: String,
    /// Tensor size in MB
    pub size_mb: usize,
    /// Write throughput in MB/s
    pub write_throughput_mb_sec: f64,
    /// Read throughput in MB/s
    pub read_throughput_mb_sec: f64,
    /// Write throughput in GB/s
    pub write_throughput_gb_sec: f64,
    /// Read throughput in GB/s
    pub read_throughput_gb_sec: f64,
    /// Write latency p50 in ms
    pub write_latency_ms: f64,
    /// Read latency p50 in ms
    pub read_latency_ms: f64,
    /// Whether write meets 2 GB/s target
    pub write_meets_target: bool,
    /// Whether read meets 2 GB/s target
    pub read_meets_target: bool,
}

/// Benchmark chunked storage method.
///
/// Tests put_tensor_chunked and get_tensor_chunked throughput.
fn bench_chunked(size_mb: usize) -> ThroughputResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("chunked.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let shape = vec![num_elements];
    
    // Warmup
    engine.put_tensor_chunked("warmup", &data, &shape, DType::Float64).ok();
    let _ = engine.get_tensor_chunked("warmup");
    
    // Measure write
    let iterations = 3;
    let mut write_latencies = Vec::with_capacity(iterations);
    let write_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        engine.put_tensor_chunked(&name, &data, &shape, DType::Float64).ok();
        write_latencies.push(op_start.elapsed());
    }
    
    let write_duration = write_start.elapsed();
    let total_bytes = data.len() * iterations;
    let write_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / write_duration.as_secs_f64();
    let write_throughput_gb_sec = write_throughput_mb_sec / 1024.0;
    let (write_p50, _, _) = calculate_percentiles(write_latencies);
    
    // Measure read
    let mut read_latencies = Vec::with_capacity(iterations);
    let read_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        let _ = engine.get_tensor_chunked(&name);
        read_latencies.push(op_start.elapsed());
    }
    
    let read_duration = read_start.elapsed();
    let read_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / read_duration.as_secs_f64();
    let read_throughput_gb_sec = read_throughput_mb_sec / 1024.0;
    let (read_p50, _, _) = calculate_percentiles(read_latencies);
    
    ThroughputResult {
        method: "chunked".to_string(),
        size_mb,
        write_throughput_mb_sec,
        read_throughput_mb_sec,
        write_throughput_gb_sec,
        read_throughput_gb_sec,
        write_latency_ms: write_p50 / 1000.0,
        read_latency_ms: read_p50 / 1000.0,
        write_meets_target: write_throughput_gb_sec >= 2.0,
        read_meets_target: read_throughput_gb_sec >= 2.0,
    }
}

/// Benchmark batched storage method.
///
/// Tests put_tensor_batched and get_tensor_batched throughput.
fn bench_batched(size_mb: usize) -> ThroughputResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("batched.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let shape = vec![num_elements];
    
    // Warmup
    engine.put_tensor_batched("warmup", &data, &shape, DType::Float64).ok();
    let _ = engine.get_tensor_batched("warmup");
    
    // Measure write
    let iterations = 3;
    let mut write_latencies = Vec::with_capacity(iterations);
    let write_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        engine.put_tensor_batched(&name, &data, &shape, DType::Float64).ok();
        write_latencies.push(op_start.elapsed());
    }
    
    let write_duration = write_start.elapsed();
    let total_bytes = data.len() * iterations;
    let write_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / write_duration.as_secs_f64();
    let write_throughput_gb_sec = write_throughput_mb_sec / 1024.0;
    let (write_p50, _, _) = calculate_percentiles(write_latencies);
    
    // Measure read
    let mut read_latencies = Vec::with_capacity(iterations);
    let read_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        let _ = engine.get_tensor_batched(&name);
        read_latencies.push(op_start.elapsed());
    }
    
    let read_duration = read_start.elapsed();
    let read_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / read_duration.as_secs_f64();
    let read_throughput_gb_sec = read_throughput_mb_sec / 1024.0;
    let (read_p50, _, _) = calculate_percentiles(read_latencies);
    
    ThroughputResult {
        method: "batched".to_string(),
        size_mb,
        write_throughput_mb_sec,
        read_throughput_mb_sec,
        write_throughput_gb_sec,
        read_throughput_gb_sec,
        write_latency_ms: write_p50 / 1000.0,
        read_latency_ms: read_p50 / 1000.0,
        write_meets_target: write_throughput_gb_sec >= 2.0,
        read_meets_target: read_throughput_gb_sec >= 2.0,
    }
}

/// Benchmark memory-mapped storage method.
///
/// Tests put_tensor_mmap and get_tensor_mmap throughput.
fn bench_mmap(size_mb: usize) -> ThroughputResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("mmap.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let shape = vec![num_elements];
    
    // Warmup
    engine.put_tensor_mmap("warmup", &data, &shape, DType::Float64).ok();
    let _ = engine.get_tensor_mmap("warmup");
    
    // Measure write
    let iterations = 3;
    let mut write_latencies = Vec::with_capacity(iterations);
    let write_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        engine.put_tensor_mmap(&name, &data, &shape, DType::Float64).ok();
        write_latencies.push(op_start.elapsed());
    }
    
    let write_duration = write_start.elapsed();
    let total_bytes = data.len() * iterations;
    let write_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / write_duration.as_secs_f64();
    let write_throughput_gb_sec = write_throughput_mb_sec / 1024.0;
    let (write_p50, _, _) = calculate_percentiles(write_latencies);
    
    // Measure read
    let mut read_latencies = Vec::with_capacity(iterations);
    let read_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        let _ = engine.get_tensor_mmap(&name);
        read_latencies.push(op_start.elapsed());
    }
    
    let read_duration = read_start.elapsed();
    let read_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / read_duration.as_secs_f64();
    let read_throughput_gb_sec = read_throughput_mb_sec / 1024.0;
    let (read_p50, _, _) = calculate_percentiles(read_latencies);
    
    ThroughputResult {
        method: "mmap".to_string(),
        size_mb,
        write_throughput_mb_sec,
        read_throughput_mb_sec,
        write_throughput_gb_sec,
        read_throughput_gb_sec,
        write_latency_ms: write_p50 / 1000.0,
        read_latency_ms: read_p50 / 1000.0,
        write_meets_target: write_throughput_gb_sec >= 2.0,
        read_meets_target: read_throughput_gb_sec >= 2.0,
    }
}

/// Benchmark optimized memory-mapped storage method (async flush).
///
/// Tests put_tensor_mmap_fast and get_tensor_mmap throughput.
/// This method uses async flush for maximum write throughput.
fn bench_mmap_fast(size_mb: usize) -> ThroughputResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("mmap_fast.db");
    
    let db = SynaDB::new(&db_path).expect("Failed to create database");
    let mut engine = TensorEngine::new(db);
    
    let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
    let data = generate_tensor_data(num_elements, DType::Float64, 42);
    let shape = vec![num_elements];
    
    // Warmup
    engine.put_tensor_mmap_fast("warmup", &data, &shape, DType::Float64).ok();
    let _ = engine.get_tensor_mmap("warmup");
    
    // Measure write
    let iterations = 3;
    let mut write_latencies = Vec::with_capacity(iterations);
    let write_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        engine.put_tensor_mmap_fast(&name, &data, &shape, DType::Float64).ok();
        write_latencies.push(op_start.elapsed());
    }
    
    let write_duration = write_start.elapsed();
    let total_bytes = data.len() * iterations;
    let write_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / write_duration.as_secs_f64();
    let write_throughput_gb_sec = write_throughput_mb_sec / 1024.0;
    let (write_p50, _, _) = calculate_percentiles(write_latencies);
    
    // Measure read (uses same get_tensor_mmap as regular mmap)
    let mut read_latencies = Vec::with_capacity(iterations);
    let read_start = Instant::now();
    
    for i in 0..iterations {
        let name = format!("tensor_{}", i);
        let op_start = Instant::now();
        let _ = engine.get_tensor_mmap(&name);
        read_latencies.push(op_start.elapsed());
    }
    
    let read_duration = read_start.elapsed();
    let read_throughput_mb_sec = (total_bytes as f64 / 1024.0 / 1024.0) / read_duration.as_secs_f64();
    let read_throughput_gb_sec = read_throughput_mb_sec / 1024.0;
    let (read_p50, _, _) = calculate_percentiles(read_latencies);
    
    ThroughputResult {
        method: "mmap_fast".to_string(),
        size_mb,
        write_throughput_mb_sec,
        read_throughput_mb_sec,
        write_throughput_gb_sec,
        read_throughput_gb_sec,
        write_latency_ms: write_p50 / 1000.0,
        read_latency_ms: read_p50 / 1000.0,
        write_meets_target: write_throughput_gb_sec >= 2.0,
        read_meets_target: read_throughput_gb_sec >= 2.0,
    }
}

/// Print throughput comparison table.
fn print_throughput_comparison(results: &[ThroughputResult]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           TENSOR THROUGHPUT COMPARISON (Target: 2 GB/s)                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Method   │ Size   │ Write MB/s │ Write GB/s │ Read MB/s  │ Read GB/s  │ Write OK │ Read OK   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    
    for r in results {
        let write_status = if r.write_meets_target { "✓" } else { "✗" };
        let read_status = if r.read_meets_target { "✓" } else { "✗" };
        
        println!(
            "║ {:8} │ {:4} MB │ {:10.1} │ {:10.3} │ {:10.1} │ {:10.3} │    {}     │    {}      ║",
            r.method,
            r.size_mb,
            r.write_throughput_mb_sec,
            r.write_throughput_gb_sec,
            r.read_throughput_mb_sec,
            r.read_throughput_gb_sec,
            write_status,
            read_status
        );
    }
    
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    
    // Summary
    let any_write_meets = results.iter().any(|r| r.write_meets_target);
    let any_read_meets = results.iter().any(|r| r.read_meets_target);
    
    println!("\n=== Summary ===");
    if any_write_meets {
        let best_write = results.iter()
            .max_by(|a, b| a.write_throughput_gb_sec.partial_cmp(&b.write_throughput_gb_sec).unwrap())
            .unwrap();
        println!("✓ WRITE target met: {} method achieves {:.2} GB/s at {} MB", 
            best_write.method, best_write.write_throughput_gb_sec, best_write.size_mb);
    } else {
        let best_write = results.iter()
            .max_by(|a, b| a.write_throughput_gb_sec.partial_cmp(&b.write_throughput_gb_sec).unwrap())
            .unwrap();
        println!("✗ WRITE target NOT met: best is {} method at {:.2} GB/s ({:.0}% of target)", 
            best_write.method, best_write.write_throughput_gb_sec, best_write.write_throughput_gb_sec / 2.0 * 100.0);
    }
    
    if any_read_meets {
        let best_read = results.iter()
            .max_by(|a, b| a.read_throughput_gb_sec.partial_cmp(&b.read_throughput_gb_sec).unwrap())
            .unwrap();
        println!("✓ READ target met: {} method achieves {:.2} GB/s at {} MB", 
            best_read.method, best_read.read_throughput_gb_sec, best_read.size_mb);
    } else {
        let best_read = results.iter()
            .max_by(|a, b| a.read_throughput_gb_sec.partial_cmp(&b.read_throughput_gb_sec).unwrap())
            .unwrap();
        println!("✗ READ target NOT met: best is {} method at {:.2} GB/s ({:.0}% of target)", 
            best_read.method, best_read.read_throughput_gb_sec, best_read.read_throughput_gb_sec / 2.0 * 100.0);
    }
}

/// Run tensor throughput comparison benchmark.
///
/// Benchmarks all tensor storage methods (chunked, batched, mmap) at various
/// sizes up to 1GB to validate 2 GB/s throughput target.
///
/// # Validation Criteria
///
/// - At least one method achieves 2 GB/s write throughput
/// - At least one method achieves 2 GB/s read throughput
/// - Benchmark runs on 1GB tensor without OOM
///
/// _Requirements: 9.3_
pub fn run_tensor_throughput_comparison() -> Vec<BenchmarkResult> {
    let sizes_mb = [10, 50, 100, 500, 1000]; // Up to 1GB tensors
    let mut throughput_results = Vec::new();
    let mut benchmark_results = Vec::new();
    
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    TENSOR THROUGHPUT VALIDATION BENCHMARK (Target: 2 GB/s)                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    
    for &size_mb in &sizes_mb {
        println!("\n=== {} MB Tensor ===", size_mb);
        
        // Test chunked method
        print!("  Testing chunked... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let chunked_result = bench_chunked(size_mb);
        println!("Write: {:.2} GB/s, Read: {:.2} GB/s", 
            chunked_result.write_throughput_gb_sec, chunked_result.read_throughput_gb_sec);
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("tensor_chunked_{}mb_write", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig {
                warmup_iterations: 1,
                measurement_iterations: 3,
                value_size_bytes: size_mb * 1024 * 1024,
                thread_count: 1,
                sync_on_write: false,
            },
            throughput_ops_sec: chunked_result.write_throughput_mb_sec,
            latency_p50_us: chunked_result.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        throughput_results.push(chunked_result);
        
        // Test batched method
        print!("  Testing batched... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let batched_result = bench_batched(size_mb);
        println!("Write: {:.2} GB/s, Read: {:.2} GB/s", 
            batched_result.write_throughput_gb_sec, batched_result.read_throughput_gb_sec);
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("tensor_batched_{}mb_write", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig {
                warmup_iterations: 1,
                measurement_iterations: 3,
                value_size_bytes: size_mb * 1024 * 1024,
                thread_count: 1,
                sync_on_write: false,
            },
            throughput_ops_sec: batched_result.write_throughput_mb_sec,
            latency_p50_us: batched_result.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        throughput_results.push(batched_result);
        
        // Test mmap method
        print!("  Testing mmap... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mmap_result = bench_mmap(size_mb);
        println!("Write: {:.2} GB/s, Read: {:.2} GB/s", 
            mmap_result.write_throughput_gb_sec, mmap_result.read_throughput_gb_sec);
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("tensor_mmap_{}mb_write", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig {
                warmup_iterations: 1,
                measurement_iterations: 3,
                value_size_bytes: size_mb * 1024 * 1024,
                thread_count: 1,
                sync_on_write: false,
            },
            throughput_ops_sec: mmap_result.write_throughput_mb_sec,
            latency_p50_us: mmap_result.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        throughput_results.push(mmap_result);
        
        // Test mmap_fast method (async flush for maximum throughput)
        print!("  Testing mmap_fast... ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mmap_fast_result = bench_mmap_fast(size_mb);
        println!("Write: {:.2} GB/s, Read: {:.2} GB/s", 
            mmap_fast_result.write_throughput_gb_sec, mmap_fast_result.read_throughput_gb_sec);
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("tensor_mmap_fast_{}mb_write", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig {
                warmup_iterations: 1,
                measurement_iterations: 3,
                value_size_bytes: size_mb * 1024 * 1024,
                thread_count: 1,
                sync_on_write: false,
            },
            throughput_ops_sec: mmap_fast_result.write_throughput_mb_sec,
            latency_p50_us: mmap_fast_result.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        throughput_results.push(mmap_fast_result);
    }
    
    // Print comparison table
    print_throughput_comparison(&throughput_results);
    
    // Validate targets
    let any_write_meets = throughput_results.iter().any(|r| r.write_meets_target);
    let any_read_meets = throughput_results.iter().any(|r| r.read_meets_target);
    let ran_1gb = throughput_results.iter().any(|r| r.size_mb == 1000);
    
    println!("\n=== Validation Results ===");
    println!("  [{}] At least one method achieves 2 GB/s write", if any_write_meets { "✓" } else { "✗" });
    println!("  [{}] At least one method achieves 2 GB/s read", if any_read_meets { "✓" } else { "✗" });
    println!("  [{}] Benchmark runs on 1GB tensor without OOM", if ran_1gb { "✓" } else { "✗" });
    
    benchmark_results
}

/// Run a quick throughput validation (smaller sizes for CI).
///
/// Tests only 10MB and 100MB tensors for faster CI runs.
pub fn run_quick_throughput_validation() -> Vec<BenchmarkResult> {
    let sizes_mb = [10, 100];
    let mut throughput_results = Vec::new();
    let mut benchmark_results = Vec::new();
    
    println!("\n=== Quick Throughput Validation (Target: 2 GB/s) ===\n");
    
    for &size_mb in &sizes_mb {
        println!("Testing {} MB tensor...", size_mb);
        
        let chunked = bench_chunked(size_mb);
        println!("  chunked: W={:.2} GB/s, R={:.2} GB/s", 
            chunked.write_throughput_gb_sec, chunked.read_throughput_gb_sec);
        throughput_results.push(chunked.clone());
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("quick_chunked_{}mb", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig::default(),
            throughput_ops_sec: chunked.write_throughput_mb_sec,
            latency_p50_us: chunked.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        let batched = bench_batched(size_mb);
        println!("  batched: W={:.2} GB/s, R={:.2} GB/s", 
            batched.write_throughput_gb_sec, batched.read_throughput_gb_sec);
        throughput_results.push(batched.clone());
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("quick_batched_{}mb", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig::default(),
            throughput_ops_sec: batched.write_throughput_mb_sec,
            latency_p50_us: batched.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        let mmap = bench_mmap(size_mb);
        println!("  mmap: W={:.2} GB/s, R={:.2} GB/s", 
            mmap.write_throughput_gb_sec, mmap.read_throughput_gb_sec);
        throughput_results.push(mmap.clone());
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("quick_mmap_{}mb", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig::default(),
            throughput_ops_sec: mmap.write_throughput_mb_sec,
            latency_p50_us: mmap.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
        
        let mmap_fast = bench_mmap_fast(size_mb);
        println!("  mmap_fast: W={:.2} GB/s, R={:.2} GB/s", 
            mmap_fast.write_throughput_gb_sec, mmap_fast.read_throughput_gb_sec);
        throughput_results.push(mmap_fast.clone());
        
        benchmark_results.push(BenchmarkResult {
            benchmark: format!("quick_mmap_fast_{}mb", size_mb),
            database: "Syna".to_string(),
            config: BenchmarkConfig::default(),
            throughput_ops_sec: mmap_fast.write_throughput_mb_sec,
            latency_p50_us: mmap_fast.write_latency_ms * 1000.0,
            latency_p95_us: 0.0,
            latency_p99_us: 0.0,
            memory_mb: 0.0,
            disk_mb: 0.0,
            duration_secs: 0.0,
        });
    }
    
    // Find best results
    let best_write = throughput_results.iter()
        .max_by(|a, b| a.write_throughput_gb_sec.partial_cmp(&b.write_throughput_gb_sec).unwrap())
        .unwrap();
    let best_read = throughput_results.iter()
        .max_by(|a, b| a.read_throughput_gb_sec.partial_cmp(&b.read_throughput_gb_sec).unwrap())
        .unwrap();
    
    println!("\nBest write: {} at {} MB = {:.2} GB/s", 
        best_write.method, best_write.size_mb, best_write.write_throughput_gb_sec);
    println!("Best read: {} at {} MB = {:.2} GB/s", 
        best_read.method, best_read.size_mb, best_read.read_throughput_gb_sec);
    
    benchmark_results
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_put_benchmark() {
        let config = TensorBenchConfig {
            num_elements: 100,
            dtype: DType::Float64,
            iterations: 2,
            warmup_iterations: 1,
        };
        
        let result = run_tensor_put_benchmark(&config);
        
        assert_eq!(result.database, "Syna");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_tensor_get_benchmark() {
        let config = TensorBenchConfig {
            num_elements: 100,
            dtype: DType::Float64,
            iterations: 2,
            warmup_iterations: 1,
        };
        
        let result = run_tensor_get_benchmark(&config);
        
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_tensor_roundtrip_benchmark() {
        let config = TensorBenchConfig {
            num_elements: 100,
            dtype: DType::Float64,
            iterations: 2,
            warmup_iterations: 0,
        };
        
        let result = run_tensor_roundtrip_benchmark(&config);
        
        assert!(result.throughput_ops_sec > 0.0);
    }
    
    #[test]
    fn test_generate_tensor_data() {
        let data1 = generate_tensor_data(10, DType::Float64, 42);
        let data2 = generate_tensor_data(10, DType::Float64, 42);
        let data3 = generate_tensor_data(10, DType::Float64, 43);
        
        assert_eq!(data1.len(), 80); // 10 * 8 bytes
        assert_eq!(data1, data2); // Same seed = same data
        assert_ne!(data1, data3); // Different seed = different data
    }
}

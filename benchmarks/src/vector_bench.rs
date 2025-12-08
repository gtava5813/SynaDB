//! Vector store performance benchmarks.
//!
//! This module implements benchmarks for vector storage and similarity search operations.
//! It tests:
//! - Vector insert throughput (target: 100K/sec for 768-dim vectors)
//! - Vector search latency (target: <10ms for 1M vectors)
//! - HNSW index build time
//!
//! _Requirements: 9.1, 9.2 (Performance Guarantees)_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use synadb::distance::DistanceMetric;
use synadb::vector::{VectorConfig, VectorStore};
use std::time::Instant;
use tempfile::tempdir;

/// Standard vector dimensions for benchmarking
pub const VECTOR_DIMENSIONS: &[u16] = &[128, 384, 768, 1536];

/// Generate a random vector with deterministic seed
fn generate_vector(dims: u16, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Run vector insert benchmark
///
/// Tests sequential vector insert performance.
/// Target: 100K vectors/sec for 768-dim float32 vectors.
///
/// _Requirements: 9.1_
pub fn run_vector_insert_benchmark(config: &VectorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("vectors.db");
    
    let vec_config = VectorConfig {
        dimensions: config.dimensions,
        metric: DistanceMetric::Cosine,
        index_threshold: usize::MAX, // Disable auto-indexing for pure insert benchmark
        ..Default::default()
    };
    
    let mut store = VectorStore::new(&db_path, vec_config)
        .expect("Failed to create vector store");
    
    // Pre-generate vectors for consistent timing
    let vectors: Vec<Vec<f32>> = (0..config.num_vectors)
        .map(|i| generate_vector(config.dimensions, i as u64))
        .collect();
    
    // Warmup phase
    for i in 0..config.warmup_iterations.min(config.num_vectors / 10) {
        let key = format!("warmup/{}", i);
        store.insert(&key, &vectors[i % vectors.len()]).ok();
    }
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.num_vectors);
    let start = Instant::now();
    
    for i in 0..config.num_vectors {
        let key = format!("vec/{}", i);
        let op_start = Instant::now();
        store.insert(&key, &vectors[i]).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.num_vectors as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    // Get disk usage
    let disk_mb = std::fs::metadata(&db_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);
    
    // Estimate memory (vectors cached in memory)
    let memory_mb = (config.num_vectors * config.dimensions as usize * 4) as f64 / 1024.0 / 1024.0;
    
    BenchmarkResult {
        benchmark: format!("vector_insert_{}dim", config.dimensions),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.num_vectors,
            value_size_bytes: config.dimensions as usize * 4,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb,
        disk_mb,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run vector search benchmark (brute force)
///
/// Tests brute-force k-nearest neighbor search performance.
///
/// _Requirements: 9.2_
pub fn run_vector_search_brute_force_benchmark(config: &VectorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("vectors.db");
    
    let vec_config = VectorConfig {
        dimensions: config.dimensions,
        metric: DistanceMetric::Cosine,
        index_threshold: usize::MAX, // Force brute force
        ..Default::default()
    };
    
    let mut store = VectorStore::new(&db_path, vec_config)
        .expect("Failed to create vector store");
    
    // Insert vectors
    println!("  Inserting {} vectors...", config.num_vectors);
    for i in 0..config.num_vectors {
        let key = format!("vec/{}", i);
        let vector = generate_vector(config.dimensions, i as u64);
        store.insert(&key, &vector).ok();
    }
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.num_queries);
    let start = Instant::now();
    
    for query in &queries {
        let op_start = Instant::now();
        store.search(query, config.k).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.num_queries as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    BenchmarkResult {
        benchmark: format!("vector_search_brute_{}dim_{}vectors", config.dimensions, config.num_vectors),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: config.num_queries,
            value_size_bytes: config.dimensions as usize * 4,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Run vector search benchmark with HNSW index
///
/// Tests HNSW approximate nearest neighbor search performance.
/// Target: <10ms for 1M vectors.
///
/// _Requirements: 9.2_
pub fn run_vector_search_hnsw_benchmark(config: &VectorBenchConfig) -> BenchmarkResult {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("vectors.db");
    
    let vec_config = VectorConfig {
        dimensions: config.dimensions,
        metric: DistanceMetric::Cosine,
        index_threshold: 100, // Enable HNSW for small datasets too
        ..Default::default()
    };
    
    let mut store = VectorStore::new(&db_path, vec_config)
        .expect("Failed to create vector store");
    
    // Insert vectors
    println!("  Inserting {} vectors...", config.num_vectors);
    for i in 0..config.num_vectors {
        let key = format!("vec/{}", i);
        let vector = generate_vector(config.dimensions, i as u64);
        store.insert(&key, &vector).ok();
    }
    
    // Build HNSW index
    println!("  Building HNSW index...");
    let index_start = Instant::now();
    store.build_index().expect("Failed to build index");
    let index_duration = index_start.elapsed();
    println!("  Index built in {:.2}s", index_duration.as_secs_f64());
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Measurement phase
    let mut latencies = Vec::with_capacity(config.num_queries);
    let start = Instant::now();
    
    for query in &queries {
        let op_start = Instant::now();
        store.search(query, config.k).ok();
        latencies.push(op_start.elapsed());
    }
    
    let total_duration = start.elapsed();
    let throughput = config.num_queries as f64 / total_duration.as_secs_f64();
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    BenchmarkResult {
        benchmark: format!("vector_search_hnsw_{}dim_{}vectors", config.dimensions, config.num_vectors),
        database: "Syna".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: 0,
            measurement_iterations: config.num_queries,
            value_size_bytes: config.dimensions as usize * 4,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: throughput,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        memory_mb: 0.0,
        disk_mb: 0.0,
        duration_secs: total_duration.as_secs_f64(),
    }
}

/// Configuration for vector benchmarks
#[derive(Debug, Clone)]
pub struct VectorBenchConfig {
    /// Number of dimensions per vector
    pub dimensions: u16,
    /// Number of vectors to insert/index
    pub num_vectors: usize,
    /// Number of search queries to run
    pub num_queries: usize,
    /// Number of nearest neighbors to retrieve
    pub k: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for VectorBenchConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            num_vectors: 10_000,
            num_queries: 100,
            k: 10,
            warmup_iterations: 100,
        }
    }
}

/// Run all vector benchmarks and print results
pub fn run_all_vector_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    println!("\n=== Vector Insert Benchmarks ===");
    
    // Test different dimensions
    for &dims in &[128u16, 384, 768, 1536] {
        let config = VectorBenchConfig {
            dimensions: dims,
            num_vectors: 10_000,
            warmup_iterations: 100,
            ..Default::default()
        };
        
        println!("\nRunning vector insert benchmark ({} dimensions)...", dims);
        let result = run_vector_insert_benchmark(&config);
        print_vector_result(&result);
        results.push(result);
    }
    
    println!("\n=== Vector Search Benchmarks (Brute Force) ===");
    
    // Test different dataset sizes with brute force
    for &num_vectors in &[1_000usize, 5_000, 10_000] {
        let config = VectorBenchConfig {
            dimensions: 768,
            num_vectors,
            num_queries: 100,
            k: 10,
            ..Default::default()
        };
        
        println!("\nRunning brute force search benchmark ({} vectors)...", num_vectors);
        let result = run_vector_search_brute_force_benchmark(&config);
        print_vector_result(&result);
        results.push(result);
    }
    
    println!("\n=== Vector Search Benchmarks (HNSW) ===");
    
    // Test HNSW with larger datasets
    for &num_vectors in &[10_000usize, 50_000, 100_000] {
        let config = VectorBenchConfig {
            dimensions: 768,
            num_vectors,
            num_queries: 100,
            k: 10,
            ..Default::default()
        };
        
        println!("\nRunning HNSW search benchmark ({} vectors)...", num_vectors);
        let result = run_vector_search_hnsw_benchmark(&config);
        print_vector_result(&result);
        
        // Check against target: <10ms for search
        let target_met = result.latency_p50_us < 10_000.0;
        if target_met {
            println!("  ✓ Target met: p50 latency < 10ms");
        } else {
            println!("  ✗ Target NOT met: p50 latency >= 10ms");
        }
        
        results.push(result);
    }
    
    results
}

/// Print vector benchmark result
fn print_vector_result(result: &BenchmarkResult) {
    println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
    println!("  Latency p50: {:.1} μs ({:.2} ms)", result.latency_p50_us, result.latency_p50_us / 1000.0);
    println!("  Latency p95: {:.1} μs ({:.2} ms)", result.latency_p95_us, result.latency_p95_us / 1000.0);
    println!("  Latency p99: {:.1} μs ({:.2} ms)", result.latency_p99_us, result.latency_p99_us / 1000.0);
    if result.disk_mb > 0.0 {
        println!("  Disk usage: {:.2} MB", result.disk_mb);
    }
}

/// Run quick vector benchmark for CI/testing
#[allow(dead_code)]
pub fn run_quick_vector_benchmark() -> BenchmarkResult {
    let config = VectorBenchConfig {
        dimensions: 128,
        num_vectors: 1_000,
        num_queries: 10,
        k: 10,
        warmup_iterations: 10,
    };
    
    run_vector_insert_benchmark(&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_insert_benchmark() {
        let config = VectorBenchConfig {
            dimensions: 64,
            num_vectors: 100,
            warmup_iterations: 10,
            ..Default::default()
        };
        
        let result = run_vector_insert_benchmark(&config);
        
        assert_eq!(result.database, "Syna");
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_vector_search_brute_force_benchmark() {
        let config = VectorBenchConfig {
            dimensions: 64,
            num_vectors: 100,
            num_queries: 10,
            k: 5,
            ..Default::default()
        };
        
        let result = run_vector_search_brute_force_benchmark(&config);
        
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_vector_search_hnsw_benchmark() {
        let config = VectorBenchConfig {
            dimensions: 64,
            num_vectors: 500,
            num_queries: 10,
            k: 5,
            ..Default::default()
        };
        
        let result = run_vector_search_hnsw_benchmark(&config);
        
        assert!(result.throughput_ops_sec > 0.0);
        assert!(result.latency_p50_us > 0.0);
    }
    
    #[test]
    fn test_generate_vector() {
        let v1 = generate_vector(128, 42);
        let v2 = generate_vector(128, 42);
        let v3 = generate_vector(128, 43);
        
        assert_eq!(v1.len(), 128);
        assert_eq!(v1, v2); // Same seed = same vector
        assert_ne!(v1, v3); // Different seed = different vector
    }
}

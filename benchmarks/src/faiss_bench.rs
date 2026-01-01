//! FAISS vs HNSW benchmark comparison
//!
//! This module compares the performance of SynaDB's native HNSW index
//! against FAISS indexes (Flat and IVF) for vector similarity search.
//!
//! Metrics compared:
//! - Insert throughput (vectors/sec)
//! - Search latency (ms for top-10)
//! - Memory usage (MB)
//! - Recall@10 accuracy
//!
//! _Requirements: 9.2 (Performance Guarantees)_

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use std::time::Instant;
use tempfile::tempdir;

/// Configuration for FAISS vs HNSW benchmarks
#[derive(Debug, Clone)]
pub struct FaissVsHnswConfig {
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

impl Default for FaissVsHnswConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            num_vectors: 100_000,
            num_queries: 100,
            k: 10,
            warmup_iterations: 10,
        }
    }
}

/// Results from a single index benchmark
#[derive(Debug, Clone)]
pub struct IndexBenchResult {
    /// Name of the index type
    pub index_name: String,
    /// Insert throughput (vectors/sec)
    pub insert_throughput: f64,
    /// Search latency p50 (ms)
    pub search_latency_p50_ms: f64,
    /// Search latency p99 (ms)
    pub search_latency_p99_ms: f64,
    /// Memory usage (MB)
    pub memory_mb: f64,
    /// Recall@k accuracy (0.0 - 1.0)
    pub recall_at_k: f64,
    /// Index build time (seconds)
    pub build_time_secs: f64,
}

/// Generate a random vector with deterministic seed
fn generate_vector(dims: u16, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Benchmark HNSW index
pub fn bench_hnsw(config: &FaissVsHnswConfig) -> IndexBenchResult {
    use synadb::distance::DistanceMetric;
    use synadb::vector::{VectorConfig, VectorStore};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("hnsw_bench.db");
    
    let vec_config = VectorConfig {
        dimensions: config.dimensions,
        metric: DistanceMetric::Cosine,
        index_threshold: 100, // Enable HNSW early
        ..Default::default()
    };
    
    let mut store = VectorStore::new(&db_path, vec_config)
        .expect("Failed to create vector store");
    
    // Pre-generate vectors
    let vectors: Vec<Vec<f32>> = (0..config.num_vectors)
        .map(|i| generate_vector(config.dimensions, i as u64))
        .collect();
    
    // Measure insert throughput
    let insert_start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        let key = format!("vec/{}", i);
        store.insert(&key, vector).ok();
    }
    let insert_duration = insert_start.elapsed();
    let insert_throughput = config.num_vectors as f64 / insert_duration.as_secs_f64();
    
    // Build HNSW index
    let build_start = Instant::now();
    store.build_index().expect("Failed to build HNSW index");
    let build_time_secs = build_start.elapsed().as_secs_f64();
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Measure search latency
    let mut latencies = Vec::with_capacity(config.num_queries);
    for query in &queries {
        let op_start = Instant::now();
        store.search(query, config.k).ok();
        latencies.push(op_start.elapsed());
    }
    
    let (p50, _p95, p99) = calculate_percentiles(latencies);
    
    // Estimate memory usage (vectors + index overhead)
    let vector_memory = (config.num_vectors * config.dimensions as usize * 4) as f64 / 1024.0 / 1024.0;
    let index_overhead = vector_memory * 0.3; // HNSW typically adds ~30% overhead
    let memory_mb = vector_memory + index_overhead;
    
    // Calculate recall (HNSW vs brute force ground truth)
    let recall = calculate_hnsw_recall(&mut store, &vectors, &queries, config.k);
    
    IndexBenchResult {
        index_name: "HNSW".to_string(),
        insert_throughput,
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        memory_mb,
        recall_at_k: recall,
        build_time_secs,
    }
}

/// Calculate recall@k for HNSW by comparing against brute force
fn calculate_hnsw_recall(
    store: &mut synadb::vector::VectorStore,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    use synadb::distance::DistanceMetric;
    
    let metric = DistanceMetric::Cosine;
    let mut total_recall = 0.0;
    
    for query in queries {
        // Get HNSW results
        let hnsw_results = store.search(query, k).unwrap_or_default();
        let hnsw_keys: std::collections::HashSet<_> = hnsw_results.iter()
            .map(|r| r.key.clone())
            .collect();
        
        // Calculate brute force ground truth
        let mut distances: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, metric.distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let ground_truth: std::collections::HashSet<_> = distances.iter()
            .take(k)
            .map(|(i, _)| format!("vec/{}", i))
            .collect();
        
        // Calculate recall for this query
        let matches = hnsw_keys.intersection(&ground_truth).count();
        total_recall += matches as f64 / k as f64;
    }
    
    total_recall / queries.len() as f64
}

/// Benchmark FAISS index (when feature is enabled)
#[cfg(feature = "faiss")]
pub fn bench_faiss(index_type: &str, config: &FaissVsHnswConfig) -> IndexBenchResult {
    use synadb::distance::DistanceMetric;
    use synadb::faiss_index::{FaissConfig, FaissIndex};
    
    let faiss_config = FaissConfig {
        index_type: index_type.to_string(),
        train_size: config.num_vectors.min(10000),
        nprobe: 10,
        use_gpu: false,
    };
    
    let mut index = FaissIndex::new(config.dimensions, DistanceMetric::Cosine, faiss_config)
        .expect("Failed to create FAISS index");
    
    // Pre-generate vectors
    let vectors: Vec<Vec<f32>> = (0..config.num_vectors)
        .map(|i| generate_vector(config.dimensions, i as u64))
        .collect();
    
    // Measure insert throughput
    let insert_start = Instant::now();
    for (i, vector) in vectors.iter().enumerate() {
        let key = format!("vec/{}", i);
        index.insert(&key, vector).ok();
    }
    let insert_duration = insert_start.elapsed();
    let insert_throughput = config.num_vectors as f64 / insert_duration.as_secs_f64();
    
    // Build time is included in insert for FAISS (training happens during insert)
    let build_time_secs = 0.0;
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Measure search latency
    let mut latencies = Vec::with_capacity(config.num_queries);
    for query in &queries {
        let op_start = Instant::now();
        index.search(query, config.k).ok();
        latencies.push(op_start.elapsed());
    }
    
    let (p50, _p95, p99) = calculate_percentiles(latencies);
    
    // Estimate memory usage
    let vector_memory = (config.num_vectors * config.dimensions as usize * 4) as f64 / 1024.0 / 1024.0;
    let index_overhead = if index_type.contains("IVF") { 0.1 } else { 0.0 }; // IVF adds ~10% overhead
    let memory_mb = vector_memory * (1.0 + index_overhead);
    
    // Calculate recall (FAISS vs brute force ground truth)
    let recall = calculate_faiss_recall(&mut index, &vectors, &queries, config.k);
    
    IndexBenchResult {
        index_name: format!("FAISS-{}", index_type),
        insert_throughput,
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        memory_mb,
        recall_at_k: recall,
        build_time_secs,
    }
}

/// Calculate recall@k for FAISS by comparing against brute force
#[cfg(feature = "faiss")]
fn calculate_faiss_recall(
    index: &mut synadb::faiss_index::FaissIndex,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    use synadb::distance::DistanceMetric;
    
    let metric = DistanceMetric::Cosine;
    let mut total_recall = 0.0;
    
    for query in queries {
        // Get FAISS results
        let faiss_results = index.search(query, k).unwrap_or_default();
        let faiss_keys: std::collections::HashSet<_> = faiss_results.iter()
            .map(|(key, _)| key.clone())
            .collect();
        
        // Calculate brute force ground truth
        let mut distances: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, metric.distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let ground_truth: std::collections::HashSet<_> = distances.iter()
            .take(k)
            .map(|(i, _)| format!("vec/{}", i))
            .collect();
        
        // Calculate recall for this query
        let matches = faiss_keys.intersection(&ground_truth).count();
        total_recall += matches as f64 / k as f64;
    }
    
    total_recall / queries.len() as f64
}

/// Stub for FAISS benchmark when feature is not enabled
#[cfg(not(feature = "faiss"))]
pub fn bench_faiss(_index_type: &str, config: &FaissVsHnswConfig) -> IndexBenchResult {
    println!("  [FAISS not enabled - compile with --features faiss]");
    IndexBenchResult {
        index_name: format!("FAISS-{} (disabled)", _index_type),
        insert_throughput: 0.0,
        search_latency_p50_ms: 0.0,
        search_latency_p99_ms: 0.0,
        memory_mb: 0.0,
        recall_at_k: 0.0,
        build_time_secs: 0.0,
    }
}

/// Print comparison table
pub fn print_comparison(results: &[IndexBenchResult]) {
    println!("\n┌─────────────────────┬───────────────┬──────────────┬──────────────┬────────────┬────────────┐");
    println!("│ Index               │ Insert (v/s)  │ Search p50   │ Search p99   │ Memory MB  │ Recall@10  │");
    println!("├─────────────────────┼───────────────┼──────────────┼──────────────┼────────────┼────────────┤");
    
    for result in results {
        println!(
            "│ {:19} │ {:>13.0} │ {:>10.2}ms │ {:>10.2}ms │ {:>10.1} │ {:>9.1}% │",
            result.index_name,
            result.insert_throughput,
            result.search_latency_p50_ms,
            result.search_latency_p99_ms,
            result.memory_mb,
            result.recall_at_k * 100.0
        );
    }
    
    println!("└─────────────────────┴───────────────┴──────────────┴──────────────┴────────────┴────────────┘");
}

/// Run FAISS vs HNSW benchmark comparison
///
/// Compares performance across different vector counts and index types.
///
/// _Requirements: 9.2_
pub fn run_faiss_vs_hnsw_benchmark() {
    let vector_counts = [100_000, 1_000_000];
    let dimensions = 768;

    for count in vector_counts {
        println!("\n=== {} vectors, {} dimensions ===", count, dimensions);

        let config = FaissVsHnswConfig {
            dimensions,
            num_vectors: count,
            num_queries: 100,
            k: 10,
            warmup_iterations: 10,
        };

        // Benchmark HNSW
        println!("\nBenchmarking HNSW...");
        let hnsw_results = bench_hnsw(&config);
        println!("  Insert: {:.0} vectors/sec", hnsw_results.insert_throughput);
        println!("  Search p50: {:.2}ms", hnsw_results.search_latency_p50_ms);
        println!("  Build time: {:.2}s", hnsw_results.build_time_secs);

        // Benchmark FAISS Flat
        println!("\nBenchmarking FAISS Flat...");
        let faiss_flat_results = bench_faiss("Flat", &config);
        if faiss_flat_results.insert_throughput > 0.0 {
            println!("  Insert: {:.0} vectors/sec", faiss_flat_results.insert_throughput);
            println!("  Search p50: {:.2}ms", faiss_flat_results.search_latency_p50_ms);
        }

        // Benchmark FAISS IVF
        println!("\nBenchmarking FAISS IVF1024,Flat...");
        let faiss_ivf_results = bench_faiss("IVF1024,Flat", &config);
        if faiss_ivf_results.insert_throughput > 0.0 {
            println!("  Insert: {:.0} vectors/sec", faiss_ivf_results.insert_throughput);
            println!("  Search p50: {:.2}ms", faiss_ivf_results.search_latency_p50_ms);
        }

        // Print comparison
        print_comparison(&[hnsw_results, faiss_flat_results, faiss_ivf_results]);
    }
}

/// Run quick FAISS vs HNSW benchmark (smaller dataset for CI)
pub fn run_quick_faiss_vs_hnsw_benchmark() -> Vec<IndexBenchResult> {
    let config = FaissVsHnswConfig {
        dimensions: 128,
        num_vectors: 10_000,
        num_queries: 50,
        k: 10,
        warmup_iterations: 5,
    };

    println!("\n=== Quick FAISS vs HNSW Benchmark ===");
    println!("  Vectors: {}", config.num_vectors);
    println!("  Dimensions: {}", config.dimensions);
    println!("  Queries: {}", config.num_queries);

    let mut results = Vec::new();

    // Benchmark HNSW
    println!("\nBenchmarking HNSW...");
    let hnsw_results = bench_hnsw(&config);
    results.push(hnsw_results.clone());

    // Benchmark FAISS Flat
    println!("Benchmarking FAISS Flat...");
    let faiss_flat_results = bench_faiss("Flat", &config);
    results.push(faiss_flat_results.clone());

    // Benchmark FAISS IVF
    println!("Benchmarking FAISS IVF256,Flat...");
    let faiss_ivf_results = bench_faiss("IVF256,Flat", &config);
    results.push(faiss_ivf_results.clone());

    // Print comparison
    print_comparison(&results);

    results
}

/// Convert IndexBenchResult to BenchmarkResult for unified reporting
pub fn to_benchmark_result(result: &IndexBenchResult, config: &FaissVsHnswConfig) -> BenchmarkResult {
    BenchmarkResult {
        benchmark: format!("faiss_vs_hnsw_{}", result.index_name.to_lowercase().replace("-", "_")),
        database: result.index_name.clone(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.num_queries,
            value_size_bytes: config.dimensions as usize * 4,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: result.insert_throughput,
        latency_p50_us: result.search_latency_p50_ms * 1000.0,
        latency_p95_us: result.search_latency_p99_ms * 1000.0, // Using p99 as p95 approximation
        latency_p99_us: result.search_latency_p99_ms * 1000.0,
        memory_mb: result.memory_mb,
        disk_mb: 0.0,
        duration_secs: result.build_time_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_hnsw() {
        let config = FaissVsHnswConfig {
            dimensions: 64,
            num_vectors: 500,
            num_queries: 10,
            k: 5,
            warmup_iterations: 2,
        };

        let result = bench_hnsw(&config);

        assert_eq!(result.index_name, "HNSW");
        assert!(result.insert_throughput > 0.0);
        assert!(result.search_latency_p50_ms > 0.0);
        assert!(result.recall_at_k >= 0.0 && result.recall_at_k <= 1.0);
    }

    #[test]
    fn test_bench_faiss_stub() {
        // This tests the stub when FAISS is not enabled
        let config = FaissVsHnswConfig {
            dimensions: 64,
            num_vectors: 100,
            num_queries: 5,
            k: 5,
            warmup_iterations: 1,
        };

        let result = bench_faiss("Flat", &config);

        // When FAISS is not enabled, we get a stub result
        #[cfg(not(feature = "faiss"))]
        {
            assert!(result.index_name.contains("disabled"));
            assert_eq!(result.insert_throughput, 0.0);
        }

        #[cfg(feature = "faiss")]
        {
            assert!(result.insert_throughput > 0.0);
        }
    }

    #[test]
    fn test_generate_vector_deterministic() {
        let v1 = generate_vector(128, 42);
        let v2 = generate_vector(128, 42);
        let v3 = generate_vector(128, 43);

        assert_eq!(v1.len(), 128);
        assert_eq!(v1, v2); // Same seed = same vector
        assert_ne!(v1, v3); // Different seed = different vector
    }

    #[test]
    fn test_print_comparison() {
        let results = vec![
            IndexBenchResult {
                index_name: "HNSW".to_string(),
                insert_throughput: 50000.0,
                search_latency_p50_ms: 1.5,
                search_latency_p99_ms: 5.0,
                memory_mb: 100.0,
                recall_at_k: 0.95,
                build_time_secs: 2.0,
            },
            IndexBenchResult {
                index_name: "FAISS-Flat".to_string(),
                insert_throughput: 100000.0,
                search_latency_p50_ms: 10.0,
                search_latency_p99_ms: 15.0,
                memory_mb: 80.0,
                recall_at_k: 1.0,
                build_time_secs: 0.0,
            },
        ];

        // Just verify it doesn't panic
        print_comparison(&results);
    }
}

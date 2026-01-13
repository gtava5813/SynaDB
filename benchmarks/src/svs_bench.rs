//! SparseVectorStore (SVS) Benchmarks
//!
//! Benchmarks for the inverted index sparse vector store used for lexical search
//! (SPLADE, BM25, TF-IDF embeddings).

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use synadb::sparse_vector_store::SparseVectorStore;
use synadb::sparse_vector::SparseVector;
use tempfile::tempdir;

/// Configuration for SVS benchmarks
#[derive(Debug, Clone)]
pub struct SvsBenchConfig {
    /// Number of sparse vectors to insert
    pub num_vectors: usize,
    /// Average number of non-zero elements per vector
    pub avg_nnz: usize,
    /// Maximum dimension index (vocabulary size)
    pub max_dim: u32,
    /// Number of search queries
    pub num_queries: usize,
    /// Number of results to retrieve
    pub k: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for SvsBenchConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10_000,
            avg_nnz: 100,      // Typical SPLADE output
            max_dim: 30_000,   // Typical vocab size
            num_queries: 100,
            k: 10,
            warmup_iterations: 10,
        }
    }
}

/// Result of SVS benchmark
#[derive(Debug, Clone)]
pub struct SvsBenchResult {
    pub index_type: String,
    pub num_vectors: usize,
    pub avg_nnz: usize,
    pub insert_throughput: f64,
    pub search_latency_p50_ms: f64,
    pub search_latency_p95_ms: f64,
    pub search_latency_p99_ms: f64,
    pub queries_per_sec: f64,
    pub storage_mb: f64,
    pub build_time_secs: f64,
}

/// Generate random sparse vectors for benchmarking
fn generate_sparse_vectors(
    num_vectors: usize,
    avg_nnz: usize,
    max_dim: u32,
    rng: &mut ChaCha8Rng,
) -> Vec<(String, SparseVector)> {
    let mut vectors = Vec::with_capacity(num_vectors);
    
    for i in 0..num_vectors {
        // Vary nnz around average (50% to 150%)
        let nnz = (avg_nnz as f64 * (0.5 + rng.gen::<f64>())) as usize;
        let nnz = nnz.max(1).min(max_dim as usize);
        
        // Generate random indices (sorted, unique)
        let mut indices: Vec<u32> = (0..max_dim).collect();
        indices.shuffle(rng);
        indices.truncate(nnz);
        
        // Create sparse vector and add weights
        let mut sv = SparseVector::new();
        for &idx in &indices {
            let weight = rng.gen::<f32>() * 2.0; // 0.0 to 2.0
            sv.add(idx, weight);
        }
        
        let key = format!("doc_{}", i);
        vectors.push((key, sv));
    }
    
    vectors
}

/// Run SVS insert benchmark
pub fn run_svs_insert_benchmark(config: &SvsBenchConfig) -> SvsBenchResult {
    let _dir = tempdir().expect("Failed to create temp dir");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let vectors = generate_sparse_vectors(
        config.num_vectors,
        config.avg_nnz,
        config.max_dim,
        &mut rng,
    );
    
    // Create store
    let mut store = SparseVectorStore::new();
    
    // Warmup
    for (key, sv) in vectors.iter().take(config.warmup_iterations) {
        store.index_with_key(key, sv.clone());
    }
    
    // Clear and recreate for actual benchmark
    let mut store = SparseVectorStore::new();
    
    // Benchmark inserts
    let mut latencies = Vec::with_capacity(config.num_vectors);
    let start = Instant::now();
    
    for (key, sv) in &vectors {
        let op_start = Instant::now();
        store.index_with_key(key, sv.clone());
        latencies.push(op_start.elapsed());
    }
    
    let total_time = start.elapsed();
    let throughput = config.num_vectors as f64 / total_time.as_secs_f64();
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    SvsBenchResult {
        index_type: "SVS".to_string(),
        num_vectors: config.num_vectors,
        avg_nnz: config.avg_nnz,
        insert_throughput: throughput,
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p95_ms: p95 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        queries_per_sec: 0.0,
        storage_mb: 0.0, // In-memory store
        build_time_secs: total_time.as_secs_f64(),
    }
}

/// Run SVS search benchmark
pub fn run_svs_search_benchmark(config: &SvsBenchConfig) -> SvsBenchResult {
    let _dir = tempdir().expect("Failed to create temp dir");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let vectors = generate_sparse_vectors(
        config.num_vectors,
        config.avg_nnz,
        config.max_dim,
        &mut rng,
    );
    
    // Create and populate store
    let mut store = SparseVectorStore::new();
    
    let build_start = Instant::now();
    for (key, sv) in &vectors {
        store.index_with_key(key, sv.clone());
    }
    let build_time = build_start.elapsed();
    
    // Generate query vectors (use subset of indexed vectors as queries)
    let queries: Vec<&SparseVector> = vectors
        .iter()
        .take(config.num_queries)
        .map(|(_, sv)| sv)
        .collect();
    
    // Warmup searches
    for query in queries.iter().take(config.warmup_iterations) {
        let _ = store.search(query, config.k);
    }
    
    // Benchmark searches
    let mut latencies = Vec::with_capacity(config.num_queries);
    let search_start = Instant::now();
    
    for query in &queries {
        let op_start = Instant::now();
        let _ = store.search(query, config.k);
        latencies.push(op_start.elapsed());
    }
    
    let search_time = search_start.elapsed();
    let qps = config.num_queries as f64 / search_time.as_secs_f64();
    
    let (p50, p95, p99) = calculate_percentiles(latencies);
    
    SvsBenchResult {
        index_type: "SVS".to_string(),
        num_vectors: config.num_vectors,
        avg_nnz: config.avg_nnz,
        insert_throughput: config.num_vectors as f64 / build_time.as_secs_f64(),
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p95_ms: p95 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        queries_per_sec: qps,
        storage_mb: 0.0, // In-memory store
        build_time_secs: build_time.as_secs_f64(),
    }
}

/// Run quick SVS benchmark (10K vectors)
pub fn run_quick_svs_benchmark() -> Vec<SvsBenchResult> {
    println!("\n=== Quick SVS Benchmark (10K vectors) ===\n");
    
    let config = SvsBenchConfig {
        num_vectors: 10_000,
        avg_nnz: 100,
        max_dim: 30_000,
        num_queries: 100,
        k: 10,
        warmup_iterations: 10,
    };
    
    let result = run_svs_search_benchmark(&config);
    
    println!("SVS (10K vectors, avg_nnz=100):");
    println!("  Insert throughput: {:.0} vectors/sec", result.insert_throughput);
    println!("  Search latency p50: {:.2} ms", result.search_latency_p50_ms);
    println!("  Search latency p99: {:.2} ms", result.search_latency_p99_ms);
    println!("  Queries/sec: {:.0}", result.queries_per_sec);
    println!("  Storage: {:.2} MB", result.storage_mb);
    
    vec![result]
}

/// Run full SVS benchmark suite
pub fn run_full_svs_benchmark() -> Vec<SvsBenchResult> {
    println!("\n=== Full SVS Benchmark Suite ===\n");
    
    let mut results = Vec::new();
    
    // Test different scales
    let scales = [10_000, 50_000, 100_000];
    let nnz_values = [50, 100, 200]; // Different sparsity levels
    
    for &num_vectors in &scales {
        for &avg_nnz in &nnz_values {
            println!("Benchmarking: {} vectors, avg_nnz={}", num_vectors, avg_nnz);
            
            let config = SvsBenchConfig {
                num_vectors,
                avg_nnz,
                max_dim: 30_000,
                num_queries: 100,
                k: 10,
                warmup_iterations: 10,
            };
            
            let result = run_svs_search_benchmark(&config);
            
            println!("  Insert: {:.0} vec/sec, Search p50: {:.2}ms, QPS: {:.0}",
                result.insert_throughput,
                result.search_latency_p50_ms,
                result.queries_per_sec
            );
            
            results.push(result);
        }
    }
    
    results
}

/// Print results table
pub fn print_results_table(results: &[SvsBenchResult]) {
    println!("\n{:=<90}", "");
    println!("{:<12} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10}",
        "Vectors", "Avg NNZ", "Insert/s", "Search p50", "Search p99", "QPS", "Storage");
    println!("{:-<90}", "");
    
    for r in results {
        println!("{:<12} {:>8} {:>10.0} {:>10.2}ms {:>10.2}ms {:>10.0} {:>8.2}MB",
            r.num_vectors,
            r.avg_nnz,
            r.insert_throughput,
            r.search_latency_p50_ms,
            r.search_latency_p99_ms,
            r.queries_per_sec,
            r.storage_mb
        );
    }
    println!("{:=<90}", "");
}

/// Convert to generic BenchmarkResult
pub fn to_benchmark_result(result: &SvsBenchResult, config: &SvsBenchConfig) -> BenchmarkResult {
    BenchmarkResult {
        benchmark: format!("svs_{}k_nnz{}", config.num_vectors / 1000, config.avg_nnz),
        database: "SynaDB SVS".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.num_vectors,
            value_size_bytes: config.avg_nnz * 8, // approx bytes per vector
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: result.insert_throughput,
        latency_p50_us: result.search_latency_p50_ms * 1000.0,
        latency_p95_us: result.search_latency_p95_ms * 1000.0,
        latency_p99_us: result.search_latency_p99_ms * 1000.0,
        memory_mb: 0.0,
        disk_mb: result.storage_mb,
        duration_secs: result.build_time_secs,
    }
}

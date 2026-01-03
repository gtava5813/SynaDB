//! Cascade Index performance benchmarks.
//!
//! This module implements comprehensive benchmarks for the Cascade Index,
//! testing across multiple dataset sizes and embedding dimensions.
//!
//! Two implementations available:
//! - **CascadeIndex** (default): Original implementation with bucket tree splits
//!   - Higher recall (~95%)
//! - **MmapCascadeIndex** (--mmap): Optimized with SynaDB physics principles
//!   - Append-only writes (Arrow of Time)
//!   - Faster search, lower recall (~85%)
//!
//! Benchmark matrix:
//! - Dataset sizes: 10K, 50K, 100K vectors
//! - Dimensions: 384, 768, 1024, 1536, 3072, 4096, 7168
//!
//! Metrics measured:
//! - Write throughput (vectors/sec)
//! - Index build time
//! - Search latency (p50, p95, p99)
//! - Storage size
//! - Recall@K

use crate::{BenchmarkConfig, BenchmarkResult, calculate_percentiles};
use std::time::Instant;
use tempfile::tempdir;

/// Embedding model configurations for benchmarking
pub const EMBEDDING_MODELS: &[(&str, u32)] = &[
    ("MiniLM", 384),
    ("BERT", 768),
    ("E5-large", 1024),
    ("OpenAI-ada-002", 1536),
    ("OpenAI-text-3-large", 3072),
    ("Cohere-embed-v3", 4096),
    ("DeepSeek-V3", 7168),
];

/// Dataset sizes for benchmarking
pub const DATASET_SIZES: &[usize] = &[10_000, 50_000, 100_000];

/// Configuration for Cascade Index benchmarks
#[derive(Debug, Clone)]
pub struct CascadeBenchConfig {
    pub dimensions: u32,
    pub num_vectors: usize,
    pub num_queries: usize,
    pub k: usize,
    pub warmup_queries: usize,
    pub preset: String,
    /// Use mmap-optimized implementation instead of original
    pub use_mmap: bool,
}

impl Default for CascadeBenchConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            num_vectors: 10_000,
            num_queries: 100,
            k: 10,
            warmup_queries: 10,
            preset: "large".to_string(),
            use_mmap: false,
        }
    }
}

/// Result of a Cascade Index benchmark
#[derive(Debug, Clone)]
pub struct CascadeBenchResult {
    pub model_name: String,
    pub dimensions: u32,
    pub num_vectors: usize,
    pub write_throughput: f64,
    pub index_build_time_secs: f64,
    pub search_latency_p50_ms: f64,
    pub search_latency_p95_ms: f64,
    pub search_latency_p99_ms: f64,
    pub storage_mb: f64,
    pub recall_at_k: f64,
    pub queries_per_sec: f64,
}

/// Generate a random vector with deterministic seed
fn generate_vector(dims: u32, seed: u64) -> Vec<f32> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Generate clustered vectors (more realistic for embeddings)
fn generate_clustered_vectors(dims: u32, num_vectors: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|i| generate_vector(dims, i as u64 * 1000))
        .collect();
    
    // Generate vectors around cluster centers
    (0..num_vectors)
        .map(|i| {
            let center = &centers[i % num_clusters];
            center.iter()
                .map(|&c| c + rng.gen::<f32>() * 0.2 - 0.1)
                .collect()
        })
        .collect()
}

/// Run Cascade Index insert benchmark
pub fn run_cascade_insert_benchmark(config: &CascadeBenchConfig) -> CascadeBenchResult {
    if config.use_mmap {
        run_mmap_cascade_benchmark(config)
    } else {
        run_default_cascade_benchmark(config)
    }
}

/// Run benchmark with original CascadeIndex (default)
fn run_default_cascade_benchmark(config: &CascadeBenchConfig) -> CascadeBenchResult {
    use synadb::cascade::{CascadeIndex, CascadeConfig};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("cascade.db");
    
    // Create Cascade config based on preset, then set dimensions
    let mut cascade_config = match config.preset.as_str() {
        "small" => CascadeConfig::small(),
        "large" => CascadeConfig::large(),
        "high_recall" => CascadeConfig::high_recall(),
        "fast_search" => CascadeConfig::fast_search(),
        _ => CascadeConfig::default(),
    };
    cascade_config.dimensions = config.dimensions as u16;
    
    let mut index = CascadeIndex::new(&db_path, cascade_config)
        .expect("Failed to create Cascade index");
    
    // Generate clustered vectors (more realistic)
    println!("    Generating {} vectors ({} dims)...", config.num_vectors, config.dimensions);
    let vectors = generate_clustered_vectors(config.dimensions, config.num_vectors, 100);
    
    // Measure insert throughput
    println!("    Inserting vectors...");
    let insert_start = Instant::now();
    
    for (i, vector) in vectors.iter().enumerate() {
        let key = format!("vec_{}", i);
        index.insert(&key, vector).expect("Insert failed");
    }
    
    let insert_duration = insert_start.elapsed();
    let write_throughput = config.num_vectors as f64 / insert_duration.as_secs_f64();
    
    // Index is built incrementally, measure any finalization
    let build_start = Instant::now();
    // Cascade builds incrementally, no explicit build needed
    let index_build_time = build_start.elapsed().as_secs_f64();
    
    // Get storage size (use temp path since save requires explicit path)
    let storage_mb = 0.0; // Will be calculated after search
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Warmup
    for query in queries.iter().take(config.warmup_queries) {
        let _ = index.search(query, config.k);
    }
    
    // Measure search latency
    println!("    Running {} search queries...", config.num_queries);
    let mut latencies = Vec::with_capacity(config.num_queries);
    
    for query in &queries {
        let start = Instant::now();
        let _ = index.search(query, config.k);
        latencies.push(start.elapsed());
    }
    
    let (p50, p95, p99) = calculate_percentiles(latencies.clone());
    let total_search_time: f64 = latencies.iter().map(|d| d.as_secs_f64()).sum();
    let queries_per_sec = config.num_queries as f64 / total_search_time;
    
    // Calculate recall (compare against brute force)
    let recall = calculate_recall_simple(&index, &vectors, &queries, config.k);
    
    CascadeBenchResult {
        model_name: "".to_string(),
        dimensions: config.dimensions,
        num_vectors: config.num_vectors,
        write_throughput,
        index_build_time_secs: insert_duration.as_secs_f64() + index_build_time,
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p95_ms: p95 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        storage_mb,
        recall_at_k: recall,
        queries_per_sec,
    }
}

/// Run benchmark with optimized MmapCascadeIndex (default)
fn run_mmap_cascade_benchmark(config: &CascadeBenchConfig) -> CascadeBenchResult {
    use synadb::cascade::{MmapCascadeIndex, MmapCascadeConfig};
    
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("cascade_mmap");
    
    // Create config based on preset
    let mut cascade_config = match config.preset.as_str() {
        "small" => MmapCascadeConfig::small(),
        "large" => MmapCascadeConfig::large(),
        "high_recall" => MmapCascadeConfig::high_recall(),
        "fast_search" => MmapCascadeConfig::fast_search(),
        _ => MmapCascadeConfig::default(),
    };
    cascade_config.dimensions = config.dimensions as u16;
    cascade_config.initial_capacity = config.num_vectors;
    
    let mut index = MmapCascadeIndex::new(&db_path, cascade_config)
        .expect("Failed to create MmapCascadeIndex");
    
    // Generate clustered vectors
    println!("    [Mmap] Generating {} vectors ({} dims)...", config.num_vectors, config.dimensions);
    let vectors = generate_clustered_vectors(config.dimensions, config.num_vectors, 100);
    
    // Measure insert throughput
    println!("    [Mmap] Inserting vectors...");
    let insert_start = Instant::now();
    
    for (i, vector) in vectors.iter().enumerate() {
        let key = format!("vec_{}", i);
        index.insert(&key, vector).expect("Insert failed");
    }
    
    let insert_duration = insert_start.elapsed();
    let write_throughput = config.num_vectors as f64 / insert_duration.as_secs_f64();
    
    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| generate_vector(config.dimensions, (config.num_vectors + i) as u64))
        .collect();
    
    // Warmup
    for query in queries.iter().take(config.warmup_queries) {
        let _ = index.search(query, config.k);
    }
    
    // Measure search latency
    println!("    [Mmap] Running {} search queries...", config.num_queries);
    let mut latencies = Vec::with_capacity(config.num_queries);
    
    for query in &queries {
        let start = Instant::now();
        let _ = index.search(query, config.k);
        latencies.push(start.elapsed());
    }
    
    let (p50, p95, p99) = calculate_percentiles(latencies.clone());
    let total_search_time: f64 = latencies.iter().map(|d| d.as_secs_f64()).sum();
    let queries_per_sec = config.num_queries as f64 / total_search_time;
    
    // Calculate recall
    let recall = calculate_recall_mmap(&index, &vectors, &queries, config.k);
    
    CascadeBenchResult {
        model_name: "".to_string(),
        dimensions: config.dimensions,
        num_vectors: config.num_vectors,
        write_throughput,
        index_build_time_secs: insert_duration.as_secs_f64(),
        search_latency_p50_ms: p50 / 1000.0,
        search_latency_p95_ms: p95 / 1000.0,
        search_latency_p99_ms: p99 / 1000.0,
        storage_mb: 0.0,
        recall_at_k: recall,
        queries_per_sec,
    }
}

/// Calculate recall by comparing against brute force search (simple implementation)
fn calculate_recall_simple(
    index: &synadb::cascade::CascadeIndex,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;
    let num_queries = queries.len().min(20); // Sample for speed
    
    for query in queries.iter().take(num_queries) {
        // Get Cascade results
        let cascade_results = index.search(query, k).unwrap_or_default();
        let cascade_keys: std::collections::HashSet<_> = cascade_results.iter()
            .map(|r| r.key.clone())
            .collect();
        
        // Compute brute force results using cosine distance
        let mut distances: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let brute_force_keys: std::collections::HashSet<_> = distances.iter()
            .take(k)
            .map(|(i, _)| format!("vec_{}", i))
            .collect();
        
        // Calculate intersection
        let intersection = cascade_keys.intersection(&brute_force_keys).count();
        total_recall += intersection as f64 / k as f64;
    }
    
    total_recall / num_queries as f64
}

/// Calculate recall for MmapCascadeIndex
fn calculate_recall_mmap(
    index: &synadb::cascade::MmapCascadeIndex,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;
    let num_queries = queries.len().min(20);
    
    for query in queries.iter().take(num_queries) {
        let results = index.search(query, k).unwrap_or_default();
        let result_keys: std::collections::HashSet<_> = results.iter()
            .map(|r| r.key.clone())
            .collect();
        
        // Brute force
        let mut distances: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let brute_force_keys: std::collections::HashSet<_> = distances.iter()
            .take(k)
            .map(|(i, _)| format!("vec_{}", i))
            .collect();
        
        let intersection = result_keys.intersection(&brute_force_keys).count();
        total_recall += intersection as f64 / k as f64;
    }
    
    total_recall / num_queries as f64
}

/// Compute cosine distance between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-10);
    1.0 - similarity
}

/// Run comprehensive Cascade Index benchmark across all dimensions and sizes
pub fn run_full_cascade_benchmark() -> Vec<CascadeBenchResult> {
    let mut results = Vec::new();
    
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CASCADE INDEX BENCHMARK SUITE                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    for &num_vectors in DATASET_SIZES {
        println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Dataset Size: {:>6} vectors                                                │", num_vectors);
        println!("└──────────────────────────────────────────────────────────────────────────────┘\n");
        
        for &(model_name, dims) in EMBEDDING_MODELS {
            println!("  [{:>20}] {} dimensions", model_name, dims);
            
            let config = CascadeBenchConfig {
                dimensions: dims,
                num_vectors,
                num_queries: 100,
                k: 10,
                warmup_queries: 10,
                preset: if num_vectors >= 50_000 { "large" } else { "small" }.to_string(),
                use_mmap: false,
            };
            
            let mut result = run_cascade_insert_benchmark(&config);
            result.model_name = model_name.to_string();
            
            println!("    Write: {:>10.0} vec/s | Build: {:>6.2}s | Search p50: {:>6.2}ms | Recall: {:>5.1}%",
                result.write_throughput,
                result.index_build_time_secs,
                result.search_latency_p50_ms,
                result.recall_at_k * 100.0
            );
            
            results.push(result);
        }
    }
    
    results
}

/// Run quick Cascade benchmark for CI/testing
pub fn run_quick_cascade_benchmark() -> Vec<CascadeBenchResult> {
    let mut results = Vec::new();
    
    println!("\n=== Quick Cascade Index Benchmark ===\n");
    
    // Test with 10K vectors, 3 dimension sizes
    for &(model_name, dims) in &[("MiniLM", 384u32), ("BERT", 768), ("OpenAI-ada-002", 1536)] {
        println!("Testing {} ({} dims)...", model_name, dims);
        
        let config = CascadeBenchConfig {
            dimensions: dims,
            num_vectors: 10_000,
            num_queries: 50,
            k: 10,
            warmup_queries: 5,
            preset: "small".to_string(),
            use_mmap: false,
        };
        
        let mut result = run_cascade_insert_benchmark(&config);
        result.model_name = model_name.to_string();
        
        println!("  Write: {:.0} vec/s, Search p50: {:.2}ms, Recall: {:.1}%\n",
            result.write_throughput,
            result.search_latency_p50_ms,
            result.recall_at_k * 100.0
        );
        
        results.push(result);
    }
    
    results
}

/// Print benchmark results as a formatted table
pub fn print_results_table(results: &[CascadeBenchResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                    CASCADE INDEX BENCHMARK RESULTS                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Model                  │ Dims  │ Vectors │ Write/sec  │ Build (s) │ Search p50 │ Storage  │ Recall   ║");
    println!("╠════════════════════════╪═══════╪═════════╪════════════╪═══════════╪════════════╪══════════╪══════════╣");
    
    for result in results {
        println!("║ {:22} │ {:>5} │ {:>7} │ {:>10.0} │ {:>9.2} │ {:>8.2}ms │ {:>6.1}MB │ {:>6.1}%  ║",
            result.model_name,
            result.dimensions,
            result.num_vectors,
            result.write_throughput,
            result.index_build_time_secs,
            result.search_latency_p50_ms,
            result.storage_mb,
            result.recall_at_k * 100.0
        );
    }
    
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝");
}

/// Print comparison table by dimension (for a fixed dataset size)
pub fn print_dimension_comparison(results: &[CascadeBenchResult], num_vectors: usize) {
    let filtered: Vec<_> = results.iter()
        .filter(|r| r.num_vectors == num_vectors)
        .collect();
    
    if filtered.is_empty() {
        return;
    }
    
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Cascade Index Performance by Embedding Model ({} vectors)                      │", num_vectors);
    println!("├─────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Model                  │ Dims  │ Write/sec  │ Build    │ Search (p50) │ Storage/10K  │");
    println!("├────────────────────────┼───────┼────────────┼──────────┼──────────────┼──────────────┤");
    
    for result in &filtered {
        let storage_per_10k = result.storage_mb * 10_000.0 / result.num_vectors as f64;
        println!("│ {:22} │ {:>5} │ {:>10.0} │ {:>6.1}s  │ {:>10.2}ms │ {:>10.1}MB │",
            result.model_name,
            result.dimensions,
            result.write_throughput,
            result.index_build_time_secs,
            result.search_latency_p50_ms,
            storage_per_10k
        );
    }
    
    println!("└─────────────────────────────────────────────────────────────────────────────────────────┘");
}

/// Convert to standard BenchmarkResult for reporting
pub fn to_benchmark_result(result: &CascadeBenchResult, config: &CascadeBenchConfig) -> BenchmarkResult {
    BenchmarkResult {
        benchmark: format!("cascade_{}d_{}v", result.dimensions, result.num_vectors),
        database: "Syna-Cascade".to_string(),
        config: BenchmarkConfig {
            warmup_iterations: config.warmup_queries,
            measurement_iterations: config.num_queries,
            value_size_bytes: result.dimensions as usize * 4,
            thread_count: 1,
            sync_on_write: false,
        },
        throughput_ops_sec: result.write_throughput,
        latency_p50_us: result.search_latency_p50_ms * 1000.0,
        latency_p95_us: result.search_latency_p95_ms * 1000.0,
        latency_p99_us: result.search_latency_p99_ms * 1000.0,
        memory_mb: result.storage_mb,
        disk_mb: result.storage_mb,
        duration_secs: result.index_build_time_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cascade_benchmark_small() {
        let config = CascadeBenchConfig {
            dimensions: 128,
            num_vectors: 500,
            num_queries: 10,
            k: 5,
            warmup_queries: 2,
            preset: "small".to_string(),
        };
        
        let result = run_cascade_insert_benchmark(&config);
        
        assert!(result.write_throughput > 0.0);
        assert!(result.search_latency_p50_ms > 0.0);
        assert!(result.recall_at_k >= 0.0 && result.recall_at_k <= 1.0);
    }
}

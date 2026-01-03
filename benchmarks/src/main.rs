//! Syna Benchmark Suite
//!
//! Comprehensive benchmarks comparing Syna against other embedded databases.

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::time::Duration;

mod config;
mod report;
mod write_bench;
mod read_bench;
mod mixed_bench;
mod storage_bench;
mod competitors;
mod reproducibility;
mod vector_bench;
mod tensor_bench;
mod faiss_bench;
mod cascade_bench;

pub use config::*;
pub use report::*;
pub use reproducibility::*;

/// Syna Benchmark Suite
#[derive(Parser)]
#[command(name = "SYNA_bench")]
#[command(about = "Benchmark suite for Syna database")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run write performance benchmarks
    Write {
        /// Number of warmup iterations
        #[arg(long, default_value = "1000")]
        warmup: usize,
        
        /// Number of measurement iterations
        #[arg(long, default_value = "10000")]
        iterations: usize,
        
        /// Value sizes to test (comma-separated)
        #[arg(long, default_value = "64,1024,65536")]
        sizes: String,
        
        /// Compare sync vs async writes
        #[arg(long, default_value = "false")]
        compare_sync: bool,
    },
    
    /// Run read performance benchmarks
    Read {
        /// Number of warmup iterations
        #[arg(long, default_value = "1000")]
        warmup: usize,
        
        /// Number of measurement iterations
        #[arg(long, default_value = "10000")]
        iterations: usize,
        
        /// Number of threads to test
        #[arg(long, default_value = "1,4,8")]
        threads: String,
    },
    
    /// Run mixed workload benchmarks (YCSB)
    Mixed {
        /// Workload type: A, B, C, D, F, or timeseries
        #[arg(long, default_value = "A")]
        workload: String,
        
        /// Number of operations
        #[arg(long, default_value = "100000")]
        operations: usize,
    },
    
    /// Run storage efficiency benchmarks
    Storage {
        /// Number of entries to write
        #[arg(long, default_value = "100000")]
        entries: usize,
    },
    
    /// Run vector store benchmarks (insert and search)
    Vector {
        /// Vector dimensions
        #[arg(long, default_value = "768")]
        dimensions: u16,
        
        /// Number of vectors to insert
        #[arg(long, default_value = "10000")]
        num_vectors: usize,
        
        /// Number of search queries
        #[arg(long, default_value = "100")]
        num_queries: usize,
        
        /// Number of nearest neighbors to retrieve
        #[arg(long, default_value = "10")]
        k: usize,
        
        /// Run full benchmark suite
        #[arg(long, default_value = "false")]
        full: bool,
    },
    
    /// Run tensor engine benchmarks (batch operations)
    /// NOTE: Current TensorEngine stores elements individually (O(n) writes).
    /// Use --full to see blob-based throughput potential.
    Tensor {
        /// Number of tensor elements (default 1000 for per-element storage)
        #[arg(long, default_value = "1000")]
        num_elements: usize,
        
        /// Number of iterations
        #[arg(long, default_value = "3")]
        iterations: usize,
        
        /// Run full benchmark suite including blob storage comparison
        #[arg(long, default_value = "false")]
        full: bool,
    },
    
    /// Run tensor throughput validation benchmark (2 GB/s target)
    /// Tests chunked, batched, and mmap storage methods at various sizes.
    Throughput {
        /// Run quick validation (10MB, 100MB only)
        #[arg(long, default_value = "false")]
        quick: bool,
    },
    
    /// Run FAISS vs HNSW comparison benchmark
    Faiss {
        /// Vector dimensions
        #[arg(long, default_value = "768")]
        dimensions: u16,
        
        /// Number of vectors to insert
        #[arg(long, default_value = "100000")]
        num_vectors: usize,
        
        /// Number of search queries
        #[arg(long, default_value = "100")]
        num_queries: usize,
        
        /// Number of nearest neighbors to retrieve
        #[arg(long, default_value = "10")]
        k: usize,
        
        /// Run quick benchmark (smaller dataset)
        #[arg(long, default_value = "false")]
        quick: bool,
        
        /// Run full benchmark suite (100K and 1M vectors)
        #[arg(long, default_value = "false")]
        full: bool,
    },
    
    /// Run Cascade Index benchmarks (LSH + bucket tree + graph)
    Cascade {
        /// Vector dimensions
        #[arg(long, default_value = "768")]
        dimensions: u32,
        
        /// Number of vectors to insert
        #[arg(long, default_value = "10000")]
        num_vectors: usize,
        
        /// Number of search queries
        #[arg(long, default_value = "100")]
        num_queries: usize,
        
        /// Number of nearest neighbors to retrieve
        #[arg(long, default_value = "10")]
        k: usize,
        
        /// Configuration preset: small, large, high_recall, fast_search
        #[arg(long, default_value = "large")]
        preset: String,
        
        /// Run quick benchmark (10K vectors, 3 dimensions)
        #[arg(long, default_value = "false")]
        quick: bool,
        
        /// Run full benchmark suite (10K, 50K, 100K × 7 dimensions)
        #[arg(long, default_value = "false")]
        full: bool,
        
        /// Use mmap-optimized implementation (faster search, lower recall)
        #[arg(long, default_value = "false")]
        mmap: bool,
    },
    
    /// Run all benchmarks and generate report
    All {
        /// Output directory for reports
        #[arg(long, default_value = "results")]
        output: String,
    },
}

/// Result of a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark: String,
    pub database: String,
    pub config: BenchmarkConfig,
    pub throughput_ops_sec: f64,
    pub latency_p50_us: f64,
    pub latency_p95_us: f64,
    pub latency_p99_us: f64,
    pub memory_mb: f64,
    pub disk_mb: f64,
    pub duration_secs: f64,
}

/// Configuration for a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub value_size_bytes: usize,
    pub thread_count: usize,
    pub sync_on_write: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 1000,
            measurement_iterations: 10000,
            value_size_bytes: 1024,
            thread_count: 1,
            sync_on_write: false,
        }
    }
}

/// Calculate latency percentiles from a sorted list of durations
pub fn calculate_percentiles(mut latencies: Vec<Duration>) -> (f64, f64, f64) {
    if latencies.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    latencies.sort();
    let len = latencies.len();
    
    let p50 = latencies[len * 50 / 100].as_secs_f64() * 1_000_000.0;
    let p95 = latencies[len * 95 / 100].as_secs_f64() * 1_000_000.0;
    let p99 = latencies[len * 99 / 100].as_secs_f64() * 1_000_000.0;
    
    (p50, p95, p99)
}

fn main() {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Write { warmup, iterations, sizes, compare_sync } => {
            let sizes: Vec<usize> = sizes
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            
            println!("Running write benchmarks...");
            println!("  Warmup: {} iterations", warmup);
            println!("  Measurement: {} iterations", iterations);
            println!("  Value sizes: {:?} bytes", sizes);
            
            for size in sizes {
                let config = BenchmarkConfig {
                    warmup_iterations: warmup,
                    measurement_iterations: iterations,
                    value_size_bytes: size,
                    ..Default::default()
                };
                
                let result = write_bench::run_SYNA_write(&config);
                println!("\nSyna ({}B values):", size);
                println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs", result.latency_p50_us);
                println!("  Latency p95: {:.1} μs", result.latency_p95_us);
                println!("  Latency p99: {:.1} μs", result.latency_p99_us);
                
                // Run sync vs async comparison if requested
                if compare_sync {
                    let (async_result, sync_result) = write_bench::run_sync_vs_async_comparison(&config);
                    write_bench::print_sync_comparison(&async_result, &sync_result);
                }
            }
        }
        
        Commands::Read { warmup, iterations, threads } => {
            let threads: Vec<usize> = threads
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            
            println!("Running read benchmarks...");
            println!("  Warmup: {} iterations", warmup);
            println!("  Measurement: {} iterations", iterations);
            println!("  Thread counts: {:?}", threads);
            
            for thread_count in threads {
                let config = BenchmarkConfig {
                    warmup_iterations: warmup,
                    measurement_iterations: iterations,
                    thread_count,
                    ..Default::default()
                };
                
                let result = read_bench::run_SYNA_read(&config);
                println!("\nSyna ({} threads):", thread_count);
                println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs", result.latency_p50_us);
            }
        }
        
        Commands::Mixed { workload, operations } => {
            println!("Running mixed workload benchmark...");
            println!("  Workload: {}", workload);
            println!("  Operations: {}", operations);
            
            let result = mixed_bench::run_ycsb(&workload, operations);
            println!("\nResults:");
            println!("  Throughput: {:.0} ops/sec", result.throughput_ops_sec);
        }
        
        Commands::Storage { entries } => {
            println!("Running storage efficiency benchmark...");
            println!("  Entries: {}", entries);
            
            // Run full storage suite
            storage_bench::run_full_storage_suite(entries);
        }
        
        Commands::Vector { dimensions, num_vectors, num_queries, k, full } => {
            println!("Running vector benchmarks...");
            
            if full {
                // Run full vector benchmark suite
                let results = vector_bench::run_all_vector_benchmarks();
                println!("\n=== Vector Benchmark Summary ===");
                println!("Total benchmarks run: {}", results.len());
            } else {
                // Run specific configuration
                let config = vector_bench::VectorBenchConfig {
                    dimensions,
                    num_vectors,
                    num_queries,
                    k,
                    warmup_iterations: 100,
                };
                
                println!("  Dimensions: {}", dimensions);
                println!("  Vectors: {}", num_vectors);
                println!("  Queries: {}", num_queries);
                println!("  K: {}", k);
                
                println!("\n--- Vector Insert Benchmark ---");
                let insert_result = vector_bench::run_vector_insert_benchmark(&config);
                println!("  Throughput: {:.0} ops/sec", insert_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs", insert_result.latency_p50_us);
                println!("  Latency p99: {:.1} μs", insert_result.latency_p99_us);
                
                // Check target: 100K/sec for 768-dim
                if dimensions == 768 && insert_result.throughput_ops_sec >= 100_000.0 {
                    println!("  ✓ Target met: >= 100K vectors/sec");
                } else if dimensions == 768 {
                    println!("  ✗ Target NOT met: < 100K vectors/sec");
                }
                
                println!("\n--- Vector Search Benchmark (Brute Force) ---");
                let search_bf_result = vector_bench::run_vector_search_brute_force_benchmark(&config);
                println!("  Throughput: {:.0} queries/sec", search_bf_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs ({:.2} ms)", search_bf_result.latency_p50_us, search_bf_result.latency_p50_us / 1000.0);
                println!("  Latency p99: {:.1} μs ({:.2} ms)", search_bf_result.latency_p99_us, search_bf_result.latency_p99_us / 1000.0);
                
                println!("\n--- Vector Search Benchmark (HNSW) ---");
                let search_hnsw_result = vector_bench::run_vector_search_hnsw_benchmark(&config);
                println!("  Throughput: {:.0} queries/sec", search_hnsw_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs ({:.2} ms)", search_hnsw_result.latency_p50_us, search_hnsw_result.latency_p50_us / 1000.0);
                println!("  Latency p99: {:.1} μs ({:.2} ms)", search_hnsw_result.latency_p99_us, search_hnsw_result.latency_p99_us / 1000.0);
                
                // Check target: <10ms for search
                if search_hnsw_result.latency_p50_us < 10_000.0 {
                    println!("  ✓ Target met: p50 latency < 10ms");
                } else {
                    println!("  ✗ Target NOT met: p50 latency >= 10ms");
                }
            }
        }
        
        Commands::Tensor { num_elements, iterations, full } => {
            println!("Running tensor benchmarks...");
            
            if full {
                // Run full tensor benchmark suite
                let results = tensor_bench::run_all_tensor_benchmarks();
                println!("\n=== Tensor Benchmark Summary ===");
                println!("Total benchmarks run: {}", results.len());
            } else {
                // Run specific configuration
                let config = tensor_bench::TensorBenchConfig {
                    num_elements,
                    dtype: synadb::tensor::DType::Float64,
                    iterations,
                    warmup_iterations: 2,
                };
                
                let data_size_mb = (num_elements * 8) as f64 / 1024.0 / 1024.0;
                println!("  Elements: {}", num_elements);
                println!("  Data size: {:.2} MB", data_size_mb);
                println!("  Iterations: {}", iterations);
                
                println!("\n--- Tensor Put Benchmark ---");
                let put_result = tensor_bench::run_tensor_put_benchmark(&config);
                println!("  Throughput: {:.2} MB/s", put_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs ({:.2} ms)", put_result.latency_p50_us, put_result.latency_p50_us / 1000.0);
                
                println!("\n--- Tensor Get Benchmark ---");
                let get_result = tensor_bench::run_tensor_get_benchmark(&config);
                println!("  Throughput: {:.2} MB/s", get_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs ({:.2} ms)", get_result.latency_p50_us, get_result.latency_p50_us / 1000.0);
                
                // Check target: 1 GB/s
                let throughput_gb_sec = get_result.throughput_ops_sec / 1024.0;
                if throughput_gb_sec >= 1.0 {
                    println!("  ✓ Target met: throughput >= 1 GB/s");
                } else {
                    println!("  ✗ Target NOT met: throughput < 1 GB/s ({:.3} GB/s)", throughput_gb_sec);
                }
                
                println!("\n--- Tensor Round-Trip Benchmark ---");
                let rt_result = tensor_bench::run_tensor_roundtrip_benchmark(&config);
                println!("  Throughput: {:.2} MB/s", rt_result.throughput_ops_sec);
                println!("  Latency p50: {:.1} μs ({:.2} ms)", rt_result.latency_p50_us, rt_result.latency_p50_us / 1000.0);
            }
        }
        
        Commands::Throughput { quick } => {
            println!("Running tensor throughput validation benchmark...");
            println!("Target: 2 GB/s for both read and write operations\n");
            
            if quick {
                // Quick validation with smaller sizes
                let results = tensor_bench::run_quick_throughput_validation();
                println!("\n=== Quick Throughput Validation Complete ===");
                println!("Total benchmarks run: {}", results.len());
            } else {
                // Full validation up to 1GB tensors
                let results = tensor_bench::run_tensor_throughput_comparison();
                println!("\n=== Full Throughput Validation Complete ===");
                println!("Total benchmarks run: {}", results.len());
            }
        }
        
        Commands::Faiss { dimensions, num_vectors, num_queries, k, quick, full } => {
            println!("Running FAISS vs HNSW benchmark...");
            
            if full {
                // Run full benchmark suite with 100K and 1M vectors
                faiss_bench::run_faiss_vs_hnsw_benchmark();
            } else if quick {
                // Run quick benchmark with smaller dataset
                let results = faiss_bench::run_quick_faiss_vs_hnsw_benchmark();
                println!("\n=== Quick FAISS vs HNSW Benchmark Complete ===");
                println!("Total index types benchmarked: {}", results.len());
            } else {
                // Run specific configuration
                let config = faiss_bench::FaissVsHnswConfig {
                    dimensions,
                    num_vectors,
                    num_queries,
                    k,
                    warmup_iterations: 10,
                };
                
                println!("  Dimensions: {}", dimensions);
                println!("  Vectors: {}", num_vectors);
                println!("  Queries: {}", num_queries);
                println!("  K: {}", k);
                
                let mut results = Vec::new();
                
                // Benchmark HNSW
                println!("\n--- HNSW Benchmark ---");
                let hnsw_result = faiss_bench::bench_hnsw(&config);
                println!("  Insert throughput: {:.0} vectors/sec", hnsw_result.insert_throughput);
                println!("  Search latency p50: {:.2} ms", hnsw_result.search_latency_p50_ms);
                println!("  Search latency p99: {:.2} ms", hnsw_result.search_latency_p99_ms);
                println!("  Memory usage: {:.1} MB", hnsw_result.memory_mb);
                println!("  Recall@{}: {:.1}%", k, hnsw_result.recall_at_k * 100.0);
                println!("  Build time: {:.2}s", hnsw_result.build_time_secs);
                results.push(hnsw_result);
                
                // Benchmark FAISS Flat
                println!("\n--- FAISS Flat Benchmark ---");
                let faiss_flat_result = faiss_bench::bench_faiss("Flat", &config);
                if faiss_flat_result.insert_throughput > 0.0 {
                    println!("  Insert throughput: {:.0} vectors/sec", faiss_flat_result.insert_throughput);
                    println!("  Search latency p50: {:.2} ms", faiss_flat_result.search_latency_p50_ms);
                    println!("  Search latency p99: {:.2} ms", faiss_flat_result.search_latency_p99_ms);
                    println!("  Memory usage: {:.1} MB", faiss_flat_result.memory_mb);
                    println!("  Recall@{}: {:.1}%", k, faiss_flat_result.recall_at_k * 100.0);
                }
                results.push(faiss_flat_result);
                
                // Benchmark FAISS IVF
                println!("\n--- FAISS IVF1024,Flat Benchmark ---");
                let faiss_ivf_result = faiss_bench::bench_faiss("IVF1024,Flat", &config);
                if faiss_ivf_result.insert_throughput > 0.0 {
                    println!("  Insert throughput: {:.0} vectors/sec", faiss_ivf_result.insert_throughput);
                    println!("  Search latency p50: {:.2} ms", faiss_ivf_result.search_latency_p50_ms);
                    println!("  Search latency p99: {:.2} ms", faiss_ivf_result.search_latency_p99_ms);
                    println!("  Memory usage: {:.1} MB", faiss_ivf_result.memory_mb);
                    println!("  Recall@{}: {:.1}%", k, faiss_ivf_result.recall_at_k * 100.0);
                }
                results.push(faiss_ivf_result);
                
                // Print comparison table
                faiss_bench::print_comparison(&results);
                
                // Check targets
                println!("\n=== Performance Target Check ===");
                let hnsw = &results[0];
                
                // Target: <10ms for search
                if hnsw.search_latency_p50_ms < 10.0 {
                    println!("  ✓ HNSW search latency target met: p50 < 10ms ({:.2}ms)", hnsw.search_latency_p50_ms);
                } else {
                    println!("  ✗ HNSW search latency target NOT met: p50 >= 10ms ({:.2}ms)", hnsw.search_latency_p50_ms);
                }
                
                // Target: >90% recall
                if hnsw.recall_at_k >= 0.9 {
                    println!("  ✓ HNSW recall target met: >= 90% ({:.1}%)", hnsw.recall_at_k * 100.0);
                } else {
                    println!("  ✗ HNSW recall target NOT met: < 90% ({:.1}%)", hnsw.recall_at_k * 100.0);
                }
            }
        }
        
        Commands::All { output } => {
            println!("Running full benchmark suite...");
            println!("  Output directory: {}", output);
            
            // Run all benchmarks and generate report
            let mut results = run_all_benchmarks();
            
            // Add vector benchmarks
            println!("\n=== Running Vector Benchmarks ===");
            results.extend(vector_bench::run_all_vector_benchmarks());
            
            // Add tensor benchmarks
            println!("\n=== Running Tensor Benchmarks ===");
            results.extend(tensor_bench::run_all_tensor_benchmarks());
            
            // Add FAISS vs HNSW benchmarks
            println!("\n=== Running FAISS vs HNSW Benchmarks ===");
            let faiss_results = faiss_bench::run_quick_faiss_vs_hnsw_benchmark();
            let faiss_config = faiss_bench::FaissVsHnswConfig::default();
            for result in &faiss_results {
                results.push(faiss_bench::to_benchmark_result(result, &faiss_config));
            }
            
            // Add Cascade Index benchmarks
            println!("\n=== Running Cascade Index Benchmarks ===");
            let cascade_results = cascade_bench::run_quick_cascade_benchmark();
            let cascade_config = cascade_bench::CascadeBenchConfig::default();
            for result in &cascade_results {
                results.push(cascade_bench::to_benchmark_result(result, &cascade_config));
            }
            
            report::generate_report(&results, &output);
            
            println!("\nReport generated in {}/", output);
        }
        
        Commands::Cascade { dimensions, num_vectors, num_queries, k, preset, quick, full, mmap } => {
            println!("Running Cascade Index benchmarks...");
            if mmap {
                println!("  Mode: Mmap (optimized, faster search)");
            } else {
                println!("  Mode: Default (original, higher recall)");
            }
            
            if full {
                // Run full benchmark suite across all dimensions and sizes
                let results = cascade_bench::run_full_cascade_benchmark();
                cascade_bench::print_results_table(&results);
                
                // Print per-size comparison tables
                for &size in cascade_bench::DATASET_SIZES {
                    cascade_bench::print_dimension_comparison(&results, size);
                }
                
                println!("\n=== Cascade Index Benchmark Complete ===");
                println!("Total configurations tested: {}", results.len());
            } else if quick {
                // Run quick benchmark
                let results = cascade_bench::run_quick_cascade_benchmark();
                cascade_bench::print_results_table(&results);
            } else {
                // Run specific configuration
                let config = cascade_bench::CascadeBenchConfig {
                    dimensions,
                    num_vectors,
                    num_queries,
                    k,
                    warmup_queries: 10,
                    preset,
                    use_mmap: mmap,
                };
                
                println!("  Dimensions: {}", dimensions);
                println!("  Vectors: {}", num_vectors);
                println!("  Queries: {}", num_queries);
                println!("  K: {}", k);
                println!("  Preset: {}", config.preset);
                
                let result = cascade_bench::run_cascade_insert_benchmark(&config);
                
                println!("\n=== Cascade Index Results ===");
                println!("  Write throughput: {:.0} vectors/sec", result.write_throughput);
                println!("  Index build time: {:.2}s", result.index_build_time_secs);
                println!("  Search latency p50: {:.2}ms", result.search_latency_p50_ms);
                println!("  Search latency p95: {:.2}ms", result.search_latency_p95_ms);
                println!("  Search latency p99: {:.2}ms", result.search_latency_p99_ms);
                println!("  Queries/sec: {:.0}", result.queries_per_sec);
                println!("  Storage: {:.2}MB", result.storage_mb);
                println!("  Recall@{}: {:.1}%", k, result.recall_at_k * 100.0);
                
                // Check targets
                println!("\n=== Performance Target Check ===");
                if result.search_latency_p50_ms < 10.0 {
                    println!("  ✓ Search latency target met: p50 < 10ms ({:.2}ms)", result.search_latency_p50_ms);
                } else {
                    println!("  ✗ Search latency target NOT met: p50 >= 10ms ({:.2}ms)", result.search_latency_p50_ms);
                }
                
                if result.recall_at_k >= 0.90 {
                    println!("  ✓ Recall target met: >= 90% ({:.1}%)", result.recall_at_k * 100.0);
                } else {
                    println!("  ✗ Recall target NOT met: < 90% ({:.1}%)", result.recall_at_k * 100.0);
                }
            }
        }
    }
}

fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // Write benchmarks
    for size in [64, 1024, 65536] {
        let config = BenchmarkConfig {
            value_size_bytes: size,
            ..Default::default()
        };
        results.push(write_bench::run_SYNA_write(&config));
    }
    
    // Read benchmarks
    for threads in [1, 4, 8] {
        let config = BenchmarkConfig {
            thread_count: threads,
            ..Default::default()
        };
        results.push(read_bench::run_SYNA_read(&config));
    }
    
    // Mixed workloads
    for workload in ["A", "B", "C"] {
        results.push(mixed_bench::run_ycsb(workload, 10000));
    }
    
    // Storage
    results.push(storage_bench::run_storage_bench(10000));
    
    results
}



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
            
            report::generate_report(&results, &output);
            
            println!("\nReport generated in {}/", output);
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



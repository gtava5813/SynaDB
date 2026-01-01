//! Comprehensive benchmarks for AI-Native features using Criterion.
//!
//! This module provides Criterion-based benchmarks for:
//! - Vector insert throughput across different dimensions
//! - Vector search performance (brute force and HNSW)
//! - Tensor operations throughput
//! - Model registry operations
//! - Experiment tracking operations
//!
//! **Feature: syna-ai-native, Task 28.1: Comprehensive Benchmark Suite**
//! **Validates: Requirements 9.1, 9.2**

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use synadb::distance::DistanceMetric;
use synadb::experiment::{ExperimentTracker, RunStatus};
use synadb::model_registry::ModelRegistry;
use synadb::tensor::{DType, TensorEngine};
use synadb::vector::{VectorConfig, VectorStore};
use synadb::SynaDB;
use tempfile::tempdir;

// ============================================================================
// Vector Benchmarks
// ============================================================================

/// Generate a deterministic random vector for benchmarking
fn generate_vector(dims: u16, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

/// Benchmark vector insert performance across different dimensions.
///
/// Target: 100K vectors/sec for 768-dim float32 vectors.
///
/// _Requirements: 9.1_
fn bench_vector_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insert");

    for dims in [128u16, 384, 768, 1536].iter() {
        // Set throughput to measure vectors per second
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("dims", dims), dims, |b, &dims| {
            let dir = tempdir().unwrap();
            let config = VectorConfig {
                dimensions: dims,
                metric: DistanceMetric::Cosine,
                index_threshold: usize::MAX, // Disable auto-indexing for pure insert benchmark
                ..Default::default()
            };
            let mut store = VectorStore::new(dir.path().join("bench.db"), config).unwrap();

            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let vector: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            let mut i = 0u64;

            b.iter(|| {
                store.insert(&format!("v{}", i), &vector).unwrap();
                i += 1;
            });
        });
    }
    group.finish();
}

/// Benchmark vector search performance with brute force.
///
/// _Requirements: 9.2_
fn bench_vector_search_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_brute_force");

    for n_vectors in [1000usize, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("vectors", n_vectors),
            n_vectors,
            |b, &n| {
                // Setup: insert n vectors
                let dir = tempdir().unwrap();
                let config = VectorConfig {
                    dimensions: 768,
                    metric: DistanceMetric::Cosine,
                    index_threshold: usize::MAX, // Force brute force
                    ..Default::default()
                };
                let mut store = VectorStore::new(dir.path().join("bench.db"), config).unwrap();

                // Insert vectors
                for i in 0..n {
                    let v = generate_vector(768, i as u64);
                    store.insert(&format!("v{}", i), &v).unwrap();
                }

                let query = generate_vector(768, 999999);

                b.iter(|| store.search(&query, 10).unwrap());
            },
        );
    }
    group.finish();
}

/// Benchmark vector search performance with HNSW index.
///
/// Target: <10ms for 1M vectors.
///
/// _Requirements: 9.2_
fn bench_vector_search_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_hnsw");
    // Allow longer measurement time for HNSW benchmarks
    group.measurement_time(std::time::Duration::from_secs(10));

    for n_vectors in [10000usize, 50000, 100000].iter() {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("vectors", n_vectors),
            n_vectors,
            |b, &n| {
                // Setup: insert n vectors and build HNSW index
                let dir = tempdir().unwrap();
                let config = VectorConfig {
                    dimensions: 768,
                    metric: DistanceMetric::Cosine,
                    index_threshold: 100, // Enable HNSW
                    ..Default::default()
                };
                let mut store = VectorStore::new(dir.path().join("bench.db"), config).unwrap();

                // Insert vectors
                for i in 0..n {
                    let v = generate_vector(768, i as u64);
                    store.insert(&format!("v{}", i), &v).unwrap();
                }

                // Build HNSW index
                store.build_index().unwrap();

                let query = generate_vector(768, 999999);

                b.iter(|| store.search(&query, 10).unwrap());
            },
        );
    }
    group.finish();
}

// ============================================================================
// Tensor Benchmarks
// ============================================================================

/// Generate deterministic tensor data
fn generate_tensor_data(num_elements: usize, seed: u64) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0u8; num_elements * 8]; // f64 = 8 bytes

    for i in 0..num_elements {
        let value: f64 = rng.gen::<f64>() * 100.0;
        let bytes = value.to_le_bytes();
        data[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }

    data
}

/// Benchmark tensor put operations (chunked storage).
///
/// Target: 1 GB/s throughput.
///
/// _Requirements: 9.3_
fn bench_tensor_put_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_put_chunked");

    // Test different tensor sizes: 1MB, 10MB
    for size_mb in [1usize, 10].iter() {
        let num_elements = size_mb * 1024 * 1024 / 8; // f64 elements
        let data_size = num_elements * 8;

        group.throughput(Throughput::Bytes(data_size as u64));

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_mb),
            &num_elements,
            |b, &num_elements| {
                let dir = tempdir().unwrap();
                let db = SynaDB::new(dir.path().join("tensor.db")).unwrap();
                let mut engine = TensorEngine::new(db);

                let data = generate_tensor_data(num_elements, 42);
                let shape = vec![num_elements];
                let mut i = 0u64;

                b.iter(|| {
                    let name = format!("tensor_{}", i);
                    engine
                        .put_tensor_chunked(&name, &data, &shape, DType::Float64)
                        .unwrap();
                    i += 1;
                });
            },
        );
    }
    group.finish();
}

/// Benchmark tensor get operations (chunked storage).
///
/// Target: 1 GB/s throughput.
///
/// _Requirements: 9.3_
fn bench_tensor_get_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_get_chunked");

    for size_mb in [1usize, 10].iter() {
        let num_elements = size_mb * 1024 * 1024 / 8;
        let data_size = num_elements * 8;

        group.throughput(Throughput::Bytes(data_size as u64));

        group.bench_with_input(
            BenchmarkId::new("size_mb", size_mb),
            &num_elements,
            |b, &num_elements| {
                let dir = tempdir().unwrap();
                let db = SynaDB::new(dir.path().join("tensor.db")).unwrap();
                let mut engine = TensorEngine::new(db);

                let data = generate_tensor_data(num_elements, 42);
                let shape = vec![num_elements];

                // Store the tensor first
                engine
                    .put_tensor_chunked("test_tensor", &data, &shape, DType::Float64)
                    .unwrap();

                b.iter(|| engine.get_tensor_chunked("test_tensor").unwrap());
            },
        );
    }
    group.finish();
}

// ============================================================================
// Model Registry Benchmarks
// ============================================================================

/// Benchmark model save operations.
///
/// _Requirements: 4.1, 4.2, 4.3_
fn bench_model_save(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_registry_save");

    // Test different model sizes: 1KB, 100KB, 1MB
    for size_kb in [1usize, 100, 1024].iter() {
        let model_size = size_kb * 1024;

        group.throughput(Throughput::Bytes(model_size as u64));

        group.bench_with_input(
            BenchmarkId::new("size_kb", size_kb),
            &model_size,
            |b, &model_size| {
                let dir = tempdir().unwrap();
                let mut registry = ModelRegistry::new(dir.path().join("models.db")).unwrap();

                // Generate model data
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let model_data: Vec<u8> = (0..model_size).map(|_| rng.gen()).collect();

                let mut metadata = HashMap::new();
                metadata.insert("accuracy".to_string(), "0.95".to_string());

                let mut i = 0u64;

                b.iter(|| {
                    let name = format!("model_{}", i);
                    registry.save_model(&name, &model_data, metadata.clone()).unwrap();
                    i += 1;
                });
            },
        );
    }
    group.finish();
}

/// Benchmark model load operations (with checksum verification).
///
/// _Requirements: 4.4_
fn bench_model_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_registry_load");

    for size_kb in [1usize, 100, 1024].iter() {
        let model_size = size_kb * 1024;

        group.throughput(Throughput::Bytes(model_size as u64));

        group.bench_with_input(
            BenchmarkId::new("size_kb", size_kb),
            &model_size,
            |b, &model_size| {
                let dir = tempdir().unwrap();
                let mut registry = ModelRegistry::new(dir.path().join("models.db")).unwrap();

                // Generate and save model data
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let model_data: Vec<u8> = (0..model_size).map(|_| rng.gen()).collect();

                let mut metadata = HashMap::new();
                metadata.insert("accuracy".to_string(), "0.95".to_string());

                registry
                    .save_model("test_model", &model_data, metadata)
                    .unwrap();

                b.iter(|| registry.load_model("test_model", None).unwrap());
            },
        );
    }
    group.finish();
}

// ============================================================================
// Experiment Tracking Benchmarks
// ============================================================================

/// Benchmark experiment metric logging.
///
/// _Requirements: 5.3_
fn bench_experiment_log_metric(c: &mut Criterion) {
    let mut group = c.benchmark_group("experiment_log_metric");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_metric", |b| {
        let dir = tempdir().unwrap();
        let mut tracker = ExperimentTracker::new(dir.path().join("experiments.db")).unwrap();

        let run_id = tracker.start_run("benchmark_exp", vec![]).unwrap();
        let mut step = 0u64;

        b.iter(|| {
            tracker.log_metric(&run_id, "loss", 0.5, Some(step)).unwrap();
            step += 1;
        });

        tracker.end_run(&run_id, RunStatus::Completed).ok();
    });

    group.finish();
}

/// Benchmark experiment parameter logging.
///
/// _Requirements: 5.2_
fn bench_experiment_log_param(c: &mut Criterion) {
    let mut group = c.benchmark_group("experiment_log_param");
    group.throughput(Throughput::Elements(1));

    group.bench_function("single_param", |b| {
        let dir = tempdir().unwrap();
        let mut tracker = ExperimentTracker::new(dir.path().join("experiments.db")).unwrap();

        let run_id = tracker.start_run("benchmark_exp", vec![]).unwrap();
        let mut i = 0u64;

        b.iter(|| {
            let key = format!("param_{}", i);
            tracker.log_param(&run_id, &key, "value").unwrap();
            i += 1;
        });

        tracker.end_run(&run_id, RunStatus::Completed).ok();
    });

    group.finish();
}

/// Benchmark experiment run creation.
///
/// _Requirements: 5.1_
fn bench_experiment_start_run(c: &mut Criterion) {
    let mut group = c.benchmark_group("experiment_start_run");
    group.throughput(Throughput::Elements(1));

    group.bench_function("new_run", |b| {
        let dir = tempdir().unwrap();
        let mut tracker = ExperimentTracker::new(dir.path().join("experiments.db")).unwrap();

        b.iter(|| {
            let run_id = tracker
                .start_run("benchmark_exp", vec!["bench".to_string()])
                .unwrap();
            tracker.end_run(&run_id, RunStatus::Completed).ok();
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups and Main
// ============================================================================

criterion_group!(
    vector_benches,
    bench_vector_insert,
    bench_vector_search_brute_force,
    bench_vector_search_hnsw,
);

criterion_group!(
    tensor_benches,
    bench_tensor_put_chunked,
    bench_tensor_get_chunked,
);

criterion_group!(model_benches, bench_model_save, bench_model_load,);

criterion_group!(
    experiment_benches,
    bench_experiment_log_metric,
    bench_experiment_log_param,
    bench_experiment_start_run,
);

criterion_main!(
    vector_benches,
    tensor_benches,
    model_benches,
    experiment_benches
);

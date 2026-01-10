//! Benchmark for Hybrid Hot/Cold Vector Architecture
//!
//! Tests HybridVectorStore combining GWI (hot) + Cascade (cold)

use std::time::Instant;
use synadb::arch::{HybridConfig, HybridVectorStore};
use synadb::cascade::CascadeConfig;
use synadb::gwi::GwiConfig;
use tempfile::tempdir;

/// Generate random vectors for testing
fn generate_vectors(count: usize, dims: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            (0..dims)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (i * dims + j).hash(&mut hasher);
                    let h = hasher.finish();
                    ((h % 1000) as f32 / 1000.0) - 0.5
                })
                .collect()
        })
        .collect()
}

/// Normalize a vector to unit length
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[test]
fn bench_hybrid_ingest_and_search() {
    let dir = tempdir().unwrap();
    let hot_path = dir.path().join("hot.gwi");
    let cold_path = dir.path().join("cold.cascade");

    const DIMS: usize = 128;
    const SAMPLE_SIZE: usize = 1000;
    const INGEST_SIZE: usize = 5000;
    const SEARCH_K: usize = 10;
    const NUM_QUERIES: usize = 100;

    println!("\n=== Hybrid Hot/Cold Architecture Benchmark ===");
    println!("Dimensions: {}", DIMS);
    println!("Sample vectors (for attractors): {}", SAMPLE_SIZE);
    println!("Vectors to ingest: {}", INGEST_SIZE);
    println!("Search k: {}", SEARCH_K);
    println!("Number of queries: {}", NUM_QUERIES);

    // Generate sample vectors for attractor initialization
    let mut samples = generate_vectors(SAMPLE_SIZE, DIMS);
    for v in &mut samples {
        normalize(v);
    }
    let sample_refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();

    // Generate vectors to ingest
    let mut vectors = generate_vectors(INGEST_SIZE, DIMS);
    for v in &mut vectors {
        normalize(v);
    }

    // Generate query vectors
    let mut queries = generate_vectors(NUM_QUERIES, DIMS);
    for q in &mut queries {
        normalize(q);
    }

    // Create hybrid config
    let config = HybridConfig {
        hot: GwiConfig {
            dimensions: DIMS as u16,
            branching_factor: 8,
            num_levels: 3,
            nprobe: 10,
            initial_capacity: INGEST_SIZE,
            ..Default::default()
        },
        cold: CascadeConfig {
            dimensions: DIMS as u16,
            num_bits: 6,
            num_tables: 8,
            num_probes: 8,
            ..Default::default()
        },
    };

    // Create hybrid store
    let start = Instant::now();
    let mut store = HybridVectorStore::new(&hot_path, &cold_path, config).unwrap();
    let create_time = start.elapsed();
    println!("\nCreate time: {:?}", create_time);

    // Initialize hot layer attractors
    let start = Instant::now();
    store.initialize_hot(&sample_refs).unwrap();
    let init_time = start.elapsed();
    println!("Initialize attractors: {:?}", init_time);

    // Ingest to hot layer
    let start = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        let key = format!("doc_{}", i);
        store.ingest(&key, v).unwrap();
    }
    let ingest_time = start.elapsed();
    let ingest_rate = INGEST_SIZE as f64 / ingest_time.as_secs_f64();
    println!(
        "Hot ingest: {:?} ({:.0} vectors/sec)",
        ingest_time, ingest_rate
    );

    // Search hot layer only
    let start = Instant::now();
    for q in &queries {
        let _ = store.search_hot(q, SEARCH_K);
    }
    let hot_search_time = start.elapsed();
    let hot_search_avg = hot_search_time.as_micros() as f64 / NUM_QUERIES as f64;
    println!(
        "Hot search ({}x): {:?} (avg {:.1}µs/query)",
        NUM_QUERIES, hot_search_time, hot_search_avg
    );

    // Promote to cold
    let start = Instant::now();
    let promoted = store.promote_to_cold().unwrap();
    let promote_time = start.elapsed();
    println!("Promote to cold: {:?} ({} vectors)", promote_time, promoted);

    // Search cold layer only
    let start = Instant::now();
    for q in &queries {
        let _ = store.search_cold(q, SEARCH_K);
    }
    let cold_search_time = start.elapsed();
    let cold_search_avg = cold_search_time.as_micros() as f64 / NUM_QUERIES as f64;
    println!(
        "Cold search ({}x): {:?} (avg {:.1}µs/query)",
        NUM_QUERIES, cold_search_time, cold_search_avg
    );

    // Unified search (both layers)
    let start = Instant::now();
    for q in &queries {
        let results = store.search(q, SEARCH_K).unwrap();
        assert!(!results.is_empty(), "Search should return results");
    }
    let unified_search_time = start.elapsed();
    let unified_search_avg = unified_search_time.as_micros() as f64 / NUM_QUERIES as f64;
    println!(
        "Unified search ({}x): {:?} (avg {:.1}µs/query)",
        NUM_QUERIES, unified_search_time, unified_search_avg
    );

    // Summary
    println!("\n=== Summary ===");
    println!("Hot count: {}", store.hot_count());
    println!("Cold count: {}", store.cold_count());
    println!("Total count: {}", store.len());

    // Verify counts
    assert_eq!(store.hot_count(), INGEST_SIZE);
    assert_eq!(store.cold_count(), INGEST_SIZE);
    assert_eq!(store.len(), INGEST_SIZE * 2); // Both layers have the data

    println!("\n✓ Benchmark complete!");
}

#[test]
fn test_hybrid_basic_operations() {
    let dir = tempdir().unwrap();
    let hot_path = dir.path().join("hot.gwi");
    let cold_path = dir.path().join("cold.cascade");

    const DIMS: usize = 64;

    // Generate sample vectors
    let mut samples = generate_vectors(100, DIMS);
    for v in &mut samples {
        normalize(v);
    }
    let sample_refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();

    let config = HybridConfig {
        hot: GwiConfig {
            dimensions: DIMS as u16,
            branching_factor: 4,
            num_levels: 2,
            nprobe: 5,
            initial_capacity: 1000,
            ..Default::default()
        },
        cold: CascadeConfig {
            dimensions: DIMS as u16,
            num_bits: 4,
            num_tables: 4,
            num_probes: 4,
            ..Default::default()
        },
    };

    let mut store = HybridVectorStore::new(&hot_path, &cold_path, config).unwrap();
    store.initialize_hot(&sample_refs).unwrap();

    // Test empty state
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    // Ingest some vectors
    let mut v1 = vec![0.1f32; DIMS];
    normalize(&mut v1);
    store.ingest("key1", &v1).unwrap();

    let mut v2 = vec![0.2f32; DIMS];
    normalize(&mut v2);
    store.ingest("key2", &v2).unwrap();

    assert_eq!(store.hot_count(), 2);
    assert_eq!(store.cold_count(), 0);
    assert!(!store.is_empty());

    // Search hot
    let results = store.search(&v1, 5).unwrap();
    assert!(!results.is_empty());

    // Promote to cold
    let promoted = store.promote_to_cold().unwrap();
    assert_eq!(promoted, 2);
    assert_eq!(store.cold_count(), 2);

    // Search unified (should find in both)
    let results = store.search(&v1, 5).unwrap();
    assert!(!results.is_empty());

    println!("✓ Basic operations test passed!");
}

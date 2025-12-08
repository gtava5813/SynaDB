//! Property-based tests for Vector similarity search correctness.
//!
//! **Feature: Syna-ai-native, Property 18: Similarity Search Correctness**
//! **Validates: Requirements 1.4**

use proptest::prelude::*;
use synadb::distance::DistanceMetric;
use synadb::vector::{VectorConfig, VectorStore};
use tempfile::tempdir;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// **Feature: Syna-ai-native, Property 18: Similarity Search Correctness**
    ///
    /// For brute-force search, results must be the actual k nearest neighbors.
    /// This property verifies that:
    /// 1. The search returns exactly min(k, n_vectors) results
    /// 2. Results are sorted by score (ascending = most similar first)
    /// 3. The returned results are actually the k nearest vectors
    ///
    /// **Validates: Requirements 1.4**
    #[test]
    fn prop_search_returns_k_nearest(
        k in 1usize..=10,
        n_vectors in 10usize..100,
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 128,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config.clone()).unwrap();

        // Insert vectors with deterministic values based on index
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..n_vectors {
            let vec: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect();
            let key = format!("v{}", i);
            store.insert(&key, &vec).unwrap();
            vectors.push((key, vec));
        }

        // Query with a deterministic vector
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let results = store.search(&query, k).unwrap();

        // Verify we got k results (or all if n < k)
        let expected_count = k.min(n_vectors);
        prop_assert_eq!(results.len(), expected_count,
            "Expected {} results, got {}", expected_count, results.len());

        // Verify results are sorted by score (ascending = most similar first)
        for i in 1..results.len() {
            prop_assert!(results[i-1].score <= results[i].score,
                "Results not sorted: {} > {} at positions {} and {}",
                results[i-1].score, results[i].score, i-1, i);
        }

        // Verify these are actually the k nearest (brute force check)
        let mut all_distances: Vec<(String, f32)> = vectors.iter()
            .map(|(k, v)| (k.clone(), config.metric.distance(&query, v)))
            .collect();
        all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, result) in results.iter().enumerate() {
            prop_assert_eq!(&result.key, &all_distances[i].0,
                "Wrong result at position {}: expected key '{}', got '{}'",
                i, all_distances[i].0, result.key);
        }
    }

    /// Test that search works correctly with Euclidean distance metric.
    #[test]
    fn prop_search_euclidean_metric(
        k in 1usize..=5,
        n_vectors in 5usize..50,
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_euclidean.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config.clone()).unwrap();

        // Insert vectors
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..n_vectors {
            let vec: Vec<f32> = (0..64).map(|j| (i as f32 + j as f32 * 0.01)).collect();
            let key = format!("e{}", i);
            store.insert(&key, &vec).unwrap();
            vectors.push((key, vec));
        }

        // Query
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.01).collect();
        let results = store.search(&query, k).unwrap();

        // Verify count
        let expected_count = k.min(n_vectors);
        prop_assert_eq!(results.len(), expected_count);

        // Verify sorted
        for i in 1..results.len() {
            prop_assert!(results[i-1].score <= results[i].score,
                "Euclidean results not sorted");
        }

        // Verify correctness
        let mut all_distances: Vec<(String, f32)> = vectors.iter()
            .map(|(k, v)| (k.clone(), config.metric.distance(&query, v)))
            .collect();
        all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, result) in results.iter().enumerate() {
            prop_assert_eq!(&result.key, &all_distances[i].0,
                "Wrong Euclidean result at position {}", i);
        }
    }

    /// Test that search works correctly with DotProduct distance metric.
    #[test]
    fn prop_search_dot_product_metric(
        k in 1usize..=5,
        n_vectors in 5usize..50,
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_dot.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::DotProduct,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config.clone()).unwrap();

        // Insert vectors
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..n_vectors {
            let vec: Vec<f32> = (0..64).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
            let key = format!("d{}", i);
            store.insert(&key, &vec).unwrap();
            vectors.push((key, vec));
        }

        // Query
        let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.2).cos()).collect();
        let results = store.search(&query, k).unwrap();

        // Verify count
        let expected_count = k.min(n_vectors);
        prop_assert_eq!(results.len(), expected_count);

        // Verify sorted
        for i in 1..results.len() {
            prop_assert!(results[i-1].score <= results[i].score,
                "DotProduct results not sorted");
        }

        // Verify correctness
        let mut all_distances: Vec<(String, f32)> = vectors.iter()
            .map(|(k, v)| (k.clone(), config.metric.distance(&query, v)))
            .collect();
        all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (i, result) in results.iter().enumerate() {
            prop_assert_eq!(&result.key, &all_distances[i].0,
                "Wrong DotProduct result at position {}", i);
        }
    }
}

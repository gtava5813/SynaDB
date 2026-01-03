//! Property tests for Cascade Index
//!
//! **Feature: Cascade Index, Property 19: Insert-Search Round-Trip**
//! **Feature: Cascade Index, Property 20: Search Returns K Results**

use proptest::prelude::*;
use synadb::cascade::{CascadeConfig, CascadeIndex};
use tempfile::tempdir;

/// Generate a random vector of given dimensions
fn arb_vector(dims: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-1.0f32..1.0f32, dims)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// **Property 19: Insert-Search Round-Trip**
    /// Inserted vectors should be retrievable via search
    #[test]
    fn prop_cascade_insert_search_roundtrip(
        vectors in proptest::collection::vec(arb_vector(64), 10..50),
        query_idx in 0usize..10,
    ) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cascade");

        let config = CascadeConfig {
            dimensions: 64,
            num_bits: 6,
            num_tables: 4,
            m: 8,
            ..Default::default()
        };

        let mut index = CascadeIndex::new(&path, config).unwrap();

        // Insert vectors
        let mut keys = Vec::new();
        for (i, vec) in vectors.iter().enumerate() {
            let key = format!("v{}", i);
            index.insert(&key, vec).unwrap();
            keys.push(key);
        }

        // Search for one of the inserted vectors
        let query_idx = query_idx % vectors.len();
        let results = index.search(&vectors[query_idx], 5).unwrap();

        // The query vector itself should be in results (exact match)
        let found = results.iter().any(|r| r.key == keys[query_idx]);
        prop_assert!(found, "Query vector not found in results");
    }

    /// **Property 20: Search Returns At Most K Results**
    /// Search should never return more than k results
    #[test]
    fn prop_cascade_search_returns_at_most_k_results(
        vectors in proptest::collection::vec(arb_vector(64), 20..100),
        k in 1usize..20,
    ) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cascade");

        let config = CascadeConfig {
            dimensions: 64,
            num_bits: 6,
            num_tables: 4,
            m: 8,
            ..Default::default()
        };

        let mut index = CascadeIndex::new(&path, config).unwrap();

        // Insert vectors
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(&format!("v{}", i), vec).unwrap();
        }

        // Search
        let query = &vectors[0];
        let results = index.search(query, k).unwrap();

        // Should return at most k results (may return fewer for ANN)
        prop_assert!(results.len() <= k, "Got {} results, expected at most {}", results.len(), k);

        // Results should be non-empty if we have vectors
        prop_assert!(!results.is_empty(), "Search returned no results");
    }
}

#[test]
fn test_cascade_persistence_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("persist.cascade");

    let config = CascadeConfig {
        dimensions: 32,
        num_bits: 6,
        num_tables: 2,
        m: 4,
        ..Default::default()
    };

    // Create and populate
    {
        let mut index = CascadeIndex::new(&path, config.clone()).unwrap();
        for i in 0..50 {
            let vec: Vec<f32> = (0..32).map(|j| ((i + j) % 10) as f32 / 10.0).collect();
            index.insert(&format!("k{}", i), &vec).unwrap();
        }
        index.flush().unwrap();
    }

    // Reload and verify
    {
        let index = CascadeIndex::load(&path).unwrap();
        assert_eq!(index.len(), 50);

        // Search should work
        let query: Vec<f32> = (0..32).map(|j| (j % 10) as f32 / 10.0).collect();
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
    }
}

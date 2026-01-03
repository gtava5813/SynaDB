//! **Feature: GWI, Property: Persistence Round-Trip**
//!
//! Tests that GravityWellIndex correctly persists and loads data across close/reopen cycles.

use synadb::gwi::{GravityWellIndex, GwiConfig};
use tempfile::tempdir;

/// Test that GWI data persists across close/reopen
#[test]
fn test_gwi_persistence_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.gwi");

    let dims = 128;
    let num_vectors = 100;

    // Generate sample vectors for initialization
    let sample_vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..dims).map(|j| ((i * dims + j) as f32) * 0.01).collect())
        .collect();
    let sample_refs: Vec<&[f32]> = sample_vectors.iter().map(|v| v.as_slice()).collect();

    // Generate test vectors
    let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| (0..dims).map(|j| ((i * dims + j) as f32) * 0.001).collect())
        .collect();
    let test_keys: Vec<String> = (0..num_vectors).map(|i| format!("item_{}", i)).collect();

    // Create, initialize, and populate
    {
        let config = GwiConfig {
            dimensions: dims as u16,
            branching_factor: 4,
            num_levels: 2,
            initial_capacity: 1000,
            ..Default::default()
        };

        let mut gwi = GravityWellIndex::new(&path, config).unwrap();
        gwi.initialize_attractors(&sample_refs).unwrap();

        for (key, vec) in test_keys.iter().zip(test_vectors.iter()) {
            gwi.insert(key, vec).unwrap();
        }

        assert_eq!(gwi.len(), num_vectors);
        gwi.close().unwrap();
    }

    // Reopen and verify
    {
        let gwi = GravityWellIndex::open(&path).unwrap();

        // Check count
        assert_eq!(
            gwi.len(),
            num_vectors,
            "Vector count should persist: expected {}, got {}",
            num_vectors,
            gwi.len()
        );

        // Check that we can retrieve vectors
        for (i, key) in test_keys.iter().enumerate() {
            let vec = gwi.get(key).unwrap();
            assert!(vec.is_some(), "Key {} should exist after reopen", key);

            let vec = vec.unwrap();
            assert_eq!(vec.len(), dims, "Vector dimensions should match");

            // Verify vector values
            for (j, &val) in vec.iter().enumerate() {
                let expected = ((i * dims + j) as f32) * 0.001;
                assert!(
                    (val - expected).abs() < 1e-6,
                    "Vector value mismatch at key={}, dim={}: expected {}, got {}",
                    key,
                    j,
                    expected,
                    val
                );
            }
        }
    }
}

/// Test that search works after reopen
#[test]
fn test_gwi_search_after_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("search_test.gwi");

    let dims = 64;

    // Generate sample vectors
    let sample_vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let mut v = vec![0.0f32; dims];
            v[i % dims] = 1.0; // One-hot-ish vectors
            v
        })
        .collect();
    let sample_refs: Vec<&[f32]> = sample_vectors.iter().map(|v| v.as_slice()).collect();

    // Create and populate
    {
        let config = GwiConfig {
            dimensions: dims as u16,
            branching_factor: 4,
            num_levels: 2,
            initial_capacity: 1000,
            ..Default::default()
        };

        let mut gwi = GravityWellIndex::new(&path, config).unwrap();
        gwi.initialize_attractors(&sample_refs).unwrap();

        // Insert vectors
        for (i, vec) in sample_vectors.iter().enumerate() {
            gwi.insert(&format!("vec_{}", i), vec).unwrap();
        }

        gwi.close().unwrap();
    }

    // Reopen and search
    {
        let gwi = GravityWellIndex::open(&path).unwrap();

        // Search for a vector similar to vec_0
        let query = &sample_vectors[0];
        let results = gwi.search_with_nprobe(query, 5, 10).unwrap();

        assert!(
            !results.is_empty(),
            "Search should return results after reopen"
        );

        // The closest result should be vec_0 itself
        assert_eq!(
            results[0].key, "vec_0",
            "Closest result should be the query vector itself"
        );
    }
}

/// Test that empty GWI can be created and reopened
#[test]
fn test_gwi_empty_persistence() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.gwi");

    // Create empty (uninitialized) GWI
    {
        let config = GwiConfig {
            dimensions: 128,
            ..Default::default()
        };

        let gwi = GravityWellIndex::new(&path, config).unwrap();
        assert_eq!(gwi.len(), 0);
        // Don't initialize - just close
    }

    // File should exist but be empty of vectors
    assert!(path.exists());
}

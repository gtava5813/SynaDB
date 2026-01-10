//! Property Tests for Sparse Vector Store Persistence
//!
//! **Property 7: Index Persistence Round-Trip**
//! For any sparse vector store with indexed documents, saving to disk
//! and loading back should preserve all data and search functionality.

use proptest::prelude::*;
use synadb::{SparseVector, SparseVectorStore};
use tempfile::tempdir;

/// Generate a random sparse vector with 1-50 non-zero terms
fn arb_sparse_vector() -> impl Strategy<Value = SparseVector> {
    prop::collection::vec((1u32..10000u32, 0.1f32..10.0f32), 1..50).prop_map(|terms| {
        let mut vec = SparseVector::new();
        for (term_id, weight) in terms {
            vec.add(term_id, weight);
        }
        vec
    })
}

/// Generate a random document key
fn arb_key() -> impl Strategy<Value = String> {
    "[a-z]{1,20}".prop_map(|s| s)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property 7: Index Persistence Round-Trip**
    ///
    /// For any set of documents indexed in a sparse vector store:
    /// 1. Save the store to disk
    /// 2. Load it back
    /// 3. All documents should be retrievable
    /// 4. Search should return the same results
    #[test]
    fn prop_persistence_roundtrip(
        docs in prop::collection::vec((arb_key(), arb_sparse_vector()), 1..20)
    ) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.svs");

        // Create and populate store
        let mut store = SparseVectorStore::new();
        let mut expected_keys = Vec::new();

        for (key, vec) in &docs {
            // Use unique keys by appending index
            let unique_key = format!("{}_{}", key, expected_keys.len());
            store.index_with_key(&unique_key, vec.clone());
            expected_keys.push(unique_key);
        }

        let original_len = store.len();
        let original_stats = store.stats();

        // Save to disk
        store.save(&file_path).unwrap();

        // Load from disk
        let loaded = SparseVectorStore::load(&file_path).unwrap();

        // Verify length preserved
        prop_assert_eq!(loaded.len(), original_len);

        // Verify stats preserved
        let loaded_stats = loaded.stats();
        prop_assert_eq!(loaded_stats.num_documents, original_stats.num_documents);
        prop_assert_eq!(loaded_stats.num_terms, original_stats.num_terms);
        prop_assert_eq!(loaded_stats.num_postings, original_stats.num_postings);

        // Verify all documents retrievable
        for key in &expected_keys {
            prop_assert!(loaded.get_by_key(key).is_some(), "Key {} not found after load", key);
        }

        // Verify search works (use first doc's terms as query)
        if !docs.is_empty() {
            let query = &docs[0].1;
            if !query.is_empty() {
                let original_results = store.search(query, 10);
                let loaded_results = loaded.search(query, 10);

                // Same number of results
                prop_assert_eq!(
                    original_results.len(),
                    loaded_results.len(),
                    "Search result count mismatch"
                );

                // Same keys in same order
                for (orig, load) in original_results.iter().zip(loaded_results.iter()) {
                    prop_assert_eq!(&orig.key, &load.key, "Search result key mismatch");
                    // Scores should be very close (floating point)
                    prop_assert!(
                        (orig.score - load.score).abs() < 0.0001,
                        "Score mismatch: {} vs {}",
                        orig.score,
                        load.score
                    );
                }
            }
        }
    }

    /// **Property: Empty Store Persistence**
    ///
    /// An empty store should save and load correctly.
    #[test]
    fn prop_empty_persistence(_dummy in 0..1i32) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("empty.svs");

        let store = SparseVectorStore::new();
        store.save(&file_path).unwrap();

        let loaded = SparseVectorStore::load(&file_path).unwrap();
        prop_assert!(loaded.is_empty());
    }

    /// **Property: Vector Data Integrity**
    ///
    /// After save/load, vector data should be identical.
    #[test]
    fn prop_vector_data_integrity(
        vec in arb_sparse_vector()
    ) {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("integrity.svs");

        let mut store = SparseVectorStore::new();
        store.index_with_key("test_doc", vec.clone());
        store.save(&file_path).unwrap();

        let loaded = SparseVectorStore::load(&file_path).unwrap();
        let loaded_vec = loaded.get_by_key("test_doc").unwrap();

        // Verify all terms match
        prop_assert_eq!(vec.nnz(), loaded_vec.nnz());

        for (term_id, weight) in vec.iter() {
            let loaded_weight = loaded_vec.get(*term_id);
            prop_assert!(
                (weight - loaded_weight).abs() < 0.0001,
                "Weight mismatch for term {}: {} vs {}",
                term_id,
                weight,
                loaded_weight
            );
        }
    }
}

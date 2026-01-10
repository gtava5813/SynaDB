//! Property tests for SparseVectorStore
//!
//! **Feature: Sparse Vector Store (SVS), Properties 4-8**
//!
//! Tests universal invariants for sparse vector store operations.

use proptest::prelude::*;
use synadb::{SparseVector, SparseVectorStore};

/// Strategy to generate arbitrary sparse vectors
fn arb_sparse_vector() -> impl Strategy<Value = SparseVector> {
    prop::collection::vec((0u32..10000u32, 0.001f32..10.0f32), 1..50).prop_map(|pairs| {
        let mut vec = SparseVector::new();
        for (term_id, weight) in pairs {
            vec.add(term_id, weight);
        }
        vec
    })
}

/// Strategy to generate a store with documents
fn arb_store_with_docs() -> impl Strategy<Value = (SparseVectorStore, Vec<(String, SparseVector)>)>
{
    prop::collection::vec(arb_sparse_vector(), 1..20).prop_map(|vecs| {
        let mut store = SparseVectorStore::new();
        let mut docs = Vec::new();
        for (i, vec) in vecs.into_iter().enumerate() {
            let key = format!("doc_{}", i);
            store.index_with_key(&key, vec.clone());
            docs.push((key, vec));
        }
        (store, docs)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property 4: Search Result Ordering**
    ///
    /// Search results are always sorted by score descending.
    ///
    /// _Validates: Requirement 2.6 (Top-k Ordering)_
    #[test]
    fn prop_search_results_ordered(
        (store, _docs) in arb_store_with_docs(),
        query in arb_sparse_vector()
    ) {
        let results = store.search(&query, 100);

        // Results should be sorted by score descending
        for i in 1..results.len() {
            prop_assert!(
                results[i - 1].score >= results[i].score,
                "Results not sorted: {} < {} at positions {}, {}",
                results[i - 1].score, results[i].score, i - 1, i
            );
        }
    }

    /// **Property 5: Indexed Document Retrieval**
    ///
    /// Any indexed document can be retrieved by its key.
    ///
    /// _Validates: Requirement 2.3 (Document Storage)_
    #[test]
    fn prop_indexed_document_retrievable(
        (store, docs) in arb_store_with_docs()
    ) {
        for (key, original_vec) in docs {
            let retrieved = store.get_by_key(&key);
            prop_assert!(retrieved.is_some(), "Document {} not found", key);

            let retrieved = retrieved.unwrap();
            prop_assert_eq!(
                original_vec.nnz(), retrieved.nnz(),
                "Document {} has wrong nnz", key
            );

            // All weights should match
            for (term_id, weight) in original_vec.iter() {
                prop_assert!(
                    (retrieved.get(*term_id) - weight).abs() < 1e-6,
                    "Weight mismatch for term {} in doc {}", term_id, key
                );
            }
        }
    }

    /// **Property 6: Delete Removes from Search**
    ///
    /// After deleting a document, it should not appear in search results.
    ///
    /// _Validates: Requirement 2.7 (Delete Operation)_
    #[test]
    fn prop_delete_removes_from_search(
        (mut store, docs) in arb_store_with_docs(),
        query in arb_sparse_vector()
    ) {
        // Pick a document to delete (first one)
        if let Some((key, _)) = docs.first() {
            let deleted = store.delete(key);
            prop_assert!(deleted, "Delete should succeed for {}", key);

            // Document should not be retrievable
            prop_assert!(store.get_by_key(key).is_none(), "Deleted doc {} still retrievable", key);

            // Document should not appear in search results
            let results = store.search(&query, 100);
            for r in &results {
                prop_assert_ne!(&r.key, key, "Deleted doc {} found in search results", key);
            }
        }
    }

    /// **Property 7: Search Score Correctness**
    ///
    /// Search scores equal the dot product between query and document.
    ///
    /// _Validates: Requirement 2.5 (Dot Product Scoring)_
    #[test]
    fn prop_search_score_equals_dot_product(
        (store, docs) in arb_store_with_docs(),
        query in arb_sparse_vector()
    ) {
        let results = store.search(&query, 100);

        for result in results {
            // Find the original document
            let doc_vec = docs.iter()
                .find(|(k, _)| k == &result.key)
                .map(|(_, v)| v);

            if let Some(doc_vec) = doc_vec {
                let expected_score = query.dot(doc_vec);
                prop_assert!(
                    (result.score - expected_score).abs() < 1e-5,
                    "Score mismatch for {}: expected {}, got {}",
                    result.key, expected_score, result.score
                );
            }
        }
    }

    /// **Property 8: Stats Consistency**
    ///
    /// Index statistics are consistent with actual content.
    ///
    /// _Validates: Requirement 2.8 (Statistics)_
    #[test]
    fn prop_stats_consistent(
        (store, docs) in arb_store_with_docs()
    ) {
        let stats = store.stats();

        prop_assert_eq!(
            stats.num_documents, docs.len(),
            "num_documents mismatch"
        );

        // Total postings should equal sum of all document nnz
        let expected_postings: usize = docs.iter().map(|(_, v)| v.nnz()).sum();
        prop_assert_eq!(
            stats.num_postings, expected_postings,
            "num_postings mismatch"
        );

        // Average doc length
        let expected_avg = expected_postings as f32 / docs.len() as f32;
        prop_assert!(
            (stats.avg_doc_length - expected_avg).abs() < 1e-5,
            "avg_doc_length mismatch: expected {}, got {}",
            expected_avg, stats.avg_doc_length
        );
    }

    /// **Property 9: Top-K Limit**
    ///
    /// Search returns at most k results.
    ///
    /// _Validates: Requirement 2.6 (Top-k)_
    #[test]
    fn prop_search_respects_k_limit(
        (store, _docs) in arb_store_with_docs(),
        query in arb_sparse_vector(),
        k in 1usize..10
    ) {
        let results = store.search(&query, k);
        prop_assert!(
            results.len() <= k,
            "Got {} results but k={}", results.len(), k
        );
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_store_search() {
        let store = SparseVectorStore::new();
        let mut query = SparseVector::new();
        query.add(100, 1.0);

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query_search() {
        let mut store = SparseVectorStore::new();
        let mut doc = SparseVector::new();
        doc.add(100, 1.0);
        store.index_with_key("doc1", doc);

        let query = SparseVector::new();
        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_overlap_search() {
        let mut store = SparseVectorStore::new();
        let mut doc = SparseVector::new();
        doc.add(100, 1.0);
        store.index_with_key("doc1", doc);

        let mut query = SparseVector::new();
        query.add(999, 1.0);

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut store = SparseVectorStore::new();
        assert!(!store.delete("nonexistent"));
    }

    #[test]
    fn test_replace_document() {
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 1.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(200, 2.0);
        store.index_with_key("doc1", doc2);

        assert_eq!(store.len(), 1);

        let retrieved = store.get_by_key("doc1").unwrap();
        assert_eq!(retrieved.get(100), 0.0);
        assert_eq!(retrieved.get(200), 2.0);
    }
}

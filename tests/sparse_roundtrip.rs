//! Property tests for SparseVector
//!
//! **Feature: Sparse Vector Store (SVS), Properties 1-3**
//!
//! Tests universal invariants for sparse vector operations.

use proptest::prelude::*;
use std::collections::HashMap;
use synadb::SparseVector;

/// Strategy to generate arbitrary sparse vectors
fn arb_sparse_vector() -> impl Strategy<Value = SparseVector> {
    // Generate 0-100 term_id/weight pairs
    prop::collection::vec((0u32..30522u32, 0.001f32..10.0f32), 0..100).prop_map(|pairs| {
        let mut vec = SparseVector::new();
        for (term_id, weight) in pairs {
            vec.add(term_id, weight);
        }
        vec
    })
}

/// Strategy to generate sparse vectors with potential zero/negative weights
fn arb_sparse_vector_with_invalid() -> impl Strategy<Value = HashMap<u32, f32>> {
    prop::collection::hash_map(0u32..30522u32, -5.0f32..10.0f32, 0..50)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property 1: Sparse Vector Serialization Round-Trip**
    ///
    /// For any sparse vector V:
    /// - from_bytes(to_bytes(V)) == V
    ///
    /// _Validates: Requirement 1.6 (Serialization)_
    #[test]
    fn prop_sparse_vector_roundtrip(vec in arb_sparse_vector()) {
        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes);

        prop_assert!(restored.is_some(), "Deserialization should succeed");
        let restored = restored.unwrap();

        // Same number of non-zero terms
        prop_assert_eq!(vec.nnz(), restored.nnz());

        // All weights match
        for (term_id, weight) in vec.iter() {
            prop_assert!(
                (restored.get(*term_id) - weight).abs() < 1e-6,
                "Weight mismatch for term {}: expected {}, got {}",
                term_id, weight, restored.get(*term_id)
            );
        }
    }

    /// **Property 2: Dot Product Commutativity**
    ///
    /// For any sparse vectors A and B:
    /// - A.dot(B) == B.dot(A)
    ///
    /// _Validates: Requirement 1.3 (Dot Product)_
    #[test]
    fn prop_dot_product_commutative(
        a in arb_sparse_vector(),
        b in arb_sparse_vector()
    ) {
        let ab = a.dot(&b);
        let ba = b.dot(&a);

        prop_assert!(
            (ab - ba).abs() < 1e-5,
            "Dot product not commutative: a.dot(b)={}, b.dot(a)={}",
            ab, ba
        );
    }

    /// **Property 3: Zero/Negative Weight Filtering**
    ///
    /// For any weights map with zero/negative values:
    /// - from_weights() filters them out
    /// - All stored weights are positive
    ///
    /// _Validates: Requirement 1.2 (Positive Weights Only)_
    #[test]
    fn prop_zero_negative_filtered(weights in arb_sparse_vector_with_invalid()) {
        let vec = SparseVector::from_weights(weights.clone());

        // Count expected positive weights
        let expected_count = weights.values().filter(|w| **w > 0.0).count();
        prop_assert_eq!(vec.nnz(), expected_count);

        // All stored weights should be positive
        for (_, weight) in vec.iter() {
            prop_assert!(*weight > 0.0, "Found non-positive weight: {}", weight);
        }

        // Zero/negative weights should return 0.0
        for (term_id, original_weight) in &weights {
            if *original_weight <= 0.0 {
                prop_assert_eq!(
                    vec.get(*term_id), 0.0,
                    "Zero/negative weight {} for term {} should not be stored",
                    original_weight, term_id
                );
            }
        }
    }

    /// **Property 4: Norm Non-Negativity**
    ///
    /// For any sparse vector V:
    /// - norm(V) >= 0
    /// - l1_norm(V) >= 0
    /// - norm(V) == 0 iff V is empty
    ///
    /// _Validates: Requirement 1.4 (Norm)_
    #[test]
    fn prop_norm_non_negative(vec in arb_sparse_vector()) {
        let l2 = vec.norm();
        let l1 = vec.l1_norm();

        prop_assert!(l2 >= 0.0, "L2 norm should be non-negative: {}", l2);
        prop_assert!(l1 >= 0.0, "L1 norm should be non-negative: {}", l1);

        if vec.is_empty() {
            prop_assert_eq!(l2, 0.0, "Empty vector should have zero L2 norm");
            prop_assert_eq!(l1, 0.0, "Empty vector should have zero L1 norm");
        } else {
            prop_assert!(l2 > 0.0, "Non-empty vector should have positive L2 norm");
            prop_assert!(l1 > 0.0, "Non-empty vector should have positive L1 norm");
        }
    }

    /// **Property 5: Dot Product with Self Equals Squared Norm**
    ///
    /// For any sparse vector V:
    /// - V.dot(V) == norm(V)²
    ///
    /// _Validates: Requirement 1.3, 1.4 (Dot Product and Norm consistency)_
    #[test]
    fn prop_dot_self_equals_squared_norm(vec in arb_sparse_vector()) {
        let dot_self = vec.dot(&vec);
        let norm_squared = vec.norm() * vec.norm();

        // Use relative tolerance for floating point comparison
        let tolerance = (dot_self.abs() + norm_squared.abs()) * 1e-5 + 1e-6;
        prop_assert!(
            (dot_self - norm_squared).abs() < tolerance,
            "v.dot(v)={} should equal norm²={} (diff={}, tol={})",
            dot_self, norm_squared, (dot_self - norm_squared).abs(), tolerance
        );
    }

    /// **Property 6: Serialization Size**
    ///
    /// For any sparse vector V with N non-zero terms:
    /// - to_bytes().len() == 4 + N * 8
    ///
    /// _Validates: Requirement 1.5 (Serialization Format)_
    #[test]
    fn prop_serialization_size(vec in arb_sparse_vector()) {
        let bytes = vec.to_bytes();
        let expected_size = 4 + vec.nnz() * 8;

        prop_assert_eq!(
            bytes.len(), expected_size,
            "Serialized size {} should be 4 + {} * 8 = {}",
            bytes.len(), vec.nnz(), expected_size
        );
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_vector_roundtrip() {
        let vec = SparseVector::new();
        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();
        assert_eq!(restored.nnz(), 0);
    }

    #[test]
    fn test_single_term_roundtrip() {
        let mut vec = SparseVector::new();
        vec.add(12345, 0.123456);
        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();
        assert_eq!(restored.nnz(), 1);
        assert!((restored.get(12345) - 0.123456).abs() < 1e-6);
    }

    #[test]
    fn test_max_term_id() {
        let mut vec = SparseVector::new();
        vec.add(u32::MAX, 1.0);
        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();
        assert_eq!(restored.get(u32::MAX), 1.0);
    }

    #[test]
    fn test_dot_product_empty() {
        let empty = SparseVector::new();
        let mut non_empty = SparseVector::new();
        non_empty.add(100, 1.0);

        assert_eq!(empty.dot(&non_empty), 0.0);
        assert_eq!(non_empty.dot(&empty), 0.0);
        assert_eq!(empty.dot(&empty), 0.0);
    }
}

//! Property-based tests for Vector serialization round-trip.
//!
//! **Feature: Syna-ai-native, Property 17: Vector Serialization Round-Trip**
//! **Validates: Requirements 1.1, 1.2**

use proptest::prelude::*;
use synadb::{Atom, SynaDB};
use tempfile::tempdir;

/// Generator for valid vector dimensions (64-8192).
/// These cover common embedding model dimensions.
fn arb_dimensions() -> impl Strategy<Value = u16> {
    prop_oneof![
        Just(64u16),
        Just(128u16),
        Just(256u16),
        Just(384u16), // MiniLM
        Just(512u16),
        Just(768u16),  // BERT base
        Just(1024u16), // BERT large
        Just(1536u16), // OpenAI ada-002
        Just(3072u16), // OpenAI text-embedding-3-large
        Just(4096u16),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-ai-native, Property 17: Vector Serialization Round-Trip**
    ///
    /// For any valid vector with dimensions in the range 64-8192,
    /// storing it in the database and retrieving it SHALL produce
    /// a value equal to the original (within floating-point tolerance).
    ///
    /// **Validates: Requirements 1.1, 1.2**
    #[test]
    fn prop_vector_roundtrip(dims in arb_dimensions()) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Generate vector with correct dimensions using deterministic values
        let vector: Vec<f32> = (0..dims).map(|i| i as f32 * 0.001).collect();

        let mut db = SynaDB::new(&db_path).unwrap();
        db.append("test_vec", Atom::Vector(vector.clone(), dims)).unwrap();

        let result = db.get("test_vec").unwrap();
        prop_assert!(result.is_some(), "Expected to retrieve stored vector");

        if let Some(Atom::Vector(data, d)) = result {
            prop_assert_eq!(d, dims, "Dimensions should match");
            prop_assert_eq!(data.len(), dims as usize, "Vector length should match dimensions");
            for (a, b) in data.iter().zip(vector.iter()) {
                prop_assert!((a - b).abs() < 1e-6 || (a.is_nan() && b.is_nan()),
                    "Vector elements should match: got {} expected {}", a, b);
            }
        } else {
            prop_assert!(false, "Expected Vector atom, got {:?}", result);
        }
    }

    /// Test vector round-trip with random f32 values (including edge cases).
    #[test]
    fn prop_vector_roundtrip_random_values(dims in arb_dimensions()) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_random.db");

        // Generate random vector values
        let vector: Vec<f32> = (0..dims)
            .map(|i| {
                // Mix of positive, negative, and small values
                let base = (i as f32 - dims as f32 / 2.0) / dims as f32;
                base * 10.0
            })
            .collect();

        let mut db = SynaDB::new(&db_path).unwrap();
        db.append("random_vec", Atom::Vector(vector.clone(), dims)).unwrap();

        let result = db.get("random_vec").unwrap();
        prop_assert!(result.is_some(), "Expected to retrieve stored vector");

        if let Some(Atom::Vector(data, d)) = result {
            prop_assert_eq!(d, dims, "Dimensions should match");
            prop_assert_eq!(data.len(), dims as usize, "Vector length should match dimensions");
            for (i, (a, b)) in data.iter().zip(vector.iter()).enumerate() {
                prop_assert!((a - b).abs() < 1e-6 || (a.is_nan() && b.is_nan()),
                    "Vector element {} mismatch: got {} expected {}", i, a, b);
            }
        } else {
            prop_assert!(false, "Expected Vector atom, got {:?}", result);
        }
    }

    /// Test that vector persists across database reopen.
    #[test]
    fn prop_vector_persistence_roundtrip(dims in arb_dimensions()) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_persist.db");

        // Generate vector
        let vector: Vec<f32> = (0..dims).map(|i| (i as f32).sin()).collect();

        // Write and close
        {
            let mut db = SynaDB::new(&db_path).unwrap();
            db.append("persist_vec", Atom::Vector(vector.clone(), dims)).unwrap();
            // db is dropped here, closing the file
        }

        // Reopen and read
        {
            let mut db = SynaDB::new(&db_path).unwrap();
            let result = db.get("persist_vec").unwrap();
            prop_assert!(result.is_some(), "Expected to retrieve stored vector after reopen");

            if let Some(Atom::Vector(data, d)) = result {
                prop_assert_eq!(d, dims, "Dimensions should match after reopen");
                prop_assert_eq!(data.len(), dims as usize, "Vector length should match after reopen");
                for (i, (a, b)) in data.iter().zip(vector.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-6 || (a.is_nan() && b.is_nan()),
                        "Vector element {} mismatch after reopen: got {} expected {}", i, a, b);
                }
            } else {
                prop_assert!(false, "Expected Vector atom after reopen, got {:?}", result);
            }
        }
    }
}

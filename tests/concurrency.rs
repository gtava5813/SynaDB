//! Property-based tests for concurrent write operations.
//!
//! **Feature: Syna-db, Property 10: Concurrent Writes Preserve All Data**
//! **Validates: Requirements 8.1**

use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use synadb::{close_db, open_db, with_db, Atom};
use tempfile::tempdir;

/// Generator for arbitrary Atom values (excluding NaN floats for equality testing).
fn arb_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        1 => Just(Atom::Null),
        10 => any::<f64>().prop_filter("filter NaN", |f| !f.is_nan()).prop_map(Atom::Float),
        10 => any::<i64>().prop_map(Atom::Int),
        10 => ".{0,50}".prop_map(|s: String| Atom::Text(s)),
        10 => prop::collection::vec(any::<u8>(), 0..100).prop_map(Atom::Bytes),
    ]
}

/// Generator for valid keys: non-empty UTF-8 strings (1-50 chars).
fn arb_key() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_]{1,50}"
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 10: Concurrent Writes Preserve All Data**
    ///
    /// For any set of key-value pairs written concurrently from multiple threads,
    /// after all writes complete, every key SHALL be retrievable with its correct
    /// value (no lost writes).
    #[test]
    fn prop_concurrent_writes_preserve_all_data(
        pairs in prop::collection::vec((arb_key(), arb_atom()), 8..32)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("concurrent_test.db");
        let db_path_str = db_path.to_str().unwrap().to_string();

        // Open the database
        open_db(&db_path_str).expect("failed to open db");

        // Use unique keys by appending index to avoid conflicts
        let unique_pairs: Vec<(String, Atom)> = pairs
            .into_iter()
            .enumerate()
            .map(|(i, (key, atom))| (format!("{}_{}", key, i), atom))
            .collect();

        // Split pairs into chunks for different threads
        let num_threads = 4;
        let chunk_size = (unique_pairs.len() + num_threads - 1) / num_threads;
        let chunks: Vec<Vec<(String, Atom)>> = unique_pairs
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        // Track expected final values (last write per key wins)
        let expected: HashMap<String, Atom> = unique_pairs
            .iter()
            .cloned()
            .collect();

        // Share the path across threads
        let path = Arc::new(db_path_str.clone());

        // Spawn threads to write concurrently
        thread::scope(|s| {
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    let path = Arc::clone(&path);
                    s.spawn(move || {
                        for (key, atom) in chunk {
                            with_db(&path, |db| {
                                db.append(&key, atom)
                            }).expect("append should succeed");
                        }
                    })
                })
                .collect();

            // Wait for all threads to complete
            for handle in handles {
                handle.join().expect("thread should not panic");
            }
        });

        // Verify all keys are readable with correct values
        for (key, expected_atom) in &expected {
            let result = with_db(&db_path_str, |db| {
                db.get(key)
            }).expect("get should succeed");

            prop_assert!(result.is_some(), "key '{}' should exist after concurrent writes", key);
            prop_assert_eq!(
                result.unwrap(),
                expected_atom.clone(),
                "key '{}' should have correct value",
                key
            );
        }

        // Verify no data corruption - all keys should have valid Atoms
        let all_keys = with_db(&db_path_str, |db| {
            Ok(db.keys())
        }).expect("keys should succeed");

        prop_assert_eq!(
            all_keys.len(),
            expected.len(),
            "should have exactly {} keys, got {}",
            expected.len(),
            all_keys.len()
        );

        // Close the database
        close_db(&db_path_str).expect("failed to close db");
    }
}

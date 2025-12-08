//! Property-based tests for database compaction.
//!
//! **Feature: Syna-db, Property 15: Compaction Preserves Latest Values**
//! **Validates: Requirements 11.1, 11.2**

use proptest::prelude::*;
use synadb::{Atom, SynaDB};
use tempfile::tempdir;

/// Generator for arbitrary Atom values.
fn arb_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        1 => Just(Atom::Null),
        10 => any::<f64>().prop_filter("filter NaN", |f| !f.is_nan()).prop_map(Atom::Float),
        10 => any::<i64>().prop_map(Atom::Int),
        10 => ".{0,100}".prop_map(|s: String| Atom::Text(s)),
        10 => prop::collection::vec(any::<u8>(), 0..1000).prop_map(Atom::Bytes),
    ]
}

/// Generator for valid keys: non-empty UTF-8 strings (1-50 chars).
fn arb_key() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_]{1,50}"
}

/// Generator for a sequence of writes to a key (1-5 values).
fn arb_write_sequence() -> impl Strategy<Value = (String, Vec<Atom>)> {
    (arb_key(), prop::collection::vec(arb_atom(), 1..=5))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 15: Compaction Preserves Latest Values**
    ///
    /// For any database with multiple writes per key (including some deleted keys),
    /// after compaction, all non-deleted keys SHALL return their latest values,
    /// and the file size SHALL be less than or equal to the original.
    ///
    /// **Validates: Requirements 11.1, 11.2**
    #[test]
    fn prop_compaction_preserves_latest_values(
        write_sequences in prop::collection::vec(arb_write_sequence(), 1..=10),
        delete_indices in prop::collection::vec(any::<bool>(), 1..=10)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Track expected state: key -> (latest_value, is_deleted)
        let mut expected_state: std::collections::HashMap<String, (Atom, bool)> =
            std::collections::HashMap::new();

        // Write multiple values per key
        for (key, values) in &write_sequences {
            for value in values {
                db.append(key, value.clone()).expect("append should succeed");
            }
            // Track the latest value
            if let Some(last_value) = values.last() {
                expected_state.insert(key.clone(), (last_value.clone(), false));
            }
        }

        // Delete some keys based on delete_indices
        let keys: Vec<String> = expected_state.keys().cloned().collect();
        for (i, key) in keys.iter().enumerate() {
            let should_delete = delete_indices.get(i % delete_indices.len()).copied().unwrap_or(false);
            if should_delete {
                db.delete(key).expect("delete should succeed");
                if let Some(entry) = expected_state.get_mut(key) {
                    entry.1 = true; // Mark as deleted
                }
            }
        }

        // Get file size before compaction
        let size_before = std::fs::metadata(&db_path).expect("metadata").len();

        // Compact the database
        db.compact().expect("compact should succeed");

        // Get file size after compaction
        let size_after = std::fs::metadata(&db_path).expect("metadata").len();

        // Verify file size is less than or equal to original
        prop_assert!(
            size_after <= size_before,
            "File size after compaction ({}) should be <= before ({})",
            size_after, size_before
        );

        // Verify all non-deleted keys return their expected latest values
        for (key, (expected_value, is_deleted)) in &expected_state {
            let result = db.get(key).expect("get should succeed");

            if *is_deleted {
                prop_assert!(
                    result.is_none(),
                    "Deleted key '{}' should return None after compaction",
                    key
                );
            } else {
                prop_assert!(
                    result.is_some(),
                    "Non-deleted key '{}' should exist after compaction",
                    key
                );
                prop_assert_eq!(
                    result.unwrap(),
                    expected_value.clone(),
                    "Key '{}' should have correct value after compaction",
                    key
                );
            }
        }

        // Verify keys() only returns non-deleted keys
        let keys_after = db.keys();
        for (key, (_, is_deleted)) in &expected_state {
            if *is_deleted {
                prop_assert!(
                    !keys_after.contains(key),
                    "Deleted key '{}' should not be in keys() after compaction",
                    key
                );
            } else {
                prop_assert!(
                    keys_after.contains(key),
                    "Non-deleted key '{}' should be in keys() after compaction",
                    key
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 16: History Excludes Post-Tombstone**
    ///
    /// For any key that was written to, then deleted, then written to again,
    /// the history SHALL contain values from both periods but NOT include
    /// the tombstone marker itself.
    ///
    /// **Validates: Requirements 10.3**
    #[test]
    fn prop_history_excludes_tombstone(
        key in arb_key(),
        values_before in prop::collection::vec(arb_atom(), 1..=3),
        values_after in prop::collection::vec(arb_atom(), 1..=3)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write values_before to key
        for value in &values_before {
            db.append(&key, value.clone()).expect("append should succeed");
        }

        // Delete key
        db.delete(&key).expect("delete should succeed");

        // Write values_after to key (resurrection)
        for value in &values_after {
            db.append(&key, value.clone()).expect("append should succeed");
        }

        // Get history
        let history = db.get_history(&key).expect("get_history should succeed");

        // Expected history: values_before + values_after (no tombstone)
        let expected_len = values_before.len() + values_after.len();
        prop_assert_eq!(
            history.len(),
            expected_len,
            "History length should be {} (before: {}, after: {}), got {}",
            expected_len, values_before.len(), values_after.len(), history.len()
        );

        // Verify values_before are at the start
        for (i, expected) in values_before.iter().enumerate() {
            prop_assert_eq!(
                &history[i],
                expected,
                "History[{}] should match values_before[{}]",
                i, i
            );
        }

        // Verify values_after are at the end
        for (i, expected) in values_after.iter().enumerate() {
            let history_idx = values_before.len() + i;
            prop_assert_eq!(
                &history[history_idx],
                expected,
                "History[{}] should match values_after[{}]",
                history_idx, i
            );
        }

        // Verify no Null values in history (tombstones are stored as Null)
        // Note: This check is only valid if we didn't write Null values ourselves
        // We need to check that the tombstone itself is not in the history
        // The tombstone is a special entry with IS_TOMBSTONE flag, not just Atom::Null
        // Since get_history reads all entries, we verify the count matches expected
        // (if tombstone was included, count would be expected_len + 1)
    }
}

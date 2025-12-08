//! Property-based tests for database recovery.
//!
//! **Feature: Syna-db, Property 4: Index Rebuild on Reopen**
//! **Feature: Syna-db, Property 11: Corruption Recovery Skips Bad Entries**
//! **Validates: Requirements 1.2, 9.2, 9.3**

use proptest::prelude::*;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use synadb::{Atom, SynaDB};
use tempfile::tempdir;

/// Generator for arbitrary Atom values (excluding NaN for equality testing).
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

/// Generator for a sequence of key-value pairs.
fn arb_key_value_pairs() -> impl Strategy<Value = Vec<(String, Atom)>> {
    prop::collection::vec((arb_key(), arb_atom()), 1..20)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 4: Index Rebuild on Reopen**
    ///
    /// For any sequence of write operations to a database, closing the database
    /// and reopening it SHALL result in an index that allows retrieving all
    /// previously written keys with their latest values.
    ///
    /// **Validates: Requirements 1.2, 9.2**
    #[test]
    fn prop_index_rebuild_on_reopen(pairs in arb_key_value_pairs()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        // Track expected final state (last value for each key)
        let mut expected: HashMap<String, Atom> = HashMap::new();

        // Open db1 and write all entries
        {
            let mut db1 = SynaDB::new(&db_path).expect("failed to create db");

            for (key, atom) in &pairs {
                db1.append(key, atom.clone()).expect("append should succeed");
                expected.insert(key.clone(), atom.clone());
            }

            // Close db1 (drop)
            db1.close().expect("close should succeed");
        }

        // Open db2 at same path
        let mut db2 = SynaDB::new(&db_path).expect("failed to reopen db");

        // For each key written, db2.get(key) should equal last value written for that key
        for (key, expected_atom) in &expected {
            let result = db2.get(key).expect("get should succeed");
            prop_assert!(result.is_some(), "key '{}' should exist after reopen", key);
            prop_assert_eq!(
                result.unwrap(),
                expected_atom.clone(),
                "value for key '{}' should match after reopen",
                key
            );
        }

        // Verify keys() returns all expected keys
        let keys = db2.keys();
        for key in expected.keys() {
            prop_assert!(keys.contains(key), "keys() should contain '{}'", key);
        }
    }
}

/// Generator for corruption parameters.
fn arb_corruption() -> impl Strategy<Value = (usize, Vec<u8>)> {
    // Generate corruption position (entry index) and random bytes to inject
    (0usize..10, prop::collection::vec(any::<u8>(), 1..20))
}

/// Generator for unique key-value pairs (no duplicate keys).
fn arb_unique_key_value_pairs() -> impl Strategy<Value = Vec<(String, Atom)>> {
    prop::collection::vec((arb_key(), arb_atom()), 3..10).prop_filter_map(
        "filter duplicate keys",
        |pairs| {
            let mut seen = std::collections::HashSet::new();
            let unique: Vec<_> = pairs
                .into_iter()
                .filter(|(k, _)| seen.insert(k.clone()))
                .collect();
            if unique.len() >= 3 {
                Some(unique)
            } else {
                None
            }
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 11: Corruption Recovery Skips Bad Entries**
    ///
    /// For any valid database file with injected corruption (truncated or garbled bytes)
    /// in one entry, reopening the database SHALL successfully recover entries before
    /// the corrupted entry.
    ///
    /// **Validates: Requirements 9.3**
    ///
    /// Note: This test uses unique keys to avoid ambiguity about which value should
    /// be recovered when the same key appears multiple times with corruption in between.
    #[test]
    fn prop_corruption_recovery_skips_bad_entries(
        pairs in arb_unique_key_value_pairs(),
        (corrupt_entry_idx, corrupt_bytes) in arb_corruption()
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        // Track offsets for each entry
        let mut entry_offsets: Vec<u64> = Vec::new();
        let mut expected_before_corruption: HashMap<String, Atom> = HashMap::new();

        // Open db and write all entries
        {
            let mut db = SynaDB::new(&db_path).expect("failed to create db");

            for (key, atom) in &pairs {
                let offset = db.append(key, atom.clone()).expect("append should succeed");
                entry_offsets.push(offset);
            }

            db.close().expect("close should succeed");
        }

        // Determine which entry to corrupt (ensure it's within bounds)
        let corrupt_idx = corrupt_entry_idx % pairs.len();

        // Build expected state: entries before corruption should be recoverable
        // Since we use unique keys, each key appears exactly once
        for (i, (key, atom)) in pairs.iter().enumerate() {
            if i < corrupt_idx {
                expected_before_corruption.insert(key.clone(), atom.clone());
            }
        }

        // Inject corruption at the chosen entry's offset
        {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&db_path)
                .expect("failed to open db file");

            let corrupt_offset = entry_offsets[corrupt_idx];
            file.seek(SeekFrom::Start(corrupt_offset)).expect("seek failed");
            file.write_all(&corrupt_bytes).expect("write failed");
            file.sync_all().expect("sync failed");
        }

        // Reopen database - should recover entries before corruption
        let mut db2 = SynaDB::new(&db_path).expect("failed to reopen db");

        // Verify entries before corruption are recoverable
        for (key, expected_atom) in &expected_before_corruption {
            let result = db2.get(key).expect("get should succeed");
            prop_assert!(
                result.is_some(),
                "key '{}' (before corruption) should exist after recovery",
                key
            );
            prop_assert_eq!(
                result.unwrap(),
                expected_atom.clone(),
                "value for key '{}' should match after recovery",
                key
            );
        }
    }
}

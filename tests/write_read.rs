//! Property-based tests for write-read round-trip.
//!
//! **Feature: Syna-db, Property 3: Write-Read Round-Trip**
//! **Validates: Requirements 3.1, 3.2, 10.3**

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

/// Generator for valid keys: non-empty UTF-8 strings (1-100 chars).
fn arb_key() -> impl Strategy<Value = String> {
    ".{1,100}".prop_filter("non-empty key", |s: &String| !s.is_empty())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 3: Write-Read Round-Trip**
    ///
    /// For any key-value pair where the key is a non-empty UTF-8 string and the value
    /// is any valid Atom, writing to the database and then reading back SHALL return
    /// an Atom equal to the original value.
    #[test]
    fn prop_write_read_roundtrip(key in arb_key(), atom in arb_atom()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write the value
        db.append(&key, atom.clone()).expect("append should succeed");

        // Read it back
        let result = db.get(&key).expect("get should succeed");

        prop_assert!(result.is_some(), "key should exist after write");
        prop_assert_eq!(result.unwrap(), atom, "read value should equal written value");
    }

    /// **Feature: Syna-db, Property 3: Write-Read Round-Trip**
    ///
    /// For multiple writes to the same key, the last write wins.
    #[test]
    fn prop_last_write_wins(
        key in arb_key(),
        atoms in prop::collection::vec(arb_atom(), 2..10)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write all values to the same key
        for atom in &atoms {
            db.append(&key, atom.clone()).expect("append should succeed");
        }

        // Read should return the last value
        let result = db.get(&key).expect("get should succeed");
        let expected = atoms.last().unwrap();

        prop_assert!(result.is_some(), "key should exist after writes");
        prop_assert_eq!(result.unwrap(), expected.clone(), "should return last written value");
    }
}

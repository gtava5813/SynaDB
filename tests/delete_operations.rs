//! Property-based tests for delete operations.
//!
//! **Feature: Syna-db, Property 13: Delete Makes Key Unreadable**
//! **Feature: Syna-db, Property 14: Delete-Write Resurrection**
//! **Validates: Requirements 10.1, 10.2, 10.4, 10.5**

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

    /// **Feature: Syna-db, Property 13: Delete Makes Key Unreadable**
    ///
    /// For any key with a value, after calling delete on that key, reading the key
    /// SHALL return null (not found), and the key SHALL NOT appear in the list of keys.
    ///
    /// **Validates: Requirements 10.1, 10.2, 10.5**
    #[test]
    fn prop_delete_makes_key_unreadable(key in arb_key(), atom in arb_atom()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write the value
        db.append(&key, atom.clone()).expect("append should succeed");

        // Verify it's readable
        let result = db.get(&key).expect("get should succeed");
        prop_assert!(result.is_some(), "key should exist after write");
        prop_assert_eq!(result.unwrap(), atom, "read value should equal written value");

        // Verify key is in keys list
        let keys_before = db.keys();
        prop_assert!(keys_before.contains(&key), "key should be in keys() before delete");

        // Verify exists returns true
        prop_assert!(db.exists(&key), "exists() should return true before delete");

        // Delete the key
        db.delete(&key).expect("delete should succeed");

        // Verify it's no longer readable
        let result_after = db.get(&key).expect("get should succeed");
        prop_assert!(result_after.is_none(), "key should not exist after delete");

        // Verify key is not in keys list
        let keys_after = db.keys();
        prop_assert!(!keys_after.contains(&key), "key should not be in keys() after delete");

        // Verify exists returns false
        prop_assert!(!db.exists(&key), "exists() should return false after delete");
    }

    /// **Feature: Syna-db, Property 14: Delete-Write Resurrection**
    ///
    /// For any deleted key, writing a new value to that key SHALL make the key
    /// readable again with the new value.
    ///
    /// **Validates: Requirements 10.4**
    #[test]
    fn prop_delete_write_resurrection(
        key in arb_key(),
        atom1 in arb_atom(),
        atom2 in arb_atom()
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write first value
        db.append(&key, atom1.clone()).expect("append should succeed");

        // Verify it's readable
        let result1 = db.get(&key).expect("get should succeed");
        prop_assert!(result1.is_some(), "key should exist after first write");

        // Delete the key
        db.delete(&key).expect("delete should succeed");

        // Verify it's deleted
        let result_deleted = db.get(&key).expect("get should succeed");
        prop_assert!(result_deleted.is_none(), "key should not exist after delete");

        // Write second value (resurrection)
        db.append(&key, atom2.clone()).expect("append should succeed");

        // Verify it's readable with the new value
        let result2 = db.get(&key).expect("get should succeed");
        prop_assert!(result2.is_some(), "key should exist after resurrection");
        prop_assert_eq!(result2.unwrap(), atom2, "read value should equal second written value");

        // Verify key is in keys list
        let keys = db.keys();
        prop_assert!(keys.contains(&key), "key should be in keys() after resurrection");

        // Verify exists returns true
        prop_assert!(db.exists(&key), "exists() should return true after resurrection");
    }
}

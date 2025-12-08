//! Property-based tests for database instance isolation.
//!
//! **Feature: Syna-db, Property 5: Database Instance Isolation**
//! **Validates: Requirements 1.5**

use proptest::prelude::*;
use synadb::{close_db, open_db, with_db, Atom, SynaDB};
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

    /// **Feature: Syna-db, Property 5: Database Instance Isolation**
    ///
    /// For any two database instances opened with different file paths, writes to one
    /// database SHALL NOT affect reads from the other database.
    #[test]
    fn prop_database_instance_isolation(
        key in arb_key(),
        atom1 in arb_atom(),
        atom2 in arb_atom()
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path1 = dir.path().join("test1.db");
        let db_path2 = dir.path().join("test2.db");

        // Open two separate database instances
        let mut db1 = SynaDB::new(&db_path1).expect("failed to create db1");
        let mut db2 = SynaDB::new(&db_path2).expect("failed to create db2");

        // Write (key, atom1) to db1
        db1.append(&key, atom1.clone()).expect("append to db1 should succeed");

        // Write (key, atom2) to db2
        db2.append(&key, atom2.clone()).expect("append to db2 should succeed");

        // Read key from db1 - should equal atom1
        let result1 = db1.get(&key).expect("get from db1 should succeed");
        prop_assert!(result1.is_some(), "key should exist in db1");
        prop_assert_eq!(result1.unwrap(), atom1.clone(), "db1 should return atom1");

        // Read key from db2 - should equal atom2
        let result2 = db2.get(&key).expect("get from db2 should succeed");
        prop_assert!(result2.is_some(), "key should exist in db2");
        prop_assert_eq!(result2.unwrap(), atom2.clone(), "db2 should return atom2");

        // Verify writes to one db never affect the other
        // Write a new value to db1
        let new_atom = Atom::Text("new_value".to_string());
        db1.append(&key, new_atom.clone()).expect("second append to db1 should succeed");

        // db2 should still have atom2
        let result2_after = db2.get(&key).expect("get from db2 after db1 write should succeed");
        prop_assert!(result2_after.is_some(), "key should still exist in db2");
        prop_assert_eq!(result2_after.unwrap(), atom2, "db2 should still return atom2 after db1 write");

        // db1 should have the new value
        let result1_after = db1.get(&key).expect("get from db1 after write should succeed");
        prop_assert!(result1_after.is_some(), "key should exist in db1");
        prop_assert_eq!(result1_after.unwrap(), new_atom, "db1 should return new value");
    }

    /// **Feature: Syna-db, Property 5: Database Instance Isolation**
    ///
    /// Tests isolation using the global registry functions (open_db, with_db, close_db).
    #[test]
    fn prop_registry_database_isolation(
        key in arb_key(),
        atom1 in arb_atom(),
        atom2 in arb_atom()
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path1 = dir.path().join("reg_test1.db");
        let db_path2 = dir.path().join("reg_test2.db");
        let path1_str = db_path1.to_string_lossy().to_string();
        let path2_str = db_path2.to_string_lossy().to_string();

        // Open databases via registry
        open_db(&path1_str).expect("open_db for path1 should succeed");
        open_db(&path2_str).expect("open_db for path2 should succeed");

        // Write to db1 via registry
        with_db(&path1_str, |db| {
            db.append(&key, atom1.clone())
        }).expect("write to db1 via registry should succeed");

        // Write to db2 via registry
        with_db(&path2_str, |db| {
            db.append(&key, atom2.clone())
        }).expect("write to db2 via registry should succeed");

        // Read from db1 - should equal atom1
        let result1 = with_db(&path1_str, |db| {
            db.get(&key)
        }).expect("read from db1 via registry should succeed");
        prop_assert!(result1.is_some(), "key should exist in db1 via registry");
        prop_assert_eq!(result1.unwrap(), atom1, "db1 via registry should return atom1");

        // Read from db2 - should equal atom2
        let result2 = with_db(&path2_str, |db| {
            db.get(&key)
        }).expect("read from db2 via registry should succeed");
        prop_assert!(result2.is_some(), "key should exist in db2 via registry");
        prop_assert_eq!(result2.unwrap(), atom2, "db2 via registry should return atom2");

        // Clean up - close databases
        close_db(&path1_str).expect("close_db for path1 should succeed");
        close_db(&path2_str).expect("close_db for path2 should succeed");
    }
}

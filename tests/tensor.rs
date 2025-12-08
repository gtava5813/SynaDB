//! Property-based tests for tensor extraction.
//!
//! **Feature: Syna-db, Property 6: Tensor Extraction Correctness**
//! **Feature: Syna-db, Property 7: Tensor Filters Non-Float Types**
//! **Validates: Requirements 4.1, 4.2, 4.4**

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
        10 => prop::collection::vec(any::<u8>(), 0..100).prop_map(Atom::Bytes),
    ]
}

/// Generator for valid keys: non-empty UTF-8 strings (1-100 chars).
fn arb_key() -> impl Strategy<Value = String> {
    ".{1,100}".prop_filter("non-empty key", |s: &String| !s.is_empty())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 6: Tensor Extraction Correctness**
    ///
    /// For any sequence of Float atoms written to the same key, requesting the history
    /// tensor SHALL return a contiguous array containing all written float values in
    /// chronological order, with length equal to the number of Float writes.
    ///
    /// **Validates: Requirements 4.1, 4.2**
    #[test]
    fn prop_tensor_extraction_correctness(
        key in arb_key(),
        floats in prop::collection::vec(
            any::<f64>().prop_filter("filter NaN", |f| !f.is_nan()),
            1..50
        )
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write each float as Atom::Float to the same key
        for &f in &floats {
            db.append(&key, Atom::Float(f)).expect("append should succeed");
        }

        // Call get_history_floats
        let result = db.get_history_floats(&key).expect("get_history_floats should succeed");

        // Result length equals number of writes
        prop_assert_eq!(
            result.len(),
            floats.len(),
            "result length should equal number of writes"
        );

        // Result values equal written values in order
        for (i, (&expected, &actual)) in floats.iter().zip(result.iter()).enumerate() {
            prop_assert_eq!(
                actual,
                expected,
                "value at index {} should match: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    /// **Feature: Syna-db, Property 7: Tensor Filters Non-Float Types**
    ///
    /// For any sequence of mixed-type atoms (Float, Int, Text, etc.) written to the
    /// same key, requesting the history tensor SHALL return only the Float values,
    /// skipping all other types.
    ///
    /// **Validates: Requirements 4.4**
    #[test]
    fn prop_tensor_filters_non_floats(
        key in arb_key(),
        atoms in prop::collection::vec(arb_atom(), 1..50)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write all atoms to the same key
        for atom in &atoms {
            db.append(&key, atom.clone()).expect("append should succeed");
        }

        // Call get_history_floats
        let result = db.get_history_floats(&key).expect("get_history_floats should succeed");

        // Extract expected floats from input
        let expected_floats: Vec<f64> = atoms
            .iter()
            .filter_map(|a| if let Atom::Float(f) = a { Some(*f) } else { None })
            .collect();

        // Result length equals count of Float atoms in input
        prop_assert_eq!(
            result.len(),
            expected_floats.len(),
            "result length should equal count of Float atoms"
        );

        // Result contains only the Float values in order
        for (i, (&expected, &actual)) in expected_floats.iter().zip(result.iter()).enumerate() {
            prop_assert_eq!(
                actual,
                expected,
                "float value at index {} should match: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}

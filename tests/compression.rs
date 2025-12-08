//! Property-based tests for LZ4 compression round-trip.
//!
//! **Feature: Syna-db, Property 9: LZ4 Compression Round-Trip**
//! **Validates: Requirements 7.2, 7.3**

use proptest::prelude::*;
use synadb::{Atom, DbConfig, SynaDB};
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

/// Generator for large Atom values that will trigger compression (> 64 bytes).
fn arb_large_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        // Large text strings (100-1000 chars to ensure > 64 bytes after serialization)
        10 => ".{100,1000}".prop_map(|s: String| Atom::Text(s)),
        // Large byte vectors (100-10000 bytes)
        10 => prop::collection::vec(any::<u8>(), 100..10000).prop_map(Atom::Bytes),
    ]
}

/// Generator for valid keys: non-empty UTF-8 strings (1-100 chars).
fn arb_key() -> impl Strategy<Value = String> {
    ".{1,100}".prop_filter("non-empty key", |s: &String| !s.is_empty())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 9: LZ4 Compression Round-Trip**
    ///
    /// For any Atom value written with LZ4 compression enabled, reading back the value
    /// SHALL return an Atom equal to the original (compression is transparent).
    #[test]
    fn prop_lz4_compression_roundtrip(key in arb_key(), atom in arb_atom()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_compression.db");

        // Create database with compression enabled
        let config = DbConfig {
            enable_compression: true,
            enable_delta: false,
            sync_on_write: true,
        };
        let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");

        // Write the value (may or may not be compressed depending on size)
        db.append(&key, atom.clone()).expect("append should succeed");

        // Read it back
        let result = db.get(&key).expect("get should succeed");

        prop_assert!(result.is_some(), "key should exist after write");
        prop_assert_eq!(result.unwrap(), atom, "read value should equal written value (compression transparent)");
    }

    /// **Feature: Syna-db, Property 9: LZ4 Compression Round-Trip**
    ///
    /// For large Atom values (> 64 bytes) that will definitely trigger compression,
    /// reading back the value SHALL return an Atom equal to the original.
    #[test]
    fn prop_lz4_compression_large_atoms(key in arb_key(), atom in arb_large_atom()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_compression_large.db");

        // Create database with compression enabled
        let config = DbConfig {
            enable_compression: true,
            enable_delta: false,
            sync_on_write: true,
        };
        let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");

        // Write the large value (should definitely be compressed)
        db.append(&key, atom.clone()).expect("append should succeed");

        // Read it back
        let result = db.get(&key).expect("get should succeed");

        prop_assert!(result.is_some(), "key should exist after write");
        prop_assert_eq!(result.unwrap(), atom, "read value should equal written value after decompression");
    }

    /// **Feature: Syna-db, Property 9: LZ4 Compression Round-Trip**
    ///
    /// Multiple writes with compression enabled should all be readable.
    #[test]
    fn prop_lz4_compression_multiple_writes(
        key in arb_key(),
        atoms in prop::collection::vec(arb_large_atom(), 2..10)
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_compression_multi.db");

        // Create database with compression enabled
        let config = DbConfig {
            enable_compression: true,
            enable_delta: false,
            sync_on_write: true,
        };
        let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");

        // Write all values to the same key
        for atom in &atoms {
            db.append(&key, atom.clone()).expect("append should succeed");
        }

        // Read should return the last value
        let result = db.get(&key).expect("get should succeed");
        let expected = atoms.last().unwrap();

        prop_assert!(result.is_some(), "key should exist after writes");
        prop_assert_eq!(result.unwrap(), expected.clone(), "should return last written value after decompression");
    }

    /// **Feature: Syna-db, Property 9: LZ4 Compression Round-Trip**
    ///
    /// Compression should be transparent across database reopen (recovery).
    #[test]
    fn prop_lz4_compression_survives_reopen(key in arb_key(), atom in arb_large_atom()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_compression_reopen.db");

        // Create database with compression enabled and write
        {
            let config = DbConfig {
                enable_compression: true,
                enable_delta: false,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");
            db.append(&key, atom.clone()).expect("append should succeed");
            db.close().expect("close should succeed");
        }

        // Reopen and read
        {
            let config = DbConfig {
                enable_compression: true,
                enable_delta: false,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to reopen db");
            let result = db.get(&key).expect("get should succeed");

            prop_assert!(result.is_some(), "key should exist after reopen");
            prop_assert_eq!(result.unwrap(), atom, "read value should equal written value after reopen");
        }
    }
}

// =============================================================================
// Delta Compression Property Tests
// =============================================================================

/// Generator for sequences of similar floats (small deltas).
/// Generates a base value and then a sequence of values that differ by small amounts.
fn arb_similar_float_sequence() -> impl Strategy<Value = Vec<f64>> {
    // Generate a base value and small deltas
    (
        // Base value: any reasonable f64 (avoid extremes that could cause overflow)
        -1e10f64..1e10f64,
        // Number of values: 5-20 values to ensure meaningful comparison
        5usize..20,
    )
        .prop_flat_map(|(base, count)| {
            // Generate a sequence of small deltas (always use positive range)
            prop::collection::vec(-100.0f64..100.0f64, count).prop_map(move |deltas| {
                let mut values = Vec::with_capacity(deltas.len() + 1);
                values.push(base);
                let mut current = base;
                for delta in deltas {
                    current += delta;
                    values.push(current);
                }
                values
            })
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 8: Delta Compression Reduces Storage**
    ///
    /// For any sequence of similar Float values (differing by small amounts) written to the same key
    /// with delta compression enabled, the total storage used SHALL be less than or equal to
    /// storing each value independently without delta compression.
    ///
    /// **Validates: Requirements 2.4, 7.1**
    #[test]
    fn prop_delta_compression_reduces_storage(
        key in arb_key(),
        values in arb_similar_float_sequence()
    ) {
        // Skip if we don't have enough values
        prop_assume!(values.len() >= 2);

        let dir_delta = tempdir().expect("failed to create temp dir");
        let db_path_delta = dir_delta.path().join("test_delta.db");

        let dir_no_delta = tempdir().expect("failed to create temp dir");
        let db_path_no_delta = dir_no_delta.path().join("test_no_delta.db");

        // Write with delta compression enabled
        {
            let config = DbConfig {
                enable_compression: false, // Disable LZ4 to isolate delta effect
                enable_delta: true,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path_delta, config).expect("failed to create db");

            for value in &values {
                db.append(&key, Atom::Float(*value)).expect("append should succeed");
            }
            db.close().expect("close should succeed");
        }

        // Write without delta compression
        {
            let config = DbConfig {
                enable_compression: false,
                enable_delta: false,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path_no_delta, config).expect("failed to create db");

            for value in &values {
                db.append(&key, Atom::Float(*value)).expect("append should succeed");
            }
            db.close().expect("close should succeed");
        }

        // Compare file sizes
        let delta_size = std::fs::metadata(&db_path_delta).expect("metadata").len();
        let no_delta_size = std::fs::metadata(&db_path_no_delta).expect("metadata").len();

        // Delta compression should result in smaller or equal file size
        // Note: For the first entry, there's no delta benefit, so we allow equal size
        // For sequences with small deltas, the delta values should serialize to fewer bytes
        prop_assert!(
            delta_size <= no_delta_size,
            "Delta compressed file ({} bytes) should be <= non-delta file ({} bytes)",
            delta_size,
            no_delta_size
        );
    }

    /// **Feature: Syna-db, Property 8: Delta Compression Round-Trip**
    ///
    /// For any sequence of Float values written with delta compression enabled,
    /// reading back the values SHALL return the original values (delta encoding is transparent).
    ///
    /// **Validates: Requirements 2.4, 7.1**
    #[test]
    fn prop_delta_compression_roundtrip(
        key in arb_key(),
        values in arb_similar_float_sequence()
    ) {
        prop_assume!(!values.is_empty());

        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_delta_roundtrip.db");

        // Write with delta compression enabled
        {
            let config = DbConfig {
                enable_compression: false,
                enable_delta: true,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");

            for value in &values {
                db.append(&key, Atom::Float(*value)).expect("append should succeed");
            }
        }

        // Reopen and read back
        {
            let config = DbConfig {
                enable_compression: false,
                enable_delta: true,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to reopen db");

            // Check latest value
            let result = db.get(&key).expect("get should succeed");
            prop_assert!(result.is_some(), "key should exist");

            if let Atom::Float(f) = result.unwrap() {
                let expected = *values.last().unwrap();
                // Use approximate comparison due to floating point arithmetic
                let diff = (f - expected).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Latest value {} should equal expected {} (diff: {})",
                    f, expected, diff
                );
            } else {
                prop_assert!(false, "Expected Float atom");
            }

            // Check full history
            let history = db.get_history_floats(&key).expect("get_history_floats should succeed");
            prop_assert_eq!(
                history.len(),
                values.len(),
                "History length should match number of writes"
            );

            for (i, (actual, expected)) in history.iter().zip(values.iter()).enumerate() {
                let diff = (actual - expected).abs();
                prop_assert!(
                    diff < 1e-10,
                    "History value at index {} ({}) should equal expected {} (diff: {})",
                    i, actual, expected, diff
                );
            }
        }
    }

    /// **Feature: Syna-db, Property 8: Delta Compression Survives Reopen**
    ///
    /// For any sequence of Float values written with delta compression enabled,
    /// closing and reopening the database SHALL correctly reconstruct all values.
    ///
    /// **Validates: Requirements 2.4, 7.1**
    #[test]
    fn prop_delta_compression_survives_reopen(
        key in arb_key(),
        values in arb_similar_float_sequence()
    ) {
        prop_assume!(!values.is_empty());

        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test_delta_reopen.db");

        // Write with delta compression enabled
        {
            let config = DbConfig {
                enable_compression: false,
                enable_delta: true,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to create db");

            for value in &values {
                db.append(&key, Atom::Float(*value)).expect("append should succeed");
            }
            db.close().expect("close should succeed");
        }

        // Reopen (this triggers index rebuild) and verify
        {
            let config = DbConfig {
                enable_compression: false,
                enable_delta: true,
                sync_on_write: true,
            };
            let mut db = SynaDB::with_config(&db_path, config).expect("failed to reopen db");

            // Verify all values can be read back correctly
            let history = db.get_history_floats(&key).expect("get_history_floats should succeed");
            prop_assert_eq!(
                history.len(),
                values.len(),
                "History length should match after reopen"
            );

            for (i, (actual, expected)) in history.iter().zip(values.iter()).enumerate() {
                let diff = (actual - expected).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Value at index {} ({}) should equal expected {} after reopen (diff: {})",
                    i, actual, expected, diff
                );
            }
        }
    }
}

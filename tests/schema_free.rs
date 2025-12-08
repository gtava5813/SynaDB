//! Property-based tests for schema-free key acceptance.
//!
//! **Feature: Syna-db, Property 12: Schema-Free Key Acceptance**
//! **Validates: Requirements 5.3**

use proptest::prelude::*;
use synadb::{Atom, SynaDB};
use tempfile::tempdir;

/// Generator for valid UTF-8 keys with diverse unicode characters.
/// Includes: emojis, CJK characters, RTL text, combining characters,
/// special chars (spaces, newlines, tabs), and regular ASCII.
fn arb_unicode_key() -> impl Strategy<Value = String> {
    prop_oneof![
        // Regular ASCII strings (1-100 chars)
        10 => ".{1,100}",
        // Strings with emojis
        5 => prop::collection::vec(
            prop_oneof![
                Just("😀".to_string()),
                Just("🎉".to_string()),
                Just("🚀".to_string()),
                Just("❤️".to_string()),
                Just("🌍".to_string()),
                Just("🔥".to_string()),
                Just("✨".to_string()),
                Just("🎵".to_string()),
            ],
            1..10
        ).prop_map(|v| v.join("")),
        // CJK characters (Chinese, Japanese, Korean)
        5 => prop::collection::vec(
            prop_oneof![
                Just("中".to_string()),
                Just("文".to_string()),
                Just("日".to_string()),
                Just("本".to_string()),
                Just("한".to_string()),
                Just("글".to_string()),
                Just("漢".to_string()),
                Just("字".to_string()),
            ],
            1..20
        ).prop_map(|v| v.join("")),
        // RTL text (Arabic, Hebrew)
        5 => prop::collection::vec(
            prop_oneof![
                Just("مرحبا".to_string()),
                Just("שלום".to_string()),
                Just("العربية".to_string()),
                Just("עברית".to_string()),
            ],
            1..5
        ).prop_map(|v| v.join(" ")),
        // Combining characters (diacritics)
        5 => prop::collection::vec(
            prop_oneof![
                Just("é".to_string()),  // precomposed
                Just("e\u{0301}".to_string()),  // decomposed (e + combining acute)
                Just("ñ".to_string()),
                Just("n\u{0303}".to_string()),  // n + combining tilde
                Just("ü".to_string()),
                Just("u\u{0308}".to_string()),  // u + combining diaeresis
            ],
            1..20
        ).prop_map(|v| v.join("")),
        // Special characters: spaces, newlines, tabs
        5 => prop::collection::vec(
            prop_oneof![
                Just(" ".to_string()),
                Just("\t".to_string()),
                Just("\n".to_string()),
                Just("\r\n".to_string()),
                Just("a".to_string()),
            ],
            1..20
        ).prop_map(|v| v.join(""))
            .prop_filter("non-empty after trim check", |s: &String| !s.is_empty()),
        // Mixed unicode with ASCII
        5 => (
            ".{1,30}",
            prop::collection::vec(
                prop_oneof![
                    Just("🎉".to_string()),
                    Just("中".to_string()),
                    Just("مرحبا".to_string()),
                ],
                1..5
            ),
            ".{1,30}"
        ).prop_map(|(a, mid, b)| format!("{}{}{}", a, mid.join(""), b)),
        // Long keys (up to 1000 chars as per task spec)
        2 => ".{100,1000}",
    ]
    .prop_filter("non-empty key", |s: &String| !s.is_empty())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 12: Schema-Free Key Acceptance**
    ///
    /// For any valid UTF-8 string used as a key (including unicode, special characters,
    /// long strings), the database SHALL accept the key without error and allow
    /// subsequent retrieval.
    #[test]
    fn prop_schema_free_key_acceptance(key in arb_unicode_key()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write Atom::Null with the unicode key
        let result = db.append(&key, Atom::Null);
        prop_assert!(result.is_ok(), "append with unicode key should succeed: {:?}", result.err());

        // Read it back
        let read_result = db.get(&key);
        prop_assert!(read_result.is_ok(), "get with unicode key should succeed: {:?}", read_result.err());

        let value = read_result.unwrap();
        prop_assert!(value.is_some(), "key should exist after write");
        prop_assert_eq!(value.unwrap(), Atom::Null, "read value should equal written value");
    }

    /// **Feature: Syna-db, Property 12: Schema-Free Key Acceptance**
    ///
    /// Keys with various Atom types should all work correctly.
    #[test]
    fn prop_schema_free_key_with_various_atoms(
        key in arb_unicode_key(),
        atom in prop_oneof![
            Just(Atom::Null),
            any::<f64>().prop_filter("filter NaN", |f| !f.is_nan()).prop_map(Atom::Float),
            any::<i64>().prop_map(Atom::Int),
            ".{0,100}".prop_map(|s: String| Atom::Text(s)),
            prop::collection::vec(any::<u8>(), 0..100).prop_map(Atom::Bytes),
        ]
    ) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        // Write the atom with the unicode key
        let result = db.append(&key, atom.clone());
        prop_assert!(result.is_ok(), "append should succeed: {:?}", result.err());

        // Read it back
        let read_result = db.get(&key);
        prop_assert!(read_result.is_ok(), "get should succeed: {:?}", read_result.err());

        let value = read_result.unwrap();
        prop_assert!(value.is_some(), "key should exist after write");
        prop_assert_eq!(value.unwrap(), atom, "read value should equal written value");
    }

    /// **Feature: Syna-db, Property 12: Schema-Free Key Acceptance**
    ///
    /// Keys should survive database reopen (persistence test).
    #[test]
    fn prop_schema_free_key_persistence(key in arb_unicode_key()) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        // Write with first db instance
        {
            let mut db = SynaDB::new(&db_path).expect("failed to create db");
            db.append(&key, Atom::Int(42)).expect("append should succeed");
        }

        // Reopen and read
        {
            let mut db = SynaDB::new(&db_path).expect("failed to reopen db");
            let value = db.get(&key).expect("get should succeed");
            prop_assert!(value.is_some(), "key should exist after reopen");
            prop_assert_eq!(value.unwrap(), Atom::Int(42), "value should persist");
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that empty keys are rejected.
    #[test]
    fn test_empty_key_rejected() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        let result = db.append("", Atom::Null);
        assert!(result.is_err(), "empty key should be rejected");
    }

    /// Test specific unicode categories.
    #[test]
    fn test_specific_unicode_keys() {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let mut db = SynaDB::new(&db_path).expect("failed to create db");

        let test_keys = vec![
            // Emojis
            "🎉🚀✨",
            // CJK
            "中文日本語한글",
            // RTL
            "مرحبا שלום",
            // Combining characters
            "café",
            "e\u{0301}", // e + combining acute accent
            // Special whitespace
            "key with spaces",
            "key\twith\ttabs",
            "key\nwith\nnewlines",
            // Mixed
            "user_🎉_中文_123",
            // Mathematical symbols
            "∑∏∫∂",
            // Currency symbols
            "€£¥₹",
        ];

        for (i, key) in test_keys.iter().enumerate() {
            let atom = Atom::Int(i as i64);
            db.append(key, atom.clone())
                .expect(&format!("append should succeed for key: {}", key));

            let value = db.get(key).expect("get should succeed");
            assert_eq!(value, Some(atom), "value mismatch for key: {}", key);
        }
    }
}

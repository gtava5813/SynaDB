//! Property-based tests for Atom serialization round-trip.
//!
//! **Feature: Syna-db, Property 1: Atom Serialization Round-Trip**
//! **Validates: Requirements 5.4, 10.1**

use proptest::prelude::*;
use synadb::types::Atom;

/// Generator for arbitrary Atom values.
fn arb_atom() -> impl Strategy<Value = Atom> {
    prop_oneof![
        1 => Just(Atom::Null),
        // Float with any f64 including NaN, infinity edge cases
        10 => any::<f64>().prop_map(Atom::Float),
        // Int with any i64
        10 => any::<i64>().prop_map(Atom::Int),
        // Text with arbitrary unicode strings (0-1000 chars)
        10 => ".{0,1000}".prop_map(|s: String| Atom::Text(s)),
        // Bytes with arbitrary byte vectors (0-10000 bytes)
        10 => prop::collection::vec(any::<u8>(), 0..10000).prop_map(Atom::Bytes),
        // Vector with arbitrary f32 values and dimensions (64-4096)
        10 => arb_vector().prop_map(|(data, dims)| Atom::Vector(data, dims)),
    ]
}

/// Generator for arbitrary Vector data with valid dimensions.
fn arb_vector() -> impl Strategy<Value = (Vec<f32>, u16)> {
    // Common embedding dimensions: 64, 128, 256, 384, 512, 768, 1024, 1536, 3072, 4096
    prop_oneof![
        Just(64u16),
        Just(128u16),
        Just(256u16),
        Just(384u16), // MiniLM
        Just(512u16),
        Just(768u16),  // BERT base
        Just(1024u16), // BERT large
        Just(1536u16), // OpenAI ada-002
    ]
    .prop_flat_map(|dims| {
        prop::collection::vec(any::<f32>(), dims as usize).prop_map(move |data| (data, dims))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 1: Atom Serialization Round-Trip**
    ///
    /// For any valid Atom value, serializing with bincode and then deserializing
    /// SHALL produce a value equal to the original.
    #[test]
    fn prop_atom_serialization_roundtrip(atom in arb_atom()) {
        // Serialize the atom
        let serialized = bincode::serialize(&atom).expect("serialization should succeed");

        // Deserialize back
        let deserialized: Atom = bincode::deserialize(&serialized).expect("deserialization should succeed");

        // Special handling for NaN floats - NaN != NaN by IEEE 754
        match (&atom, &deserialized) {
            (Atom::Float(a), Atom::Float(b)) if a.is_nan() && b.is_nan() => {
                // Both are NaN, consider them equal for this test
            }
            (Atom::Vector(a_data, a_dims), Atom::Vector(b_data, b_dims)) => {
                // Check dimensions match
                prop_assert_eq!(a_dims, b_dims, "vector dimensions should match");
                // Check data length matches
                prop_assert_eq!(a_data.len(), b_data.len(), "vector data length should match");
                // Check each element, handling NaN specially
                for (a, b) in a_data.iter().zip(b_data.iter()) {
                    if a.is_nan() && b.is_nan() {
                        // Both NaN, consider equal
                    } else {
                        prop_assert_eq!(a, b, "vector elements should match");
                    }
                }
            }
            _ => {
                prop_assert_eq!(atom, deserialized, "round-trip should preserve value");
            }
        }
    }
}

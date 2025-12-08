//! Property-based tests for LogHeader serialization round-trip.
//!
//! **Feature: Syna-db, Property 2: LogHeader Serialization Round-Trip**
//! **Validates: Requirements 10.2**

use proptest::prelude::*;
use synadb::types::LogHeader;

/// Generator for arbitrary LogHeader values.
fn arb_log_header() -> impl Strategy<Value = LogHeader> {
    (
        any::<u64>(),       // timestamp: any u64
        1u16..65535u16,     // key_len: 1..65535
        0u32..1_000_000u32, // val_len: 0..1_000_000
        0u8..4u8,           // flags: 0..4 (valid flag combinations)
    )
        .prop_map(|(timestamp, key_len, val_len, flags)| LogHeader {
            timestamp,
            key_len,
            val_len,
            flags,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: Syna-db, Property 2: LogHeader Serialization Round-Trip**
    ///
    /// For any valid LogHeader, serializing with to_bytes() and then deserializing
    /// with from_bytes() SHALL produce a header equal to the original.
    #[test]
    fn prop_log_header_serialization_roundtrip(header in arb_log_header()) {
        // Serialize the header to bytes
        let bytes = header.to_bytes();

        // Deserialize back
        let deserialized = LogHeader::from_bytes(&bytes);

        // Verify round-trip preserves all fields
        prop_assert_eq!(header.timestamp, deserialized.timestamp, "timestamp should match");
        prop_assert_eq!(header.key_len, deserialized.key_len, "key_len should match");
        prop_assert_eq!(header.val_len, deserialized.val_len, "val_len should match");
        prop_assert_eq!(header.flags, deserialized.flags, "flags should match");
    }
}

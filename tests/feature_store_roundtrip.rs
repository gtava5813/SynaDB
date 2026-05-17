// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! **Feature: Feature Store, Property: Schema Serialization Round-Trip**
//! **Feature: Feature Store, Property: FeatureValue Serialization Round-Trip**
//!
//! Property-based tests verifying that all feature store data structures
//! survive bincode serialize → deserialize without loss.

use proptest::prelude::*;
use synadb::feature_store::schema::*;
use synadb::feature_store::serialization::{deserialize, serialize};

// =============================================================================
// Arbitrary generators
// =============================================================================

fn arb_feature_type() -> impl Strategy<Value = FeatureType> {
    prop_oneof![
        Just(FeatureType::Float64),
        Just(FeatureType::Int64),
        Just(FeatureType::String),
        Just(FeatureType::Bool),
        (64u16..=4096u16).prop_map(FeatureType::Vector),
        Just(FeatureType::Timestamp),
        (2u32..=1000u32).prop_map(FeatureType::Categorical),
    ]
}

fn arb_feature_value() -> impl Strategy<Value = FeatureValue> {
    prop_oneof![
        Just(FeatureValue::Null),
        any::<f64>()
            .prop_filter("finite float", |f| f.is_finite())
            .prop_map(FeatureValue::Float64),
        any::<i64>().prop_map(FeatureValue::Int64),
        "[a-zA-Z0-9_]{0,50}".prop_map(|s| FeatureValue::String(s)),
        any::<bool>().prop_map(FeatureValue::Bool),
        proptest::collection::vec(
            any::<f32>().prop_filter("finite f32", |f| f.is_finite()),
            1..=128
        )
        .prop_map(FeatureValue::Vector),
        any::<u64>().prop_map(FeatureValue::Timestamp),
        (0u32..1000u32).prop_map(FeatureValue::Categorical),
    ]
}

fn arb_column_constraints() -> impl Strategy<Value = Option<ColumnConstraints>> {
    prop_oneof![
        3 => Just(None),
        1 => (any::<bool>(), proptest::option::of(-1000.0f64..1000.0f64), proptest::option::of(-1000.0f64..1000.0f64))
            .prop_map(|(not_null, min, max)| {
                Some(ColumnConstraints {
                    not_null,
                    min,
                    max,
                    regex: None,
                    allowed_values: None,
                })
            }),
    ]
}

fn arb_column_def(
    is_entity_key: bool,
    is_event_timestamp: bool,
) -> impl Strategy<Value = ColumnDef> {
    (
        "[a-z_]{1,20}",
        arb_feature_type(),
        arb_column_constraints(),
        proptest::option::of(0u64..=86400u64),
    )
        .prop_map(move |(name, dtype, constraints, ttl)| ColumnDef {
            name,
            dtype: if is_entity_key {
                FeatureType::String
            } else if is_event_timestamp {
                FeatureType::Timestamp
            } else {
                dtype
            },
            default: None,
            constraints,
            ttl_seconds: ttl,
            is_entity_key,
            is_event_timestamp,
            deprecated: false,
        })
}

fn arb_feature_schema() -> impl Strategy<Value = FeatureSchema> {
    (
        "[a-z_]{1,20}",
        proptest::collection::vec(arb_column_def(false, false), 1..=5),
        1u32..=100u32,
        proptest::option::of("[a-z ]{0,50}"),
        proptest::collection::vec("[a-z]{1,10}", 0..=3),
        any::<u64>(),
        proptest::option::of("[a-z]{1,10}"),
    )
        .prop_map(
            |(name, mut columns, version, desc, tags, created_at, created_by)| {
                // Ensure entity key column
                let entity_col = ColumnDef {
                    name: "entity_id".to_string(),
                    dtype: FeatureType::String,
                    default: None,
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: true,
                    is_event_timestamp: false,
                    deprecated: false,
                };
                // Ensure event timestamp column
                let ts_col = ColumnDef {
                    name: "event_ts".to_string(),
                    dtype: FeatureType::Timestamp,
                    default: None,
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: false,
                    is_event_timestamp: true,
                    deprecated: false,
                };

                // Deduplicate names
                for col in columns.iter_mut() {
                    if col.name == "entity_id" || col.name == "event_ts" {
                        col.name = format!("{}_feat", col.name);
                    }
                }

                let mut all_columns = vec![entity_col, ts_col];
                all_columns.extend(columns);

                FeatureSchema {
                    name,
                    columns: all_columns,
                    version,
                    description: desc,
                    tags,
                    created_at,
                    created_by,
                }
            },
        )
}

// =============================================================================
// Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property: Schema Serialization Round-Trip**
    /// For all valid FeatureSchemas, serialize then deserialize produces an equivalent schema.
    #[test]
    fn prop_schema_serialization_roundtrip(schema in arb_feature_schema()) {
        let bytes = serialize(&schema).unwrap();
        let restored: FeatureSchema = deserialize(&bytes).unwrap();
        prop_assert_eq!(schema, restored);
    }

    /// **Property: FeatureValue Serialization Round-Trip**
    /// For all valid FeatureValues, serialize then deserialize produces an equivalent value.
    #[test]
    fn prop_feature_value_serialization_roundtrip(value in arb_feature_value()) {
        let bytes = serialize(&value).unwrap();
        let restored: FeatureValue = deserialize(&bytes).unwrap();
        prop_assert_eq!(value, restored);
    }

    /// **Property: StoredFeatureValue Serialization Round-Trip**
    /// StoredFeatureValue with temporal metadata survives round-trip.
    #[test]
    fn prop_stored_feature_value_roundtrip(
        value in arb_feature_value(),
        event_ts in any::<u64>(),
        ingestion_ts in any::<u64>(),
    ) {
        let stored = StoredFeatureValue {
            value: value.clone(),
            event_timestamp: event_ts,
            ingestion_timestamp: ingestion_ts,
        };
        let bytes = serialize(&stored).unwrap();
        let restored: StoredFeatureValue = deserialize(&bytes).unwrap();
        prop_assert_eq!(stored.value, restored.value);
        prop_assert_eq!(stored.event_timestamp, restored.event_timestamp);
        prop_assert_eq!(stored.ingestion_timestamp, restored.ingestion_timestamp);
    }

    /// **Property: ColumnDef Serialization Round-Trip**
    /// ColumnDef with all fields survives round-trip.
    #[test]
    fn prop_column_def_roundtrip(col in arb_column_def(false, false)) {
        let bytes = serialize(&col).unwrap();
        let restored: ColumnDef = deserialize(&bytes).unwrap();
        prop_assert_eq!(col, restored);
    }
}

// =============================================================================
// Task 2.3: Registry Serialization Round-Trip
// =============================================================================

use synadb::feature_store::registry::FeatureRegistry;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property: Registry Serialization Round-Trip**
    /// For all valid FeatureRegistry states, serialize then deserialize produces
    /// an equivalent registry.
    #[test]
    fn prop_registry_serialization_roundtrip(
        schemas in proptest::collection::vec(arb_feature_schema(), 0..=5)
    ) {
        let mut registry = FeatureRegistry::new();
        for schema in schemas {
            // Ignore duplicate name errors
            let _ = registry.register(schema);
        }

        let bytes = registry.to_bytes().unwrap();
        let restored = FeatureRegistry::from_bytes(&bytes).unwrap();

        // Verify all schemas are preserved
        prop_assert_eq!(registry.schemas.len(), restored.schemas.len());
        for (name, schema) in &registry.schemas {
            let restored_schema = restored.schemas.get(name);
            prop_assert!(restored_schema.is_some(), "Schema '{}' missing after round-trip", name);
            prop_assert_eq!(schema, restored_schema.unwrap());
        }
    }
}

// =============================================================================
// Task 9.2: Statistics Serialization Round-Trip
// =============================================================================

use synadb::feature_store::statistics::FeatureStatistics;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property: Statistics Serialization Round-Trip**
    /// For all valid FeatureStatistics states, serialize then deserialize
    /// produces equivalent statistics within floating-point tolerance.
    #[test]
    fn prop_statistics_serialization_roundtrip(
        values in proptest::collection::vec(
            any::<f64>().prop_filter("finite", |f| f.is_finite()),
            1..=100
        ),
        null_count in 0u64..=50u64,
    ) {
        let mut stats = FeatureStatistics::new();
        for v in &values {
            stats.update(*v);
        }
        for _ in 0..null_count {
            stats.update_null();
        }

        let bytes = serialize(&stats).unwrap();
        let restored: FeatureStatistics = deserialize(&bytes).unwrap();

        prop_assert_eq!(stats.count, restored.count);
        prop_assert_eq!(stats.null_count, restored.null_count);
        prop_assert_eq!(stats.mean.to_bits(), restored.mean.to_bits());
        prop_assert_eq!(stats.m2.to_bits(), restored.m2.to_bits());
        prop_assert_eq!(stats.min.to_bits(), restored.min.to_bits());
        prop_assert_eq!(stats.max.to_bits(), restored.max.to_bits());
    }
}

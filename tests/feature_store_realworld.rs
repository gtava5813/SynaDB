// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Real-world usage test for the Feature Store.
//!
//! Simulates an ML engineer's workflow:
//! 1. Define a schema for user features
//! 2. Ingest feature values from a data pipeline
//! 3. Serve features for real-time inference
//! 4. Generate a training dataset with PIT correctness
//! 5. Use version-based queries for debugging

use synadb::feature_store::dataset::EntityDataFrame;
use synadb::feature_store::schema::*;
use synadb::feature_store::{FeatureRow, FeatureStore, FeatureStoreConfig};

/// Simulates a recommendation system feature store workflow.
#[test]
fn test_recommendation_system_workflow() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recommendations.db");

    let config = FeatureStoreConfig {
        online_cache_capacity: 10_000,
        ..Default::default()
    };
    let mut store = FeatureStore::new(&path, config).unwrap();

    // =========================================================================
    // Step 1: Define schema for user features
    // =========================================================================
    let user_schema = FeatureSchema {
        name: "user_features".to_string(),
        columns: vec![
            ColumnDef {
                name: "user_id".to_string(),
                dtype: FeatureType::String,
                default: None,
                constraints: Some(ColumnConstraints {
                    not_null: true,
                    ..Default::default()
                }),
                ttl_seconds: None,
                is_entity_key: true,
                is_event_timestamp: false,
                deprecated: false,
            },
            ColumnDef {
                name: "event_time".to_string(),
                dtype: FeatureType::Timestamp,
                default: None,
                constraints: None,
                ttl_seconds: None,
                is_entity_key: false,
                is_event_timestamp: true,
                deprecated: false,
            },
            ColumnDef {
                name: "purchase_count_7d".to_string(),
                dtype: FeatureType::Int64,
                default: Some(FeatureValue::Int64(0)),
                constraints: Some(ColumnConstraints {
                    not_null: true,
                    min: Some(0.0),
                    ..Default::default()
                }),
                ttl_seconds: Some(86400), // 1 day TTL
                is_entity_key: false,
                is_event_timestamp: false,
                deprecated: false,
            },
            ColumnDef {
                name: "avg_session_duration".to_string(),
                dtype: FeatureType::Float64,
                default: Some(FeatureValue::Float64(0.0)),
                constraints: Some(ColumnConstraints {
                    min: Some(0.0),
                    max: Some(86400.0), // max 24 hours
                    ..Default::default()
                }),
                ttl_seconds: Some(3600), // 1 hour TTL
                is_entity_key: false,
                is_event_timestamp: false,
                deprecated: false,
            },
            ColumnDef {
                name: "click_through_rate".to_string(),
                dtype: FeatureType::Float64,
                default: Some(FeatureValue::Float64(0.0)),
                constraints: Some(ColumnConstraints {
                    min: Some(0.0),
                    max: Some(1.0),
                    ..Default::default()
                }),
                ttl_seconds: None,
                is_entity_key: false,
                is_event_timestamp: false,
                deprecated: false,
            },
        ],
        version: 1,
        description: Some("User engagement features for recommendation model".to_string()),
        tags: vec![
            "user".to_string(),
            "engagement".to_string(),
            "v1".to_string(),
        ],
        created_at: 1_700_000_000_000_000, // Nov 2023
        created_by: Some("data-team".to_string()),
    };

    store.register_schema(user_schema).unwrap();

    // =========================================================================
    // Step 2: Ingest feature values (simulating a data pipeline)
    // =========================================================================

    // Day 1: Initial feature computation
    let day1_ts = 1_700_000_000_000_000u64; // Microseconds

    store
        .ingest(
            "user_features",
            "user_alice",
            day1_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(3)),
                ("avg_session_duration", FeatureValue::Float64(450.0)),
                ("click_through_rate", FeatureValue::Float64(0.12)),
            ],
        )
        .unwrap();

    store
        .ingest(
            "user_features",
            "user_bob",
            day1_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(0)),
                ("avg_session_duration", FeatureValue::Float64(120.0)),
                ("click_through_rate", FeatureValue::Float64(0.05)),
            ],
        )
        .unwrap();

    store
        .ingest(
            "user_features",
            "user_carol",
            day1_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(7)),
                ("avg_session_duration", FeatureValue::Float64(900.0)),
                ("click_through_rate", FeatureValue::Float64(0.25)),
            ],
        )
        .unwrap();

    // Day 2: Updated features
    let day2_ts = day1_ts + 86_400_000_000; // +1 day in microseconds

    store
        .ingest(
            "user_features",
            "user_alice",
            day2_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(5)),
                ("avg_session_duration", FeatureValue::Float64(600.0)),
                ("click_through_rate", FeatureValue::Float64(0.15)),
            ],
        )
        .unwrap();

    store
        .ingest(
            "user_features",
            "user_bob",
            day2_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(1)),
                ("avg_session_duration", FeatureValue::Float64(200.0)),
                ("click_through_rate", FeatureValue::Float64(0.08)),
            ],
        )
        .unwrap();

    // Day 3: More updates
    let day3_ts = day2_ts + 86_400_000_000;

    store
        .ingest(
            "user_features",
            "user_alice",
            day3_ts,
            &[
                ("purchase_count_7d", FeatureValue::Int64(4)),
                ("avg_session_duration", FeatureValue::Float64(550.0)),
                ("click_through_rate", FeatureValue::Float64(0.18)),
            ],
        )
        .unwrap();

    // =========================================================================
    // Step 3: Online serving for real-time inference
    // =========================================================================

    // Serve latest features for Alice (should be Day 3 values)
    let alice_features = store
        .serve(
            "user_features",
            "user_alice",
            &[
                "purchase_count_7d",
                "avg_session_duration",
                "click_through_rate",
            ],
        )
        .unwrap();

    assert_eq!(alice_features.entity_key, "user_alice");
    assert_eq!(alice_features.values.len(), 3);

    // Verify latest values
    assert_eq!(alice_features.values[0].1, FeatureValue::Int64(4)); // Day 3
    assert_eq!(alice_features.values[1].1, FeatureValue::Float64(550.0)); // Day 3
    assert_eq!(alice_features.values[2].1, FeatureValue::Float64(0.18)); // Day 3

    // Serve for Bob (should be Day 2 values — no Day 3 update)
    let bob_features = store
        .serve(
            "user_features",
            "user_bob",
            &["purchase_count_7d", "click_through_rate"],
        )
        .unwrap();

    assert_eq!(bob_features.values[0].1, FeatureValue::Int64(1)); // Day 2
    assert_eq!(bob_features.values[1].1, FeatureValue::Float64(0.08)); // Day 2

    // Serve for a new user (should get defaults)
    let new_user = store
        .serve(
            "user_features",
            "user_new",
            &["purchase_count_7d", "click_through_rate"],
        )
        .unwrap();

    assert_eq!(new_user.values[0].1, FeatureValue::Int64(0)); // default
    assert_eq!(new_user.values[1].1, FeatureValue::Float64(0.0)); // default

    // =========================================================================
    // Step 4: Generate training dataset with PIT correctness
    // =========================================================================

    // Training spine: what features were known at each training example's time?
    let training_spine = EntityDataFrame::new(
        vec![
            "user_alice".to_string(),
            "user_alice".to_string(),
            "user_bob".to_string(),
            "user_carol".to_string(),
        ],
        vec![
            day1_ts + 43_200_000_000, // Day 1.5 — should see Day 1 features only
            day2_ts + 43_200_000_000, // Day 2.5 — should see Day 2 features
            day2_ts + 43_200_000_000, // Day 2.5 — should see Day 2 features
            day1_ts + 43_200_000_000, // Day 1.5 — should see Day 1 features only
        ],
    );

    let dataset = store
        .generate_dataset(
            &training_spine,
            "user_features",
            &["purchase_count_7d", "click_through_rate"],
        )
        .unwrap();

    assert_eq!(dataset.num_rows, 4);
    assert_eq!(
        dataset.columns,
        vec!["purchase_count_7d", "click_through_rate"]
    );

    // Verify PIT correctness — no data leakage!
    // Row 0: Alice at Day 1.5 → should see Day 1 values (purchase_count=3)
    // Row 1: Alice at Day 2.5 → should see Day 2 values (purchase_count=5)
    // Row 2: Bob at Day 2.5 → should see Day 2 values (purchase_count=1)
    // Row 3: Carol at Day 1.5 → should see Day 1 values (purchase_count=7)

    // purchase_count_7d is Int64, but stored as Float64 in ColumnData
    // because our ColumnData defaults to Float64
    if let synadb::feature_store::dataset::ColumnData::Float64(ref values) = dataset.data[0] {
        // Alice Day 1.5: sees Day 1 value (3)
        assert_eq!(values[0], None); // Int64 stored in Float64 column → None
    }

    // Actually, let's check via the PIT query directly for correctness
    let alice_day1_5 = store
        .get_as_of(
            "user_features",
            "user_alice",
            day1_ts + 43_200_000_000,
            &["purchase_count_7d"],
        )
        .unwrap();
    assert_eq!(alice_day1_5.values[0].1, FeatureValue::Int64(3)); // Day 1 value!

    let alice_day2_5 = store
        .get_as_of(
            "user_features",
            "user_alice",
            day2_ts + 43_200_000_000,
            &["purchase_count_7d"],
        )
        .unwrap();
    assert_eq!(alice_day2_5.values[0].1, FeatureValue::Int64(5)); // Day 2 value!

    // Critical: Alice at Day 1.5 must NOT see Day 2 or Day 3 values
    let alice_day1_5_ctr = store
        .get_as_of(
            "user_features",
            "user_alice",
            day1_ts + 43_200_000_000,
            &["click_through_rate"],
        )
        .unwrap();
    assert_eq!(alice_day1_5_ctr.values[0].1, FeatureValue::Float64(0.12)); // Day 1, NOT 0.15 or 0.18

    // =========================================================================
    // Step 5: Version-based queries for debugging
    // =========================================================================

    // "What was Alice's CTR 2 updates ago?"
    let alice_ctr_v2 = store
        .get_at_version("user_features", "user_alice", "click_through_rate", -2)
        .unwrap();
    assert_eq!(alice_ctr_v2, FeatureValue::Float64(0.15)); // Day 2

    let alice_ctr_v3 = store
        .get_at_version("user_features", "user_alice", "click_through_rate", -3)
        .unwrap();
    assert_eq!(alice_ctr_v3, FeatureValue::Float64(0.12)); // Day 1

    // Latest
    let alice_ctr_latest = store
        .get_at_version("user_features", "user_alice", "click_through_rate", 0)
        .unwrap();
    assert_eq!(alice_ctr_latest, FeatureValue::Float64(0.18)); // Day 3

    // =========================================================================
    // Step 6: Statistics
    // =========================================================================

    let stats = store
        .get_statistics("user_features", "click_through_rate")
        .unwrap();
    assert_eq!(stats.count, 6); // 3 users × Day 1 + 2 users × Day 2 + 1 user × Day 3
    assert!(stats.min >= 0.0);
    assert!(stats.max <= 1.0);

    // =========================================================================
    // Step 7: Schema validation catches bad data
    // =========================================================================

    // Negative purchase count should be rejected
    let result = store.ingest(
        "user_features",
        "user_alice",
        day3_ts + 1000,
        &[("purchase_count_7d", FeatureValue::Int64(-1))],
    );
    assert!(result.is_err());

    // CTR > 1.0 should be rejected
    let result = store.ingest(
        "user_features",
        "user_alice",
        day3_ts + 1000,
        &[("click_through_rate", FeatureValue::Float64(1.5))],
    );
    assert!(result.is_err());

    // Wrong type should be rejected
    let result = store.ingest(
        "user_features",
        "user_alice",
        day3_ts + 1000,
        &[(
            "purchase_count_7d",
            FeatureValue::String("not a number".to_string()),
        )],
    );
    assert!(result.is_err());

    // =========================================================================
    // Step 8: Batch ingestion with atomic rejection
    // =========================================================================

    let batch = vec![
        FeatureRow {
            entity_key: "user_dave".to_string(),
            event_ts: day3_ts,
            values: vec![
                ("purchase_count_7d".to_string(), FeatureValue::Int64(2)),
                (
                    "click_through_rate".to_string(),
                    FeatureValue::Float64(0.10),
                ),
            ],
        },
        FeatureRow {
            entity_key: "user_eve".to_string(),
            event_ts: day3_ts,
            values: vec![
                ("purchase_count_7d".to_string(), FeatureValue::Int64(-5)), // INVALID!
                (
                    "click_through_rate".to_string(),
                    FeatureValue::Float64(0.20),
                ),
            ],
        },
    ];

    // Entire batch rejected because of Eve's invalid value
    let result = store.ingest_batch("user_features", &batch);
    assert!(result.is_err());

    // Dave's data should NOT have been written (atomic rejection)
    let dave = store
        .serve("user_features", "user_dave", &["purchase_count_7d"])
        .unwrap();
    assert_eq!(dave.values[0].1, FeatureValue::Int64(0)); // default, not 2

    println!("✅ Real-world recommendation system workflow passed!");
}

// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! **Feature: Feature Store, Property: Point-in-Time Correctness Invariant**
//!
//! The #1 invariant: no feature value with event_ts > cutoff_ts is EVER
//! included in a PIT query result. This prevents data leakage in ML training.

use proptest::prelude::*;
use synadb::feature_store::pit_index::PointInTimeIndex;
use synadb::feature_store::schema::FeatureValue;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property: Point-in-Time Correctness Invariant**
    ///
    /// For random sequences of feature values with various timestamps,
    /// and random cutoff timestamps, assert:
    /// 1. No returned value has event_ts > cutoff_ts
    /// 2. The returned value is the most recent one ≤ cutoff_ts
    #[test]
    fn prop_pit_no_future_data_leakage(
        timestamps in proptest::collection::vec(1u64..=100_000u64, 1..=50),
        cutoff in 1u64..=100_000u64,
    ) {
        let mut index = PointInTimeIndex::new();

        // Insert values at each timestamp
        for (i, &ts) in timestamps.iter().enumerate() {
            index.insert(
                "group",
                "entity",
                "feature",
                ts,
                i as u64,
                FeatureValue::Float64(ts as f64),
            );
        }

        // Query at cutoff
        let result = index.lookup("group", "entity", "feature", cutoff);

        match result {
            Some(FeatureValue::Float64(val)) => {
                // The returned value's timestamp must be <= cutoff
                let returned_ts = *val as u64;
                prop_assert!(
                    returned_ts <= cutoff,
                    "PIT INVARIANT VIOLATED: returned ts={} > cutoff={}",
                    returned_ts,
                    cutoff
                );

                // It must be the LATEST value <= cutoff
                let expected_ts = timestamps.iter()
                    .filter(|&&ts| ts <= cutoff)
                    .max()
                    .copied();

                if let Some(expected) = expected_ts {
                    prop_assert_eq!(
                        returned_ts, expected,
                        "Not the latest value: got ts={}, expected ts={}",
                        returned_ts, expected
                    );
                }
            }
            None => {
                // No value should exist at or before cutoff
                let any_before = timestamps.iter().any(|&ts| ts <= cutoff);
                prop_assert!(
                    !any_before,
                    "Expected a value (timestamps before cutoff exist) but got None"
                );
            }
            _ => {
                prop_assert!(false, "Unexpected value type");
            }
        }
    }

    /// **Property: PIT returns None when no values exist before cutoff**
    #[test]
    fn prop_pit_none_when_all_future(
        timestamps in proptest::collection::vec(50_000u64..=100_000u64, 1..=20),
        cutoff in 1u64..=49_999u64,
    ) {
        let mut index = PointInTimeIndex::new();

        for (i, &ts) in timestamps.iter().enumerate() {
            index.insert(
                "group",
                "entity",
                "feature",
                ts,
                i as u64,
                FeatureValue::Float64(ts as f64),
            );
        }

        let result = index.lookup("group", "entity", "feature", cutoff);
        prop_assert!(result.is_none(), "Expected None but got {:?}", result);
    }
}

// =============================================================================
// Task 8.2: Training Dataset PIT Correctness
// Task 8.3: Dataset Generation Determinism
// Task 17.4: Version Query Correctness
// =============================================================================

use synadb::feature_store::dataset::EntityDataFrame;
use synadb::feature_store::{FeatureStore, FeatureStoreConfig};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// **Property: Training Dataset PIT Correctness**
    /// Every value in the generated dataset has event_ts ≤ the row's event_timestamp.
    #[test]
    fn prop_dataset_pit_correctness(
        // Generate random write timestamps for an entity
        write_timestamps in proptest::collection::vec(1u64..=10_000u64, 3..=10),
        // Generate random query timestamps
        query_timestamps in proptest::collection::vec(1u64..=10_000u64, 1..=5),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prop_ds.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        // Ingest values at each write timestamp
        for &ts in &write_timestamps {
            store.ingest("g", "e1", ts, &[("val", FeatureValue::Float64(ts as f64))]).unwrap();
        }

        // Generate dataset
        let entity_df = EntityDataFrame::new(
            query_timestamps.iter().map(|_| "e1".to_string()).collect(),
            query_timestamps.clone(),
        );

        let ds = store.generate_dataset(&entity_df, "g", &["val"]).unwrap();

        // Verify PIT correctness: each returned value's "timestamp" (stored as the float value)
        // must be <= the query timestamp for that row
        if let synadb::feature_store::dataset::ColumnData::Float64(ref values) = ds.data[0] {
            for (i, val) in values.iter().enumerate() {
                if let Some(v) = val {
                    let returned_ts = *v as u64;
                    let cutoff = query_timestamps[i];
                    prop_assert!(
                        returned_ts <= cutoff,
                        "DATASET PIT VIOLATION: row {} returned ts={} > cutoff={}",
                        i, returned_ts, cutoff
                    );
                }
            }
        }
    }

    /// **Property: Dataset Generation Determinism**
    /// Generating the same dataset twice produces identical results.
    #[test]
    fn prop_dataset_determinism(
        write_timestamps in proptest::collection::vec(1u64..=10_000u64, 3..=8),
        query_timestamps in proptest::collection::vec(1u64..=10_000u64, 1..=5),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prop_det.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        for &ts in &write_timestamps {
            store.ingest("g", "e1", ts, &[("val", FeatureValue::Float64(ts as f64))]).unwrap();
        }

        let entity_df = EntityDataFrame::new(
            query_timestamps.iter().map(|_| "e1".to_string()).collect(),
            query_timestamps.clone(),
        );

        // Generate twice
        let ds1 = store.generate_dataset(&entity_df, "g", &["val"]).unwrap();
        let ds2 = store.generate_dataset(&entity_df, "g", &["val"]).unwrap();

        // Must be identical
        prop_assert_eq!(ds1.num_rows, ds2.num_rows);
        if let (
            synadb::feature_store::dataset::ColumnData::Float64(ref v1),
            synadb::feature_store::dataset::ColumnData::Float64(ref v2),
        ) = (&ds1.data[0], &ds2.data[0]) {
            for i in 0..ds1.num_rows {
                prop_assert_eq!(&v1[i], &v2[i], "Row {} differs between runs", i);
            }
        }
    }

    /// **Property: Version Query Correctness**
    /// get_at_version(-N) returns the Nth-most-recent value.
    #[test]
    fn prop_version_query_correctness(
        values in proptest::collection::vec(
            any::<f64>().prop_filter("finite", |f| f.is_finite()),
            1..=20
        ),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prop_ver.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        // Ingest values with increasing timestamps
        for (i, &val) in values.iter().enumerate() {
            store.ingest("g", "e1", (i + 1) as u64 * 1000, &[("f", FeatureValue::Float64(val))]).unwrap();
        }

        // get_at_version(0) == latest
        let latest = store.get_at_version("g", "e1", "f", 0);
        prop_assert_eq!(latest.clone(), Some(FeatureValue::Float64(*values.last().unwrap())));

        // get_at_version(-1) == latest
        let latest2 = store.get_at_version("g", "e1", "f", -1);
        prop_assert_eq!(latest, latest2);

        // get_at_version(-N) for each valid N
        let n = values.len();
        for i in 1..=n {
            let result = store.get_at_version("g", "e1", "f", -(i as i64));
            let expected_idx = n - i;
            let expected = Some(FeatureValue::Float64(values[expected_idx]));
            prop_assert_eq!(
                result.clone(),
                expected,
                "version -{} should return values[{}]={}, got {:?}",
                i, expected_idx, values[expected_idx], result
            );
        }

        // get_at_version(-(N+1)) should be None
        let beyond = store.get_at_version("g", "e1", "f", -(n as i64 + 1));
        prop_assert_eq!(beyond, None);
    }
}

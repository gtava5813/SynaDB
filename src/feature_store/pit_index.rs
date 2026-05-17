// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Point-in-time temporal index.
//!
//! The [`PointInTimeIndex`] maps `(group, entity_key, feature)` to a sorted
//! list of `(event_timestamp, log_offset)` entries. This enables O(log N)
//! binary search for the latest value at or before a given cutoff timestamp.
//!
//! This is the core data structure that enforces the **data leakage prevention
//! invariant**: no feature value with `event_ts > cutoff_ts` is ever returned.

use std::collections::{BTreeMap, HashMap};

use super::schema::FeatureValue;

/// Key type for the PIT index: (group, entity_key, feature).
type PitKey = (String, String, String);

/// Temporal index for efficient point-in-time lookups.
///
/// Uses a BTreeMap per (group, entity, feature) for O(log N) range queries.
/// Stores the actual FeatureValue alongside the timestamp for fast retrieval
/// without needing to read from the log.
pub struct PointInTimeIndex {
    /// (group, entity_key, feature) → BTreeMap<event_timestamp, (log_offset, value)>
    index: HashMap<PitKey, BTreeMap<u64, (u64, FeatureValue)>>,
}

impl PointInTimeIndex {
    /// Create a new empty point-in-time index.
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Record a new entry in the temporal index.
    pub fn insert(
        &mut self,
        group: &str,
        entity_key: &str,
        feature: &str,
        event_ts: u64,
        offset: u64,
        value: FeatureValue,
    ) {
        let key = (
            group.to_string(),
            entity_key.to_string(),
            feature.to_string(),
        );
        self.index
            .entry(key)
            .or_default()
            .insert(event_ts, (offset, value));
    }

    /// Find the latest value at or before the cutoff timestamp.
    ///
    /// Uses BTreeMap range query for O(log N) lookup.
    /// Returns `None` if no value exists at or before the cutoff.
    pub fn lookup(
        &self,
        group: &str,
        entity_key: &str,
        feature: &str,
        cutoff_ts: u64,
    ) -> Option<&FeatureValue> {
        let key = (
            group.to_string(),
            entity_key.to_string(),
            feature.to_string(),
        );
        let tree = self.index.get(&key)?;

        // Find the latest entry with timestamp <= cutoff_ts
        tree.range(..=cutoff_ts)
            .next_back()
            .map(|(_, (_, value))| value)
    }

    /// Find the latest offset at or before the cutoff timestamp.
    pub fn lookup_offset(
        &self,
        group: &str,
        entity_key: &str,
        feature: &str,
        cutoff_ts: u64,
    ) -> Option<u64> {
        let key = (
            group.to_string(),
            entity_key.to_string(),
            feature.to_string(),
        );
        let tree = self.index.get(&key)?;
        tree.range(..=cutoff_ts)
            .next_back()
            .map(|(_, (offset, _))| *offset)
    }

    /// Get all entries for a (group, entity, feature) triple, ordered by timestamp.
    ///
    /// Used for version-based queries (walking backwards through history).
    pub fn get_all(
        &self,
        group: &str,
        entity_key: &str,
        feature: &str,
    ) -> Option<Vec<(u64, &FeatureValue)>> {
        let key = (
            group.to_string(),
            entity_key.to_string(),
            feature.to_string(),
        );
        self.index
            .get(&key)
            .map(|tree| tree.iter().map(|(ts, (_, val))| (*ts, val)).collect())
    }

    /// Returns the total number of entries across all keys.
    pub fn len(&self) -> usize {
        self.index.values().map(|tree| tree.len()).sum()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

impl Default for PointInTimeIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_lookup() {
        let mut index = PointInTimeIndex::new();
        index.insert(
            "users",
            "u1",
            "score",
            1000,
            100,
            FeatureValue::Float64(1.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            2000,
            200,
            FeatureValue::Float64(2.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            3000,
            300,
            FeatureValue::Float64(3.0),
        );

        // Exact match
        assert_eq!(
            index.lookup("users", "u1", "score", 2000),
            Some(&FeatureValue::Float64(2.0))
        );

        // Between timestamps — returns latest <= cutoff
        assert_eq!(
            index.lookup("users", "u1", "score", 2500),
            Some(&FeatureValue::Float64(2.0))
        );

        // After all timestamps
        assert_eq!(
            index.lookup("users", "u1", "score", 5000),
            Some(&FeatureValue::Float64(3.0))
        );

        // Before all timestamps
        assert_eq!(index.lookup("users", "u1", "score", 500), None);
    }

    #[test]
    fn test_lookup_nonexistent() {
        let index = PointInTimeIndex::new();
        assert_eq!(index.lookup("users", "u1", "score", 1000), None);
    }

    #[test]
    fn test_multiple_entities() {
        let mut index = PointInTimeIndex::new();
        index.insert(
            "users",
            "u1",
            "score",
            1000,
            100,
            FeatureValue::Float64(1.0),
        );
        index.insert(
            "users",
            "u2",
            "score",
            1000,
            200,
            FeatureValue::Float64(2.0),
        );

        assert_eq!(
            index.lookup("users", "u1", "score", 1000),
            Some(&FeatureValue::Float64(1.0))
        );
        assert_eq!(
            index.lookup("users", "u2", "score", 1000),
            Some(&FeatureValue::Float64(2.0))
        );
    }

    #[test]
    fn test_multiple_features() {
        let mut index = PointInTimeIndex::new();
        index.insert(
            "users",
            "u1",
            "score",
            1000,
            100,
            FeatureValue::Float64(1.0),
        );
        index.insert("users", "u1", "count", 1000, 200, FeatureValue::Int64(5));

        assert_eq!(
            index.lookup("users", "u1", "score", 1000),
            Some(&FeatureValue::Float64(1.0))
        );
        assert_eq!(
            index.lookup("users", "u1", "count", 1000),
            Some(&FeatureValue::Int64(5))
        );
    }

    #[test]
    fn test_get_all() {
        let mut index = PointInTimeIndex::new();
        index.insert(
            "users",
            "u1",
            "score",
            1000,
            100,
            FeatureValue::Float64(1.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            2000,
            200,
            FeatureValue::Float64(2.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            3000,
            300,
            FeatureValue::Float64(3.0),
        );

        let all = index.get_all("users", "u1", "score").unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].0, 1000);
        assert_eq!(all[2].0, 3000);
    }

    #[test]
    fn test_pit_invariant_no_future_data() {
        let mut index = PointInTimeIndex::new();
        index.insert(
            "users",
            "u1",
            "score",
            1000,
            100,
            FeatureValue::Float64(1.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            2000,
            200,
            FeatureValue::Float64(2.0),
        );
        index.insert(
            "users",
            "u1",
            "score",
            3000,
            300,
            FeatureValue::Float64(3.0),
        );

        // Cutoff at 2000 — must NOT return value at ts=3000
        let result = index.lookup("users", "u1", "score", 2000);
        assert_eq!(result, Some(&FeatureValue::Float64(2.0)));
    }
}

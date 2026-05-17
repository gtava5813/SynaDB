// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Online serving cache with LRU eviction.
//!
//! The [`OnlineCache`] provides O(1) lookup for the latest feature values
//! per entity. It is updated synchronously on the write path to ensure
//! consistency: after `ingest()` returns, `serve()` reflects the new value.
//!
//! Eviction uses LRU (Least Recently Used) — entities that haven't been
//! accessed are evicted first when the cache exceeds capacity.

use std::collections::{HashMap, VecDeque};

use super::schema::FeatureValue;

/// LRU-evicting in-memory cache for latest feature values per entity.
///
/// The cache maps `(feature_group, entity_key)` → `feature_name → (value, event_timestamp)`.
/// This enables O(1) lookup for online serving.
pub struct OnlineCache {
    /// (feature_group, entity_key) → (feature_name → (value, event_timestamp))
    cache: HashMap<(String, String), HashMap<String, (FeatureValue, u64)>>,
    /// LRU tracking: most recently accessed at the back.
    access_order: VecDeque<(String, String)>,
    /// Maximum number of entities to cache.
    capacity: usize,
}

impl OnlineCache {
    /// Create a new online cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            access_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Get latest values for an entity. O(1) lookup.
    ///
    /// Returns a vector of (feature_name, value, event_timestamp) for the
    /// requested features. Missing features are omitted from the result.
    ///
    /// Promotes the entity in the LRU on access.
    pub fn get(
        &mut self,
        group: &str,
        entity_key: &str,
        features: &[&str],
    ) -> Option<Vec<(String, FeatureValue, u64)>> {
        let key = (group.to_string(), entity_key.to_string());

        if !self.cache.contains_key(&key) {
            return None;
        }

        // Promote in LRU
        self.promote(&key);

        let entry = self.cache.get(&key)?;

        let mut result = Vec::with_capacity(features.len());
        for feature in features {
            if let Some((value, ts)) = entry.get(*feature) {
                result.push((feature.to_string(), value.clone(), *ts));
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    /// Update cache with a new ingested value.
    ///
    /// Only updates if the new event_ts >= existing event_ts for the same
    /// feature (prevents out-of-order updates from overwriting newer values).
    pub fn put(
        &mut self,
        group: &str,
        entity_key: &str,
        feature: &str,
        value: FeatureValue,
        event_ts: u64,
    ) {
        let key = (group.to_string(), entity_key.to_string());

        let entry = self.cache.entry(key.clone()).or_default();

        // Only update if newer or equal timestamp
        let should_update = match entry.get(feature) {
            Some((_, existing_ts)) => event_ts >= *existing_ts,
            None => true,
        };

        if should_update {
            entry.insert(feature.to_string(), (value, event_ts));
        }

        // Add to LRU if new entity
        if !self.access_order.contains(&key) {
            self.access_order.push_back(key);
            self.evict_if_needed();
        }
    }

    /// Returns the number of entities currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Promote an entity to the back of the LRU (most recently used).
    fn promote(&mut self, key: &(String, String)) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
            self.access_order.push_back(key.clone());
        }
    }

    /// Evict least-recently-accessed entities when over capacity.
    fn evict_if_needed(&mut self) {
        while self.cache.len() > self.capacity {
            if let Some(evicted_key) = self.access_order.pop_front() {
                self.cache.remove(&evicted_key);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_and_get() {
        let mut cache = OnlineCache::new(100);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.95), 1000);

        let result = cache.get("users", "u1", &["score"]);
        assert!(result.is_some());
        let values = result.unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].0, "score");
        assert_eq!(values[0].1, FeatureValue::Float64(0.95));
        assert_eq!(values[0].2, 1000);
    }

    #[test]
    fn test_get_missing_entity() {
        let mut cache = OnlineCache::new(100);
        let result = cache.get("users", "nonexistent", &["score"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_get_missing_feature() {
        let mut cache = OnlineCache::new(100);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.95), 1000);

        let result = cache.get("users", "u1", &["nonexistent"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_newer_value_overwrites() {
        let mut cache = OnlineCache::new(100);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.5), 1000);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.9), 2000);

        let result = cache.get("users", "u1", &["score"]).unwrap();
        assert_eq!(result[0].1, FeatureValue::Float64(0.9));
    }

    #[test]
    fn test_older_value_does_not_overwrite() {
        let mut cache = OnlineCache::new(100);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.9), 2000);
        cache.put("users", "u1", "score", FeatureValue::Float64(0.5), 1000);

        let result = cache.get("users", "u1", &["score"]).unwrap();
        assert_eq!(result[0].1, FeatureValue::Float64(0.9));
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = OnlineCache::new(2);
        cache.put("g", "e1", "f", FeatureValue::Int64(1), 100);
        cache.put("g", "e2", "f", FeatureValue::Int64(2), 200);
        cache.put("g", "e3", "f", FeatureValue::Int64(3), 300);

        // e1 should be evicted (LRU)
        assert!(cache.get("g", "e1", &["f"]).is_none());
        assert!(cache.get("g", "e2", &["f"]).is_some());
        assert!(cache.get("g", "e3", &["f"]).is_some());
    }

    #[test]
    fn test_lru_promotion_on_access() {
        let mut cache = OnlineCache::new(2);
        cache.put("g", "e1", "f", FeatureValue::Int64(1), 100);
        cache.put("g", "e2", "f", FeatureValue::Int64(2), 200);

        // Access e1 to promote it
        cache.get("g", "e1", &["f"]);

        // Insert e3 — should evict e2 (now LRU), not e1
        cache.put("g", "e3", "f", FeatureValue::Int64(3), 300);

        assert!(cache.get("g", "e1", &["f"]).is_some());
        assert!(cache.get("g", "e2", &["f"]).is_none());
    }

    #[test]
    fn test_multiple_features_per_entity() {
        let mut cache = OnlineCache::new(100);
        cache.put("g", "e1", "f1", FeatureValue::Float64(1.0), 100);
        cache.put("g", "e1", "f2", FeatureValue::Int64(42), 100);
        cache.put("g", "e1", "f3", FeatureValue::Bool(true), 100);

        let result = cache.get("g", "e1", &["f1", "f2", "f3"]).unwrap();
        assert_eq!(result.len(), 3);
    }
}

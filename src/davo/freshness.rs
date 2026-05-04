//! FreshnessIndex (V1): HashMap-based staleness queries.
//!
//! **Deprecated.** This implementation requires O(N) scans for staleness
//! queries. Use [`super::FreshnessIndexV2`] instead, which achieves
//! O(k + log N) via a deadline-based BTreeMap secondary index.

use std::collections::HashMap;

/// HashMap-based freshness index (V1).
///
/// All staleness queries (`query_fresh`, `count_stale`) are O(N).
/// Prefer [`super::FreshnessIndexV2`] for production workloads.
#[deprecated(
    since = "1.2.0",
    note = "Use FreshnessIndexV2 for O(k + log N) staleness queries"
)]
#[derive(Debug)]
pub struct FreshnessIndex {
    /// Key -> (stored_at, decay_rate)
    entries: HashMap<String, FreshnessEntry>,
}

#[derive(Debug, Clone)]
struct FreshnessEntry {
    stored_at: u64,
    decay_rate: f32,
}

#[allow(deprecated)]
impl FreshnessIndex {
    /// Create a new freshness index
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert or update a key's freshness info
    pub fn insert(&mut self, key: &str, decay_rate: f32) {
        self.entries.insert(
            key.to_string(),
            FreshnessEntry {
                stored_at: now_micros(),
                decay_rate,
            },
        );
    }

    /// Get freshness of a key
    pub fn get_freshness(&self, key: &str) -> Option<f32> {
        self.entries.get(key).map(|e| {
            let age = now_micros().saturating_sub(e.stored_at);
            let age_secs = age as f32 / 1_000_000.0;
            (-e.decay_rate * age_secs).exp()
        })
    }

    /// Query keys matching pattern with minimum freshness
    pub fn query_fresh(&self, pattern: &str, min_freshness: f32) -> Vec<String> {
        let now = now_micros();

        self.entries
            .iter()
            .filter(|(key, entry)| {
                // Check pattern match
                if !Self::matches_pattern(key, pattern) {
                    return false;
                }

                // Check freshness
                let age = now.saturating_sub(entry.stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                let freshness = (-entry.decay_rate * age_secs).exp();

                freshness >= min_freshness
            })
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Count stale keys (below threshold)
    pub fn count_stale(&self, threshold: f32) -> usize {
        let now = now_micros();

        self.entries
            .values()
            .filter(|entry| {
                let age = now.saturating_sub(entry.stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                let freshness = (-entry.decay_rate * age_secs).exp();
                freshness < threshold
            })
            .count()
    }

    /// Average freshness across all keys
    pub fn average_freshness(&self) -> f32 {
        if self.entries.is_empty() {
            return 1.0;
        }

        let now = now_micros();
        let total: f32 = self
            .entries
            .values()
            .map(|entry| {
                let age = now.saturating_sub(entry.stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                (-entry.decay_rate * age_secs).exp()
            })
            .sum();

        total / self.entries.len() as f32
    }

    /// Number of tracked keys
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Simple pattern matching (supports * wildcard)
    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        if let Some(prefix) = pattern.strip_suffix("/*") {
            return key.starts_with(prefix);
        }

        if let Some(prefix) = pattern.strip_suffix('*') {
            return key.starts_with(prefix);
        }

        key == pattern
    }
}

#[allow(deprecated)]
impl Default for FreshnessIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to get current time in microseconds
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        assert!(FreshnessIndex::matches_pattern(
            "gradient/layer_0",
            "gradient/*"
        ));
        assert!(FreshnessIndex::matches_pattern(
            "gradient/layer_0/weights",
            "gradient/*"
        ));
        assert!(!FreshnessIndex::matches_pattern(
            "model/weights",
            "gradient/*"
        ));
        assert!(FreshnessIndex::matches_pattern("anything", "*"));
    }

    #[test]
    fn test_freshness_decay() {
        let mut index = FreshnessIndex::new();
        index.insert("test", 1.0); // Decay rate of 1/second

        // Immediately after insert, freshness should be ~1.0
        let freshness = index.get_freshness("test").unwrap();
        assert!(freshness > 0.99);
    }
}

//! FreshnessIndexV2: Scalable staleness queries with Forward Decay
//!
//! Addresses the O(N) scan problem by using a deadline-based secondary index.
//!
//! # The Problem
//!
//! In a standard HashMap, querying "all items where freshness < threshold" requires
//! O(N) iteration because freshness changes continuously as time advances.
//!
//! # The Solution: Forward Decay
//!
//! Instead of computing freshness at query time, we precompute the **deadline**
//! when each entry becomes stale:
//!
//! ```text
//! freshness(t) = e^(-λ × (t - stored_at)) < threshold
//! deadline = stored_at + (-ln(threshold) / λ)
//! ```
//!
//! For threshold = 0.5 (half-life):
//! ```text
//! deadline = stored_at + ln(2) / λ
//! ```
//!
//! # Complexity
//!
//! | Operation | HashMap Only | With Deadline Index |
//! |-----------|--------------|---------------------|
//! | Insert | O(1) | O(log N) |
//! | Point query | O(1) | O(1) |
//! | Staleness scan | O(N) | O(k + log N) |
//! | Eviction | O(N) | O(k) |
//!
//! Where k = number of stale entries.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

/// Scalable freshness index with deadline-based staleness queries
#[derive(Debug)]
pub struct FreshnessIndexV2 {
    /// Key -> (stored_at, decay_rate) for point queries
    entries: HashMap<String, FreshnessEntry>,

    /// Deadline -> Keys (sorted by when they become stale)
    /// BTreeMap enables O(log N) range queries
    deadline_index: BTreeMap<u64, HashSet<String>>,

    /// Staleness threshold for deadline computation (default: 0.5)
    threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FreshnessEntry {
    stored_at: u64,
    decay_rate: f32,
    deadline: u64,
}

/// On-disk representation of a [`FreshnessIndexV2`].
///
/// Persisted with bincode. Version field allows future format migrations.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedIndex {
    version: u32,
    threshold: f32,
    entries: Vec<(String, FreshnessEntry)>,
}

const PERSIST_VERSION: u32 = 1;

impl FreshnessIndexV2 {
    /// Create a new freshness index with default threshold (0.5)
    pub fn new() -> Self {
        Self::with_threshold(0.5)
    }

    /// Create with custom staleness threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            entries: HashMap::new(),
            deadline_index: BTreeMap::new(),
            threshold: threshold.clamp(0.001, 0.999),
        }
    }

    /// Insert or update a key's freshness info
    /// Complexity: O(log N)
    pub fn insert(&mut self, key: &str, decay_rate: f32) {
        let now = now_micros();
        let deadline = self.compute_deadline(now, decay_rate);

        // Remove old deadline if key exists
        if let Some(old) = self.entries.get(key) {
            self.remove_from_deadline_index(key, old.deadline);
        }

        // Insert new entry
        self.entries.insert(
            key.to_string(),
            FreshnessEntry {
                stored_at: now,
                decay_rate,
                deadline,
            },
        );

        // Add to deadline index
        self.deadline_index
            .entry(deadline)
            .or_default()
            .insert(key.to_string());
    }

    /// Get freshness of a key (0.0 to 1.0)
    /// Complexity: O(1)
    pub fn get_freshness(&self, key: &str) -> Option<f32> {
        self.entries.get(key).map(|e| {
            let age = now_micros().saturating_sub(e.stored_at);
            let age_secs = age as f32 / 1_000_000.0;
            (-e.decay_rate * age_secs).exp()
        })
    }

    /// Check if a key is stale (below threshold)
    /// Complexity: O(1)
    pub fn is_stale(&self, key: &str) -> Option<bool> {
        self.entries.get(key).map(|e| now_micros() >= e.deadline)
    }

    /// Query all stale keys (freshness < threshold)
    /// Complexity: O(k + log N) where k = number of stale entries
    pub fn query_stale(&self) -> Vec<String> {
        let now = now_micros();

        self.deadline_index
            .range(..=now)
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect()
    }

    /// Query all fresh keys (freshness >= threshold)
    /// Complexity: O(k + log N) where k = number of fresh entries
    pub fn query_fresh(&self) -> Vec<String> {
        let now = now_micros();

        self.deadline_index
            .range((now + 1)..)
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect()
    }

    /// Query fresh keys matching pattern
    /// Complexity: O(k) where k = fresh entries matching pattern
    pub fn query_fresh_pattern(&self, pattern: &str, min_freshness: f32) -> Vec<String> {
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

    /// Evict all stale entries and return their keys
    /// Complexity: O(k) where k = number of stale entries
    pub fn evict_stale(&mut self) -> Vec<String> {
        let now = now_micros();
        let mut evicted = Vec::new();

        // Collect all stale deadlines
        let stale_deadlines: Vec<u64> =
            self.deadline_index.range(..=now).map(|(d, _)| *d).collect();

        // Remove from both indexes
        for deadline in stale_deadlines {
            if let Some(keys) = self.deadline_index.remove(&deadline) {
                for key in keys {
                    self.entries.remove(&key);
                    evicted.push(key);
                }
            }
        }

        evicted
    }

    /// Count stale keys
    /// Complexity: O(log N + k) where k = stale entries
    pub fn count_stale(&self) -> usize {
        let now = now_micros();

        self.deadline_index
            .range(..=now)
            .map(|(_, keys)| keys.len())
            .sum()
    }

    /// Count fresh keys
    /// Complexity: O(log N + k) where k = fresh entries
    pub fn count_fresh(&self) -> usize {
        self.entries.len() - self.count_stale()
    }

    /// Average freshness across all keys
    /// Complexity: O(N) - must compute freshness for each entry
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

    /// Get the staleness threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Remove a key from the index
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.remove_from_deadline_index(key, entry.deadline);
            true
        } else {
            false
        }
    }

    /// Save the index to disk using bincode.
    ///
    /// Persists threshold and all entries (stored_at, decay_rate, deadline).
    /// The in-memory BTreeMap is rebuilt from entries on [`load`](Self::load).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let persisted = PersistedIndex {
            version: PERSIST_VERSION,
            threshold: self.threshold,
            entries: self
                .entries
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        let bytes = bincode::serialize(&persisted)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Load an index from disk.
    ///
    /// Rebuilds the deadline BTreeMap from the persisted entries so that
    /// staleness queries work immediately after load.
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let persisted: PersistedIndex = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if persisted.version != PERSIST_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported FreshnessIndexV2 persist version: {} (expected {})",
                    persisted.version, PERSIST_VERSION
                ),
            ));
        }

        let mut index = Self::with_threshold(persisted.threshold);
        for (key, entry) in persisted.entries {
            index
                .deadline_index
                .entry(entry.deadline)
                .or_default()
                .insert(key.clone());
            index.entries.insert(key, entry);
        }
        Ok(index)
    }

    /// Compute deadline when entry becomes stale
    fn compute_deadline(&self, stored_at: u64, decay_rate: f32) -> u64 {
        if decay_rate <= 0.0 {
            return u64::MAX; // Never stale (static data)
        }

        // deadline = stored_at + (-ln(threshold) / λ)
        // For threshold = 0.5: deadline = stored_at + ln(2) / λ
        let time_to_stale_secs = -self.threshold.ln() / decay_rate;
        let time_to_stale_micros = (time_to_stale_secs * 1_000_000.0) as u64;

        stored_at.saturating_add(time_to_stale_micros)
    }

    /// Remove key from deadline index
    fn remove_from_deadline_index(&mut self, key: &str, deadline: u64) {
        if let Some(keys) = self.deadline_index.get_mut(&deadline) {
            keys.remove(key);
            if keys.is_empty() {
                self.deadline_index.remove(&deadline);
            }
        }
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

impl Default for FreshnessIndexV2 {
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
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query() {
        let mut index = FreshnessIndexV2::new();

        // Insert with fast decay (will be stale quickly)
        index.insert("fast", 10.0); // ~70ms half-life

        // Insert with slow decay
        index.insert("slow", 0.0001); // ~2 hour half-life

        // Both should be fresh immediately
        assert!(index.get_freshness("fast").unwrap() > 0.99);
        assert!(index.get_freshness("slow").unwrap() > 0.99);
    }

    #[test]
    fn test_deadline_computation() {
        let index = FreshnessIndexV2::with_threshold(0.5);

        // For λ = 1.0, half-life = ln(2) ≈ 0.693 seconds
        let now = 1_000_000; // 1 second in micros
        let deadline = index.compute_deadline(now, 1.0);

        // Deadline should be ~693,147 microseconds after stored_at
        let expected = now + 693_147;
        assert!((deadline as i64 - expected as i64).abs() < 1000);
    }

    #[test]
    fn test_stale_query_efficiency() {
        let mut index = FreshnessIndexV2::new();

        // Insert 1000 entries with varying decay rates
        for i in 0..1000 {
            let decay = if i < 100 { 100.0 } else { 0.00001 }; // 100 fast, 900 slow
            index.insert(&format!("key_{}", i), decay);
        }

        // Wait a tiny bit for fast entries to become stale
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Query stale - should be O(k) not O(N)
        let stale = index.query_stale();

        // Fast entries should be stale
        assert!(stale.len() >= 50); // At least some should be stale
        assert!(stale.len() <= 200); // Not all should be stale
    }

    #[test]
    fn test_eviction() {
        let mut index = FreshnessIndexV2::new();

        // Insert with immediate staleness
        index.insert("stale1", 1000.0); // Very fast decay
        index.insert("stale2", 1000.0);
        index.insert("fresh", 0.00001); // Very slow decay

        // Wait for fast entries to become stale
        std::thread::sleep(std::time::Duration::from_millis(5));

        let evicted = index.evict_stale();

        assert!(evicted.contains(&"stale1".to_string()));
        assert!(evicted.contains(&"stale2".to_string()));
        assert!(!evicted.contains(&"fresh".to_string()));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_static_data_never_stale() {
        let mut index = FreshnessIndexV2::new();

        // Insert with zero decay (static)
        index.insert("static", 0.0);

        // Should never be stale
        assert!(!index.is_stale("static").unwrap());

        // Freshness should always be 1.0
        assert!((index.get_freshness("static").unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pattern_matching() {
        assert!(FreshnessIndexV2::matches_pattern(
            "gradient/layer_0",
            "gradient/*"
        ));
        assert!(FreshnessIndexV2::matches_pattern(
            "gradient/layer_0/weights",
            "gradient/*"
        ));
        assert!(!FreshnessIndexV2::matches_pattern(
            "model/weights",
            "gradient/*"
        ));
        assert!(FreshnessIndexV2::matches_pattern("anything", "*"));
    }

    #[test]
    fn test_remove() {
        let mut index = FreshnessIndexV2::new();

        index.insert("key1", 0.001);
        index.insert("key2", 0.001);

        assert_eq!(index.len(), 2);

        assert!(index.remove("key1"));
        assert_eq!(index.len(), 1);

        assert!(!index.remove("key1")); // Already removed
        assert!(index.get_freshness("key1").is_none());
    }
}

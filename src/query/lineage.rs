//! Data Lineage — track provenance and transformations.
//!
//! Records where data came from and how it was derived.

use serde::Serialize;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Source of a data entry.
#[derive(Debug, Clone, Serialize)]
pub enum LineageSource {
    /// Written directly by an application.
    DirectWrite { client_id: Option<String> },
    /// Computed from a query.
    Computed { query: String },
    /// Imported from an external source.
    Imported { source: String },
}

/// A single transformation in the lineage chain.
#[derive(Debug, Clone, Serialize)]
pub struct Transformation {
    /// Operation name (e.g., "AVG", "RESAMPLE").
    pub operation: String,
    /// When the transformation was applied.
    pub timestamp: u64,
    /// Input keys that were used.
    pub input_keys: Vec<String>,
}

/// Full lineage record for a key.
#[derive(Debug, Clone, Serialize)]
pub struct DataLineage {
    /// The key being traced.
    pub key: String,
    /// When this version was created.
    pub timestamp: u64,
    /// Where the data came from.
    pub source: LineageSource,
    /// Chain of transformations applied.
    pub transformations: Vec<Transformation>,
}

/// In-memory lineage tracker.
#[derive(Debug, Default)]
pub struct LineageTracker {
    records: HashMap<String, Vec<DataLineage>>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

impl LineageTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record lineage for a key.
    pub fn record(&mut self, lineage: DataLineage) {
        self.records
            .entry(lineage.key.clone())
            .or_default()
            .push(lineage);
    }

    /// Get all lineage records for a key.
    pub fn get_lineage(&self, key: &str) -> Vec<&DataLineage> {
        self.records
            .get(key)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Find all keys derived from a given source key.
    pub fn find_derived_from(&self, source_key: &str) -> Vec<String> {
        self.records
            .iter()
            .filter(|(_, lineages)| {
                lineages.iter().any(|l| {
                    l.transformations
                        .iter()
                        .any(|t| t.input_keys.contains(&source_key.to_string()))
                })
            })
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Number of tracked keys.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get_lineage() {
        let mut tracker = LineageTracker::new();
        tracker.record(DataLineage {
            key: "sensor/temp/avg".into(),
            timestamp: 1000,
            source: LineageSource::Computed {
                query: "SELECT AVG(value) FROM \"sensor/temp\"".into(),
            },
            transformations: vec![Transformation {
                operation: "AVG".into(),
                timestamp: 1000,
                input_keys: vec!["sensor/temp".into()],
            }],
        });

        let lineage = tracker.get_lineage("sensor/temp/avg");
        assert_eq!(lineage.len(), 1);
        assert!(matches!(&lineage[0].source, LineageSource::Computed { .. }));
    }

    #[test]
    fn test_find_derived_from() {
        let mut tracker = LineageTracker::new();
        tracker.record(DataLineage {
            key: "derived/a".into(),
            timestamp: 100,
            source: LineageSource::Computed {
                query: "...".into(),
            },
            transformations: vec![Transformation {
                operation: "SUM".into(),
                timestamp: 100,
                input_keys: vec!["source/x".into()],
            }],
        });
        tracker.record(DataLineage {
            key: "derived/b".into(),
            timestamp: 200,
            source: LineageSource::Computed {
                query: "...".into(),
            },
            transformations: vec![Transformation {
                operation: "AVG".into(),
                timestamp: 200,
                input_keys: vec!["source/x".into(), "source/y".into()],
            }],
        });

        let derived = tracker.find_derived_from("source/x");
        assert_eq!(derived.len(), 2);
        assert!(derived.contains(&"derived/a".to_string()));
        assert!(derived.contains(&"derived/b".to_string()));
    }

    #[test]
    fn test_empty_lineage() {
        let tracker = LineageTracker::new();
        assert!(tracker.get_lineage("nonexistent").is_empty());
        assert!(tracker.find_derived_from("x").is_empty());
    }
}

// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! # SynaDB Feature Store
//!
//! A fully embedded, zero-server feature management system for ML engineers.
//!
//! The Feature Store provides:
//! - **Typed feature schemas** with validation constraints
//! - **Point-in-time queries** (data leakage prevention by construction)
//! - **Online serving** (<1ms, in-memory cache with LRU)
//! - **Training dataset generation** (parallel point-in-time joins)
//! - **Feature views** (EQL-based derived features)
//! - **DAVO freshness integration** (features know when they're stale)
//! - **Schema evolution** (add/widen/deprecate columns)
//! - **Feature registry** (metadata catalog with lineage)
//! - **Statistics & drift detection** (Welford's, PSI)
//!
//! # Architecture
//!
//! All feature data is stored in the existing SynaDB append-only log using
//! a key namespace convention: `__fs/{group}/{entity}/{feature}`.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ FeatureStore                                             │
//! ├─────────────────────────────────────────────────────────┤
//! │ Registry │ OnlineCache │ PIT Index │ WriteBuffer │ Stats │
//! ├─────────────────────────────────────────────────────────┤
//! │                  SynaDB Core Engine                      │
//! │          Append-Only Log │ In-Memory Index               │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use synadb::feature_store::{FeatureStore, FeatureStoreConfig};
//! use synadb::feature_store::schema::*;
//!
//! // Create a feature store
//! let config = FeatureStoreConfig::default();
//! let mut store = FeatureStore::new("features.db", config).unwrap();
//!
//! // Register a schema
//! let schema = FeatureSchema {
//!     name: "user_features".to_string(),
//!     columns: vec![
//!         ColumnDef {
//!             name: "user_id".to_string(),
//!             dtype: FeatureType::String,
//!             default: None,
//!             constraints: Some(ColumnConstraints { not_null: true, ..Default::default() }),
//!             ttl_seconds: None,
//!             is_entity_key: true,
//!             is_event_timestamp: false,
//!             deprecated: false,
//!         },
//!         ColumnDef {
//!             name: "event_time".to_string(),
//!             dtype: FeatureType::Timestamp,
//!             default: None,
//!             constraints: None,
//!             ttl_seconds: None,
//!             is_entity_key: false,
//!             is_event_timestamp: true,
//!             deprecated: false,
//!         },
//!         ColumnDef {
//!             name: "purchase_count".to_string(),
//!             dtype: FeatureType::Int64,
//!             default: Some(FeatureValue::Int64(0)),
//!             constraints: Some(ColumnConstraints { min: Some(0.0), ..Default::default() }),
//!             ttl_seconds: Some(86400),
//!             is_entity_key: false,
//!             is_event_timestamp: false,
//!             deprecated: false,
//!         },
//!     ],
//!     version: 1,
//!     description: Some("User purchase features".to_string()),
//!     tags: vec!["user".to_string()],
//!     created_at: 0,
//!     created_by: None,
//! };
//! store.register_schema(schema).unwrap();
//! ```

pub mod config;
pub mod dataset;
pub mod ffi;
pub mod ingestion;
pub mod migration;
pub mod online_cache;
pub mod pit_index;
pub mod registry;
pub mod schema;
pub mod serialization;
pub mod statistics;

// Re-export primary types
pub use config::FeatureStoreConfig;
pub use registry::{FeatureRegistry, LineageEdge, MigrationOp, RegistryQuery, SchemaMigration};
pub use schema::{
    ColumnConstraints, ColumnDef, FeatureSchema, FeatureType, FeatureValue, StoredFeatureValue,
};

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::{DbConfig, SynaDB};
use crate::error::Result;

use self::ingestion::WriteAheadBuffer;
use self::online_cache::OnlineCache;
use self::pit_index::PointInTimeIndex;
use self::serialization::{deserialize, serialize};
use self::statistics::FeatureStatistics;

/// Key prefix for all feature store data in the append-only log.
#[allow(dead_code)]
const FS_KEY_PREFIX: &str = "__fs/";

/// Key for persisting the feature registry.
const REGISTRY_KEY: &str = "__fs_registry";

/// The top-level feature store managing all feature operations.
///
/// Wraps a [`SynaDB`] instance and adds typed schemas, point-in-time
/// indexing, online caching, and training dataset generation.
pub struct FeatureStore {
    /// Underlying SynaDB instance for persistence.
    db: SynaDB,
    /// Registered schemas by group name.
    schemas: HashMap<String, FeatureSchema>,
    /// In-memory cache for online serving.
    online_cache: OnlineCache,
    /// Temporal index for point-in-time queries.
    pit_index: PointInTimeIndex,
    /// Write-ahead buffer for batching writes.
    write_buffer: WriteAheadBuffer,
    /// Running statistics per feature (group:feature → stats).
    statistics: HashMap<String, FeatureStatistics>,
    /// Configuration.
    config: FeatureStoreConfig,
}

/// A row of feature values for batch ingestion.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    /// Entity key value.
    pub entity_key: String,
    /// Event timestamp.
    pub event_ts: u64,
    /// Feature name → value pairs.
    pub values: Vec<(String, FeatureValue)>,
}

/// Complete feature vector for one entity at one point in time.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Entity key.
    pub entity_key: String,
    /// Feature name → value pairs.
    pub values: Vec<(String, FeatureValue)>,
    /// Per-feature freshness scores (if DAVO enabled).
    pub freshness_scores: Option<Vec<(String, f64)>>,
    /// Staleness warnings for features below threshold.
    pub staleness_warnings: Vec<String>,
    /// Timestamp of the query.
    pub timestamp: u64,
}

impl FeatureStore {
    /// Create or open a feature store backed by a single file.
    ///
    /// If the file exists, the registry is restored from the log.
    /// If the file doesn't exist, a new empty feature store is created.
    pub fn new(path: impl AsRef<Path>, config: FeatureStoreConfig) -> Result<Self> {
        config.validate()?;

        let db_config = DbConfig {
            enable_compression: true,
            enable_delta: true,
            sync_on_write: config.sync_on_write,
        };

        let mut db =
            SynaDB::with_config(path.as_ref().to_str().unwrap_or("features.db"), db_config)?;

        // Try to restore registry from the log
        let schemas = Self::load_registry(&mut db);

        let online_cache = OnlineCache::new(config.online_cache_capacity);
        let pit_index = PointInTimeIndex::new();
        let write_buffer =
            WriteAheadBuffer::new(config.write_buffer_size, config.write_buffer_max_age_micros);

        Ok(Self {
            db,
            schemas,
            online_cache,
            pit_index,
            write_buffer,
            statistics: HashMap::new(),
            config,
        })
    }

    /// Register a new feature schema.
    ///
    /// Validates the schema and persists it to the log. Returns an error
    /// if a schema with the same name already exists.
    pub fn register_schema(&mut self, schema: FeatureSchema) -> Result<()> {
        schema.validate()?;

        if self.schemas.contains_key(&schema.name) {
            return Err(crate::error::SynaError::InvalidInput(format!(
                "Feature schema '{}' already exists. Use migrate() to modify.",
                schema.name
            )));
        }

        self.schemas.insert(schema.name.clone(), schema);
        self.persist_registry()?;
        Ok(())
    }

    /// Ingest a single feature value.
    ///
    /// Validates against the schema, writes to the log, and updates
    /// the online cache and PIT index.
    pub fn ingest(
        &mut self,
        group: &str,
        entity_key: &str,
        event_ts: u64,
        values: &[(&str, FeatureValue)],
    ) -> Result<()> {
        // Validate against schema if registered
        if let Some(schema) = self.schemas.get(group) {
            schema.validate_row(values)?;
        }

        let ingestion_ts = current_timestamp_micros();

        for (feature, value) in values {
            // Write to log
            let stored = StoredFeatureValue {
                value: value.clone(),
                event_timestamp: event_ts,
                ingestion_timestamp: ingestion_ts,
            };
            let key = format!("__fs/{}/{}/{}", group, entity_key, feature);
            let bytes = serialize(&stored)?;
            let offset = self.db.append(&key, crate::types::Atom::Bytes(bytes))?;

            // Update online cache
            self.online_cache
                .put(group, entity_key, feature, value.clone(), event_ts);

            // Update PIT index
            self.pit_index
                .insert(group, entity_key, feature, event_ts, offset, value.clone());

            // Update statistics
            let stats_key = format!("{}:{}", group, feature);
            let stats = self.statistics.entry(stats_key).or_default();
            match value {
                FeatureValue::Float64(v) => stats.update(*v),
                FeatureValue::Int64(v) => stats.update(*v as f64),
                FeatureValue::Null => stats.update_null(),
                _ => {}
            }
        }

        Ok(())
    }

    /// Ingest a batch of feature values atomically.
    ///
    /// Validates ALL values first. If any fail, the entire batch is rejected.
    pub fn ingest_batch(&mut self, group: &str, batch: &[FeatureRow]) -> Result<()> {
        // Validate all rows first
        if let Some(schema) = self.schemas.get(group) {
            let mut errors = Vec::new();
            for (i, row) in batch.iter().enumerate() {
                let refs: Vec<(&str, FeatureValue)> = row
                    .values
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect();
                if let Err(e) = schema.validate_row(&refs) {
                    errors.push(format!("Row {}: {}", i, e));
                }
            }
            if !errors.is_empty() {
                return Err(crate::error::SynaError::InvalidInput(errors.join("; ")));
            }
        }

        // Write all rows
        for row in batch {
            let refs: Vec<(&str, FeatureValue)> = row
                .values
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            self.ingest(group, &row.entity_key, row.event_ts, &refs)?;
        }

        Ok(())
    }

    /// Online serving: get latest feature values for an entity.
    ///
    /// Returns values from the in-memory cache with O(1) lookup.
    /// Missing features get schema-defined default values.
    pub fn serve(
        &mut self,
        group: &str,
        entity_key: &str,
        features: &[&str],
    ) -> Result<FeatureVector> {
        let cached = self.online_cache.get(group, entity_key, features);

        let mut values = Vec::with_capacity(features.len());

        if let Some(cached_values) = cached {
            // Build result from cache
            for feature in features {
                let found = cached_values.iter().find(|(name, _, _)| name == feature);
                match found {
                    Some((name, value, _)) => {
                        values.push((name.clone(), value.clone()));
                    }
                    None => {
                        // Use schema default
                        let default = self.get_default(group, feature);
                        values.push((feature.to_string(), default));
                    }
                }
            }
        } else {
            // Entity not in cache — return defaults
            for feature in features {
                let default = self.get_default(group, feature);
                values.push((feature.to_string(), default));
            }
        }

        Ok(FeatureVector {
            entity_key: entity_key.to_string(),
            values,
            freshness_scores: None,
            staleness_warnings: Vec::new(),
            timestamp: current_timestamp_micros(),
        })
    }

    /// Point-in-time query: get features as of a specific timestamp.
    ///
    /// Guarantees no value with event_ts > cutoff_ts is ever returned.
    pub fn get_as_of(
        &mut self,
        group: &str,
        entity_key: &str,
        cutoff_ts: u64,
        features: &[&str],
    ) -> Result<FeatureVector> {
        let mut values = Vec::with_capacity(features.len());

        for feature in features {
            let value = match self.pit_index.lookup(group, entity_key, feature, cutoff_ts) {
                Some(v) => v.clone(),
                None => self.get_default(group, feature),
            };

            values.push((feature.to_string(), value));
        }

        Ok(FeatureVector {
            entity_key: entity_key.to_string(),
            values,
            freshness_scores: None,
            staleness_warnings: Vec::new(),
            timestamp: cutoff_ts,
        })
    }

    /// Get the Nth-most-recent value for a feature.
    ///
    /// - `version = 0` or `version = -1` returns the latest value
    /// - `version = -N` returns the Nth-most-recent value
    /// - Returns `None` when the requested version doesn't exist
    pub fn get_at_version(
        &self,
        group: &str,
        entity_key: &str,
        feature: &str,
        version: i64,
    ) -> Option<FeatureValue> {
        let all = self.pit_index.get_all(group, entity_key, feature)?;

        if all.is_empty() {
            return None;
        }

        // version = 0 or -1 means latest
        let idx = if version == 0 || version == -1 {
            all.len() - 1
        } else if version < 0 {
            let n = (-version) as usize;
            if n > all.len() {
                return None;
            }
            all.len() - n
        } else {
            // Positive version: treat as 1-indexed from start
            let n = version as usize;
            if n > all.len() {
                return None;
            }
            n - 1
        };

        Some(all[idx].1.clone())
    }

    /// Get the value as it existed at a specific timestamp.
    ///
    /// Equivalent to `get_as_of` for a single feature.
    pub fn get_at_timestamp(
        &self,
        group: &str,
        entity_key: &str,
        feature: &str,
        timestamp: u64,
    ) -> Option<FeatureValue> {
        self.pit_index
            .lookup(group, entity_key, feature, timestamp)
            .cloned()
    }

    /// Generate a training dataset from an entity DataFrame.
    ///
    /// Performs point-in-time joins for each row in the DataFrame,
    /// looking up feature values as they were known at each row's timestamp.
    ///
    /// Returns a columnar dataset with per-column statistics.
    pub fn generate_dataset(
        &self,
        entity_df: &dataset::EntityDataFrame,
        group: &str,
        features: &[&str],
    ) -> Result<dataset::TrainingDataset> {
        if entity_df.entity_keys.len() != entity_df.event_timestamps.len() {
            return Err(crate::error::SynaError::InvalidInput(
                "entity_keys and event_timestamps must have the same length".to_string(),
            ));
        }

        let num_rows = entity_df.len();
        let mut columns: Vec<dataset::ColumnData> = features
            .iter()
            .map(|_| dataset::ColumnData::new_float64(num_rows))
            .collect();

        // For each row, perform PIT lookup
        for i in 0..num_rows {
            let entity_key = &entity_df.entity_keys[i];
            let cutoff_ts = entity_df.event_timestamps[i];

            for (col_idx, feature) in features.iter().enumerate() {
                let value = match self.pit_index.lookup(group, entity_key, feature, cutoff_ts) {
                    Some(v) => v.clone(),
                    None => self.get_default(group, feature),
                };
                columns[col_idx].push(&value);
            }
        }

        // Compute statistics
        let statistics: Vec<statistics::ColumnStatistics> = columns
            .iter()
            .map(dataset::compute_column_statistics)
            .collect();

        Ok(dataset::TrainingDataset {
            columns: features.iter().map(|f| f.to_string()).collect(),
            data: columns,
            statistics,
            num_rows,
        })
    }

    /// Flush the write buffer to disk.
    pub fn flush(&mut self) -> Result<()> {
        let entries = self.write_buffer.flush();
        for entry in entries {
            let stored = StoredFeatureValue {
                value: entry.value.clone(),
                event_timestamp: entry.event_ts,
                ingestion_timestamp: entry.ingestion_ts,
            };
            let key = format!(
                "__fs/{}/{}/{}",
                entry.group, entry.entity_key, entry.feature
            );
            let bytes = serialize(&stored)?;
            self.db.append(&key, crate::types::Atom::Bytes(bytes))?;
        }
        Ok(())
    }

    /// Get a registered schema by name.
    pub fn get_schema(&self, name: &str) -> Option<&FeatureSchema> {
        self.schemas.get(name)
    }

    /// List all registered schema names.
    pub fn list_schemas(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &FeatureStoreConfig {
        &self.config
    }

    /// Get statistics for a feature.
    pub fn get_statistics(&self, group: &str, feature: &str) -> Option<&FeatureStatistics> {
        let key = format!("{}:{}", group, feature);
        self.statistics.get(&key)
    }

    /// Register a schema for all keys matching a prefix.
    ///
    /// Future writes to keys matching the prefix are validated against this schema.
    /// Existing data is NOT retroactively validated.
    pub fn register_prefix_schema(&mut self, prefix: &str, schema: FeatureSchema) -> Result<()> {
        schema.validate()?;
        // Store with a special prefix marker
        let key = format!("__prefix:{}", prefix);
        self.schemas.insert(key, schema);
        self.persist_registry()?;
        Ok(())
    }

    /// Get the prefix schema that matches a key, if any.
    pub fn get_prefix_schema(&self, key: &str) -> Option<&FeatureSchema> {
        for (schema_key, schema) in &self.schemas {
            if let Some(prefix) = schema_key.strip_prefix("__prefix:") {
                if key.starts_with(prefix) {
                    return Some(schema);
                }
            }
        }
        None
    }

    /// Flush any pending writes and close the feature store.
    pub fn close(mut self) -> Result<()> {
        self.flush()?;
        Ok(())
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Get the schema-defined default for a feature, or Null.
    fn get_default(&self, group: &str, feature: &str) -> FeatureValue {
        if let Some(schema) = self.schemas.get(group) {
            if let Some(col) = schema.columns.iter().find(|c| c.name == feature) {
                if let Some(ref default) = col.default {
                    return default.clone();
                }
            }
        }
        FeatureValue::Null
    }

    /// Persist the current registry state to the log.
    fn persist_registry(&mut self) -> Result<()> {
        let data = serialize(&self.schemas)?;
        let wrapped = serialization::wrap_versioned(data);
        let bytes = serialize(&wrapped)?;
        self.db
            .append(REGISTRY_KEY, crate::types::Atom::Bytes(bytes))?;
        Ok(())
    }

    /// Load the registry from the log (latest value for REGISTRY_KEY).
    fn load_registry(db: &mut SynaDB) -> HashMap<String, FeatureSchema> {
        match db.get(REGISTRY_KEY) {
            Ok(Some(crate::types::Atom::Bytes(bytes))) => {
                match deserialize::<serialization::PersistedRegistry>(&bytes) {
                    Ok(persisted) => match serialization::unwrap_versioned(&persisted) {
                        Ok(data) => {
                            deserialize::<HashMap<String, FeatureSchema>>(data).unwrap_or_default()
                        }
                        Err(_) => HashMap::new(),
                    },
                    Err(_) => HashMap::new(),
                }
            }
            _ => HashMap::new(),
        }
    }
}

/// Get the current timestamp in microseconds.
fn current_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> FeatureSchema {
        FeatureSchema {
            name: "test_group".to_string(),
            columns: vec![
                ColumnDef {
                    name: "user_id".to_string(),
                    dtype: FeatureType::String,
                    default: None,
                    constraints: None,
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
                    name: "score".to_string(),
                    dtype: FeatureType::Float64,
                    default: Some(FeatureValue::Float64(0.0)),
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: false,
                    is_event_timestamp: false,
                    deprecated: false,
                },
            ],
            version: 1,
            description: Some("Test feature group".to_string()),
            tags: vec!["test".to_string()],
            created_at: 1_000_000,
            created_by: Some("test".to_string()),
        }
    }

    #[test]
    fn test_create_feature_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let store = FeatureStore::new(&path, config).unwrap();
        assert!(store.list_schemas().is_empty());
    }

    #[test]
    fn test_register_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        let schema = test_schema();
        store.register_schema(schema.clone()).unwrap();

        assert_eq!(store.list_schemas(), vec!["test_group"]);
        assert_eq!(store.get_schema("test_group"), Some(&schema));
    }

    #[test]
    fn test_duplicate_schema_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        let schema = test_schema();
        store.register_schema(schema.clone()).unwrap();
        assert!(store.register_schema(schema).is_err());
    }

    #[test]
    fn test_registry_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");

        // Create and register
        {
            let config = FeatureStoreConfig::default();
            let mut store = FeatureStore::new(&path, config).unwrap();
            store.register_schema(test_schema()).unwrap();
        }

        // Reopen and verify
        {
            let config = FeatureStoreConfig::default();
            let store = FeatureStore::new(&path, config).unwrap();
            assert_eq!(store.list_schemas(), vec!["test_group"]);
            assert_eq!(store.get_schema("test_group").unwrap().name, "test_group");
        }
    }

    #[test]
    fn test_invalid_config_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig {
            online_cache_capacity: 0,
            ..Default::default()
        };
        assert!(FeatureStore::new(&path, config).is_err());
    }

    #[test]
    fn test_ingest_and_serve() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();
        store.register_schema(test_schema()).unwrap();

        // Ingest a value
        store
            .ingest(
                "test_group",
                "user1",
                1000,
                &[("score", FeatureValue::Float64(0.95))],
            )
            .unwrap();

        // Serve it back
        let result = store.serve("test_group", "user1", &["score"]).unwrap();
        assert_eq!(result.entity_key, "user1");
        assert_eq!(result.values.len(), 1);
        assert_eq!(result.values[0].0, "score");
        assert_eq!(result.values[0].1, FeatureValue::Float64(0.95));
    }

    #[test]
    fn test_serve_default_for_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();
        store.register_schema(test_schema()).unwrap();

        // Serve without ingesting — should get default
        let result = store.serve("test_group", "user1", &["score"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Float64(0.0));
    }

    #[test]
    fn test_ingest_validation_rejects_wrong_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();
        store.register_schema(test_schema()).unwrap();

        // Try to ingest a string where float is expected
        let result = store.ingest(
            "test_group",
            "user1",
            1000,
            &[("score", FeatureValue::String("bad".to_string()))],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_batch_atomic_rejection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();
        store.register_schema(test_schema()).unwrap();

        let batch = vec![
            FeatureRow {
                entity_key: "u1".to_string(),
                event_ts: 1000,
                values: vec![("score".to_string(), FeatureValue::Float64(0.5))],
            },
            FeatureRow {
                entity_key: "u2".to_string(),
                event_ts: 2000,
                values: vec![("score".to_string(), FeatureValue::String("bad".to_string()))],
            },
        ];

        // Entire batch should be rejected
        let result = store.ingest_batch("test_group", &batch);
        assert!(result.is_err());

        // First row should NOT have been written
        let served = store.serve("test_group", "u1", &["score"]).unwrap();
        assert_eq!(served.values[0].1, FeatureValue::Float64(0.0)); // default
    }

    #[test]
    fn test_statistics_updated_on_ingest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        store
            .ingest("g", "e1", 1000, &[("val", FeatureValue::Float64(10.0))])
            .unwrap();
        store
            .ingest("g", "e2", 2000, &[("val", FeatureValue::Float64(20.0))])
            .unwrap();
        store
            .ingest("g", "e3", 3000, &[("val", FeatureValue::Float64(30.0))])
            .unwrap();

        let stats = store.get_statistics("g", "val").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 30.0);
    }

    #[test]
    fn test_cache_consistency_after_ingest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        // Ingest and immediately serve — cache must be consistent
        store
            .ingest("g", "e1", 1000, &[("f", FeatureValue::Int64(42))])
            .unwrap();
        let result = store.serve("g", "e1", &["f"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Int64(42));

        // Update with newer timestamp
        store
            .ingest("g", "e1", 2000, &[("f", FeatureValue::Int64(99))])
            .unwrap();
        let result = store.serve("g", "e1", &["f"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Int64(99));
    }

    #[test]
    fn test_get_as_of_pit_correctness() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        // Ingest values at different timestamps
        store
            .ingest("g", "e1", 1000, &[("score", FeatureValue::Float64(1.0))])
            .unwrap();
        store
            .ingest("g", "e1", 2000, &[("score", FeatureValue::Float64(2.0))])
            .unwrap();
        store
            .ingest("g", "e1", 3000, &[("score", FeatureValue::Float64(3.0))])
            .unwrap();

        // PIT query at ts=2000 should return 2.0 (not 3.0!)
        let result = store.get_as_of("g", "e1", 2000, &["score"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Float64(2.0));

        // PIT query at ts=1500 should return 1.0
        let result = store.get_as_of("g", "e1", 1500, &["score"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Float64(1.0));

        // PIT query at ts=500 should return default (Null)
        let result = store.get_as_of("g", "e1", 500, &["score"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Null);

        // PIT query at ts=5000 should return latest (3.0)
        let result = store.get_as_of("g", "e1", 5000, &["score"]).unwrap();
        assert_eq!(result.values[0].1, FeatureValue::Float64(3.0));
    }

    #[test]
    fn test_generate_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        // Ingest feature history for two entities
        store
            .ingest("g", "u1", 1000, &[("score", FeatureValue::Float64(1.0))])
            .unwrap();
        store
            .ingest("g", "u1", 2000, &[("score", FeatureValue::Float64(2.0))])
            .unwrap();
        store
            .ingest("g", "u2", 1500, &[("score", FeatureValue::Float64(1.5))])
            .unwrap();
        store
            .ingest("g", "u2", 2500, &[("score", FeatureValue::Float64(2.5))])
            .unwrap();

        // Generate dataset with PIT semantics
        let entity_df = dataset::EntityDataFrame::new(
            vec!["u1".to_string(), "u2".to_string(), "u1".to_string()],
            vec![1500, 2000, 3000],
        );

        let ds = store.generate_dataset(&entity_df, "g", &["score"]).unwrap();
        assert_eq!(ds.num_rows, 3);
        assert_eq!(ds.columns, vec!["score"]);

        // Row 0: u1 at ts=1500 → score=1.0 (latest <= 1500)
        // Row 1: u2 at ts=2000 → score=1.5 (latest <= 2000)
        // Row 2: u1 at ts=3000 → score=2.0 (latest <= 3000)
        if let dataset::ColumnData::Float64(ref values) = ds.data[0] {
            assert_eq!(values[0], Some(1.0));
            assert_eq!(values[1], Some(1.5));
            assert_eq!(values[2], Some(2.0));
        } else {
            panic!("Expected Float64 column");
        }

        // Statistics should be computed
        assert_eq!(ds.statistics.len(), 1);
        assert_eq!(ds.statistics[0].count, 3);
    }

    #[test]
    fn test_get_at_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        store
            .ingest("g", "e1", 1000, &[("f", FeatureValue::Float64(1.0))])
            .unwrap();
        store
            .ingest("g", "e1", 2000, &[("f", FeatureValue::Float64(2.0))])
            .unwrap();
        store
            .ingest("g", "e1", 3000, &[("f", FeatureValue::Float64(3.0))])
            .unwrap();

        // Latest (version 0 or -1)
        assert_eq!(
            store.get_at_version("g", "e1", "f", 0),
            Some(FeatureValue::Float64(3.0))
        );
        assert_eq!(
            store.get_at_version("g", "e1", "f", -1),
            Some(FeatureValue::Float64(3.0))
        );

        // Previous versions
        assert_eq!(
            store.get_at_version("g", "e1", "f", -2),
            Some(FeatureValue::Float64(2.0))
        );
        assert_eq!(
            store.get_at_version("g", "e1", "f", -3),
            Some(FeatureValue::Float64(1.0))
        );

        // Non-existent version
        assert_eq!(store.get_at_version("g", "e1", "f", -4), None);

        // Non-existent key
        assert_eq!(store.get_at_version("g", "nonexistent", "f", 0), None);
    }

    #[test]
    fn test_get_at_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        store
            .ingest("g", "e1", 1000, &[("f", FeatureValue::Float64(1.0))])
            .unwrap();
        store
            .ingest("g", "e1", 2000, &[("f", FeatureValue::Float64(2.0))])
            .unwrap();

        assert_eq!(
            store.get_at_timestamp("g", "e1", "f", 1500),
            Some(FeatureValue::Float64(1.0))
        );
        assert_eq!(
            store.get_at_timestamp("g", "e1", "f", 2000),
            Some(FeatureValue::Float64(2.0))
        );
        assert_eq!(store.get_at_timestamp("g", "e1", "f", 500), None);
    }

    #[test]
    fn test_prefix_schema_registration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fs.db");
        let config = FeatureStoreConfig::default();
        let mut store = FeatureStore::new(&path, config).unwrap();

        let schema = test_schema();
        store
            .register_prefix_schema("session:", schema.clone())
            .unwrap();

        // Should find schema for matching prefix
        assert!(store.get_prefix_schema("session:abc").is_some());
        assert!(store.get_prefix_schema("session:xyz/123").is_some());

        // Should not find schema for non-matching prefix
        assert!(store.get_prefix_schema("other:abc").is_none());
    }
}

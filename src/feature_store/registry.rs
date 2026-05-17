// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Feature Registry — metadata catalog for all feature definitions.
//!
//! The registry stores:
//! - All registered [`FeatureSchema`] definitions
//! - Schema migration history
//! - Feature view definitions (future)
//! - Feature lineage graph (future)
//!
//! The registry is persisted as a single bincode blob in the append-only log.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SynaError};

use super::schema::FeatureSchema;
use super::serialization::{deserialize, serialize};

/// A schema migration operation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MigrationOp {
    /// Add a new column with a default value for historical queries.
    AddColumn(super::schema::ColumnDef),
    /// Widen a column's type (e.g., Int64 → Float64).
    WidenType {
        /// Column name to widen.
        column: String,
        /// Original type.
        from: super::schema::FeatureType,
        /// New wider type.
        to: super::schema::FeatureType,
    },
    /// Mark a column as deprecated.
    DeprecateColumn {
        /// Column name to deprecate.
        column: String,
    },
    /// Update the default value for a column.
    UpdateDefault {
        /// Column name.
        column: String,
        /// New default value.
        new_default: super::schema::FeatureValue,
    },
}

/// A schema migration record.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SchemaMigration {
    /// Target schema version after this migration.
    pub version: u32,
    /// Operations to apply.
    pub operations: Vec<MigrationOp>,
    /// When the migration was applied (Unix microseconds).
    pub applied_at: u64,
    /// Human-readable description.
    pub description: Option<String>,
}

/// An edge in the feature lineage graph.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LineageEdge {
    /// Source feature reference (group:feature).
    pub source: String,
    /// Target feature reference (group:feature).
    pub target: String,
    /// Transformation description.
    pub transformation: String,
}

/// Query parameters for searching the registry.
#[derive(Debug, Clone, Default)]
pub struct RegistryQuery {
    /// Filter by name (substring match).
    pub name: Option<String>,
    /// Filter by tag.
    pub tag: Option<String>,
    /// Filter by entity key type.
    pub entity_type: Option<super::schema::FeatureType>,
    /// Filter by feature type (any column matches).
    pub feature_type: Option<super::schema::FeatureType>,
}

/// Metadata catalog for all feature definitions.
///
/// Stores schemas, migrations, and lineage. Serializable for persistence.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct FeatureRegistry {
    /// All registered schemas by name.
    pub schemas: HashMap<String, FeatureSchema>,
    /// Schema migration history by schema name.
    pub migrations: HashMap<String, Vec<SchemaMigration>>,
    /// Feature lineage graph.
    pub lineage: HashMap<String, Vec<LineageEdge>>,
}

impl FeatureRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new schema. Errors if name already exists.
    pub fn register(&mut self, schema: FeatureSchema) -> Result<()> {
        schema.validate()?;

        if self.schemas.contains_key(&schema.name) {
            return Err(SynaError::InvalidInput(format!(
                "Feature schema '{}' already registered",
                schema.name
            )));
        }

        self.schemas.insert(schema.name.clone(), schema);
        Ok(())
    }

    /// Get a schema by name.
    pub fn get(&self, name: &str) -> Option<&FeatureSchema> {
        self.schemas.get(name)
    }

    /// List all schema names.
    pub fn list(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// Search the registry by query parameters.
    pub fn search(&self, query: &RegistryQuery) -> Vec<&FeatureSchema> {
        self.schemas
            .values()
            .filter(|schema| {
                // Name filter (substring match)
                if let Some(ref name) = query.name {
                    if !schema.name.contains(name.as_str()) {
                        return false;
                    }
                }

                // Tag filter
                if let Some(ref tag) = query.tag {
                    if !schema.tags.contains(tag) {
                        return false;
                    }
                }

                // Feature type filter (any column matches)
                if let Some(ref ft) = query.feature_type {
                    let has_type = schema
                        .columns
                        .iter()
                        .any(|c| std::mem::discriminant(&c.dtype) == std::mem::discriminant(ft));
                    if !has_type {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Get the migration history for a schema.
    pub fn get_migrations(&self, name: &str) -> Option<&Vec<SchemaMigration>> {
        self.migrations.get(name)
    }

    /// Get lineage for a feature reference.
    pub fn get_lineage(&self, feature_ref: &str) -> Option<&Vec<LineageEdge>> {
        self.lineage.get(feature_ref)
    }

    /// Record a lineage edge.
    pub fn add_lineage(&mut self, source: &str, target: &str, transformation: &str) {
        let edge = LineageEdge {
            source: source.to_string(),
            target: target.to_string(),
            transformation: transformation.to_string(),
        };
        self.lineage
            .entry(target.to_string())
            .or_default()
            .push(edge);
    }

    /// Serialize the entire registry to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serialize(self)
    }

    /// Deserialize a registry from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        deserialize(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_store::schema::*;

    fn test_schema(name: &str) -> FeatureSchema {
        FeatureSchema {
            name: name.to_string(),
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
                    default: None,
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: false,
                    is_event_timestamp: false,
                    deprecated: false,
                },
            ],
            version: 1,
            description: Some("Test schema".to_string()),
            tags: vec!["test".to_string()],
            created_at: 1_000_000,
            created_by: Some("tester".to_string()),
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = FeatureRegistry::new();
        let schema = test_schema("users");
        registry.register(schema.clone()).unwrap();

        assert_eq!(registry.get("users"), Some(&schema));
        assert_eq!(registry.get("nonexistent"), None);
    }

    #[test]
    fn test_duplicate_registration_rejected() {
        let mut registry = FeatureRegistry::new();
        registry.register(test_schema("users")).unwrap();
        assert!(registry.register(test_schema("users")).is_err());
    }

    #[test]
    fn test_list_schemas() {
        let mut registry = FeatureRegistry::new();
        registry.register(test_schema("users")).unwrap();
        registry.register(test_schema("products")).unwrap();

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["products", "users"]);
    }

    #[test]
    fn test_search_by_name() {
        let mut registry = FeatureRegistry::new();
        registry.register(test_schema("user_features")).unwrap();
        registry.register(test_schema("product_features")).unwrap();

        let query = RegistryQuery {
            name: Some("user".to_string()),
            ..Default::default()
        };
        let results = registry.search(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "user_features");
    }

    #[test]
    fn test_search_by_tag() {
        let mut registry = FeatureRegistry::new();
        let mut schema = test_schema("tagged");
        schema.tags = vec!["ml".to_string(), "production".to_string()];
        registry.register(schema).unwrap();
        registry.register(test_schema("untagged")).unwrap();

        let query = RegistryQuery {
            tag: Some("ml".to_string()),
            ..Default::default()
        };
        let results = registry.search(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "tagged");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut registry = FeatureRegistry::new();
        registry.register(test_schema("users")).unwrap();
        registry.register(test_schema("products")).unwrap();
        registry.add_lineage("users:score", "derived:combined", "avg");

        let bytes = registry.to_bytes().unwrap();
        let restored = FeatureRegistry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.schemas.len(), 2);
        assert_eq!(restored.get("users").unwrap().name, "users");
        assert_eq!(restored.get("products").unwrap().name, "products");
        assert!(restored.get_lineage("derived:combined").is_some());
    }

    #[test]
    fn test_lineage() {
        let mut registry = FeatureRegistry::new();
        registry.add_lineage("raw:temp", "derived:avg_temp", "moving_avg(1h)");
        registry.add_lineage("raw:humidity", "derived:avg_temp", "correlation");

        let lineage = registry.get_lineage("derived:avg_temp").unwrap();
        assert_eq!(lineage.len(), 2);
        assert_eq!(lineage[0].source, "raw:temp");
        assert_eq!(lineage[1].source, "raw:humidity");
    }
}

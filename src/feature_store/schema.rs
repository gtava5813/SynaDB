// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Feature schema definitions and validation.
//!
//! This module defines the type system for feature stores:
//! - [`FeatureType`] — supported column types
//! - [`FeatureValue`] — runtime values matching the type system
//! - [`ColumnDef`] — column definitions with constraints
//! - [`FeatureSchema`] — complete schema for a feature group
//!
//! Schemas enforce data quality at ingestion time and enable
//! schema-aware optimizations for serving and dataset generation.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SynaError};

/// Supported feature column types.
///
/// Each type maps to a specific binary representation for efficient
/// serialization with bincode.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FeatureType {
    /// 64-bit floating point number.
    Float64,
    /// 64-bit signed integer.
    Int64,
    /// UTF-8 string.
    String,
    /// Boolean value.
    Bool,
    /// Fixed-dimension float vector (for embeddings).
    Vector(u16),
    /// Unix timestamp in microseconds.
    Timestamp,
    /// Categorical value with bounded cardinality.
    Categorical(u32),
}

impl FeatureType {
    /// Returns a human-readable name for the type.
    pub fn type_name(&self) -> &'static str {
        match self {
            FeatureType::Float64 => "Float64",
            FeatureType::Int64 => "Int64",
            FeatureType::String => "String",
            FeatureType::Bool => "Bool",
            FeatureType::Vector(_) => "Vector",
            FeatureType::Timestamp => "Timestamp",
            FeatureType::Categorical(_) => "Categorical",
        }
    }
}

/// A single typed feature value.
///
/// This enum mirrors [`FeatureType`] at the value level. Each variant
/// carries the actual data for one feature of one entity at one point in time.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FeatureValue {
    /// Absence of value.
    Null,
    /// 64-bit floating point number.
    Float64(f64),
    /// 64-bit signed integer.
    Int64(i64),
    /// UTF-8 string.
    String(String),
    /// Boolean value.
    Bool(bool),
    /// Float vector (embedding).
    Vector(Vec<f32>),
    /// Unix timestamp in microseconds.
    Timestamp(u64),
    /// Categorical value (index into category set).
    Categorical(u32),
}

impl FeatureValue {
    /// Returns the [`FeatureType`] this value corresponds to, or None for Null.
    pub fn feature_type(&self) -> Option<FeatureType> {
        match self {
            FeatureValue::Null => None,
            FeatureValue::Float64(_) => Some(FeatureType::Float64),
            FeatureValue::Int64(_) => Some(FeatureType::Int64),
            FeatureValue::String(_) => Some(FeatureType::String),
            FeatureValue::Bool(_) => Some(FeatureType::Bool),
            FeatureValue::Vector(v) => Some(FeatureType::Vector(v.len() as u16)),
            FeatureValue::Timestamp(_) => Some(FeatureType::Timestamp),
            FeatureValue::Categorical(_) => Some(FeatureType::Categorical(0)),
        }
    }

    /// Returns true if this value is Null.
    pub fn is_null(&self) -> bool {
        matches!(self, FeatureValue::Null)
    }

    /// Returns the float value if this is Float64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FeatureValue::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the integer value if this is Int64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FeatureValue::Int64(v) => Some(*v),
            _ => None,
        }
    }
}

/// A feature value with temporal metadata for storage.
///
/// This is what gets serialized to the append-only log. The two timestamps
/// enable point-in-time queries (event_timestamp) and lineage tracking
/// (ingestion_timestamp).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StoredFeatureValue {
    /// The actual feature value.
    pub value: FeatureValue,
    /// When the event occurred (used for PIT queries).
    pub event_timestamp: u64,
    /// When the value was ingested into the store.
    pub ingestion_timestamp: u64,
}

/// Validation constraints for a feature column.
///
/// Constraints are checked at ingestion time. Any violation causes
/// the entire batch to be rejected.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ColumnConstraints {
    /// If true, null values are rejected.
    pub not_null: bool,
    /// Minimum value (for Float64 and Int64).
    pub min: Option<f64>,
    /// Maximum value (for Float64 and Int64).
    pub max: Option<f64>,
    /// Regex pattern (for String values).
    pub regex: Option<String>,
    /// Allowed values (for String and Categorical).
    pub allowed_values: Option<Vec<String>>,
}

/// A single column definition in a feature schema.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// Column name (unique within schema).
    pub name: String,
    /// Data type for this column.
    pub dtype: FeatureType,
    /// Default value when missing during retrieval.
    pub default: Option<FeatureValue>,
    /// Validation constraints applied at ingestion.
    pub constraints: Option<ColumnConstraints>,
    /// Time-to-live in seconds (for DAVO freshness integration).
    pub ttl_seconds: Option<u64>,
    /// Whether this column is the entity key.
    pub is_entity_key: bool,
    /// Whether this column is the event timestamp.
    pub is_event_timestamp: bool,
    /// Whether this column is deprecated.
    pub deprecated: bool,
}

/// A typed schema for a feature group.
///
/// Each feature group has exactly one entity key column and one event
/// timestamp column. All other columns are feature columns with typed
/// values and optional constraints.
///
/// # Examples
///
/// ```rust
/// use synadb::feature_store::schema::*;
///
/// let schema = FeatureSchema {
///     name: "user_features".to_string(),
///     columns: vec![
///         ColumnDef {
///             name: "user_id".to_string(),
///             dtype: FeatureType::String,
///             default: None,
///             constraints: Some(ColumnConstraints { not_null: true, ..Default::default() }),
///             ttl_seconds: None,
///             is_entity_key: true,
///             is_event_timestamp: false,
///             deprecated: false,
///         },
///         ColumnDef {
///             name: "event_time".to_string(),
///             dtype: FeatureType::Timestamp,
///             default: None,
///             constraints: None,
///             ttl_seconds: None,
///             is_entity_key: false,
///             is_event_timestamp: true,
///             deprecated: false,
///         },
///         ColumnDef {
///             name: "purchase_count".to_string(),
///             dtype: FeatureType::Int64,
///             default: Some(FeatureValue::Int64(0)),
///             constraints: Some(ColumnConstraints { min: Some(0.0), ..Default::default() }),
///             ttl_seconds: Some(86400),
///             is_entity_key: false,
///             is_event_timestamp: false,
///             deprecated: false,
///         },
///     ],
///     version: 1,
///     description: Some("User purchase features".to_string()),
///     tags: vec!["user".to_string(), "purchase".to_string()],
///     created_at: 0,
///     created_by: None,
/// };
///
/// assert!(schema.validate().is_ok());
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FeatureSchema {
    /// Unique name for this feature group.
    pub name: String,
    /// Column definitions.
    pub columns: Vec<ColumnDef>,
    /// Schema version (monotonically increasing on migration).
    pub version: u32,
    /// Human-readable description.
    pub description: Option<String>,
    /// Tags for discovery and filtering.
    pub tags: Vec<String>,
    /// Creation timestamp (Unix microseconds).
    pub created_at: u64,
    /// Creator identifier.
    pub created_by: Option<String>,
}

impl FeatureSchema {
    /// Validate that the schema is well-formed.
    ///
    /// Checks:
    /// - Exactly one entity key column
    /// - Exactly one event timestamp column
    /// - No duplicate column names
    /// - Name is non-empty
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(SynaError::InvalidInput(
                "Feature schema name cannot be empty".to_string(),
            ));
        }

        if self.columns.is_empty() {
            return Err(SynaError::InvalidInput(
                "Feature schema must have at least one column".to_string(),
            ));
        }

        // Check for exactly one entity key
        let entity_key_count = self.columns.iter().filter(|c| c.is_entity_key).count();
        if entity_key_count != 1 {
            return Err(SynaError::InvalidInput(format!(
                "Feature schema must have exactly one entity key column, found {}",
                entity_key_count
            )));
        }

        // Check for exactly one event timestamp
        let event_ts_count = self.columns.iter().filter(|c| c.is_event_timestamp).count();
        if event_ts_count != 1 {
            return Err(SynaError::InvalidInput(format!(
                "Feature schema must have exactly one event timestamp column, found {}",
                event_ts_count
            )));
        }

        // Check for duplicate column names
        let mut seen = std::collections::HashSet::new();
        for col in &self.columns {
            if col.name.is_empty() {
                return Err(SynaError::InvalidInput(
                    "Column name cannot be empty".to_string(),
                ));
            }
            if !seen.insert(&col.name) {
                return Err(SynaError::InvalidInput(format!(
                    "Duplicate column name: '{}'",
                    col.name
                )));
            }
        }

        Ok(())
    }

    /// Validate a row of values against this schema.
    ///
    /// Checks type compatibility and constraint satisfaction for each provided value.
    /// Values not in the row are checked against not_null constraints.
    pub fn validate_row(&self, values: &[(&str, FeatureValue)]) -> Result<()> {
        for (name, value) in values {
            let col = self.columns.iter().find(|c| c.name == *name);
            let col = match col {
                Some(c) => c,
                None => {
                    return Err(SynaError::InvalidInput(format!(
                        "Unknown column '{}' in schema '{}'",
                        name, self.name
                    )));
                }
            };

            // Skip entity key and event timestamp — they're handled separately
            if col.is_entity_key || col.is_event_timestamp {
                continue;
            }

            // Check null constraint
            if value.is_null() {
                if let Some(ref constraints) = col.constraints {
                    if constraints.not_null {
                        return Err(SynaError::InvalidInput(format!(
                            "Column '{}': null value not allowed (not_null constraint)",
                            name
                        )));
                    }
                }
                continue;
            }

            // Check type compatibility
            if !is_type_compatible(&col.dtype, value) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}': expected type {}, got {:?}",
                    name,
                    col.dtype.type_name(),
                    value
                )));
            }

            // Check constraints
            if let Some(ref constraints) = col.constraints {
                validate_constraints(name, value, constraints)?;
            }
        }

        // Check not_null columns that are missing from the row
        for col in &self.columns {
            if col.is_entity_key || col.is_event_timestamp {
                continue;
            }
            if let Some(ref constraints) = col.constraints {
                if constraints.not_null {
                    let provided = values.iter().any(|(n, _)| *n == col.name);
                    if !provided && col.default.is_none() {
                        return Err(SynaError::InvalidInput(format!(
                            "Column '{}': required (not_null) but not provided and no default",
                            col.name
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the entity key column name.
    pub fn entity_key_column(&self) -> &str {
        self.columns
            .iter()
            .find(|c| c.is_entity_key)
            .map(|c| c.name.as_str())
            .unwrap_or("")
    }

    /// Get the event timestamp column name.
    pub fn event_timestamp_column(&self) -> &str {
        self.columns
            .iter()
            .find(|c| c.is_event_timestamp)
            .map(|c| c.name.as_str())
            .unwrap_or("")
    }

    /// Get feature columns (excluding entity key and event timestamp).
    pub fn feature_columns(&self) -> Vec<&ColumnDef> {
        self.columns
            .iter()
            .filter(|c| !c.is_entity_key && !c.is_event_timestamp)
            .collect()
    }
}

/// Check if a value is compatible with the expected type.
fn is_type_compatible(dtype: &FeatureType, value: &FeatureValue) -> bool {
    match (dtype, value) {
        (FeatureType::Float64, FeatureValue::Float64(_)) => true,
        (FeatureType::Int64, FeatureValue::Int64(_)) => true,
        (FeatureType::String, FeatureValue::String(_)) => true,
        (FeatureType::Bool, FeatureValue::Bool(_)) => true,
        (FeatureType::Vector(expected_dims), FeatureValue::Vector(v)) => {
            v.len() == *expected_dims as usize
        }
        (FeatureType::Timestamp, FeatureValue::Timestamp(_)) => true,
        (FeatureType::Categorical(max_card), FeatureValue::Categorical(v)) => *v < *max_card,
        _ => false,
    }
}

/// Validate a value against column constraints.
fn validate_constraints(
    name: &str,
    value: &FeatureValue,
    constraints: &ColumnConstraints,
) -> Result<()> {
    // Min/max checks for numeric types
    if let Some(min) = constraints.min {
        let numeric_val = match value {
            FeatureValue::Float64(v) => Some(*v),
            FeatureValue::Int64(v) => Some(*v as f64),
            _ => None,
        };
        if let Some(v) = numeric_val {
            if v < min {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}': value {} is below minimum {}",
                    name, v, min
                )));
            }
        }
    }

    if let Some(max) = constraints.max {
        let numeric_val = match value {
            FeatureValue::Float64(v) => Some(*v),
            FeatureValue::Int64(v) => Some(*v as f64),
            _ => None,
        };
        if let Some(v) = numeric_val {
            if v > max {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}': value {} is above maximum {}",
                    name, v, max
                )));
            }
        }
    }

    // Regex check for strings
    if let Some(ref pattern) = constraints.regex {
        if let FeatureValue::String(s) = value {
            let re = regex::Regex::new(pattern).map_err(|e| {
                SynaError::InvalidInput(format!(
                    "Column '{}': invalid regex '{}': {}",
                    name, pattern, e
                ))
            })?;
            if !re.is_match(s) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}': value '{}' does not match regex '{}'",
                    name, s, pattern
                )));
            }
        }
    }

    // Allowed values check
    if let Some(ref allowed) = constraints.allowed_values {
        let str_val = match value {
            FeatureValue::String(s) => Some(s.as_str()),
            _ => None,
        };
        if let Some(s) = str_val {
            if !allowed.iter().any(|a| a == s) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}': value '{}' not in allowed values {:?}",
                    name, s, allowed
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_schema() -> FeatureSchema {
        FeatureSchema {
            name: "test".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    dtype: FeatureType::String,
                    default: None,
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: true,
                    is_event_timestamp: false,
                    deprecated: false,
                },
                ColumnDef {
                    name: "ts".to_string(),
                    dtype: FeatureType::Timestamp,
                    default: None,
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: false,
                    is_event_timestamp: true,
                    deprecated: false,
                },
            ],
            version: 1,
            description: None,
            tags: vec![],
            created_at: 0,
            created_by: None,
        }
    }

    #[test]
    fn test_valid_schema() {
        let schema = minimal_schema();
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_no_entity_key() {
        let mut schema = minimal_schema();
        schema.columns[0].is_entity_key = false;
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_no_event_timestamp() {
        let mut schema = minimal_schema();
        schema.columns[1].is_event_timestamp = false;
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_duplicate_column_names() {
        let mut schema = minimal_schema();
        schema.columns[1].name = "id".to_string();
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_validate_row_type_mismatch() {
        let mut schema = minimal_schema();
        schema.columns.push(ColumnDef {
            name: "score".to_string(),
            dtype: FeatureType::Float64,
            default: None,
            constraints: None,
            ttl_seconds: None,
            is_entity_key: false,
            is_event_timestamp: false,
            deprecated: false,
        });

        let result = schema.validate_row(&[("score", FeatureValue::String("bad".to_string()))]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_row_not_null() {
        let mut schema = minimal_schema();
        schema.columns.push(ColumnDef {
            name: "score".to_string(),
            dtype: FeatureType::Float64,
            default: None,
            constraints: Some(ColumnConstraints {
                not_null: true,
                ..Default::default()
            }),
            ttl_seconds: None,
            is_entity_key: false,
            is_event_timestamp: false,
            deprecated: false,
        });

        let result = schema.validate_row(&[("score", FeatureValue::Null)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_row_min_max() {
        let mut schema = minimal_schema();
        schema.columns.push(ColumnDef {
            name: "score".to_string(),
            dtype: FeatureType::Float64,
            default: None,
            constraints: Some(ColumnConstraints {
                min: Some(0.0),
                max: Some(100.0),
                ..Default::default()
            }),
            ttl_seconds: None,
            is_entity_key: false,
            is_event_timestamp: false,
            deprecated: false,
        });

        assert!(schema
            .validate_row(&[("score", FeatureValue::Float64(50.0))])
            .is_ok());
        assert!(schema
            .validate_row(&[("score", FeatureValue::Float64(-1.0))])
            .is_err());
        assert!(schema
            .validate_row(&[("score", FeatureValue::Float64(101.0))])
            .is_err());
    }

    #[test]
    fn test_entity_key_column() {
        let schema = minimal_schema();
        assert_eq!(schema.entity_key_column(), "id");
    }

    #[test]
    fn test_event_timestamp_column() {
        let schema = minimal_schema();
        assert_eq!(schema.event_timestamp_column(), "ts");
    }
}

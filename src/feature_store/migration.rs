// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Schema evolution and migration.
//!
//! Supports:
//! - Adding new columns with default values
//! - Widening types (Int64 → Float64)
//! - Deprecating columns
//! - Updating default values
//!
//! Rejects:
//! - Narrowing types (Float64 → Int64)
//! - Removing columns with existing data

use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, SynaError};

use super::registry::{MigrationOp, SchemaMigration};
use super::schema::{FeatureSchema, FeatureType};

/// Apply a migration to a schema, returning the updated schema.
///
/// Validates that the migration is safe (no narrowing, no removal of
/// columns with data) before applying.
pub fn apply_migration(schema: &mut FeatureSchema, migration: &SchemaMigration) -> Result<()> {
    // Validate all operations first
    for op in &migration.operations {
        validate_operation(schema, op)?;
    }

    // Apply all operations
    for op in &migration.operations {
        apply_operation(schema, op)?;
    }

    // Bump version
    schema.version = migration.version;

    Ok(())
}

/// Validate a single migration operation against the current schema.
fn validate_operation(schema: &FeatureSchema, op: &MigrationOp) -> Result<()> {
    match op {
        MigrationOp::AddColumn(col_def) => {
            // Check column doesn't already exist
            if schema.columns.iter().any(|c| c.name == col_def.name) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}' already exists in schema '{}'",
                    col_def.name, schema.name
                )));
            }
            Ok(())
        }
        MigrationOp::WidenType { column, from, to } => {
            // Find the column
            let col = schema.columns.iter().find(|c| c.name == *column);
            let col = col.ok_or_else(|| {
                SynaError::InvalidInput(format!(
                    "Column '{}' not found in schema '{}'",
                    column, schema.name
                ))
            })?;

            // Verify current type matches 'from'
            if col.dtype != *from {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}' has type {:?}, expected {:?}",
                    column, col.dtype, from
                )));
            }

            // Only allow widening (Int64 → Float64)
            if !is_widening(from, to) {
                return Err(SynaError::InvalidInput(format!(
                    "Cannot narrow type from {:?} to {:?} for column '{}'",
                    from, to, column
                )));
            }

            Ok(())
        }
        MigrationOp::DeprecateColumn { column } => {
            // Check column exists
            if !schema.columns.iter().any(|c| c.name == *column) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}' not found in schema '{}'",
                    column, schema.name
                )));
            }
            Ok(())
        }
        MigrationOp::UpdateDefault { column, .. } => {
            // Check column exists
            if !schema.columns.iter().any(|c| c.name == *column) {
                return Err(SynaError::InvalidInput(format!(
                    "Column '{}' not found in schema '{}'",
                    column, schema.name
                )));
            }
            Ok(())
        }
    }
}

/// Apply a single migration operation to the schema.
fn apply_operation(schema: &mut FeatureSchema, op: &MigrationOp) -> Result<()> {
    match op {
        MigrationOp::AddColumn(col_def) => {
            schema.columns.push(col_def.clone());
            Ok(())
        }
        MigrationOp::WidenType { column, to, .. } => {
            if let Some(col) = schema.columns.iter_mut().find(|c| c.name == *column) {
                col.dtype = to.clone();
            }
            Ok(())
        }
        MigrationOp::DeprecateColumn { column } => {
            if let Some(col) = schema.columns.iter_mut().find(|c| c.name == *column) {
                col.deprecated = true;
            }
            Ok(())
        }
        MigrationOp::UpdateDefault {
            column,
            new_default,
        } => {
            if let Some(col) = schema.columns.iter_mut().find(|c| c.name == *column) {
                col.default = Some(new_default.clone());
            }
            Ok(())
        }
    }
}

/// Check if a type widening is valid.
fn is_widening(from: &FeatureType, to: &FeatureType) -> bool {
    matches!((from, to), (FeatureType::Int64, FeatureType::Float64))
}

/// Create a migration with the current timestamp.
pub fn create_migration(
    version: u32,
    operations: Vec<MigrationOp>,
    description: Option<String>,
) -> SchemaMigration {
    let applied_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);

    SchemaMigration {
        version,
        operations,
        applied_at,
        description,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_store::schema::*;

    fn base_schema() -> FeatureSchema {
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
                ColumnDef {
                    name: "count".to_string(),
                    dtype: FeatureType::Int64,
                    default: Some(FeatureValue::Int64(0)),
                    constraints: None,
                    ttl_seconds: None,
                    is_entity_key: false,
                    is_event_timestamp: false,
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
    fn test_add_column() {
        let mut schema = base_schema();
        let migration = create_migration(
            2,
            vec![MigrationOp::AddColumn(ColumnDef {
                name: "score".to_string(),
                dtype: FeatureType::Float64,
                default: Some(FeatureValue::Float64(0.0)),
                constraints: None,
                ttl_seconds: None,
                is_entity_key: false,
                is_event_timestamp: false,
                deprecated: false,
            })],
            Some("Add score column".to_string()),
        );

        apply_migration(&mut schema, &migration).unwrap();
        assert_eq!(schema.version, 2);
        assert_eq!(schema.columns.len(), 4);
        assert_eq!(schema.columns[3].name, "score");
    }

    #[test]
    fn test_widen_type() {
        let mut schema = base_schema();
        let migration = create_migration(
            2,
            vec![MigrationOp::WidenType {
                column: "count".to_string(),
                from: FeatureType::Int64,
                to: FeatureType::Float64,
            }],
            None,
        );

        apply_migration(&mut schema, &migration).unwrap();
        assert_eq!(schema.columns[2].dtype, FeatureType::Float64);
    }

    #[test]
    fn test_narrow_type_rejected() {
        let mut schema = base_schema();
        // Change count to Float64 first
        schema.columns[2].dtype = FeatureType::Float64;

        let migration = create_migration(
            2,
            vec![MigrationOp::WidenType {
                column: "count".to_string(),
                from: FeatureType::Float64,
                to: FeatureType::Int64,
            }],
            None,
        );

        assert!(apply_migration(&mut schema, &migration).is_err());
    }

    #[test]
    fn test_deprecate_column() {
        let mut schema = base_schema();
        let migration = create_migration(
            2,
            vec![MigrationOp::DeprecateColumn {
                column: "count".to_string(),
            }],
            None,
        );

        apply_migration(&mut schema, &migration).unwrap();
        assert!(schema.columns[2].deprecated);
    }

    #[test]
    fn test_update_default() {
        let mut schema = base_schema();
        let migration = create_migration(
            2,
            vec![MigrationOp::UpdateDefault {
                column: "count".to_string(),
                new_default: FeatureValue::Int64(42),
            }],
            None,
        );

        apply_migration(&mut schema, &migration).unwrap();
        assert_eq!(schema.columns[2].default, Some(FeatureValue::Int64(42)));
    }

    #[test]
    fn test_add_duplicate_column_rejected() {
        let mut schema = base_schema();
        let migration = create_migration(
            2,
            vec![MigrationOp::AddColumn(ColumnDef {
                name: "count".to_string(), // Already exists
                dtype: FeatureType::Float64,
                default: None,
                constraints: None,
                ttl_seconds: None,
                is_entity_key: false,
                is_event_timestamp: false,
                deprecated: false,
            })],
            None,
        );

        assert!(apply_migration(&mut schema, &migration).is_err());
    }
}

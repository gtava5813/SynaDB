// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Model registry for versioned artifact storage.
//!
//! This module provides:
//! - [`ModelRegistry`] - The main registry struct for storing and versioning ML models
//! - [`ModelVersion`] - Metadata for a model version
//! - [`ModelStage`] - Deployment stage enum (Development, Staging, Production, Archived)
//!
//! # Usage
//!
//! ```rust,no_run
//! use synadb::model_registry::{ModelRegistry, ModelStage};
//! use std::collections::HashMap;
//!
//! let mut registry = ModelRegistry::new("models.db").unwrap();
//!
//! // Save a model
//! let model_data = vec![0u8; 1024]; // Your model bytes
//! let metadata = HashMap::new();
//! let version = registry.save_model("my_model", &model_data, metadata).unwrap();
//!
//! // Load a model
//! let (data, version_info) = registry.load_model("my_model", None).unwrap();
//! ```
//!
//! _Requirements: 4.1, 4.2_

use crate::engine::SynaDB;
use crate::error::Result;
use crate::types::Atom;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Model Stage Enum
// =============================================================================

/// Model deployment stage.
///
/// Models progress through stages as they are validated and deployed:
/// - `Development` - Initial stage for new models
/// - `Staging` - Models being tested before production
/// - `Production` - Models actively serving predictions
/// - `Archived` - Retired models kept for reference
///
/// # Examples
///
/// ```rust
/// use synadb::model_registry::ModelStage;
///
/// let stage = ModelStage::Development;
/// assert_eq!(stage, ModelStage::Development);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelStage {
    /// Initial stage for new models under development.
    #[default]
    Development,
    /// Models being tested before production deployment.
    Staging,
    /// Models actively serving predictions in production.
    Production,
    /// Retired models kept for reference and rollback.
    Archived,
}

impl std::fmt::Display for ModelStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStage::Development => write!(f, "Development"),
            ModelStage::Staging => write!(f, "Staging"),
            ModelStage::Production => write!(f, "Production"),
            ModelStage::Archived => write!(f, "Archived"),
        }
    }
}

// =============================================================================
// Model Version Metadata
// =============================================================================

/// Metadata for a model version.
///
/// Contains all information about a specific version of a model,
/// including its checksum for integrity verification.
///
/// # Fields
///
/// - `name` - The model name
/// - `version` - Version number (auto-incremented)
/// - `created_at` - Unix timestamp when the model was saved
/// - `checksum` - SHA-256 hash of the model data for integrity verification
/// - `size_bytes` - Size of the model data in bytes
/// - `metadata` - User-provided key-value metadata
/// - `stage` - Current deployment stage
///
/// _Requirements: 4.1, 4.2, 4.3_
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// The model name.
    pub name: String,
    /// Version number (auto-incremented starting from 1).
    pub version: u32,
    /// Unix timestamp (seconds since epoch) when the model was saved.
    pub created_at: u64,
    /// SHA-256 checksum of the model data for integrity verification.
    pub checksum: String,
    /// Size of the model data in bytes.
    pub size_bytes: u64,
    /// User-provided key-value metadata (e.g., accuracy, framework, description).
    pub metadata: HashMap<String, String>,
    /// Current deployment stage.
    pub stage: ModelStage,
}

impl ModelVersion {
    /// Creates a new ModelVersion with the given parameters.
    pub fn new(
        name: String,
        version: u32,
        created_at: u64,
        checksum: String,
        size_bytes: u64,
        metadata: HashMap<String, String>,
        stage: ModelStage,
    ) -> Self {
        Self {
            name,
            version,
            created_at,
            checksum,
            size_bytes,
            metadata,
            stage,
        }
    }
}

// =============================================================================
// Model Registry
// =============================================================================

/// Model registry for storing and versioning ML models.
///
/// The registry provides:
/// - Automatic version numbering
/// - SHA-256 checksum computation and verification
/// - Stage management (Development → Staging → Production → Archived)
/// - Metadata storage for each version
///
/// # Storage Format
///
/// Models are stored in the underlying SynaDB with the following key structure:
/// - `model/{name}/v{version}/data` - The model binary data
/// - `model/{name}/v{version}/meta` - JSON metadata (ModelVersion)
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::model_registry::{ModelRegistry, ModelStage};
/// use std::collections::HashMap;
///
/// // Create a new registry
/// let mut registry = ModelRegistry::new("models.db").unwrap();
///
/// // Save a model with metadata
/// let model_data = vec![0u8; 1024];
/// let mut metadata = HashMap::new();
/// metadata.insert("accuracy".to_string(), "0.95".to_string());
/// metadata.insert("framework".to_string(), "pytorch".to_string());
///
/// let version = registry.save_model("classifier", &model_data, metadata).unwrap();
/// println!("Saved model version: {}", version.version);
///
/// // Load the latest version
/// let (data, info) = registry.load_model("classifier", None).unwrap();
/// println!("Loaded {} bytes, checksum: {}", data.len(), info.checksum);
///
/// // Promote to production
/// registry.set_stage("classifier", version.version, ModelStage::Production).unwrap();
/// ```
///
/// _Requirements: 4.1, 4.2_
pub struct ModelRegistry {
    /// The underlying database for storage.
    db: SynaDB,
}

impl ModelRegistry {
    /// Creates or opens a model registry at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    ///
    /// # Returns
    ///
    /// * `Ok(ModelRegistry)` - The opened registry
    /// * `Err(SynaError)` - If the database cannot be opened
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::ModelRegistry;
    ///
    /// let registry = ModelRegistry::new("models.db").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = SynaDB::new(path)?;
        Ok(Self { db })
    }

    /// Returns a reference to the underlying database.
    ///
    /// This is useful for advanced operations or debugging.
    pub fn db(&self) -> &SynaDB {
        &self.db
    }

    /// Returns a mutable reference to the underlying database.
    ///
    /// This is useful for advanced operations or debugging.
    pub fn db_mut(&mut self) -> &mut SynaDB {
        &mut self.db
    }

    /// Saves a model with automatic versioning and checksum computation.
    ///
    /// This method:
    /// 1. Computes a SHA-256 checksum of the model data for integrity verification
    /// 2. Automatically assigns the next version number
    /// 3. Stores the model data and metadata in the database
    ///
    /// # Arguments
    ///
    /// * `name` - The model name (used as identifier)
    /// * `data` - The raw model bytes (weights, serialized model, etc.)
    /// * `metadata` - User-provided key-value metadata (e.g., accuracy, framework)
    ///
    /// # Returns
    ///
    /// * `Ok(ModelVersion)` - The created version metadata including checksum
    /// * `Err(SynaError)` - If storage fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::ModelRegistry;
    /// use std::collections::HashMap;
    ///
    /// let mut registry = ModelRegistry::new("models.db").unwrap();
    ///
    /// let model_data = vec![0u8; 1024]; // Your model bytes
    /// let mut metadata = HashMap::new();
    /// metadata.insert("accuracy".to_string(), "0.95".to_string());
    /// metadata.insert("framework".to_string(), "pytorch".to_string());
    ///
    /// let version = registry.save_model("classifier", &model_data, metadata).unwrap();
    /// println!("Saved version {} with checksum {}", version.version, version.checksum);
    /// ```
    ///
    /// _Requirements: 4.1, 4.2, 4.3_
    pub fn save_model(
        &mut self,
        name: &str,
        data: &[u8],
        metadata: HashMap<String, String>,
    ) -> Result<ModelVersion> {
        // 1. Compute SHA-256 checksum
        let mut hasher = Sha256::new();
        hasher.update(data);
        let checksum = format!("{:x}", hasher.finalize());

        // 2. Get next version number
        let version = self.get_next_version(name);

        // 3. Create version metadata
        let model_version = ModelVersion {
            name: name.to_string(),
            version,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            checksum: checksum.clone(),
            size_bytes: data.len() as u64,
            metadata,
            stage: ModelStage::Development,
        };

        // 4. Store model data
        let data_key = format!("model/{}/v{}/data", name, version);
        self.db.append(&data_key, Atom::Bytes(data.to_vec()))?;

        // 5. Store metadata as JSON
        let meta_key = format!("model/{}/v{}/meta", name, version);
        let meta_json = serde_json::to_string(&model_version)
            .map_err(|e| crate::error::SynaError::InvalidPath(e.to_string()))?;
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(model_version)
    }

    /// Gets the next version number for a model.
    ///
    /// Scans existing versions and returns max + 1, or 1 if no versions exist.
    fn get_next_version(&self, name: &str) -> u32 {
        let prefix = format!("model/{}/v", name);
        let versions: Vec<u32> = self
            .db
            .keys()
            .iter()
            .filter(|k| k.starts_with(&prefix) && k.ends_with("/meta"))
            .filter_map(|k| k.strip_prefix(&prefix)?.strip_suffix("/meta")?.parse().ok())
            .collect();
        versions.into_iter().max().unwrap_or(0) + 1
    }

    /// Gets the latest version number for a model.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// * `Ok(u32)` - The latest version number
    /// * `Err(SynaError::ModelNotFound)` - If no versions exist for this model
    fn get_latest_version(&self, name: &str) -> Result<u32> {
        let prefix = format!("model/{}/v", name);
        self.db
            .keys()
            .iter()
            .filter(|k| k.starts_with(&prefix) && k.ends_with("/meta"))
            .filter_map(|k| k.strip_prefix(&prefix)?.strip_suffix("/meta")?.parse().ok())
            .max()
            .ok_or_else(|| crate::error::SynaError::ModelNotFound(name.to_string()))
    }

    /// Loads a model by name and optional version with checksum verification.
    ///
    /// This method:
    /// 1. Determines the version to load (latest if not specified)
    /// 2. Loads the model metadata
    /// 3. Loads the model data
    /// 4. Verifies the checksum matches the stored checksum
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    /// * `version` - Optional version number (loads latest if None)
    ///
    /// # Returns
    ///
    /// * `Ok((Vec<u8>, ModelVersion))` - The model data and metadata
    /// * `Err(SynaError::ModelNotFound)` - If the model or version doesn't exist
    /// * `Err(SynaError::ChecksumMismatch)` - If the data is corrupted
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::ModelRegistry;
    /// use std::collections::HashMap;
    ///
    /// let mut registry = ModelRegistry::new("models.db").unwrap();
    ///
    /// // Save a model first
    /// let model_data = vec![0u8; 1024];
    /// registry.save_model("classifier", &model_data, HashMap::new()).unwrap();
    ///
    /// // Load the latest version
    /// let (data, info) = registry.load_model("classifier", None).unwrap();
    /// println!("Loaded {} bytes", data.len());
    ///
    /// // Load a specific version
    /// let (data, info) = registry.load_model("classifier", Some(1)).unwrap();
    /// println!("Loaded version {}", info.version);
    /// ```
    ///
    /// _Requirements: 4.4_
    pub fn load_model(
        &mut self,
        name: &str,
        version: Option<u32>,
    ) -> Result<(Vec<u8>, ModelVersion)> {
        // 1. Determine version to load
        let v = match version {
            Some(v) => v,
            None => self.get_latest_version(name)?,
        };

        // 2. Load metadata
        let meta_key = format!("model/{}/v{}/meta", name, v);
        let meta_json = match self.db.get(&meta_key)? {
            Some(Atom::Text(s)) => s,
            _ => return Err(crate::error::SynaError::ModelNotFound(name.to_string())),
        };
        let model_version: ModelVersion = serde_json::from_str(&meta_json)
            .map_err(|e| crate::error::SynaError::InvalidPath(e.to_string()))?;

        // 3. Load data
        let data_key = format!("model/{}/v{}/data", name, v);
        let data = match self.db.get(&data_key)? {
            Some(Atom::Bytes(b)) => b,
            _ => return Err(crate::error::SynaError::ModelNotFound(name.to_string())),
        };

        // 4. Verify checksum
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let computed = format!("{:x}", hasher.finalize());

        if computed != model_version.checksum {
            return Err(crate::error::SynaError::ChecksumMismatch {
                expected: model_version.checksum.clone(),
                got: computed,
            });
        }

        Ok((data, model_version))
    }

    /// Lists all versions of a model.
    ///
    /// Returns all versions sorted by version number in ascending order.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<ModelVersion>)` - List of all versions sorted by version number
    /// * `Err(SynaError)` - If reading from the database fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::ModelRegistry;
    /// use std::collections::HashMap;
    ///
    /// let mut registry = ModelRegistry::new("models.db").unwrap();
    ///
    /// // Save multiple versions
    /// let data = vec![1u8, 2, 3];
    /// registry.save_model("model", &data, HashMap::new()).unwrap();
    /// registry.save_model("model", &data, HashMap::new()).unwrap();
    ///
    /// // List all versions
    /// let versions = registry.list_versions("model").unwrap();
    /// for v in versions {
    ///     println!("Version {}: stage={}", v.version, v.stage);
    /// }
    /// ```
    ///
    /// _Requirements: 4.5_
    pub fn list_versions(&mut self, name: &str) -> Result<Vec<ModelVersion>> {
        let prefix = format!("model/{}/v", name);
        let mut versions = Vec::new();

        for key in self.db.keys() {
            if key.starts_with(&prefix) && key.ends_with("/meta") {
                if let Some(Atom::Text(json)) = self.db.get(&key)? {
                    let v: ModelVersion = serde_json::from_str(&json)
                        .map_err(|e| crate::error::SynaError::InvalidPath(e.to_string()))?;
                    versions.push(v);
                }
            }
        }

        versions.sort_by_key(|v| v.version);
        Ok(versions)
    }

    /// Sets the deployment stage for a model version.
    ///
    /// Updates the stage of an existing model version. This is used to
    /// promote models through the deployment pipeline:
    /// Development → Staging → Production → Archived
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    /// * `version` - The version number to update
    /// * `stage` - The new deployment stage
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the stage was updated successfully
    /// * `Err(SynaError::ModelNotFound)` - If the model or version doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::{ModelRegistry, ModelStage};
    /// use std::collections::HashMap;
    ///
    /// let mut registry = ModelRegistry::new("models.db").unwrap();
    ///
    /// // Save a model
    /// let data = vec![1u8, 2, 3];
    /// let version = registry.save_model("model", &data, HashMap::new()).unwrap();
    ///
    /// // Promote to staging
    /// registry.set_stage("model", version.version, ModelStage::Staging).unwrap();
    ///
    /// // Promote to production
    /// registry.set_stage("model", version.version, ModelStage::Production).unwrap();
    /// ```
    ///
    /// _Requirements: 4.7_
    pub fn set_stage(&mut self, name: &str, version: u32, stage: ModelStage) -> Result<()> {
        let meta_key = format!("model/{}/v{}/meta", name, version);

        let meta_json = match self.db.get(&meta_key)? {
            Some(Atom::Text(s)) => s,
            _ => return Err(crate::error::SynaError::ModelNotFound(name.to_string())),
        };

        let mut model_version: ModelVersion = serde_json::from_str(&meta_json)
            .map_err(|e| crate::error::SynaError::InvalidPath(e.to_string()))?;
        model_version.stage = stage;

        let updated_json = serde_json::to_string(&model_version)
            .map_err(|e| crate::error::SynaError::InvalidPath(e.to_string()))?;
        self.db.append(&meta_key, Atom::Text(updated_json))?;

        Ok(())
    }

    /// Gets the production model for a name.
    ///
    /// Searches through all versions of a model and returns the one
    /// that is currently in the Production stage.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// * `Ok(Some(ModelVersion))` - The production version if one exists
    /// * `Ok(None)` - If no version is in production
    /// * `Err(SynaError)` - If reading from the database fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::model_registry::{ModelRegistry, ModelStage};
    /// use std::collections::HashMap;
    ///
    /// let mut registry = ModelRegistry::new("models.db").unwrap();
    ///
    /// // Save and promote a model
    /// let data = vec![1u8, 2, 3];
    /// let version = registry.save_model("model", &data, HashMap::new()).unwrap();
    /// registry.set_stage("model", version.version, ModelStage::Production).unwrap();
    ///
    /// // Get the production model
    /// if let Some(prod) = registry.get_production("model").unwrap() {
    ///     println!("Production version: {}", prod.version);
    /// }
    /// ```
    ///
    /// _Requirements: 4.7_
    pub fn get_production(&mut self, name: &str) -> Result<Option<ModelVersion>> {
        let versions = self.list_versions(name)?;
        Ok(versions
            .into_iter()
            .find(|v| v.stage == ModelStage::Production))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_stage_default() {
        let stage = ModelStage::default();
        assert_eq!(stage, ModelStage::Development);
    }

    #[test]
    fn test_model_stage_display() {
        assert_eq!(format!("{}", ModelStage::Development), "Development");
        assert_eq!(format!("{}", ModelStage::Staging), "Staging");
        assert_eq!(format!("{}", ModelStage::Production), "Production");
        assert_eq!(format!("{}", ModelStage::Archived), "Archived");
    }

    #[test]
    fn test_model_version_new() {
        let mut metadata = HashMap::new();
        metadata.insert("accuracy".to_string(), "0.95".to_string());

        let version = ModelVersion::new(
            "test_model".to_string(),
            1,
            1234567890,
            "abc123".to_string(),
            1024,
            metadata.clone(),
            ModelStage::Development,
        );

        assert_eq!(version.name, "test_model");
        assert_eq!(version.version, 1);
        assert_eq!(version.created_at, 1234567890);
        assert_eq!(version.checksum, "abc123");
        assert_eq!(version.size_bytes, 1024);
        assert_eq!(version.metadata.get("accuracy"), Some(&"0.95".to_string()));
        assert_eq!(version.stage, ModelStage::Development);
    }

    #[test]
    fn test_model_registry_new() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_registry.db");

        let registry = ModelRegistry::new(&db_path);
        assert!(registry.is_ok());
    }

    #[test]
    fn test_model_stage_serialization() {
        // Test that ModelStage can be serialized and deserialized
        let stage = ModelStage::Production;
        let serialized = serde_json::to_string(&stage).unwrap();
        let deserialized: ModelStage = serde_json::from_str(&serialized).unwrap();
        assert_eq!(stage, deserialized);
    }

    #[test]
    fn test_model_version_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let version = ModelVersion::new(
            "model".to_string(),
            1,
            1000,
            "checksum".to_string(),
            512,
            metadata,
            ModelStage::Staging,
        );

        let serialized = serde_json::to_string(&version).unwrap();
        let deserialized: ModelVersion = serde_json::from_str(&serialized).unwrap();

        assert_eq!(version.name, deserialized.name);
        assert_eq!(version.version, deserialized.version);
        assert_eq!(version.stage, deserialized.stage);
    }

    #[test]
    fn test_save_model_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_save.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut metadata = HashMap::new();
        metadata.insert("accuracy".to_string(), "0.95".to_string());

        let version = registry
            .save_model("test_model", &model_data, metadata)
            .unwrap();

        assert_eq!(version.name, "test_model");
        assert_eq!(version.version, 1);
        assert_eq!(version.size_bytes, 10);
        assert_eq!(version.stage, ModelStage::Development);
        assert!(!version.checksum.is_empty());
        assert_eq!(version.metadata.get("accuracy"), Some(&"0.95".to_string()));
    }

    #[test]
    fn test_save_model_auto_versioning() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_versioning.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![1u8, 2, 3];
        let metadata = HashMap::new();

        // Save first version
        let v1 = registry
            .save_model("model", &model_data, metadata.clone())
            .unwrap();
        assert_eq!(v1.version, 1);

        // Save second version
        let v2 = registry
            .save_model("model", &model_data, metadata.clone())
            .unwrap();
        assert_eq!(v2.version, 2);

        // Save third version
        let v3 = registry.save_model("model", &model_data, metadata).unwrap();
        assert_eq!(v3.version, 3);
    }

    #[test]
    fn test_save_model_checksum_consistency() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_checksum.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let metadata = HashMap::new();

        // Save the same data twice
        let v1 = registry
            .save_model("model", &model_data, metadata.clone())
            .unwrap();
        let v2 = registry.save_model("model", &model_data, metadata).unwrap();

        // Checksums should be identical for identical data
        assert_eq!(v1.checksum, v2.checksum);
    }

    #[test]
    fn test_save_model_different_data_different_checksum() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_diff_checksum.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data1 = vec![0u8, 1, 2, 3];
        let model_data2 = vec![4u8, 5, 6, 7];
        let metadata = HashMap::new();

        let v1 = registry
            .save_model("model", &model_data1, metadata.clone())
            .unwrap();
        let v2 = registry
            .save_model("model", &model_data2, metadata)
            .unwrap();

        // Checksums should be different for different data
        assert_ne!(v1.checksum, v2.checksum);
    }

    #[test]
    fn test_save_model_multiple_models() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_multi_model.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        let metadata = HashMap::new();

        // Save different models
        let v1 = registry
            .save_model("model_a", &data, metadata.clone())
            .unwrap();
        let v2 = registry
            .save_model("model_b", &data, metadata.clone())
            .unwrap();
        let v3 = registry.save_model("model_a", &data, metadata).unwrap();

        // Each model should have independent versioning
        assert_eq!(v1.version, 1);
        assert_eq!(v2.version, 1);
        assert_eq!(v3.version, 2);
    }

    #[test]
    fn test_save_model_empty_data() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_empty.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data: Vec<u8> = vec![];
        let metadata = HashMap::new();

        let version = registry
            .save_model("empty_model", &model_data, metadata)
            .unwrap();

        assert_eq!(version.size_bytes, 0);
        assert!(!version.checksum.is_empty()); // Even empty data has a checksum
    }

    #[test]
    fn test_load_model_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_load.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut metadata = HashMap::new();
        metadata.insert("accuracy".to_string(), "0.95".to_string());

        let saved_version = registry
            .save_model("test_model", &model_data, metadata)
            .unwrap();

        // Load the model
        let (loaded_data, loaded_version) = registry.load_model("test_model", None).unwrap();

        assert_eq!(loaded_data, model_data);
        assert_eq!(loaded_version.name, saved_version.name);
        assert_eq!(loaded_version.version, saved_version.version);
        assert_eq!(loaded_version.checksum, saved_version.checksum);
        assert_eq!(loaded_version.size_bytes, saved_version.size_bytes);
    }

    #[test]
    fn test_load_model_specific_version() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_load_version.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data_v1 = vec![1u8, 2, 3];
        let data_v2 = vec![4u8, 5, 6, 7];
        let metadata = HashMap::new();

        registry
            .save_model("model", &data_v1, metadata.clone())
            .unwrap();
        registry.save_model("model", &data_v2, metadata).unwrap();

        // Load specific version 1
        let (loaded_data, loaded_version) = registry.load_model("model", Some(1)).unwrap();
        assert_eq!(loaded_data, data_v1);
        assert_eq!(loaded_version.version, 1);

        // Load specific version 2
        let (loaded_data, loaded_version) = registry.load_model("model", Some(2)).unwrap();
        assert_eq!(loaded_data, data_v2);
        assert_eq!(loaded_version.version, 2);
    }

    #[test]
    fn test_load_model_latest_version() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_load_latest.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data_v1 = vec![1u8, 2, 3];
        let data_v2 = vec![4u8, 5, 6, 7];
        let data_v3 = vec![8u8, 9, 10, 11, 12];
        let metadata = HashMap::new();

        registry
            .save_model("model", &data_v1, metadata.clone())
            .unwrap();
        registry
            .save_model("model", &data_v2, metadata.clone())
            .unwrap();
        registry.save_model("model", &data_v3, metadata).unwrap();

        // Load latest (should be v3)
        let (loaded_data, loaded_version) = registry.load_model("model", None).unwrap();
        assert_eq!(loaded_data, data_v3);
        assert_eq!(loaded_version.version, 3);
    }

    #[test]
    fn test_load_model_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_not_found.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        // Try to load a model that doesn't exist
        let result = registry.load_model("nonexistent", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_version_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_version_not_found.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![1u8, 2, 3];
        let metadata = HashMap::new();

        registry.save_model("model", &model_data, metadata).unwrap();

        // Try to load a version that doesn't exist
        let result = registry.load_model("model", Some(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_checksum_verification() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_checksum_verify.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let metadata = HashMap::new();

        let saved_version = registry
            .save_model("test_model", &model_data, metadata)
            .unwrap();

        // Load and verify checksum matches
        let (loaded_data, loaded_version) = registry.load_model("test_model", None).unwrap();

        // Compute checksum of loaded data
        let mut hasher = Sha256::new();
        hasher.update(&loaded_data);
        let computed_checksum = format!("{:x}", hasher.finalize());

        assert_eq!(computed_checksum, saved_version.checksum);
        assert_eq!(computed_checksum, loaded_version.checksum);
    }

    #[test]
    fn test_load_model_empty_data() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_load_empty.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data: Vec<u8> = vec![];
        let metadata = HashMap::new();

        registry
            .save_model("empty_model", &model_data, metadata)
            .unwrap();

        // Load empty model
        let (loaded_data, loaded_version) = registry.load_model("empty_model", None).unwrap();
        assert!(loaded_data.is_empty());
        assert_eq!(loaded_version.size_bytes, 0);
    }

    #[test]
    fn test_load_model_preserves_metadata() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_load_metadata.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let model_data = vec![1u8, 2, 3];
        let mut metadata = HashMap::new();
        metadata.insert("accuracy".to_string(), "0.95".to_string());
        metadata.insert("framework".to_string(), "pytorch".to_string());
        metadata.insert("description".to_string(), "Test model".to_string());

        registry
            .save_model("model", &model_data, metadata.clone())
            .unwrap();

        // Load and verify metadata is preserved
        let (_, loaded_version) = registry.load_model("model", None).unwrap();
        assert_eq!(
            loaded_version.metadata.get("accuracy"),
            Some(&"0.95".to_string())
        );
        assert_eq!(
            loaded_version.metadata.get("framework"),
            Some(&"pytorch".to_string())
        );
        assert_eq!(
            loaded_version.metadata.get("description"),
            Some(&"Test model".to_string())
        );
    }

    // =========================================================================
    // Tests for list_versions
    // =========================================================================

    #[test]
    fn test_list_versions_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_empty.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        // List versions for a model that doesn't exist
        let versions = registry.list_versions("nonexistent").unwrap();
        assert!(versions.is_empty());
    }

    #[test]
    fn test_list_versions_single() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_single.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();

        let versions = registry.list_versions("model").unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[0].name, "model");
    }

    #[test]
    fn test_list_versions_multiple() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_multiple.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();

        let versions = registry.list_versions("model").unwrap();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);
        assert_eq!(versions[2].version, 3);
    }

    #[test]
    fn test_list_versions_sorted() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_sorted.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        // Save multiple versions
        for _ in 0..5 {
            registry.save_model("model", &data, HashMap::new()).unwrap();
        }

        let versions = registry.list_versions("model").unwrap();

        // Verify sorted by version number
        for i in 1..versions.len() {
            assert!(versions[i - 1].version < versions[i].version);
        }
    }

    #[test]
    fn test_list_versions_multiple_models() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_list_multi_model.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry
            .save_model("model_a", &data, HashMap::new())
            .unwrap();
        registry
            .save_model("model_a", &data, HashMap::new())
            .unwrap();
        registry
            .save_model("model_b", &data, HashMap::new())
            .unwrap();

        // List versions for model_a
        let versions_a = registry.list_versions("model_a").unwrap();
        assert_eq!(versions_a.len(), 2);
        assert!(versions_a.iter().all(|v| v.name == "model_a"));

        // List versions for model_b
        let versions_b = registry.list_versions("model_b").unwrap();
        assert_eq!(versions_b.len(), 1);
        assert!(versions_b.iter().all(|v| v.name == "model_b"));
    }

    // =========================================================================
    // Tests for set_stage
    // =========================================================================

    #[test]
    fn test_set_stage_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        let version = registry.save_model("model", &data, HashMap::new()).unwrap();
        assert_eq!(version.stage, ModelStage::Development);

        // Set to staging
        registry.set_stage("model", 1, ModelStage::Staging).unwrap();

        // Verify stage was updated
        let (_, loaded) = registry.load_model("model", Some(1)).unwrap();
        assert_eq!(loaded.stage, ModelStage::Staging);
    }

    #[test]
    fn test_set_stage_to_production() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_prod.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Promote to production
        registry
            .set_stage("model", 1, ModelStage::Production)
            .unwrap();

        let (_, loaded) = registry.load_model("model", Some(1)).unwrap();
        assert_eq!(loaded.stage, ModelStage::Production);
    }

    #[test]
    fn test_set_stage_to_archived() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_archived.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Archive the model
        registry
            .set_stage("model", 1, ModelStage::Archived)
            .unwrap();

        let (_, loaded) = registry.load_model("model", Some(1)).unwrap();
        assert_eq!(loaded.stage, ModelStage::Archived);
    }

    #[test]
    fn test_set_stage_model_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_not_found.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        // Try to set stage for a model that doesn't exist
        let result = registry.set_stage("nonexistent", 1, ModelStage::Production);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_stage_version_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_version_not_found.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Try to set stage for a version that doesn't exist
        let result = registry.set_stage("model", 999, ModelStage::Production);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_stage_preserves_other_metadata() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_preserves.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        let mut metadata = HashMap::new();
        metadata.insert("accuracy".to_string(), "0.95".to_string());
        metadata.insert("framework".to_string(), "pytorch".to_string());

        let original = registry.save_model("model", &data, metadata).unwrap();

        // Change stage
        registry
            .set_stage("model", 1, ModelStage::Production)
            .unwrap();

        // Verify other metadata is preserved
        let (_, loaded) = registry.load_model("model", Some(1)).unwrap();
        assert_eq!(loaded.stage, ModelStage::Production);
        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.version, original.version);
        assert_eq!(loaded.checksum, original.checksum);
        assert_eq!(loaded.size_bytes, original.size_bytes);
        assert_eq!(loaded.metadata.get("accuracy"), Some(&"0.95".to_string()));
        assert_eq!(
            loaded.metadata.get("framework"),
            Some(&"pytorch".to_string())
        );
    }

    #[test]
    fn test_set_stage_multiple_versions() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_set_stage_multi.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Set different stages for different versions
        registry
            .set_stage("model", 1, ModelStage::Archived)
            .unwrap();
        registry.set_stage("model", 2, ModelStage::Staging).unwrap();
        registry
            .set_stage("model", 3, ModelStage::Production)
            .unwrap();

        // Verify each version has correct stage
        let (_, v1) = registry.load_model("model", Some(1)).unwrap();
        let (_, v2) = registry.load_model("model", Some(2)).unwrap();
        let (_, v3) = registry.load_model("model", Some(3)).unwrap();

        assert_eq!(v1.stage, ModelStage::Archived);
        assert_eq!(v2.stage, ModelStage::Staging);
        assert_eq!(v3.stage, ModelStage::Production);
    }

    // =========================================================================
    // Tests for get_production
    // =========================================================================

    #[test]
    fn test_get_production_none() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_prod_none.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // No production version yet
        let prod = registry.get_production("model").unwrap();
        assert!(prod.is_none());
    }

    #[test]
    fn test_get_production_exists() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_prod_exists.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry
            .set_stage("model", 1, ModelStage::Production)
            .unwrap();

        let prod = registry.get_production("model").unwrap();
        assert!(prod.is_some());
        assert_eq!(prod.unwrap().version, 1);
    }

    #[test]
    fn test_get_production_multiple_versions() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_prod_multi.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Set version 2 as production
        registry
            .set_stage("model", 2, ModelStage::Production)
            .unwrap();

        let prod = registry.get_production("model").unwrap();
        assert!(prod.is_some());
        assert_eq!(prod.unwrap().version, 2);
    }

    #[test]
    fn test_get_production_nonexistent_model() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_prod_nonexistent.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        // Get production for a model that doesn't exist
        let prod = registry.get_production("nonexistent").unwrap();
        assert!(prod.is_none());
    }

    #[test]
    fn test_get_production_after_stage_change() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_get_prod_change.db");

        let mut registry = ModelRegistry::new(&db_path).unwrap();

        let data = vec![1u8, 2, 3];
        registry.save_model("model", &data, HashMap::new()).unwrap();
        registry.save_model("model", &data, HashMap::new()).unwrap();

        // Set version 1 as production
        registry
            .set_stage("model", 1, ModelStage::Production)
            .unwrap();
        let prod = registry.get_production("model").unwrap();
        assert_eq!(prod.unwrap().version, 1);

        // Archive version 1 and promote version 2
        registry
            .set_stage("model", 1, ModelStage::Archived)
            .unwrap();
        registry
            .set_stage("model", 2, ModelStage::Production)
            .unwrap();

        let prod = registry.get_production("model").unwrap();
        assert_eq!(prod.unwrap().version, 2);
    }
}

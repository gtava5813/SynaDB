//! Tensor engine for batch operations on numerical data.
//!
//! This module provides efficient batch read/write operations for numerical data,
//! enabling direct tensor extraction for ML/AI workloads.
//!
//! # Features
//!
//! - Load multiple keys as a contiguous tensor
//! - Store tensors with auto-generated keys
//! - Support for multiple data types (f32, f64, i32, i64)
//! - Pattern-based key matching for batch operations
//! - **Chunked blob storage** for high-throughput (1 GB/s target)
//!
//! # Storage Modes
//!
//! ## Per-Element Storage (Original)
//! - Each tensor element stored as separate key-value pair
//! - Good for small tensors or when individual element access is needed
//! - O(n) disk operations
//!
//! ## Chunked Blob Storage (Optimized)
//! - Tensor stored as fixed-size chunks (default 1MB each)
//! - Metadata stored separately with shape, dtype, chunk count
//! - Enables 1 GB/s+ throughput for large tensors
//! - Use `put_tensor_chunked()` and `get_tensor_chunked()`
//!
//! # Examples
//!
//! ```rust,no_run
//! use synadb::{SynaDB, Atom};
//! use synadb::tensor::{TensorEngine, DType};
//!
//! // Create database and populate with data
//! let mut db = SynaDB::new("data.db").unwrap();
//! for i in 0..100 {
//!     db.append(&format!("sensor/{:04}", i), Atom::Float(i as f64 * 0.1)).unwrap();
//! }
//!
//! // Create tensor engine
//! let mut engine = TensorEngine::new(db);
//!
//! // Load all sensor data as a tensor
//! let (data, shape) = engine.get_tensor("sensor/*", DType::Float64).unwrap();
//! assert_eq!(shape[0], 100);
//! ```
//!
//! # High-Throughput Example (Chunked)
//!
//! ```rust,no_run
//! use synadb::SynaDB;
//! use synadb::tensor::{TensorEngine, DType};
//!
//! let db = SynaDB::new("tensors.db").unwrap();
//! let mut engine = TensorEngine::new(db);
//!
//! // Store 10MB tensor as chunks (fast!)
//! let data: Vec<u8> = vec![0u8; 10 * 1024 * 1024];
//! let shape = vec![10 * 1024 * 1024 / 8]; // f64 elements
//! engine.put_tensor_chunked("model/weights", &data, &shape, DType::Float64).unwrap();
//!
//! // Load tensor back (fast!)
//! let (loaded, loaded_shape) = engine.get_tensor_chunked("model/weights").unwrap();
//! assert_eq!(loaded.len(), data.len());
//! ```
//!
//! _Requirements: 2.1, 2.2, 2.3, 2.6, 9.3_

use crate::engine::SynaDB;
use crate::error::{Result, SynaError};
use crate::types::Atom;
use serde::{Deserialize, Serialize};

/// Default chunk size for chunked tensor storage (1 MB).
/// This balances I/O efficiency with memory usage.
pub const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Metadata for a chunked tensor stored in the database.
///
/// This is stored as JSON in the metadata key for each tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMeta {
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Data type of tensor elements
    pub dtype: String,
    /// Total size in bytes
    pub total_bytes: usize,
    /// Number of chunks
    pub chunk_count: usize,
    /// Size of each chunk in bytes (last chunk may be smaller)
    pub chunk_size: usize,
}

/// Data type for tensor operations.
///
/// Specifies the numeric type for tensor data conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
}

impl DType {
    /// Returns the size in bytes for this data type.
    pub fn size(&self) -> usize {
        match self {
            DType::Float32 | DType::Int32 => 4,
            DType::Float64 | DType::Int64 => 8,
        }
    }

    /// Returns the type name as a string.
    pub fn name(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
        }
    }
}

/// Tensor engine for batch read/write operations.
///
/// Provides efficient methods for loading and storing numerical data
/// as contiguous tensors, optimized for ML/AI workloads.
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::SynaDB;
/// use synadb::tensor::{TensorEngine, DType};
///
/// let db = SynaDB::new("data.db").unwrap();
/// let mut engine = TensorEngine::new(db);
///
/// // Load data matching a pattern
/// let (data, shape) = engine.get_tensor("prefix/*", DType::Float64).unwrap();
/// ```
pub struct TensorEngine {
    db: SynaDB,
}

impl TensorEngine {
    /// Creates a new TensorEngine wrapping the given database.
    ///
    /// # Arguments
    ///
    /// * `db` - The SynaDB instance to use for storage
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::TensorEngine;
    ///
    /// let db = SynaDB::new("data.db").unwrap();
    /// let engine = TensorEngine::new(db);
    /// ```
    pub fn new(db: SynaDB) -> Self {
        Self { db }
    }

    /// Returns a reference to the underlying database.
    pub fn db(&self) -> &SynaDB {
        &self.db
    }

    /// Returns a mutable reference to the underlying database.
    pub fn db_mut(&mut self) -> &mut SynaDB {
        &mut self.db
    }

    /// Consumes the TensorEngine and returns the underlying database.
    pub fn into_db(self) -> SynaDB {
        self.db
    }

    /// Load all values matching pattern as a contiguous tensor.
    ///
    /// Keys are matched using glob-style patterns:
    /// - `prefix/*` matches all keys starting with `prefix/`
    /// - `prefix*` matches all keys starting with `prefix`
    /// - `exact_key` matches only that exact key
    ///
    /// # Arguments
    ///
    /// * `pattern` - Glob-style pattern to match keys
    /// * `dtype` - Target data type for the tensor
    ///
    /// # Returns
    ///
    /// A tuple of (data, shape) where:
    /// - `data` is a byte vector containing the tensor data
    /// - `shape` is a vector of dimensions (currently 1D: \[n_elements\])
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::{SynaDB, Atom};
    /// use synadb::tensor::{TensorEngine, DType};
    ///
    /// let mut db = SynaDB::new("data.db").unwrap();
    /// db.append("data/0", Atom::Float(1.0)).unwrap();
    /// db.append("data/1", Atom::Float(2.0)).unwrap();
    ///
    /// let mut engine = TensorEngine::new(db);
    /// let (data, shape) = engine.get_tensor("data/*", DType::Float64).unwrap();
    /// assert_eq!(shape, vec![2]);
    /// ```
    ///
    /// _Requirements: 2.1, 2.2, 2.3_
    pub fn get_tensor(&mut self, pattern: &str, dtype: DType) -> Result<(Vec<u8>, Vec<usize>)> {
        // 1. Find all keys matching pattern
        let keys = self.match_keys(pattern);

        // 2. Load values and convert to requested dtype
        let mut data = Vec::new();
        let mut count = 0;

        for key in &keys {
            if let Some(atom) = self.db.get(key)? {
                if self.append_atom_as_dtype(&mut data, &atom, dtype)? {
                    count += 1;
                }
            }
        }

        // 3. Return data and shape
        let shape = vec![count];
        Ok((data, shape))
    }

    /// Store tensor with auto-generated keys.
    ///
    /// Each element in the tensor is stored with a key of the form
    /// `{key_prefix}{index:08}` where index is zero-padded to 8 digits.
    ///
    /// # Arguments
    ///
    /// * `key_prefix` - Prefix for generated keys
    /// * `data` - Raw byte data of the tensor
    /// * `shape` - Shape of the tensor (dimensions)
    /// * `dtype` - Data type of the tensor elements
    ///
    /// # Returns
    ///
    /// The number of elements stored.
    ///
    /// # Errors
    ///
    /// Returns `SynaError::ShapeMismatch` if the data size doesn't match
    /// the expected size based on shape and dtype.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::{TensorEngine, DType};
    ///
    /// let db = SynaDB::new("data.db").unwrap();
    /// let mut engine = TensorEngine::new(db);
    ///
    /// // Store 4 float64 values
    /// let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
    ///     .iter()
    ///     .flat_map(|f| f.to_le_bytes())
    ///     .collect();
    /// let count = engine.put_tensor("values/", &data, &[4], DType::Float64).unwrap();
    /// assert_eq!(count, 4);
    /// ```
    ///
    /// _Requirements: 2.6_
    pub fn put_tensor(
        &mut self,
        key_prefix: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        let element_size = dtype.size();
        let n_elements = shape.iter().product::<usize>();
        let expected_bytes = n_elements * element_size;

        if data.len() != expected_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: expected_bytes,
            });
        }

        // Store each element with sequential key
        let mut count = 0;
        for i in 0..n_elements {
            let key = format!("{}{:08}", key_prefix, i);
            let start = i * element_size;
            let end = start + element_size;
            let atom = self.bytes_to_atom(&data[start..end], dtype)?;
            self.db.append(&key, atom)?;
            count += 1;
        }

        Ok(count)
    }

    /// Match keys against a glob-style pattern.
    ///
    /// Supports:
    /// - `prefix/*` - matches keys starting with `prefix/`
    /// - `prefix*` - matches keys starting with `prefix`
    /// - `exact` - matches only the exact key
    fn match_keys(&self, pattern: &str) -> Vec<String> {
        let keys = self.db.keys();

        if let Some(prefix_without_star) = pattern.strip_suffix("/*") {
            // Match keys starting with prefix/
            let prefix = format!("{}/", prefix_without_star);
            let mut matched: Vec<String> = keys
                .into_iter()
                .filter(|k| k.starts_with(&prefix))
                .collect();
            matched.sort(); // Sort for consistent ordering
            matched
        } else if let Some(prefix) = pattern.strip_suffix('*') {
            // Match keys starting with prefix
            let mut matched: Vec<String> =
                keys.into_iter().filter(|k| k.starts_with(prefix)).collect();
            matched.sort();
            matched
        } else {
            // Exact match
            keys.into_iter().filter(|k| k == pattern).collect()
        }
    }

    /// Append an Atom value to the data buffer, converting to the target dtype.
    ///
    /// Returns true if the value was successfully converted and appended,
    /// false if the value type couldn't be converted (e.g., Text to Float).
    fn append_atom_as_dtype(&self, data: &mut Vec<u8>, atom: &Atom, dtype: DType) -> Result<bool> {
        match (atom, dtype) {
            // Float to Float64 (native)
            (Atom::Float(f), DType::Float64) => {
                data.extend_from_slice(&f.to_le_bytes());
                Ok(true)
            }
            // Float to Float32 (downcast)
            (Atom::Float(f), DType::Float32) => {
                data.extend_from_slice(&(*f as f32).to_le_bytes());
                Ok(true)
            }
            // Float to Int64 (truncate)
            (Atom::Float(f), DType::Int64) => {
                data.extend_from_slice(&(*f as i64).to_le_bytes());
                Ok(true)
            }
            // Float to Int32 (truncate and downcast)
            (Atom::Float(f), DType::Int32) => {
                data.extend_from_slice(&(*f as i32).to_le_bytes());
                Ok(true)
            }
            // Int to Int64 (native)
            (Atom::Int(i), DType::Int64) => {
                data.extend_from_slice(&i.to_le_bytes());
                Ok(true)
            }
            // Int to Int32 (downcast)
            (Atom::Int(i), DType::Int32) => {
                data.extend_from_slice(&(*i as i32).to_le_bytes());
                Ok(true)
            }
            // Int to Float64 (upcast)
            (Atom::Int(i), DType::Float64) => {
                data.extend_from_slice(&(*i as f64).to_le_bytes());
                Ok(true)
            }
            // Int to Float32 (convert)
            (Atom::Int(i), DType::Float32) => {
                data.extend_from_slice(&(*i as f32).to_le_bytes());
                Ok(true)
            }
            // Vector elements - extract first element for scalar conversion
            (Atom::Vector(vec, _), DType::Float32) if !vec.is_empty() => {
                data.extend_from_slice(&vec[0].to_le_bytes());
                Ok(true)
            }
            (Atom::Vector(vec, _), DType::Float64) if !vec.is_empty() => {
                data.extend_from_slice(&(vec[0] as f64).to_le_bytes());
                Ok(true)
            }
            // Non-numeric types are skipped
            (Atom::Null, _) | (Atom::Text(_), _) | (Atom::Bytes(_), _) => Ok(false),
            // Empty vectors are skipped
            (Atom::Vector(vec, _), _) if vec.is_empty() => Ok(false),
            // Remaining vector conversions
            (Atom::Vector(vec, _), DType::Int32) if !vec.is_empty() => {
                data.extend_from_slice(&(vec[0] as i32).to_le_bytes());
                Ok(true)
            }
            (Atom::Vector(vec, _), DType::Int64) if !vec.is_empty() => {
                data.extend_from_slice(&(vec[0] as i64).to_le_bytes());
                Ok(true)
            }
            // Catch-all for any remaining patterns
            _ => Ok(false),
        }
    }

    /// Convert raw bytes to an Atom based on the dtype.
    fn bytes_to_atom(&self, bytes: &[u8], dtype: DType) -> Result<Atom> {
        match dtype {
            DType::Float32 => {
                let arr: [u8; 4] = bytes.try_into().map_err(|_| SynaError::ShapeMismatch {
                    data_size: bytes.len(),
                    expected_size: 4,
                })?;
                Ok(Atom::Float(f32::from_le_bytes(arr) as f64))
            }
            DType::Float64 => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SynaError::ShapeMismatch {
                    data_size: bytes.len(),
                    expected_size: 8,
                })?;
                Ok(Atom::Float(f64::from_le_bytes(arr)))
            }
            DType::Int32 => {
                let arr: [u8; 4] = bytes.try_into().map_err(|_| SynaError::ShapeMismatch {
                    data_size: bytes.len(),
                    expected_size: 4,
                })?;
                Ok(Atom::Int(i32::from_le_bytes(arr) as i64))
            }
            DType::Int64 => {
                let arr: [u8; 8] = bytes.try_into().map_err(|_| SynaError::ShapeMismatch {
                    data_size: bytes.len(),
                    expected_size: 8,
                })?;
                Ok(Atom::Int(i64::from_le_bytes(arr)))
            }
        }
    }

    // =========================================================================
    // Chunked Blob Storage (High-Throughput)
    // =========================================================================

    /// Store tensor as chunked blobs for high throughput.
    ///
    /// This method stores the tensor data as multiple fixed-size chunks,
    /// enabling 1 GB/s+ throughput for large tensors. Each chunk is stored
    /// as a separate `Atom::Bytes` entry, with metadata stored separately.
    ///
    /// # Storage Layout
    ///
    /// ```text
    /// {name}/meta     -> TensorMeta (JSON) with shape, dtype, chunk info
    /// {name}/chunk/0  -> Atom::Bytes (first chunk, up to chunk_size bytes)
    /// {name}/chunk/1  -> Atom::Bytes (second chunk)
    /// ...
    /// ```
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the tensor (e.g., "model/weights")
    /// * `data` - Raw byte data of the tensor
    /// * `shape` - Shape of the tensor (dimensions)
    /// * `dtype` - Data type of the tensor elements
    ///
    /// # Returns
    ///
    /// The number of chunks written.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::{TensorEngine, DType};
    ///
    /// let db = SynaDB::new("data.db").unwrap();
    /// let mut engine = TensorEngine::new(db);
    ///
    /// // Store 4 float64 values as chunked tensor
    /// let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
    ///     .iter()
    ///     .flat_map(|f| f.to_le_bytes())
    ///     .collect();
    /// let chunks = engine.put_tensor_chunked("values", &data, &[4], DType::Float64).unwrap();
    /// assert_eq!(chunks, 1); // Small data fits in one chunk
    /// ```
    ///
    /// _Requirements: 2.6, 9.3_
    pub fn put_tensor_chunked(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        self.put_tensor_chunked_with_size(name, data, shape, dtype, DEFAULT_CHUNK_SIZE)
    }

    /// Store tensor as chunked blobs with custom chunk size.
    ///
    /// Same as [`put_tensor_chunked`](Self::put_tensor_chunked) but allows
    /// specifying a custom chunk size for tuning performance.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the tensor
    /// * `data` - Raw byte data of the tensor
    /// * `shape` - Shape of the tensor (dimensions)
    /// * `dtype` - Data type of the tensor elements
    /// * `chunk_size` - Size of each chunk in bytes
    ///
    /// # Returns
    ///
    /// The number of chunks written.
    pub fn put_tensor_chunked_with_size(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
        chunk_size: usize,
    ) -> Result<usize> {
        let element_size = dtype.size();
        let n_elements = shape.iter().product::<usize>();
        let expected_bytes = n_elements * element_size;

        if data.len() != expected_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: expected_bytes,
            });
        }

        // Calculate number of chunks
        let chunk_count = data.len().div_ceil(chunk_size);

        // Create and store metadata
        let meta = TensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.name().to_string(),
            total_bytes: data.len(),
            chunk_count,
            chunk_size,
        };

        let meta_json = serde_json::to_string(&meta)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to serialize metadata: {}", e)))?;
        let meta_key = format!("{}/meta", name);
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        // Store data chunks
        for (i, chunk_data) in data.chunks(chunk_size).enumerate() {
            let chunk_key = format!("{}/chunk/{}", name, i);
            self.db
                .append(&chunk_key, Atom::Bytes(chunk_data.to_vec()))?;
        }

        Ok(chunk_count)
    }

    /// Load tensor from chunked blob storage.
    ///
    /// This method loads a tensor that was stored using [`put_tensor_chunked`](Self::put_tensor_chunked).
    /// It reads the metadata first, then loads all chunks and concatenates them.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor (e.g., "model/weights")
    ///
    /// # Returns
    ///
    /// A tuple of (data, shape) where:
    /// - `data` is a byte vector containing the tensor data
    /// - `shape` is a vector of dimensions
    ///
    /// # Errors
    ///
    /// Returns `SynaError::KeyNotFound` if the tensor metadata doesn't exist.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::{TensorEngine, DType};
    ///
    /// let db = SynaDB::new("data.db").unwrap();
    /// let mut engine = TensorEngine::new(db);
    ///
    /// // Store tensor
    /// let original: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
    ///     .iter()
    ///     .flat_map(|f| f.to_le_bytes())
    ///     .collect();
    /// engine.put_tensor_chunked("values", &original, &[4], DType::Float64).unwrap();
    ///
    /// // Load tensor
    /// let (loaded, shape) = engine.get_tensor_chunked("values").unwrap();
    /// assert_eq!(loaded, original);
    /// assert_eq!(shape, vec![4]);
    /// ```
    ///
    /// _Requirements: 2.1, 9.3_
    pub fn get_tensor_chunked(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
        // Load metadata
        let meta_key = format!("{}/meta", name);
        let meta_atom = self
            .db
            .get(&meta_key)?
            .ok_or_else(|| SynaError::KeyNotFound(meta_key.clone()))?;

        let meta_json = match meta_atom {
            Atom::Text(s) => s,
            _ => {
                return Err(SynaError::TypeConversion {
                    from_type: meta_atom.type_name(),
                    to_type: "Text",
                })
            }
        };

        let meta: TensorMeta = serde_json::from_str(&meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        // Pre-allocate buffer for all data
        let mut data = Vec::with_capacity(meta.total_bytes);

        // Load all chunks
        for i in 0..meta.chunk_count {
            let chunk_key = format!("{}/chunk/{}", name, i);
            let chunk_atom = self
                .db
                .get(&chunk_key)?
                .ok_or_else(|| SynaError::KeyNotFound(chunk_key.clone()))?;

            match chunk_atom {
                Atom::Bytes(bytes) => data.extend_from_slice(&bytes),
                _ => {
                    return Err(SynaError::TypeConversion {
                        from_type: chunk_atom.type_name(),
                        to_type: "Bytes",
                    })
                }
            }
        }

        // Verify total size
        if data.len() != meta.total_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: meta.total_bytes,
            });
        }

        Ok((data, meta.shape))
    }

    /// Get metadata for a chunked tensor without loading the data.
    ///
    /// This is useful for inspecting tensor properties without the overhead
    /// of loading the full tensor data.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor
    ///
    /// # Returns
    ///
    /// The tensor metadata including shape, dtype, and chunk information.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::{TensorEngine, DType};
    ///
    /// let db = SynaDB::new("data.db").unwrap();
    /// let mut engine = TensorEngine::new(db);
    ///
    /// // Store tensor
    /// let data: Vec<u8> = vec![0u8; 1024];
    /// engine.put_tensor_chunked("tensor", &data, &[128], DType::Float64).unwrap();
    ///
    /// // Get metadata only
    /// let meta = engine.get_tensor_meta("tensor").unwrap();
    /// assert_eq!(meta.shape, vec![128]);
    /// assert_eq!(meta.dtype, "float64");
    /// ```
    pub fn get_tensor_meta(&mut self, name: &str) -> Result<TensorMeta> {
        let meta_key = format!("{}/meta", name);
        let meta_atom = self
            .db
            .get(&meta_key)?
            .ok_or_else(|| SynaError::KeyNotFound(meta_key.clone()))?;

        let meta_json = match meta_atom {
            Atom::Text(s) => s,
            _ => {
                return Err(SynaError::TypeConversion {
                    from_type: meta_atom.type_name(),
                    to_type: "Text",
                })
            }
        };

        let meta: TensorMeta = serde_json::from_str(&meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        Ok(meta)
    }

    /// Delete a chunked tensor and all its chunks.
    ///
    /// This removes the metadata and all chunk entries for the tensor.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor to delete
    ///
    /// # Returns
    ///
    /// The number of entries deleted (metadata + chunks).
    pub fn delete_tensor_chunked(&mut self, name: &str) -> Result<usize> {
        // Try to load metadata to get chunk count
        let meta_key = format!("{}/meta", name);
        let chunk_count = match self.db.get(&meta_key)? {
            Some(Atom::Text(json)) => {
                if let Ok(meta) = serde_json::from_str::<TensorMeta>(&json) {
                    meta.chunk_count
                } else {
                    0
                }
            }
            _ => 0,
        };

        let mut deleted = 0;

        // Delete metadata
        if self.db.delete(&meta_key).is_ok() {
            deleted += 1;
        }

        // Delete all chunks
        for i in 0..chunk_count {
            let chunk_key = format!("{}/chunk/{}", name, i);
            if self.db.delete(&chunk_key).is_ok() {
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::Float32.size(), 4);
        assert_eq!(DType::Float64.size(), 8);
        assert_eq!(DType::Int32.size(), 4);
        assert_eq!(DType::Int64.size(), 8);
    }

    #[test]
    fn test_get_tensor_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let (data, shape) = engine.get_tensor("nonexistent/*", DType::Float64).unwrap();
        assert!(data.is_empty());
        assert_eq!(shape, vec![0]);
    }

    #[test]
    fn test_get_tensor_floats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut db = SynaDB::new(&db_path).unwrap();

        // Insert some float values
        db.append("data/00", Atom::Float(1.0)).unwrap();
        db.append("data/01", Atom::Float(2.0)).unwrap();
        db.append("data/02", Atom::Float(3.0)).unwrap();

        let mut engine = TensorEngine::new(db);
        let (data, shape) = engine.get_tensor("data/*", DType::Float64).unwrap();

        assert_eq!(shape, vec![3]);
        assert_eq!(data.len(), 3 * 8); // 3 f64 values

        // Verify values
        let values: Vec<f64> = data
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_put_tensor() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Create tensor data
        let values = vec![1.0f64, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Store tensor
        let count = engine
            .put_tensor("tensor/", &data, &[4], DType::Float64)
            .unwrap();
        assert_eq!(count, 4);

        // Verify stored values
        let db = engine.db_mut();
        assert_eq!(db.get("tensor/00000000").unwrap(), Some(Atom::Float(1.0)));
        assert_eq!(db.get("tensor/00000001").unwrap(), Some(Atom::Float(2.0)));
        assert_eq!(db.get("tensor/00000002").unwrap(), Some(Atom::Float(3.0)));
        assert_eq!(db.get("tensor/00000003").unwrap(), Some(Atom::Float(4.0)));
    }

    #[test]
    fn test_put_tensor_shape_mismatch() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Data doesn't match shape
        let data = vec![0u8; 16]; // 16 bytes
        let result = engine.put_tensor("tensor/", &data, &[4], DType::Float64); // expects 32 bytes

        assert!(matches!(result, Err(SynaError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_roundtrip_float64() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Store tensor
        let original = vec![1.5f64, 2.5, 3.5];
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        engine
            .put_tensor("rt/", &data, &[3], DType::Float64)
            .unwrap();

        // Load tensor
        let (loaded_data, shape) = engine.get_tensor("rt/*", DType::Float64).unwrap();
        assert_eq!(shape, vec![3]);

        let loaded: Vec<f64> = loaded_data
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(loaded, original);
    }

    #[test]
    fn test_int_to_float_conversion() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut db = SynaDB::new(&db_path).unwrap();

        // Insert int values
        db.append("int/0", Atom::Int(10)).unwrap();
        db.append("int/1", Atom::Int(20)).unwrap();

        let mut engine = TensorEngine::new(db);
        let (data, shape) = engine.get_tensor("int/*", DType::Float64).unwrap();

        assert_eq!(shape, vec![2]);
        let values: Vec<f64> = data
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![10.0, 20.0]);
    }

    #[test]
    fn test_skip_non_numeric() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut db = SynaDB::new(&db_path).unwrap();

        // Insert mixed values
        db.append("mix/0", Atom::Float(1.0)).unwrap();
        db.append("mix/1", Atom::Text("skip me".to_string()))
            .unwrap();
        db.append("mix/2", Atom::Float(2.0)).unwrap();
        db.append("mix/3", Atom::Null).unwrap();
        db.append("mix/4", Atom::Float(3.0)).unwrap();

        let mut engine = TensorEngine::new(db);
        let (data, shape) = engine.get_tensor("mix/*", DType::Float64).unwrap();

        // Only numeric values should be included
        assert_eq!(shape, vec![3]);
        let values: Vec<f64> = data
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    // =========================================================================
    // Chunked Storage Tests
    // =========================================================================

    #[test]
    fn test_put_get_tensor_chunked_small() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Small tensor that fits in one chunk
        let original = vec![1.0f64, 2.0, 3.0, 4.0];
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        let chunks = engine
            .put_tensor_chunked("small", &data, &[4], DType::Float64)
            .unwrap();
        assert_eq!(chunks, 1); // Should fit in one chunk

        let (loaded, shape) = engine.get_tensor_chunked("small").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn test_put_get_tensor_chunked_large() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Large tensor that spans multiple chunks (use small chunk size for testing)
        let num_elements = 10_000;
        let original: Vec<f64> = (0..num_elements).map(|i| i as f64 * 0.1).collect();
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Use 1KB chunks for testing
        let chunk_size = 1024;
        let chunks = engine
            .put_tensor_chunked_with_size(
                "large",
                &data,
                &[num_elements],
                DType::Float64,
                chunk_size,
            )
            .unwrap();

        // Should have multiple chunks
        let expected_chunks = (data.len() + chunk_size - 1) / chunk_size;
        assert_eq!(chunks, expected_chunks);
        assert!(chunks > 1);

        let (loaded, shape) = engine.get_tensor_chunked("large").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![num_elements]);

        // Verify values
        let loaded_values: Vec<f64> = loaded
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(loaded_values, original);
    }

    #[test]
    fn test_get_tensor_meta() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        engine
            .put_tensor_chunked("tensor", &data, &[2, 3], DType::Float64)
            .unwrap();

        let meta = engine.get_tensor_meta("tensor").unwrap();
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.dtype, "float64");
        assert_eq!(meta.total_bytes, 48); // 6 * 8 bytes
        assert_eq!(meta.chunk_count, 1);
    }

    #[test]
    fn test_delete_tensor_chunked() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Store tensor with multiple chunks (5000 f64 elements = 40000 bytes)
        let num_elements = 5000;
        let data: Vec<u8> = (0..num_elements)
            .map(|i| i as f64)
            .flat_map(|f| f.to_le_bytes())
            .collect();
        engine
            .put_tensor_chunked_with_size("to_delete", &data, &[num_elements], DType::Float64, 1024)
            .unwrap();

        // Verify it exists
        assert!(engine.get_tensor_meta("to_delete").is_ok());

        // Delete it
        let deleted = engine.delete_tensor_chunked("to_delete").unwrap();
        assert!(deleted > 0);

        // Verify it's gone
        assert!(engine.get_tensor_meta("to_delete").is_err());
    }

    #[test]
    fn test_chunked_shape_mismatch() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Data doesn't match shape
        let data = vec![0u8; 16]; // 16 bytes
        let result = engine.put_tensor_chunked("bad", &data, &[4], DType::Float64); // expects 32 bytes

        assert!(matches!(result, Err(SynaError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_chunked_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let result = engine.get_tensor_chunked("nonexistent");
        assert!(matches!(result, Err(SynaError::KeyNotFound(_))));
    }

    #[test]
    fn test_chunked_roundtrip_all_dtypes() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Test Float32
        let f32_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        engine
            .put_tensor_chunked("f32", &f32_data, &[4], DType::Float32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_chunked("f32").unwrap();
        assert_eq!(loaded, f32_data);

        // Test Int32
        let i32_data: Vec<u8> = vec![1i32, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_chunked("i32", &i32_data, &[4], DType::Int32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_chunked("i32").unwrap();
        assert_eq!(loaded, i32_data);

        // Test Int64
        let i64_data: Vec<u8> = vec![1i64, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_chunked("i64", &i64_data, &[4], DType::Int64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_chunked("i64").unwrap();
        assert_eq!(loaded, i64_data);
    }
}

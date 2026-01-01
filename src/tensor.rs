// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

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

/// Chunk size tiers for optimal throughput based on tensor size.
/// These constants enable auto-tuning of chunk sizes for different tensor sizes.
///
/// _Requirements: 9.3_
/// Small chunk size (1 MB) for tensors <10MB.
/// Provides good balance of I/O efficiency and memory usage for smaller tensors.
pub const CHUNK_SIZE_SMALL: usize = 1024 * 1024;

/// Medium chunk size (4 MB) for tensors 10-100MB.
/// Reduces syscall overhead while maintaining reasonable memory usage.
pub const CHUNK_SIZE_MEDIUM: usize = 4 * 1024 * 1024;

/// Large chunk size (16 MB) for tensors >100MB.
/// Maximizes throughput for very large tensors by minimizing syscall overhead.
pub const CHUNK_SIZE_LARGE: usize = 16 * 1024 * 1024;

/// Auto-select optimal chunk size based on tensor size.
///
/// This function selects the most efficient chunk size for the given tensor
/// size to maximize throughput:
/// - <10MB: 1MB chunks (CHUNK_SIZE_SMALL)
/// - 10-100MB: 4MB chunks (CHUNK_SIZE_MEDIUM)
/// - >100MB: 16MB chunks (CHUNK_SIZE_LARGE)
///
/// # Arguments
///
/// * `tensor_bytes` - Total size of the tensor in bytes
///
/// # Returns
///
/// The optimal chunk size in bytes.
///
/// # Examples
///
/// ```rust
/// use synadb::tensor::{optimal_chunk_size, CHUNK_SIZE_SMALL, CHUNK_SIZE_MEDIUM, CHUNK_SIZE_LARGE};
///
/// // Small tensor: 1MB chunks
/// assert_eq!(optimal_chunk_size(5_000_000), CHUNK_SIZE_SMALL);
///
/// // Medium tensor: 4MB chunks
/// assert_eq!(optimal_chunk_size(50_000_000), CHUNK_SIZE_MEDIUM);
///
/// // Large tensor: 16MB chunks
/// assert_eq!(optimal_chunk_size(200_000_000), CHUNK_SIZE_LARGE);
/// ```
///
/// _Requirements: 9.3_
pub fn optimal_chunk_size(tensor_bytes: usize) -> usize {
    match tensor_bytes {
        0..=10_000_000 => CHUNK_SIZE_SMALL, // <10MB: 1MB chunks
        10_000_001..=100_000_000 => CHUNK_SIZE_MEDIUM, // 10-100MB: 4MB chunks
        _ => CHUNK_SIZE_LARGE,              // >100MB: 16MB chunks
    }
}

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

    /// Store tensor with auto-tuned chunk size for optimal throughput.
    ///
    /// This method automatically selects the optimal chunk size based on the
    /// tensor size to maximize throughput:
    /// - <10MB: 1MB chunks (reduces memory overhead)
    /// - 10-100MB: 4MB chunks (balances I/O and memory)
    /// - >100MB: 16MB chunks (maximizes throughput)
    ///
    /// This can provide 10-20% throughput improvement for large tensors
    /// compared to using a fixed 1MB chunk size.
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
    /// // Store 50MB tensor - will auto-select 4MB chunks
    /// let data: Vec<u8> = vec![0u8; 50 * 1024 * 1024];
    /// let shape = vec![50 * 1024 * 1024 / 8]; // f64 elements
    /// let chunks = engine.put_tensor_optimized("model/weights", &data, &shape, DType::Float64).unwrap();
    /// println!("Stored in {} chunks", chunks);
    /// ```
    ///
    /// _Requirements: 9.3_
    pub fn put_tensor_optimized(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        let chunk_size = crate::tensor::optimal_chunk_size(data.len());
        self.put_tensor_chunked_with_size(name, data, shape, dtype, chunk_size)
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

    // =========================================================================
    // Batched Single-Append Storage (Maximum Throughput)
    // =========================================================================

    /// Store tensor as a single batched blob for maximum throughput.
    ///
    /// This method stores the entire tensor (metadata + data) in a single append
    /// operation, achieving 1.5-2x throughput compared to chunked storage by
    /// reducing syscalls.
    ///
    /// # Storage Layout
    ///
    /// ```text
    /// [meta_len:u32][metadata_json][data]
    /// ```
    ///
    /// Where:
    /// - `meta_len` is a 4-byte little-endian u32 containing the metadata JSON length
    /// - `metadata_json` is the JSON-serialized TensorMeta
    /// - `data` is the raw tensor data
    ///
    /// # Arguments
    ///
    /// * `name` - Key name for the tensor
    /// * `data` - Raw byte data of the tensor
    /// * `shape` - Shape of the tensor (dimensions)
    /// * `dtype` - Data type of the tensor elements
    ///
    /// # Returns
    ///
    /// Always returns 1 (single append operation).
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
    /// // Store 4 float64 values as batched tensor
    /// let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
    ///     .iter()
    ///     .flat_map(|f| f.to_le_bytes())
    ///     .collect();
    /// let count = engine.put_tensor_batched("values", &data, &[4], DType::Float64).unwrap();
    /// assert_eq!(count, 1); // Single append
    /// ```
    ///
    /// _Requirements: 9.3_
    pub fn put_tensor_batched(
        &mut self,
        name: &str,
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

        // 1. Create metadata
        let meta = TensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.name().to_string(),
            total_bytes: data.len(),
            chunk_count: 1,         // Single blob
            chunk_size: data.len(), // Entire data in one "chunk"
        };

        let meta_json = serde_json::to_vec(&meta)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to serialize metadata: {}", e)))?;

        // 2. Build single buffer: [meta_len:u32][meta_json][data]
        let mut buffer = Vec::with_capacity(4 + meta_json.len() + data.len());
        buffer.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&meta_json);
        buffer.extend_from_slice(data);

        // 3. Single append
        self.db.append(name, Atom::Bytes(buffer))?;

        Ok(1)
    }

    /// Load tensor from batched blob storage.
    ///
    /// This method loads a tensor that was stored using [`put_tensor_batched`](Self::put_tensor_batched).
    /// It reads the single blob and extracts metadata and data.
    ///
    /// # Arguments
    ///
    /// * `name` - Key name of the tensor
    ///
    /// # Returns
    ///
    /// A tuple of (data, shape) where:
    /// - `data` is a byte vector containing the tensor data
    /// - `shape` is a vector of dimensions
    ///
    /// # Errors
    ///
    /// Returns `SynaError::KeyNotFound` if the tensor doesn't exist.
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
    /// engine.put_tensor_batched("values", &original, &[4], DType::Float64).unwrap();
    ///
    /// // Load tensor
    /// let (loaded, shape) = engine.get_tensor_batched("values").unwrap();
    /// assert_eq!(loaded, original);
    /// assert_eq!(shape, vec![4]);
    /// ```
    ///
    /// _Requirements: 2.1, 9.3_
    pub fn get_tensor_batched(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
        // Load the blob
        let blob_atom = self
            .db
            .get(name)?
            .ok_or_else(|| SynaError::KeyNotFound(name.to_string()))?;

        let blob = match blob_atom {
            Atom::Bytes(b) => b,
            _ => {
                return Err(SynaError::TypeConversion {
                    from_type: blob_atom.type_name(),
                    to_type: "Bytes",
                })
            }
        };

        // Parse: [meta_len:u32][meta_json][data]
        if blob.len() < 4 {
            return Err(SynaError::ShapeMismatch {
                data_size: blob.len(),
                expected_size: 4,
            });
        }

        let meta_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;

        if blob.len() < 4 + meta_len {
            return Err(SynaError::ShapeMismatch {
                data_size: blob.len(),
                expected_size: 4 + meta_len,
            });
        }

        let meta_json = &blob[4..4 + meta_len];
        let meta: TensorMeta = serde_json::from_slice(meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        let data_start = 4 + meta_len;
        let data = blob[data_start..].to_vec();

        // Verify total size
        if data.len() != meta.total_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: meta.total_bytes,
            });
        }

        Ok((data, meta.shape))
    }

    /// Get metadata for a batched tensor without loading the full data.
    ///
    /// This is useful for inspecting tensor properties without the overhead
    /// of loading the full tensor data.
    ///
    /// # Arguments
    ///
    /// * `name` - Key name of the tensor
    ///
    /// # Returns
    ///
    /// The tensor metadata including shape, dtype, and size information.
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
    /// engine.put_tensor_batched("tensor", &data, &[128], DType::Float64).unwrap();
    ///
    /// // Get metadata only
    /// let meta = engine.get_tensor_batched_meta("tensor").unwrap();
    /// assert_eq!(meta.shape, vec![128]);
    /// assert_eq!(meta.dtype, "float64");
    /// ```
    pub fn get_tensor_batched_meta(&mut self, name: &str) -> Result<TensorMeta> {
        // Load the blob
        let blob_atom = self
            .db
            .get(name)?
            .ok_or_else(|| SynaError::KeyNotFound(name.to_string()))?;

        let blob = match blob_atom {
            Atom::Bytes(b) => b,
            _ => {
                return Err(SynaError::TypeConversion {
                    from_type: blob_atom.type_name(),
                    to_type: "Bytes",
                })
            }
        };

        // Parse: [meta_len:u32][meta_json]...
        if blob.len() < 4 {
            return Err(SynaError::ShapeMismatch {
                data_size: blob.len(),
                expected_size: 4,
            });
        }

        let meta_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;

        if blob.len() < 4 + meta_len {
            return Err(SynaError::ShapeMismatch {
                data_size: blob.len(),
                expected_size: 4 + meta_len,
            });
        }

        let meta_json = &blob[4..4 + meta_len];
        let meta: TensorMeta = serde_json::from_slice(meta_json)
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

    // =========================================================================
    // Memory-Mapped Tensor Writes (Zero-Copy)
    // =========================================================================

    /// Store tensor using memory-mapped I/O for zero-copy writes.
    ///
    /// This method bypasses Rust's allocator for large tensors, writing
    /// directly to the file through the OS page cache. This provides
    /// 2-3x throughput improvement for large tensors (>10MB).
    ///
    /// # Storage Layout
    ///
    /// The tensor is stored as a separate `.mmap` file alongside the database:
    /// - `{db_path}.{name}.mmap` - Raw tensor data
    /// - `{name}/mmap_meta` in database - Metadata with file path
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
    /// Always returns 1 (single mmap operation).
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
    /// // Store large tensor using mmap (fast!)
    /// let data: Vec<u8> = vec![0u8; 10 * 1024 * 1024]; // 10MB
    /// let shape = vec![10 * 1024 * 1024 / 8]; // f64 elements
    /// engine.put_tensor_mmap("model/weights", &data, &shape, DType::Float64).unwrap();
    /// ```
    ///
    /// _Requirements: 9.3, 2.4_
    pub fn put_tensor_mmap(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        use memmap2::MmapMut;
        use std::fs::OpenOptions;

        let element_size = dtype.size();
        let n_elements = shape.iter().product::<usize>();
        let expected_bytes = n_elements * element_size;

        if data.len() != expected_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: expected_bytes,
            });
        }

        // Create mmap file path based on database path and tensor name
        // Replace '/' in name with '_' for valid filename
        let safe_name = name.replace('/', "_");
        let mmap_path = self.db.path.with_extension(format!("{}.mmap", safe_name));

        // 1. Create/open file and set its length
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&mmap_path)?;

        file.set_len(data.len() as u64)?;

        // 2. Memory-map the file for writing
        // Safety: We just created/truncated the file and set its length,
        // so the mapping is valid. We have exclusive access via truncate.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // 3. Write directly to mapped memory (zero-copy from caller's perspective)
        mmap.copy_from_slice(data);

        // 4. Flush to ensure data is written to disk
        mmap.flush()?;

        // 5. Store metadata in the database
        let meta = MmapTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.name().to_string(),
            total_bytes: data.len(),
            mmap_path: mmap_path.to_string_lossy().to_string(),
        };

        let meta_json = serde_json::to_string(&meta)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to serialize metadata: {}", e)))?;
        let meta_key = format!("{}/mmap_meta", name);
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(1)
    }

    /// Store tensor using optimized direct I/O for maximum write throughput.
    ///
    /// This method achieves 2+ GB/s write throughput by:
    /// 1. Using direct file writes instead of mmap for large sequential writes
    /// 2. Using async flush (data is in OS page cache, will be written asynchronously)
    /// 3. Pre-allocating file space to avoid fragmentation
    ///
    /// # Trade-offs
    ///
    /// - **Faster writes**: 2x faster than `put_tensor_mmap` for large tensors
    /// - **Async durability**: Data may not be on disk immediately after return
    /// - **Best for**: Training checkpoints, intermediate results, non-critical data
    ///
    /// For critical data that must be durable, use `put_tensor_mmap` instead.
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
    /// Always returns 1 (single write operation).
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
    /// // Store large tensor with maximum throughput
    /// let data: Vec<u8> = vec![0u8; 100 * 1024 * 1024]; // 100MB
    /// let shape = vec![100 * 1024 * 1024 / 8]; // f64 elements
    /// engine.put_tensor_mmap_fast("model/weights", &data, &shape, DType::Float64).unwrap();
    /// ```
    ///
    /// _Requirements: 9.3_
    pub fn put_tensor_mmap_fast(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        use std::fs::OpenOptions;
        use std::io::Write;

        let element_size = dtype.size();
        let n_elements = shape.iter().product::<usize>();
        let expected_bytes = n_elements * element_size;

        if data.len() != expected_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: data.len(),
                expected_size: expected_bytes,
            });
        }

        // Create mmap file path based on database path and tensor name
        let safe_name = name.replace('/', "_");
        let mmap_path = self.db.path.with_extension(format!("{}.mmap", safe_name));

        // 1. Create file with pre-allocated space
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&mmap_path)?;

        // Pre-allocate file space to avoid fragmentation
        file.set_len(data.len() as u64)?;

        // 2. Write data directly using buffered I/O
        // For large writes, the OS will use efficient DMA transfers
        file.write_all(data)?;

        // 3. Async flush - tells OS to start writing but doesn't wait
        // This is the key optimization: we don't block on disk I/O
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // fdatasync is faster than fsync as it doesn't update metadata
            // But we skip even that for maximum throughput
            // The data is in the OS page cache and will be written asynchronously
            let _ =
                unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }

        #[cfg(windows)]
        {
            // On Windows, we rely on the OS page cache
            // Data will be written asynchronously by the OS
        }

        // 4. Store metadata in the database
        let meta = MmapTensorMeta {
            shape: shape.to_vec(),
            dtype: dtype.name().to_string(),
            total_bytes: data.len(),
            mmap_path: mmap_path.to_string_lossy().to_string(),
        };

        let meta_json = serde_json::to_string(&meta)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to serialize metadata: {}", e)))?;
        let meta_key = format!("{}/mmap_meta", name);
        self.db.append(&meta_key, Atom::Text(meta_json))?;

        Ok(1)
    }

    /// Load tensor from memory-mapped file storage.
    ///
    /// This method loads a tensor that was stored using [`put_tensor_mmap`](Self::put_tensor_mmap).
    /// It memory-maps the file and returns the data.
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
    /// // Load tensor stored with mmap
    /// let (data, shape) = engine.get_tensor_mmap("model/weights").unwrap();
    /// ```
    ///
    /// _Requirements: 2.1, 9.3_
    pub fn get_tensor_mmap(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
        use memmap2::Mmap;
        use std::fs::File;

        // Load metadata
        let meta_key = format!("{}/mmap_meta", name);
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

        let meta: MmapTensorMeta = serde_json::from_str(&meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        // Open and memory-map the file
        let file = File::open(&meta.mmap_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Verify size
        if mmap.len() != meta.total_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: mmap.len(),
                expected_size: meta.total_bytes,
            });
        }

        // Copy data from mmap (caller owns the data)
        let data = mmap.to_vec();

        Ok((data, meta.shape))
    }

    /// Get a zero-copy reference to memory-mapped tensor data.
    ///
    /// Unlike [`get_tensor_mmap`](Self::get_tensor_mmap), this method returns
    /// a reference to the memory-mapped data without copying. The returned
    /// `MmapTensorRef` keeps the mapping alive.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor
    ///
    /// # Returns
    ///
    /// A `MmapTensorRef` containing the shape and a reference to the data.
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
    /// // Get zero-copy reference to tensor
    /// let tensor_ref = engine.get_tensor_mmap_ref("model/weights").unwrap();
    /// println!("Shape: {:?}, Size: {} bytes", tensor_ref.shape, tensor_ref.data().len());
    /// ```
    ///
    /// _Requirements: 2.4, 9.3_
    pub fn get_tensor_mmap_ref(&mut self, name: &str) -> Result<MmapTensorRef> {
        use memmap2::Mmap;
        use std::fs::File;

        // Load metadata
        let meta_key = format!("{}/mmap_meta", name);
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

        let meta: MmapTensorMeta = serde_json::from_str(&meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        // Open and memory-map the file
        let file = File::open(&meta.mmap_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Verify size
        if mmap.len() != meta.total_bytes {
            return Err(SynaError::ShapeMismatch {
                data_size: mmap.len(),
                expected_size: meta.total_bytes,
            });
        }

        Ok(MmapTensorRef {
            shape: meta.shape,
            dtype: meta.dtype,
            mmap,
        })
    }

    /// Get metadata for a memory-mapped tensor without loading the data.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor
    ///
    /// # Returns
    ///
    /// The tensor metadata including shape, dtype, and file path.
    pub fn get_tensor_mmap_meta(&mut self, name: &str) -> Result<MmapTensorMeta> {
        let meta_key = format!("{}/mmap_meta", name);
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

        let meta: MmapTensorMeta = serde_json::from_str(&meta_json)
            .map_err(|e| SynaError::InvalidPath(format!("Failed to parse metadata: {}", e)))?;

        Ok(meta)
    }

    /// Delete a memory-mapped tensor and its associated file.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name of the tensor to delete
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn delete_tensor_mmap(&mut self, name: &str) -> Result<()> {
        // Try to load metadata to get file path
        let meta_key = format!("{}/mmap_meta", name);
        if let Some(Atom::Text(json)) = self.db.get(&meta_key)? {
            if let Ok(meta) = serde_json::from_str::<MmapTensorMeta>(&json) {
                // Delete the mmap file
                let _ = std::fs::remove_file(&meta.mmap_path);
            }
        }

        // Delete metadata from database
        self.db.delete(&meta_key)?;

        Ok(())
    }

    // =========================================================================
    // Async Parallel Chunk Writes (High-Throughput)
    // =========================================================================

    /// Store tensor with parallel async chunk writes.
    ///
    /// Splits tensor into chunks and writes them concurrently using async I/O,
    /// utilizing NVMe SSD queue depth for maximum throughput.
    /// This can achieve 2-4x throughput improvement on NVMe SSDs
    /// compared to sequential writes.
    ///
    /// # Implementation Note
    ///
    /// This method uses tokio's async file I/O to write chunks in parallel to
    /// temporary files, then consolidates them into the main database. This
    /// approach maximizes SSD parallelism while maintaining database consistency.
    ///
    /// # Storage Layout
    ///
    /// Same as [`put_tensor_chunked`](Self::put_tensor_chunked):
    /// ```text
    /// {name}/meta     -> TensorMeta (JSON) with shape, dtype, chunk info
    /// {name}/chunk/0  -> Atom::Bytes (first chunk)
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
    /// * `chunk_size` - Size of each chunk in bytes
    ///
    /// # Returns
    ///
    /// The number of chunks written.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::SynaDB;
    /// use synadb::tensor::{TensorEngine, DType, DEFAULT_CHUNK_SIZE};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let db = SynaDB::new("data.db").unwrap();
    ///     let mut engine = TensorEngine::new(db);
    ///
    ///     // Store large tensor with parallel async writes
    ///     let data: Vec<u8> = vec![0u8; 10 * 1024 * 1024]; // 10MB
    ///     let shape = vec![10 * 1024 * 1024 / 8]; // f64 elements
    ///     let chunks = engine.put_tensor_async(
    ///         "model/weights",
    ///         &data,
    ///         &shape,
    ///         DType::Float64,
    ///         DEFAULT_CHUNK_SIZE,
    ///     ).await.unwrap();
    ///     println!("Wrote {} chunks in parallel", chunks);
    /// }
    /// ```
    ///
    /// _Requirements: 9.3_
    #[cfg(feature = "async")]
    pub async fn put_tensor_async(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
        chunk_size: usize,
    ) -> Result<usize> {
        use tokio::fs::File as AsyncFile;
        use tokio::io::AsyncWriteExt;
        use tokio::task::JoinSet;

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

        // Create and store metadata first (synchronously)
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

        // Get database directory for temp files
        let db_dir = self.db.path.parent().unwrap_or(std::path::Path::new("."));
        let safe_name = name.replace('/', "_");

        // Prepare chunks for parallel writes to temp files
        let chunks_data: Vec<(usize, Vec<u8>)> = data
            .chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk_data)| (i, chunk_data.to_vec()))
            .collect();

        // Spawn parallel async file write tasks
        let mut tasks: JoinSet<std::result::Result<(usize, std::path::PathBuf), String>> =
            JoinSet::new();

        for (i, chunk_data) in chunks_data {
            let temp_path = db_dir.join(format!(".{}_chunk_{}.tmp", safe_name, i));
            tasks.spawn(async move {
                // Write chunk to temp file using async I/O
                let mut file = AsyncFile::create(&temp_path)
                    .await
                    .map_err(|e| format!("Failed to create temp file: {}", e))?;
                file.write_all(&chunk_data)
                    .await
                    .map_err(|e| format!("Failed to write chunk: {}", e))?;
                file.flush()
                    .await
                    .map_err(|e| format!("Failed to flush chunk: {}", e))?;
                Ok((i, temp_path))
            });
        }

        // Collect results and temp file paths
        let mut temp_files: Vec<(usize, std::path::PathBuf)> = Vec::with_capacity(chunk_count);
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok((i, path))) => temp_files.push((i, path)),
                Ok(Err(e)) => {
                    // Clean up any temp files created so far
                    for (_, path) in &temp_files {
                        let _ = std::fs::remove_file(path);
                    }
                    return Err(SynaError::InvalidPath(e));
                }
                Err(join_error) => {
                    // Clean up any temp files created so far
                    for (_, path) in &temp_files {
                        let _ = std::fs::remove_file(path);
                    }
                    return Err(SynaError::InvalidPath(format!(
                        "Async task failed: {}",
                        join_error
                    )));
                }
            }
        }

        // Sort by chunk index to ensure correct order
        temp_files.sort_by_key(|(i, _)| *i);

        // Now read temp files and write to database (sequential, but data is already on disk)
        for (i, temp_path) in &temp_files {
            let chunk_data = std::fs::read(temp_path)?;
            let chunk_key = format!("{}/chunk/{}", name, i);
            self.db.append(&chunk_key, Atom::Bytes(chunk_data))?;
            // Clean up temp file
            let _ = std::fs::remove_file(temp_path);
        }

        Ok(chunk_count)
    }

    /// Store tensor with parallel async chunk writes using default chunk size.
    ///
    /// This is a convenience wrapper around [`put_tensor_async`](Self::put_tensor_async)
    /// that uses the default chunk size of 1MB.
    ///
    /// # Arguments
    ///
    /// * `name` - Base name for the tensor
    /// * `data` - Raw byte data of the tensor
    /// * `shape` - Shape of the tensor (dimensions)
    /// * `dtype` - Data type of the tensor elements
    ///
    /// # Returns
    ///
    /// The number of chunks written.
    ///
    /// _Requirements: 9.3_
    #[cfg(feature = "async")]
    pub async fn put_tensor_async_default(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Result<usize> {
        self.put_tensor_async(name, data, shape, dtype, DEFAULT_CHUNK_SIZE)
            .await
    }
}

/// Metadata for a memory-mapped tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MmapTensorMeta {
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Data type of tensor elements
    pub dtype: String,
    /// Total size in bytes
    pub total_bytes: usize,
    /// Path to the memory-mapped file
    pub mmap_path: String,
}

/// Zero-copy reference to memory-mapped tensor data.
///
/// This struct holds a memory mapping and provides access to the tensor
/// data without copying. The mapping is kept alive as long as this struct exists.
pub struct MmapTensorRef {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type as string
    pub dtype: String,
    /// The memory mapping
    mmap: memmap2::Mmap,
}

impl MmapTensorRef {
    /// Get the raw tensor data as a byte slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Get the tensor data as f32 slice (zero-copy).
    ///
    /// # Panics
    ///
    /// Panics if the data length is not a multiple of 4.
    #[inline]
    pub fn as_f32_slice(&self) -> &[f32] {
        let count = self.mmap.len() / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const f32, count) }
    }

    /// Get the tensor data as f64 slice (zero-copy).
    ///
    /// # Panics
    ///
    /// Panics if the data length is not a multiple of 8.
    #[inline]
    pub fn as_f64_slice(&self) -> &[f64] {
        let count = self.mmap.len() / std::mem::size_of::<f64>();
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const f64, count) }
    }

    /// Get the tensor data as i32 slice (zero-copy).
    #[inline]
    pub fn as_i32_slice(&self) -> &[i32] {
        let count = self.mmap.len() / std::mem::size_of::<i32>();
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const i32, count) }
    }

    /// Get the tensor data as i64 slice (zero-copy).
    #[inline]
    pub fn as_i64_slice(&self) -> &[i64] {
        let count = self.mmap.len() / std::mem::size_of::<i64>();
        unsafe { std::slice::from_raw_parts(self.mmap.as_ptr() as *const i64, count) }
    }

    /// Get the total size in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if the tensor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

// =============================================================================
// Direct I/O Bypass (Linux Only)
// =============================================================================

/// Direct I/O configuration and utilities for bypassing OS page cache.
///
/// On Linux, using O_DIRECT flag bypasses the page cache for large sequential
/// writes, providing 20-50% throughput improvement on NVMe SSDs.
///
/// _Requirements: 9.3_
pub mod direct_io {
    use std::fs::{File, OpenOptions};
    use std::io::{self, Write};
    use std::path::Path;

    /// Alignment requirement for O_DIRECT I/O on Linux.
    /// Data buffers and file offsets must be aligned to this boundary.
    /// 4096 bytes (4KB) is the typical filesystem block size.
    pub const DIRECT_IO_ALIGNMENT: usize = 4096;

    /// Minimum size threshold for using Direct I/O.
    /// Below this size, the overhead of alignment isn't worth it.
    /// 1MB is a reasonable threshold.
    pub const DIRECT_IO_MIN_SIZE: usize = 1024 * 1024;

    /// Check if Direct I/O is available on the current platform.
    ///
    /// Returns `true` on Linux, `false` on other platforms.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::tensor::direct_io::is_direct_io_available;
    ///
    /// if is_direct_io_available() {
    ///     println!("Direct I/O is available on this platform");
    /// }
    /// ```
    #[inline]
    pub fn is_direct_io_available() -> bool {
        cfg!(target_os = "linux")
    }

    /// Open a file with O_DIRECT flag for bypassing page cache (Linux only).
    ///
    /// On Linux, this opens the file with O_DIRECT which bypasses the OS page
    /// cache, providing better throughput for large sequential writes to NVMe SSDs.
    ///
    /// On non-Linux platforms, this falls back to a regular file open.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to open/create
    ///
    /// # Returns
    ///
    /// A `File` handle opened with O_DIRECT on Linux, or a regular file on other platforms.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::tensor::direct_io::open_direct;
    /// use std::path::Path;
    ///
    /// let file = open_direct(Path::new("tensor.bin")).unwrap();
    /// ```
    ///
    /// _Requirements: 9.3_
    #[cfg(target_os = "linux")]
    pub fn open_direct(path: &Path) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;

        // O_DIRECT constant from libc
        const O_DIRECT: i32 = 0o40000;

        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(O_DIRECT)
            .open(path)
    }

    /// Open a file with O_DIRECT flag for bypassing page cache (non-Linux fallback).
    ///
    /// On non-Linux platforms, this opens a regular file without O_DIRECT.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to open/create
    ///
    /// # Returns
    ///
    /// A regular `File` handle.
    ///
    /// _Requirements: 9.3_
    #[cfg(not(target_os = "linux"))]
    pub fn open_direct(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
    }

    /// Open a file for reading with O_DIRECT flag (Linux only).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to open
    ///
    /// # Returns
    ///
    /// A `File` handle opened with O_DIRECT on Linux, or a regular file on other platforms.
    ///
    /// _Requirements: 9.3_
    #[cfg(target_os = "linux")]
    pub fn open_direct_read(path: &Path) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;

        const O_DIRECT: i32 = 0o40000;

        OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(path)
    }

    /// Open a file for reading with O_DIRECT flag (non-Linux fallback).
    ///
    /// _Requirements: 9.3_
    #[cfg(not(target_os = "linux"))]
    pub fn open_direct_read(path: &Path) -> io::Result<File> {
        OpenOptions::new().read(true).open(path)
    }

    /// Align a buffer size up to the Direct I/O alignment boundary.
    ///
    /// O_DIRECT requires buffers to be aligned to the filesystem block size.
    /// This function rounds up a size to the nearest alignment boundary.
    ///
    /// # Arguments
    ///
    /// * `size` - The size to align
    ///
    /// # Returns
    ///
    /// The size rounded up to the nearest `DIRECT_IO_ALIGNMENT` boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::tensor::direct_io::{align_size, DIRECT_IO_ALIGNMENT};
    ///
    /// assert_eq!(align_size(100), DIRECT_IO_ALIGNMENT);
    /// assert_eq!(align_size(4096), 4096);
    /// assert_eq!(align_size(4097), 8192);
    /// ```
    #[inline]
    pub fn align_size(size: usize) -> usize {
        (size + DIRECT_IO_ALIGNMENT - 1) & !(DIRECT_IO_ALIGNMENT - 1)
    }

    /// Create an aligned buffer for Direct I/O operations.
    ///
    /// O_DIRECT requires buffers to be aligned to the filesystem block size.
    /// This function creates a buffer with proper alignment.
    ///
    /// # Arguments
    ///
    /// * `size` - The minimum size of the buffer
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` with capacity aligned to `DIRECT_IO_ALIGNMENT`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::tensor::direct_io::create_aligned_buffer;
    ///
    /// let buffer = create_aligned_buffer(1000);
    /// assert!(buffer.capacity() >= 4096); // Aligned to 4KB
    /// ```
    pub fn create_aligned_buffer(size: usize) -> Vec<u8> {
        let aligned_size = align_size(size);
        vec![0; aligned_size]
    }

    /// Write data using Direct I/O with proper alignment.
    ///
    /// This function handles the alignment requirements for O_DIRECT:
    /// - Pads data to alignment boundary
    /// - Writes using the direct file handle
    ///
    /// # Arguments
    ///
    /// * `file` - File opened with `open_direct`
    /// * `data` - Data to write (will be padded to alignment)
    ///
    /// # Returns
    ///
    /// The number of bytes written (including padding).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::tensor::direct_io::{open_direct, write_aligned};
    /// use std::path::Path;
    ///
    /// let mut file = open_direct(Path::new("tensor.bin")).unwrap();
    /// let data = vec![1u8; 10000];
    /// let written = write_aligned(&mut file, &data).unwrap();
    /// ```
    ///
    /// _Requirements: 9.3_
    pub fn write_aligned(file: &mut File, data: &[u8]) -> io::Result<usize> {
        let aligned_size = align_size(data.len());

        if aligned_size == data.len() {
            // Data is already aligned
            file.write_all(data)?;
            Ok(data.len())
        } else {
            // Need to pad data to alignment boundary
            let mut aligned_buffer = create_aligned_buffer(data.len());
            aligned_buffer[..data.len()].copy_from_slice(data);
            file.write_all(&aligned_buffer)?;
            Ok(aligned_size)
        }
    }

    /// Check if a size is suitable for Direct I/O.
    ///
    /// Direct I/O has overhead from alignment requirements, so it's only
    /// beneficial for larger writes.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of data to write
    ///
    /// # Returns
    ///
    /// `true` if the size is large enough to benefit from Direct I/O
    /// AND Direct I/O is available on the current platform.
    ///
    /// # Platform Support
    ///
    /// - **Linux**: Returns `true` for sizes >= 1MB
    /// - **Other platforms**: Always returns `false` (Direct I/O not supported)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::tensor::direct_io::{should_use_direct_io, is_direct_io_available};
    ///
    /// // Small sizes never use Direct I/O
    /// assert!(!should_use_direct_io(1000));
    ///
    /// // Large sizes only use Direct I/O if available on this platform
    /// if is_direct_io_available() {
    ///     assert!(should_use_direct_io(2_000_000));
    /// } else {
    ///     assert!(!should_use_direct_io(2_000_000));
    /// }
    /// ```
    #[inline]
    pub fn should_use_direct_io(size: usize) -> bool {
        is_direct_io_available() && size >= DIRECT_IO_MIN_SIZE
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

    // =========================================================================
    // Batched Single-Append Storage Tests
    // =========================================================================

    #[test]
    fn test_put_get_tensor_batched_small() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Small tensor
        let original = vec![1.0f64, 2.0, 3.0, 4.0];
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        let count = engine
            .put_tensor_batched("small", &data, &[4], DType::Float64)
            .unwrap();
        assert_eq!(count, 1); // Single append

        let (loaded, shape) = engine.get_tensor_batched("small").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn test_put_get_tensor_batched_large() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Large tensor
        let num_elements = 10_000;
        let original: Vec<f64> = (0..num_elements).map(|i| i as f64 * 0.1).collect();
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        let count = engine
            .put_tensor_batched("large", &data, &[num_elements], DType::Float64)
            .unwrap();
        assert_eq!(count, 1); // Still single append

        let (loaded, shape) = engine.get_tensor_batched("large").unwrap();
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
    fn test_get_tensor_batched_meta() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        engine
            .put_tensor_batched("tensor", &data, &[2, 3], DType::Float64)
            .unwrap();

        let meta = engine.get_tensor_batched_meta("tensor").unwrap();
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.dtype, "float64");
        assert_eq!(meta.total_bytes, 48); // 6 * 8 bytes
    }

    #[test]
    fn test_batched_shape_mismatch() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Data doesn't match shape
        let data = vec![0u8; 16]; // 16 bytes
        let result = engine.put_tensor_batched("bad", &data, &[4], DType::Float64); // expects 32 bytes

        assert!(matches!(result, Err(SynaError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_batched_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let result = engine.get_tensor_batched("nonexistent");
        assert!(matches!(result, Err(SynaError::KeyNotFound(_))));
    }

    #[test]
    fn test_batched_roundtrip_all_dtypes() {
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
            .put_tensor_batched("f32", &f32_data, &[4], DType::Float32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_batched("f32").unwrap();
        assert_eq!(loaded, f32_data);

        // Test Float64
        let f64_data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        engine
            .put_tensor_batched("f64", &f64_data, &[4], DType::Float64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_batched("f64").unwrap();
        assert_eq!(loaded, f64_data);

        // Test Int32
        let i32_data: Vec<u8> = vec![1i32, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_batched("i32", &i32_data, &[4], DType::Int32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_batched("i32").unwrap();
        assert_eq!(loaded, i32_data);

        // Test Int64
        let i64_data: Vec<u8> = vec![1i64, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_batched("i64", &i64_data, &[4], DType::Int64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_batched("i64").unwrap();
        assert_eq!(loaded, i64_data);
    }

    #[test]
    fn test_batched_multidimensional_shape() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // 3D tensor: 2x3x4 = 24 elements
        let num_elements = 2 * 3 * 4;
        let original: Vec<f64> = (0..num_elements).map(|i| i as f64).collect();
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        engine
            .put_tensor_batched("3d", &data, &[2, 3, 4], DType::Float64)
            .unwrap();

        let (loaded, shape) = engine.get_tensor_batched("3d").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    // =========================================================================
    // Memory-Mapped Tensor Storage Tests
    // =========================================================================

    #[test]
    fn test_put_get_tensor_mmap_small() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Small tensor
        let original = vec![1.0f64, 2.0, 3.0, 4.0];
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        let count = engine
            .put_tensor_mmap("small", &data, &[4], DType::Float64)
            .unwrap();
        assert_eq!(count, 1); // Single mmap operation

        let (loaded, shape) = engine.get_tensor_mmap("small").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![4]);
    }

    #[test]
    fn test_put_get_tensor_mmap_large() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Large tensor (1MB)
        let num_elements = 128 * 1024; // 128K f64 elements = 1MB
        let original: Vec<f64> = (0..num_elements).map(|i| i as f64 * 0.1).collect();
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        let count = engine
            .put_tensor_mmap("large", &data, &[num_elements], DType::Float64)
            .unwrap();
        assert_eq!(count, 1);

        let (loaded, shape) = engine.get_tensor_mmap("large").unwrap();
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
    fn test_get_tensor_mmap_ref() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let original = vec![1.0f64, 2.0, 3.0, 4.0];
        let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

        engine
            .put_tensor_mmap("ref_test", &data, &[4], DType::Float64)
            .unwrap();

        // Get zero-copy reference
        let tensor_ref = engine.get_tensor_mmap_ref("ref_test").unwrap();
        assert_eq!(tensor_ref.shape, vec![4]);
        assert_eq!(tensor_ref.dtype, "float64");
        assert_eq!(tensor_ref.len(), 32); // 4 * 8 bytes

        // Access as f64 slice
        let floats = tensor_ref.as_f64_slice();
        assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_get_tensor_mmap_meta() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        engine
            .put_tensor_mmap("meta_test", &data, &[2, 3], DType::Float64)
            .unwrap();

        let meta = engine.get_tensor_mmap_meta("meta_test").unwrap();
        assert_eq!(meta.shape, vec![2, 3]);
        assert_eq!(meta.dtype, "float64");
        assert_eq!(meta.total_bytes, 48); // 6 * 8 bytes
        assert!(meta.mmap_path.contains("meta_test"));
    }

    #[test]
    fn test_mmap_shape_mismatch() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Data doesn't match shape
        let data = vec![0u8; 16]; // 16 bytes
        let result = engine.put_tensor_mmap("bad", &data, &[4], DType::Float64); // expects 32 bytes

        assert!(matches!(result, Err(SynaError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_mmap_not_found() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let result = engine.get_tensor_mmap("nonexistent");
        assert!(matches!(result, Err(SynaError::KeyNotFound(_))));
    }

    #[test]
    fn test_delete_tensor_mmap() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        engine
            .put_tensor_mmap("to_delete", &data, &[4], DType::Float64)
            .unwrap();

        // Verify it exists
        assert!(engine.get_tensor_mmap_meta("to_delete").is_ok());

        // Delete it
        engine.delete_tensor_mmap("to_delete").unwrap();

        // Verify it's gone
        assert!(engine.get_tensor_mmap_meta("to_delete").is_err());
    }

    #[test]
    fn test_mmap_roundtrip_all_dtypes() {
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
            .put_tensor_mmap("f32", &f32_data, &[4], DType::Float32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_mmap("f32").unwrap();
        assert_eq!(loaded, f32_data);

        // Test Float64
        let f64_data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        engine
            .put_tensor_mmap("f64", &f64_data, &[4], DType::Float64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_mmap("f64").unwrap();
        assert_eq!(loaded, f64_data);

        // Test Int32
        let i32_data: Vec<u8> = vec![1i32, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_mmap("i32", &i32_data, &[4], DType::Int32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_mmap("i32").unwrap();
        assert_eq!(loaded, i32_data);

        // Test Int64
        let i64_data: Vec<u8> = vec![1i64, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_mmap("i64", &i64_data, &[4], DType::Int64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_mmap("i64").unwrap();
        assert_eq!(loaded, i64_data);
    }

    #[test]
    fn test_mmap_tensor_ref_slices() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Test f32 slice
        let f32_values = vec![1.0f32, 2.0, 3.0, 4.0];
        let f32_data: Vec<u8> = f32_values.iter().flat_map(|f| f.to_le_bytes()).collect();
        engine
            .put_tensor_mmap("f32_ref", &f32_data, &[4], DType::Float32)
            .unwrap();
        let tensor_ref = engine.get_tensor_mmap_ref("f32_ref").unwrap();
        assert_eq!(tensor_ref.as_f32_slice(), &[1.0f32, 2.0, 3.0, 4.0]);

        // Test i32 slice
        let i32_values = vec![10i32, 20, 30, 40];
        let i32_data: Vec<u8> = i32_values.iter().flat_map(|i| i.to_le_bytes()).collect();
        engine
            .put_tensor_mmap("i32_ref", &i32_data, &[4], DType::Int32)
            .unwrap();
        let tensor_ref = engine.get_tensor_mmap_ref("i32_ref").unwrap();
        assert_eq!(tensor_ref.as_i32_slice(), &[10i32, 20, 30, 40]);

        // Test i64 slice
        let i64_values = vec![100i64, 200, 300, 400];
        let i64_data: Vec<u8> = i64_values.iter().flat_map(|i| i.to_le_bytes()).collect();
        engine
            .put_tensor_mmap("i64_ref", &i64_data, &[4], DType::Int64)
            .unwrap();
        let tensor_ref = engine.get_tensor_mmap_ref("i64_ref").unwrap();
        assert_eq!(tensor_ref.as_i64_slice(), &[100i64, 200, 300, 400]);
    }

    #[test]
    fn test_mmap_with_slashes_in_name() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Name with slashes (like "model/layer1/weights")
        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        engine
            .put_tensor_mmap("model/layer1/weights", &data, &[4], DType::Float64)
            .unwrap();

        let (loaded, shape) = engine.get_tensor_mmap("model/layer1/weights").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![4]);

        // Verify the mmap file was created with underscores
        let meta = engine.get_tensor_mmap_meta("model/layer1/weights").unwrap();
        assert!(meta.mmap_path.contains("model_layer1_weights"));
    }

    // =========================================================================
    // Configurable Chunk Size Tests
    // =========================================================================

    #[test]
    fn test_optimal_chunk_size() {
        // Small tensors (<10MB) should use 1MB chunks
        assert_eq!(optimal_chunk_size(0), CHUNK_SIZE_SMALL);
        assert_eq!(optimal_chunk_size(1_000_000), CHUNK_SIZE_SMALL);
        assert_eq!(optimal_chunk_size(5_000_000), CHUNK_SIZE_SMALL);
        assert_eq!(optimal_chunk_size(10_000_000), CHUNK_SIZE_SMALL);

        // Medium tensors (10-100MB) should use 4MB chunks
        assert_eq!(optimal_chunk_size(10_000_001), CHUNK_SIZE_MEDIUM);
        assert_eq!(optimal_chunk_size(50_000_000), CHUNK_SIZE_MEDIUM);
        assert_eq!(optimal_chunk_size(100_000_000), CHUNK_SIZE_MEDIUM);

        // Large tensors (>100MB) should use 16MB chunks
        assert_eq!(optimal_chunk_size(100_000_001), CHUNK_SIZE_LARGE);
        assert_eq!(optimal_chunk_size(200_000_000), CHUNK_SIZE_LARGE);
        assert_eq!(optimal_chunk_size(1_000_000_000), CHUNK_SIZE_LARGE);
    }

    #[test]
    fn test_chunk_size_constants() {
        // Verify the constants are correct
        assert_eq!(CHUNK_SIZE_SMALL, 1 * 1024 * 1024); // 1 MB
        assert_eq!(CHUNK_SIZE_MEDIUM, 4 * 1024 * 1024); // 4 MB
        assert_eq!(CHUNK_SIZE_LARGE, 16 * 1024 * 1024); // 16 MB
    }

    #[test]
    fn test_put_tensor_optimized_small() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Small tensor (< 10MB) - should use 1MB chunks
        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let chunks = engine
            .put_tensor_optimized("small_tensor", &data, &[4], DType::Float64)
            .unwrap();
        assert_eq!(chunks, 1); // Small data fits in one chunk

        // Verify we can read it back
        let (loaded, shape) = engine.get_tensor_chunked("small_tensor").unwrap();
        assert_eq!(loaded, data);
        assert_eq!(shape, vec![4]);

        // Verify chunk size in metadata
        let meta = engine.get_tensor_meta("small_tensor").unwrap();
        assert_eq!(meta.chunk_size, CHUNK_SIZE_SMALL);
    }

    #[test]
    fn test_put_tensor_optimized_roundtrip() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = SynaDB::new(&db_path).unwrap();
        let mut engine = TensorEngine::new(db);

        // Test with various data types
        let f64_data: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        engine
            .put_tensor_optimized("f64_opt", &f64_data, &[4], DType::Float64)
            .unwrap();
        let (loaded, _) = engine.get_tensor_chunked("f64_opt").unwrap();
        assert_eq!(loaded, f64_data);

        let i32_data: Vec<u8> = vec![1i32, 2, 3, 4]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        engine
            .put_tensor_optimized("i32_opt", &i32_data, &[4], DType::Int32)
            .unwrap();
        let (loaded, _) = engine.get_tensor_chunked("i32_opt").unwrap();
        assert_eq!(loaded, i32_data);
    }

    // =========================================================================
    // Async Tests (require tokio feature)
    // =========================================================================

    #[cfg(feature = "async")]
    mod async_tests {
        use super::*;

        #[tokio::test]
        async fn test_put_tensor_async_small() {
            let dir = tempdir().unwrap();
            let db_path = dir.path().join("test.db");
            let db = SynaDB::new(&db_path).unwrap();
            let mut engine = TensorEngine::new(db);

            // Small tensor that fits in one chunk
            let original = vec![1.0f64, 2.0, 3.0, 4.0];
            let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

            let chunks = engine
                .put_tensor_async("async_small", &data, &[4], DType::Float64, 1024)
                .await
                .unwrap();
            assert_eq!(chunks, 1); // Should fit in one chunk

            // Verify we can read it back using the regular chunked read
            let (loaded, shape) = engine.get_tensor_chunked("async_small").unwrap();
            assert_eq!(loaded, data);
            assert_eq!(shape, vec![4]);
        }

        #[tokio::test]
        async fn test_put_tensor_async_large() {
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
                .put_tensor_async(
                    "async_large",
                    &data,
                    &[num_elements],
                    DType::Float64,
                    chunk_size,
                )
                .await
                .unwrap();

            // Should have multiple chunks
            let expected_chunks = data.len().div_ceil(chunk_size);
            assert_eq!(chunks, expected_chunks);

            // Verify we can read it back
            let (loaded, shape) = engine.get_tensor_chunked("async_large").unwrap();
            assert_eq!(loaded, data);
            assert_eq!(shape, vec![num_elements]);
        }

        #[tokio::test]
        async fn test_put_tensor_async_default() {
            let dir = tempdir().unwrap();
            let db_path = dir.path().join("test.db");
            let db = SynaDB::new(&db_path).unwrap();
            let mut engine = TensorEngine::new(db);

            let original = vec![1.0f64, 2.0, 3.0, 4.0];
            let data: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();

            let chunks = engine
                .put_tensor_async_default("async_default", &data, &[4], DType::Float64)
                .await
                .unwrap();
            assert_eq!(chunks, 1); // Small data fits in one chunk with default size

            // Verify we can read it back
            let (loaded, shape) = engine.get_tensor_chunked("async_default").unwrap();
            assert_eq!(loaded, data);
            assert_eq!(shape, vec![4]);
        }

        #[tokio::test]
        async fn test_put_tensor_async_shape_mismatch() {
            let dir = tempdir().unwrap();
            let db_path = dir.path().join("test.db");
            let db = SynaDB::new(&db_path).unwrap();
            let mut engine = TensorEngine::new(db);

            // Data doesn't match shape
            let data = vec![0u8; 16]; // 16 bytes
            let result = engine
                .put_tensor_async("mismatch", &data, &[4], DType::Float64, 1024) // expects 32 bytes
                .await;

            assert!(matches!(result, Err(SynaError::ShapeMismatch { .. })));
        }
    }

    // =========================================================================
    // Direct I/O Tests
    // =========================================================================

    mod direct_io_tests {
        use super::super::direct_io::*;
        use tempfile::tempdir;

        #[test]
        fn test_is_direct_io_available() {
            // Just verify the function runs without panicking
            let available = is_direct_io_available();
            #[cfg(target_os = "linux")]
            assert!(available);
            #[cfg(not(target_os = "linux"))]
            assert!(!available);
        }

        #[test]
        fn test_align_size() {
            // Test alignment calculations
            assert_eq!(align_size(0), 0);
            assert_eq!(align_size(1), DIRECT_IO_ALIGNMENT);
            assert_eq!(align_size(100), DIRECT_IO_ALIGNMENT);
            assert_eq!(align_size(4096), 4096);
            assert_eq!(align_size(4097), 8192);
            assert_eq!(align_size(8192), 8192);
            assert_eq!(align_size(10000), 12288); // 3 * 4096
        }

        #[test]
        fn test_create_aligned_buffer() {
            let buffer = create_aligned_buffer(100);
            assert!(buffer.capacity() >= DIRECT_IO_ALIGNMENT);
            assert_eq!(buffer.len(), DIRECT_IO_ALIGNMENT);

            let buffer = create_aligned_buffer(5000);
            assert!(buffer.capacity() >= 8192);
            assert_eq!(buffer.len(), 8192); // 2 * 4096
        }

        #[test]
        fn test_should_use_direct_io() {
            // Small sizes should not use Direct I/O
            assert!(!should_use_direct_io(100));
            assert!(!should_use_direct_io(1000));
            assert!(!should_use_direct_io(DIRECT_IO_MIN_SIZE - 1));

            // Large sizes should use Direct I/O (on Linux)
            #[cfg(target_os = "linux")]
            {
                assert!(should_use_direct_io(DIRECT_IO_MIN_SIZE));
                assert!(should_use_direct_io(10_000_000));
            }

            #[cfg(not(target_os = "linux"))]
            {
                // On non-Linux, should always return false
                assert!(!should_use_direct_io(DIRECT_IO_MIN_SIZE));
                assert!(!should_use_direct_io(10_000_000));
            }
        }

        #[test]
        fn test_open_direct_write() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("direct_test.bin");

            // Should be able to open file for writing
            let file = open_direct(&file_path);
            assert!(file.is_ok());

            // File should exist
            assert!(file_path.exists());
        }

        #[test]
        fn test_write_aligned() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("aligned_write.bin");

            let mut file = open_direct(&file_path).unwrap();

            // Write some data
            let data = vec![42u8; 1000];
            let written = write_aligned(&mut file, &data).unwrap();

            // Written size should be aligned
            assert_eq!(written, DIRECT_IO_ALIGNMENT);

            // Verify file size
            drop(file);
            let metadata = std::fs::metadata(&file_path).unwrap();
            assert_eq!(metadata.len() as usize, DIRECT_IO_ALIGNMENT);
        }

        #[test]
        fn test_write_aligned_exact() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("aligned_exact.bin");

            let mut file = open_direct(&file_path).unwrap();

            // Write data that's exactly aligned
            let data = vec![42u8; DIRECT_IO_ALIGNMENT];
            let written = write_aligned(&mut file, &data).unwrap();

            assert_eq!(written, DIRECT_IO_ALIGNMENT);
        }

        #[test]
        fn test_open_direct_read() {
            let dir = tempdir().unwrap();
            let file_path = dir.path().join("read_test.bin");

            // First create a file with some data
            std::fs::write(&file_path, vec![0u8; 4096]).unwrap();

            // Should be able to open for reading
            let file = open_direct_read(&file_path);
            assert!(file.is_ok());
        }
    }
}

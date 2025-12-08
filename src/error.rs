//! Error types for Syna database operations.
//!
//! This module provides:
//! - [`SynaError`] - The main error enum for Rust code
//! - [`Result<T>`] - A type alias for `std::result::Result<T, SynaError>`
//! - Error code constants for FFI functions

// =============================================================================
// FFI Error Codes
// =============================================================================

/// Operation completed successfully.
pub const ERR_SUCCESS: i32 = 1;

/// Generic/unspecified error.
pub const ERR_GENERIC: i32 = 0;

/// Database not found in the global registry.
/// Call [`open_db()`](crate::open_db) first.
pub const ERR_DB_NOT_FOUND: i32 = -1;

/// Invalid path (null pointer or invalid UTF-8).
pub const ERR_INVALID_PATH: i32 = -2;

/// I/O error during file operations.
pub const ERR_IO: i32 = -3;

/// Serialization/deserialization error.
pub const ERR_SERIALIZATION: i32 = -4;

/// Key not found in the database.
pub const ERR_KEY_NOT_FOUND: i32 = -5;

/// Type mismatch (e.g., reading float from int key).
pub const ERR_TYPE_MISMATCH: i32 = -6;

/// Empty key is not allowed.
pub const ERR_EMPTY_KEY: i32 = -7;

/// Key exceeds maximum length (65535 bytes).
pub const ERR_KEY_TOO_LONG: i32 = -8;

/// Internal panic occurred (should not happen in normal operation).
pub const ERR_INTERNAL_PANIC: i32 = -100;

// =============================================================================
// Rust Error Types
// =============================================================================

/// Result type alias for Syna operations.
///
/// This is equivalent to `std::result::Result<T, SynaError>`.
pub type Result<T> = std::result::Result<T, SynaError>;

/// Comprehensive error types for Syna database operations.
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::{SynaDB, SynaError, Result};
///
/// fn example() -> Result<()> {
///     let mut db = SynaDB::new("test.db")?;
///     
///     // Empty keys are rejected
///     match db.append("", 42i64.into()) {
///         Err(SynaError::EmptyKey) => println!("Empty key rejected"),
///         _ => {}
///     }
///     
///     Ok(())
/// }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum SynaError {
    /// I/O error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization or deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Database not found in the global registry.
    #[error("Database not found: {0}")]
    NotFound(String),

    /// Invalid file path.
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Key not found in the database.
    #[error("Key not found: {0}")]
    KeyNotFound(String),

    /// Corrupted entry detected at the given file offset.
    #[error("Corrupted entry at offset {0}")]
    CorruptedEntry(u64),

    /// LZ4 decompression failed.
    #[error("Decompression failed")]
    DecompressionFailed,

    /// Empty key is not allowed.
    #[error("Empty key is not allowed")]
    EmptyKey,

    /// Key exceeds maximum length (65535 bytes).
    #[error("Key too long: {0} bytes (max 65535)")]
    KeyTooLong(usize),

    /// Invalid vector dimensions (must be 64-4096).
    #[error("Invalid dimensions: {0} (must be 64-4096)")]
    InvalidDimensions(u16),

    /// Vector dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected number of dimensions
        expected: u16,
        /// Actual number of dimensions provided
        got: u16,
    },

    /// Corrupted HNSW index file.
    #[error("Corrupted HNSW index: {0}")]
    CorruptedIndex(String),

    /// Shape mismatch in tensor operations.
    #[error("Shape mismatch: data size {data_size} does not match expected size {expected_size} for shape")]
    ShapeMismatch {
        /// Actual data size in bytes
        data_size: usize,
        /// Expected data size in bytes
        expected_size: usize,
    },

    /// Type conversion error in tensor operations.
    #[error("Type conversion error: cannot convert {from_type} to {to_type}")]
    TypeConversion {
        /// Source type name
        from_type: &'static str,
        /// Target type name
        to_type: &'static str,
    },

    /// Model not found in the registry.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Checksum mismatch when loading a model.
    #[error("Checksum mismatch: expected {expected}, got {got}")]
    ChecksumMismatch {
        /// Expected checksum from metadata
        expected: String,
        /// Computed checksum from loaded data
        got: String,
    },

    /// Experiment run not found.
    #[error("Run not found: {0}")]
    RunNotFound(String),

    /// Experiment run has already ended.
    #[error("Run already ended: {0}")]
    RunAlreadyEnded(String),
}

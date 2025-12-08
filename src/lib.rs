//! # Syna DB
//!
//! An embedded, log-structured, columnar-mapped database engine written in Rust.
//!
//! Syna synthesizes the embedded simplicity of SQLite, the columnar analytical
//! speed of DuckDB, and the schema flexibility of MongoDB. It exposes a C-ABI for
//! polyglot integration and is optimized for high-throughput time-series data with
//! AI/ML tensor mapping capabilities.
//!
//! ## Features
//!
//! - **Append-only log structure** - Fast sequential writes, immutable history
//! - **Schema-free** - Store heterogeneous data types without migrations
//! - **AI/ML optimized** - Extract time-series data as contiguous tensors
//! - **C-ABI interface** - Use from Python, Node.js, C++, or any FFI-capable language
//! - **Delta & LZ4 compression** - Minimize storage for time-series data
//! - **Crash recovery** - Automatic index rebuild on open
//! - **Thread-safe** - Concurrent read/write access
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use synadb::{SynaDB, Atom, Result};
//!
//! fn main() -> Result<()> {
//!     // Open or create a database
//!     let mut db = SynaDB::new("my_data.db")?;
//!
//!     // Write different data types
//!     db.append("temperature", Atom::Float(23.5))?;
//!     db.append("count", Atom::Int(42))?;
//!     db.append("name", Atom::Text("sensor-1".to_string()))?;
//!
//!     // Read values back
//!     if let Some(temp) = db.get("temperature")? {
//!         println!("Temperature: {:?}", temp);
//!     }
//!
//!     // Extract history as tensor for ML
//!     let history = db.get_history_floats("temperature")?;
//!     println!("History: {:?}", history);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! Syna uses an append-only log structure. Each entry consists of:
//! - A fixed-size [`LogHeader`] (15 bytes) with timestamp, lengths, and flags
//! - The key as UTF-8 bytes
//! - The value serialized with bincode
//!
//! An in-memory index maps keys to file offsets for O(1) lookups.

pub mod compression;
pub mod distance;
pub mod engine;
pub mod error;
pub mod experiment;
pub mod ffi;
pub mod hnsw;
pub mod model_registry;
pub mod tensor;
pub mod types;
pub mod vector;

// Re-export commonly used types
pub use engine::{close_db, free_tensor, open_db, with_db, DbConfig, SynaDB};
pub use error::{Result, SynaError};
pub use types::{Atom, LogHeader, HEADER_SIZE, IS_COMPRESSED, IS_DELTA, IS_TOMBSTONE};

// Re-export tensor types for high-throughput operations
pub use tensor::{DType, TensorEngine, TensorMeta, DEFAULT_CHUNK_SIZE};

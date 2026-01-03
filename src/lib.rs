// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! # SynaDB
//!
//! An AI-native embedded database written in Rust.
//!
//! SynaDB synthesizes the embedded simplicity of SQLite, the columnar analytical
//! speed of DuckDB, and the schema flexibility of MongoDB. It exposes a C-ABI for
//! polyglot integration and is optimized for AI/ML workloads including vector search,
//! tensor operations, model versioning, and experiment tracking.
//!
//! ## Features
//!
//! ### Core Database
//! - **Append-only log structure** - Fast sequential writes, immutable history
//! - **Schema-free** - Store heterogeneous data types without migrations
//! - **Delta & LZ4 compression** - Minimize storage for time-series data
//! - **Crash recovery** - Automatic index rebuild on open
//! - **Thread-safe** - Concurrent read/write access with mutex-protected writes
//!
//! ### Vector Search
//! - **[`vector::VectorStore`]** - Embedding storage with similarity search
//! - **[`mmap_vector::MmapVectorStore`]** - Ultra-high-throughput vector storage (490K vectors/sec)
//! - **[`hnsw::HnswIndex`]** - O(log N) approximate nearest neighbor search
//! - **[`gwi::GravityWellIndex`]** - Novel O(N) build time index (168x faster than HNSW)
//! - **[`cascade::CascadeIndex`]** - Three-stage hybrid index (LSH + bucket tree + graph)
//! - **[`distance`]** - Cosine, Euclidean, and Dot Product metrics
//!
//! ### AI/ML Platform
//! - **[`tensor::TensorEngine`]** - Batch tensor operations with chunked storage
//! - **[`model_registry::ModelRegistry`]** - Version models with SHA-256 checksum verification
//! - **[`experiment::ExperimentTracker`]** - Log parameters, metrics, and artifacts
//!
//! ### Performance
//! - **[`mmap::MmapReader`]** - Memory-mapped I/O for zero-copy reads
//! - **[`gpu::GpuContext`]** - GPU Direct memory access (optional `gpu` feature)
//! - **[`faiss_index`](faiss_index/index.html)** - FAISS integration for billion-scale search (optional `faiss` feature)
//!
//! ### FFI
//! - **[`ffi`]** - C-ABI interface for Python, Node.js, C++, or any FFI-capable language
//!
//! ## Architecture Philosophy
//!
//! SynaDB uses a **modular architecture** where each component is optimized for its workload:
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`SynaDB`] | Core key-value store with history |
//! | [`vector::VectorStore`] | Embedding storage with HNSW search |
//! | [`mmap_vector::MmapVectorStore`] | High-throughput vector ingestion |
//! | [`gwi::GravityWellIndex`] | Fast-build vector index |
//! | [`cascade::CascadeIndex`] | Hybrid three-stage index |
//! | [`tensor::TensorEngine`] | Batch tensor operations |
//! | [`model_registry::ModelRegistry`] | Model versioning with checksums |
//! | [`experiment::ExperimentTracker`] | Experiment tracking |
//!
//! ## Quick Start
//!
//! ### Core Database
//!
//! ```rust,no_run
//! use synadb::{SynaDB, Atom, Result};
//!
//! fn main() -> Result<()> {
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
//!     // Build history for ML
//!     db.append("temperature", Atom::Float(24.1))?;
//!     db.append("temperature", Atom::Float(24.8))?;
//!
//!     // Extract as tensor
//!     let history = db.get_history_floats("temperature")?;
//!     println!("History: {:?}", history); // [23.5, 24.1, 24.8]
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Vector Store
//!
//! ```rust,no_run
//! use synadb::vector::{VectorStore, VectorConfig, SearchResult};
//! use synadb::distance::DistanceMetric;
//!
//! fn main() -> synadb::Result<()> {
//!     let config = VectorConfig {
//!         dimensions: 768,
//!         metric: DistanceMetric::Cosine,
//!         ..Default::default()
//!     };
//!     let mut store = VectorStore::new("vectors.db", config)?;
//!
//!     // Insert embeddings
//!     let embedding = vec![0.1f32; 768];
//!     store.insert("doc1", &embedding)?;
//!
//!     // Search for similar vectors
//!     let query = vec![0.1f32; 768];
//!     let results = store.search(&query, 10)?;
//!     for r in results {
//!         println!("{}: {:.4}", r.key, r.score);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### High-Throughput Vector Ingestion
//!
//! ```rust,no_run
//! use synadb::mmap_vector::{MmapVectorStore, MmapVectorConfig};
//! use synadb::distance::DistanceMetric;
//!
//! fn main() -> synadb::Result<()> {
//!     let config = MmapVectorConfig {
//!         dimensions: 768,
//!         metric: DistanceMetric::Cosine,
//!         initial_capacity: 100_000,
//!         ..Default::default()
//!     };
//!     let mut store = MmapVectorStore::new("vectors.mmap", config)?;
//!
//!     // Insert vectors one at a time or in batches
//!     let embedding = vec![0.1f32; 768];
//!     store.insert("doc_0", &embedding)?;
//!
//!     // Batch insert (490K vectors/sec)
//!     let keys: Vec<&str> = vec!["doc_1", "doc_2", "doc_3"];
//!     let vecs: Vec<Vec<f32>> = vec![vec![0.1f32; 768]; 3];
//!     let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
//!     store.insert_batch(&keys, &vec_refs)?;
//!
//!     // Build index for fast search
//!     store.build_index()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Model Registry
//!
//! ```rust,no_run
//! use synadb::model_registry::{ModelRegistry, ModelStage};
//! use std::collections::HashMap;
//!
//! fn main() -> synadb::Result<()> {
//!     let mut registry = ModelRegistry::new("models.db")?;
//!
//!     // Save model with metadata
//!     let model_data = vec![0u8; 1024];
//!     let mut metadata = HashMap::new();
//!     metadata.insert("accuracy".to_string(), "0.95".to_string());
//!     let version = registry.save_model("classifier", &model_data, metadata)?;
//!     println!("Saved v{} with checksum {}", version.version, version.checksum);
//!
//!     // Load with checksum verification
//!     let (data, info) = registry.load_model("classifier", None)?;
//!
//!     // Promote to production
//!     registry.set_stage("classifier", version.version, ModelStage::Production)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Experiment Tracking
//!
//! ```rust,no_run
//! use synadb::experiment::{ExperimentTracker, RunStatus};
//!
//! fn main() -> synadb::Result<()> {
//!     let mut tracker = ExperimentTracker::new("experiments.db")?;
//!
//!     // Start a run
//!     let run_id = tracker.start_run("mnist", vec!["baseline".to_string()])?;
//!
//!     // Log parameters and metrics
//!     tracker.log_param(&run_id, "learning_rate", "0.001")?;
//!     for epoch in 0..100 {
//!         let loss = 1.0 / (epoch + 1) as f64;
//!         tracker.log_metric(&run_id, "loss", loss, Some(epoch as u64))?;
//!     }
//!
//!     // End run
//!     tracker.end_run(&run_id, RunStatus::Completed)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `gpu` | Enable GPU Direct memory access (requires CUDA) |
//! | `faiss` | Enable FAISS integration for billion-scale search |
//! | `async` | Enable async runtime for parallel operations |
//!
//! ## Storage Architecture
//!
//! SynaDB uses an append-only log structure. Each entry consists of:
//! - A fixed-size [`LogHeader`] (15 bytes) with timestamp, lengths, and flags
//! - The key as UTF-8 bytes
//! - The value serialized with bincode
//!
//! An in-memory index maps keys to file offsets for O(1) lookups.
//!
//! An in-memory index maps keys to file offsets for O(1) lookups.

pub mod cascade;
pub mod compression;
pub mod distance;
pub mod engine;
pub mod error;
pub mod experiment;
#[cfg(feature = "faiss")]
pub mod faiss_index;
pub mod ffi;
pub mod gpu;
pub mod gwi;
pub mod hnsw;
pub mod mmap;
pub mod mmap_vector;
pub mod model_registry;
pub mod tensor;
pub mod types;
pub mod vector;

// Re-export commonly used types
pub use engine::{close_db, free_tensor, open_db, with_db, DbConfig, SynaDB};
pub use error::{Result, SynaError};
pub use types::{Atom, LogHeader, HEADER_SIZE, IS_COMPRESSED, IS_DELTA, IS_TOMBSTONE};

// Re-export tensor types for high-throughput operations
pub use tensor::{
    optimal_chunk_size, DType, MmapTensorMeta, MmapTensorRef, TensorEngine, TensorMeta,
    CHUNK_SIZE_LARGE, CHUNK_SIZE_MEDIUM, CHUNK_SIZE_SMALL, DEFAULT_CHUNK_SIZE,
};

// Re-export memory-mapped reader for zero-copy access
pub use mmap::MmapReader;

// Re-export mmap vector store for ultra-high-throughput writes
pub use mmap_vector::{MmapSearchResult, MmapVectorConfig, MmapVectorStore};

// Re-export GPU types for direct memory access
pub use gpu::{GpuContext, GpuTensor};

// Re-export Cascade Index for fast O(N) build time
pub use cascade::{CascadeConfig, CascadeIndex};

// Re-export vector store types
pub use vector::{SearchResult, VectorConfig, VectorStore};

// Re-export HNSW index types
pub use hnsw::{HnswConfig, HnswIndex};

// Re-export Gravity Well Index
pub use gwi::GravityWellIndex;

// Re-export model registry types
pub use model_registry::{ModelRegistry, ModelStage, ModelVersion};

// Re-export experiment tracking types
pub use experiment::{ExperimentTracker, Run, RunStatus};

// Re-export distance metrics
pub use distance::DistanceMetric;

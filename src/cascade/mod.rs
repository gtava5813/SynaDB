//! Cascade Index: Fast vector index with O(N) build time
//!
//! Cascade Index combines Locality-Sensitive Hashing (LSH) with adaptive buckets
//! and a sparse graph to achieve O(N) build time while maintaining high recall.
//!
//! # Two Implementations
//!
//! ## MmapCascadeIndex (Recommended)
//!
//! Optimized implementation following SynaDB physics principles:
//! - **Arrow of Time**: Append-only writes (vectors, edges, buckets)
//! - **The Observer**: Zero-copy mmap reads
//! - Faster search, better memory efficiency
//!
//! ```rust,ignore
//! use synadb::cascade::{MmapCascadeIndex, MmapCascadeConfig};
//!
//! let config = MmapCascadeConfig::default();
//! let mut index = MmapCascadeIndex::new("vectors", config)?;
//!
//! index.insert("doc1", &embedding)?;
//! let results = index.search(&query, 10)?;
//! ```
//!
//! ## CascadeIndex (Original)
//!
//! Original implementation with bucket tree splits:
//! - Higher recall (100% vs 85-90%)
//! - More complex code
//!
//! ```rust,ignore
//! use synadb::cascade::{CascadeIndex, CascadeConfig};
//!
//! let config = CascadeConfig::default();
//! let mut index = CascadeIndex::new("vectors.cascade", config)?;
//!
//! index.insert("doc1", &embedding)?;
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Benchmark Comparison
//!
//! | Implementation | Write/sec | Search p50 | Recall |
//! |----------------|-----------|------------|--------|
//! | MmapCascadeIndex | 7,886 | 0.45ms | 90% |
//! | CascadeIndex | 9,619 | 1.10ms | 100% |

// Original implementation
mod bucket;
mod config;
mod graph;
mod index;
mod lsh;

// New mmap-based implementation (SynaDB physics principles)
mod append_graph;
mod mmap_index;
mod mmap_store;
mod simple_lsh;

// Original exports
pub use bucket::{BucketConfig, BucketForest};
pub use config::CascadeConfig;
pub use graph::{CascadeGraph, GraphConfig};
pub use index::{CascadeIndex, SearchResult};
pub use lsh::HyperplaneLSH;

// New mmap-based exports (recommended)
pub use append_graph::{AppendGraph, AppendGraphConfig};
pub use mmap_index::{MmapCascadeConfig, MmapCascadeIndex, MmapCascadeStats, MmapSearchResult};
pub use mmap_store::{MmapStoreConfig, MmapVectorStorage};
pub use simple_lsh::{SimpleLSH, SimpleLSHConfig};

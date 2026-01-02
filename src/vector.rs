// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Vector store for embedding storage and similarity search.
//!
//! This module provides a high-level API for storing and searching vector embeddings,
//! built on top of the core SynaDB engine.
//!
//! # Features
//!
//! - Store vectors with dimensions from 64 to 8192
//! - Brute-force k-nearest neighbor search for small datasets
//! - HNSW index for fast approximate nearest neighbor search on large datasets
//! - Support for cosine, euclidean, and dot product distance metrics
//! - Automatic dimension validation
//! - Automatic index building when vector count exceeds threshold
//!
//! # Example
//!
//! ```rust,no_run
//! use synadb::vector::{VectorStore, VectorConfig};
//! use synadb::distance::DistanceMetric;
//!
//! // Create a vector store with 768 dimensions (BERT-sized)
//! let config = VectorConfig {
//!     dimensions: 768,
//!     metric: DistanceMetric::Cosine,
//!     ..Default::default()
//! };
//!
//! let mut store = VectorStore::new("vectors.db", config).unwrap();
//!
//! // Insert vectors
//! let embedding = vec![0.1f32; 768];
//! store.insert("doc1", &embedding).unwrap();
//!
//! // Search for similar vectors (uses HNSW if available, else brute force)
//! let query = vec![0.1f32; 768];
//! let results = store.search(&query, 5).unwrap();
//! ```

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::distance::DistanceMetric;
use crate::engine::SynaDB;
use crate::error::{Result, SynaError};
#[cfg(feature = "faiss")]
use crate::faiss_index::FaissConfig;
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::types::Atom;

/// Configuration for a vector store.
///
/// # Example
///
/// ```rust
/// use synadb::vector::{VectorConfig, IndexBackend};
/// use synadb::distance::DistanceMetric;
///
/// let config = VectorConfig {
///     dimensions: 768,
///     metric: DistanceMetric::Cosine,
///     key_prefix: "embeddings/".to_string(),
///     index_threshold: 10000,
///     backend: IndexBackend::default(),
///     sync_on_write: true,
///     checkpoint_interval_secs: 30,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Number of dimensions (64-8192).
    /// Common values: 384 (MiniLM), 768 (BERT), 1536 (OpenAI ada-002), 7168 (DeepSeek-V3).
    pub dimensions: u16,
    /// Distance metric for similarity search.
    /// Lower distance = more similar for all metrics.
    pub metric: DistanceMetric,
    /// Key prefix for vectors in the underlying database.
    /// Default: "vec/"
    pub key_prefix: String,
    /// Number of vectors at which to automatically build an HNSW index.
    /// Set to 0 to disable automatic indexing.
    /// Default: 10000
    pub index_threshold: usize,
    /// Index backend selection for similarity search.
    /// Default: HNSW with default configuration.
    ///
    /// **Requirements:** 1.5
    pub backend: IndexBackend,
    /// Sync to disk after every write operation.
    /// When `true` (default), each write calls `fsync()` for durability.
    /// Set to `false` for higher throughput at the risk of data loss on crash.
    pub sync_on_write: bool,
    /// Interval in seconds between automatic index checkpoints.
    /// Set to 0 to disable automatic checkpoints (save only on close).
    /// Default: 30 seconds
    pub checkpoint_interval_secs: u64,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            key_prefix: "vec/".to_string(),
            index_threshold: 10000,
            backend: IndexBackend::default(),
            sync_on_write: true,
            checkpoint_interval_secs: 30,
        }
    }
}

/// Index backend selection for vector similarity search.
///
/// This enum allows choosing between different indexing strategies:
/// - HNSW (default): Built-in hierarchical navigable small world graph
/// - FAISS: High-performance library for billion-scale search (requires 'faiss' feature)
/// - None: Brute-force search only (no index)
///
/// # Example
///
/// ```rust
/// use synadb::vector::IndexBackend;
/// use synadb::hnsw::HnswConfig;
///
/// // Use default HNSW index
/// let backend = IndexBackend::default();
///
/// // Use custom HNSW configuration
/// let backend = IndexBackend::Hnsw(HnswConfig {
///     m: 32,
///     ef_construction: 400,
///     ..Default::default()
/// });
///
/// // Disable indexing (brute-force only)
/// let backend = IndexBackend::None;
/// ```
///
/// **Requirements:** 1.5
#[derive(Debug, Clone)]
pub enum IndexBackend {
    /// Built-in HNSW implementation (default).
    /// Provides O(log N) approximate nearest neighbor search.
    Hnsw(HnswConfig),
    /// FAISS-backed index (requires 'faiss' feature).
    /// Supports billion-scale search and GPU acceleration.
    #[cfg(feature = "faiss")]
    Faiss(FaissConfig),
    /// No index (brute-force only).
    /// Uses O(N) linear scan for all searches.
    None,
}

impl Default for IndexBackend {
    fn default() -> Self {
        IndexBackend::Hnsw(HnswConfig::default())
    }
}

/// Result of a similarity search.
///
/// Results are sorted by score (ascending), so lower scores indicate
/// more similar vectors.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Key of the matching vector (without the prefix).
    pub key: String,
    /// Distance/similarity score (lower = more similar).
    pub score: f32,
    /// The vector data.
    pub vector: Vec<f32>,
}

/// Vector store for embedding storage and similarity search.
///
/// `VectorStore` provides a high-level API for storing and searching
/// vector embeddings. It wraps a `SynaDB` instance and adds:
///
/// - Dimension validation
/// - Brute-force k-nearest neighbor search for small datasets
/// - HNSW index for fast approximate nearest neighbor search on large datasets
/// - FAISS index for billion-scale search and GPU acceleration (requires 'faiss' feature)
/// - Key prefixing for namespace isolation
/// - Automatic index building when vector count exceeds threshold
/// - O(1) key existence checking via HashSet
/// - Checkpoint-based index persistence for high throughput
///
/// # Example
///
/// ```rust,no_run
/// use synadb::vector::{VectorStore, VectorConfig};
///
/// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
///
/// // Insert a 768-dimensional vector
/// let embedding = vec![0.1f32; 768];
/// store.insert("doc1", &embedding).unwrap();
///
/// // Search for 5 nearest neighbors (uses HNSW if available)
/// let results = store.search(&embedding, 5).unwrap();
/// for r in results {
///     println!("{}: {:.4}", r.key, r.score);
/// }
/// ```
///
/// **Requirements:** 1.5, 1.7
pub struct VectorStore {
    /// Underlying database instance.
    db: SynaDB,
    /// Path to the database file (used for index file paths).
    db_path: std::path::PathBuf,
    /// Configuration for this vector store.
    config: VectorConfig,
    /// Cached vector keys for search (includes prefix) - O(1) lookup via HashSet.
    vector_keys: HashSet<String>,
    /// Ordered list of keys for iteration (maintains insertion order for brute-force search).
    vector_keys_ordered: Vec<String>,
    /// HNSW index for fast approximate nearest neighbor search.
    /// Built automatically when vector count exceeds `config.index_threshold`.
    hnsw_index: Option<HnswIndex>,
    /// FAISS index for billion-scale search (requires 'faiss' feature).
    /// Used when `config.backend` is `IndexBackend::Faiss`.
    #[cfg(feature = "faiss")]
    faiss_index: Option<crate::faiss_index::FaissIndex>,
    /// Whether the index has unsaved changes.
    index_dirty: bool,
    /// Last time the index was checkpointed.
    last_checkpoint: Instant,
    /// Checkpoint interval from config.
    checkpoint_interval: Duration,
}

impl VectorStore {
    /// Creates or opens a vector store at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    /// * `config` - Configuration for the vector store
    ///
    /// # Errors
    ///
    /// * `SynaError::InvalidDimensions` - If dimensions are not in range 64-4096
    /// * `SynaError::Io` - If the database file cannot be opened/created
    /// * `SynaError::IndexError` - If FAISS index creation fails (when using FAISS backend)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// ```
    ///
    /// # Backend Selection
    ///
    /// The backend is determined by `config.backend`:
    /// - `IndexBackend::Hnsw(config)` - Uses built-in HNSW index (default)
    /// - `IndexBackend::Faiss(config)` - Uses FAISS index (requires 'faiss' feature)
    /// - `IndexBackend::None` - Brute-force search only
    ///
    /// **Requirements:** 1.5
    pub fn new<P: AsRef<Path>>(path: P, config: VectorConfig) -> Result<Self> {
        // Validate dimensions (64-8192, supports up to DeepSeek-V3's 7168)
        if config.dimensions < 64 || config.dimensions > 8192 {
            return Err(SynaError::InvalidDimensions(config.dimensions));
        }

        // Store the path for index file operations
        let db_path = path.as_ref().to_path_buf();

        // Create database with sync_on_write config
        let db_config = crate::engine::DbConfig {
            sync_on_write: config.sync_on_write,
            ..Default::default()
        };
        let db = SynaDB::with_config(&db_path, db_config)?;

        // Load existing vector keys from the database into both HashSet and Vec
        let vector_keys_ordered: Vec<String> = db
            .keys()
            .into_iter()
            .filter(|k| k.starts_with(&config.key_prefix))
            .collect();
        let vector_keys: HashSet<String> = vector_keys_ordered.iter().cloned().collect();

        // Try to load existing HNSW index if backend supports it
        let hnsw_index = match &config.backend {
            IndexBackend::Hnsw(_) | IndexBackend::None => {
                let hnsw_path = Self::hnsw_index_path(&db_path);
                if hnsw_path.exists() {
                    // Try to load existing index, validating dimensions and metric
                    match HnswIndex::load_validated(&hnsw_path, config.dimensions, config.metric) {
                        Ok(index) => {
                            // Verify index has expected number of vectors
                            if index.len() == vector_keys.len() {
                                Some(index)
                            } else {
                                // Index is stale, will rebuild if needed
                                None
                            }
                        }
                        Err(_) => None, // Index corrupted or incompatible, will rebuild
                    }
                } else {
                    None
                }
            }
            #[cfg(feature = "faiss")]
            IndexBackend::Faiss(_) => None, // FAISS manages its own index
        };

        // Initialize FAISS index if configured (feature-gated)
        #[cfg(feature = "faiss")]
        let faiss_index = match &config.backend {
            IndexBackend::Faiss(faiss_config) => Some(crate::faiss_index::FaissIndex::new(
                config.dimensions,
                config.metric,
                faiss_config.clone(),
            )?),
            _ => None,
        };

        // Set up checkpoint interval
        let checkpoint_interval = Duration::from_secs(config.checkpoint_interval_secs);

        Ok(Self {
            db,
            db_path,
            config,
            vector_keys,
            vector_keys_ordered,
            hnsw_index,
            #[cfg(feature = "faiss")]
            faiss_index,
            index_dirty: false,
            last_checkpoint: Instant::now(),
            checkpoint_interval,
        })
    }

    /// Returns the path to the HNSW index file for a given database path.
    fn hnsw_index_path(db_path: &Path) -> std::path::PathBuf {
        let mut hnsw_path = db_path.to_path_buf();
        let extension = match hnsw_path.extension() {
            Some(ext) => format!("{}.hnsw", ext.to_string_lossy()),
            None => "hnsw".to_string(),
        };
        hnsw_path.set_extension(extension);
        hnsw_path
    }

    /// Inserts a vector with the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - Unique identifier for the vector (without prefix)
    /// * `vector` - The vector data (must match configured dimensions)
    ///
    /// # Errors
    ///
    /// * `SynaError::DimensionMismatch` - If vector length doesn't match configured dimensions
    /// * `SynaError::Io` - If the write fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// let embedding = vec![0.1f32; 768];
    /// store.insert("doc1", &embedding).unwrap();
    /// ```
    pub fn insert(&mut self, key: &str, vector: &[f32]) -> Result<()> {
        // Validate dimensions
        if vector.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len() as u16,
            });
        }

        let full_key = format!("{}{}", self.config.key_prefix, key);
        let atom = Atom::Vector(vector.to_vec(), self.config.dimensions);

        self.db.append(&full_key, atom)?;

        // O(1) key existence check via HashSet
        let is_new_key = self.vector_keys.insert(full_key.clone());
        if is_new_key {
            self.vector_keys_ordered.push(full_key.clone());
        }

        // If we have an HNSW index, add the new vector incrementally using search-based insertion
        if self.hnsw_index.is_some() && is_new_key {
            // Use true incremental insert with O(log N) neighbor finding
            self.insert_to_hnsw_incremental(&full_key, vector);
            self.index_dirty = true;
        } else if self.hnsw_index.is_none() {
            // Check if we should build the index (auto-build when threshold reached)
            // Only build for HNSW backend when threshold is configured
            if matches!(self.config.backend, IndexBackend::Hnsw(_))
                && self.config.index_threshold > 0
                && self.vector_keys.len() >= self.config.index_threshold
            {
                self.build_index()?;
            }
        }

        // Checkpoint if interval elapsed and we have dirty changes
        if self.index_dirty 
            && self.checkpoint_interval.as_secs() > 0
            && self.last_checkpoint.elapsed() >= self.checkpoint_interval 
        {
            self.checkpoint_index()?;
        }

        Ok(())
    }

    /// Inserts multiple vectors in a single batch operation.
    /// 
    /// This is significantly faster than calling `insert()` in a loop because:
    /// - Single lock acquisition for all inserts
    /// - Deferred index building until after all vectors are inserted
    /// - Reduced FFI overhead when called from Python
    ///
    /// # Arguments
    ///
    /// * `keys` - Slice of key strings
    /// * `vectors` - Slice of vector slices (each must match configured dimensions)
    ///
    /// # Returns
    ///
    /// Number of vectors successfully inserted.
    ///
    /// # Errors
    ///
    /// * `SynaError::DimensionMismatch` - If any vector length doesn't match configured dimensions
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// let keys = vec!["doc1", "doc2", "doc3"];
    /// let embeddings: Vec<Vec<f32>> = vec![vec![0.1f32; 768]; 3];
    /// let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
    /// let count = store.insert_batch(&keys, &refs).unwrap();
    /// ```
    pub fn insert_batch(&mut self, keys: &[&str], vectors: &[&[f32]]) -> Result<usize> {
        if keys.len() != vectors.len() {
            return Err(SynaError::ShapeMismatch {
                data_size: vectors.len(),
                expected_size: keys.len(),
            });
        }

        let mut inserted = 0;
        let should_build_index = self.hnsw_index.is_none()
            && matches!(self.config.backend, IndexBackend::Hnsw(_))
            && self.config.index_threshold > 0;

        // Phase 1: Insert all vectors to storage (fast, sequential I/O)
        for (key, vector) in keys.iter().zip(vectors.iter()) {
            // Validate dimensions
            if vector.len() != self.config.dimensions as usize {
                return Err(SynaError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vector.len() as u16,
                });
            }

            let full_key = format!("{}{}", self.config.key_prefix, key);
            let atom = Atom::Vector(vector.to_vec(), self.config.dimensions);

            self.db.append(&full_key, atom)?;

            // O(1) key existence check via HashSet
            let is_new_key = self.vector_keys.insert(full_key.clone());
            if is_new_key {
                self.vector_keys_ordered.push(full_key.clone());
                inserted += 1;
            }
        }

        // Phase 2: Build or update index after all inserts
        if should_build_index && self.vector_keys.len() >= self.config.index_threshold {
            // Build index once after batch insert
            self.build_index()?;
        }
        // Note: When index exists, we skip incremental updates in batch mode
        // for maximum write throughput. Call build_index() to rebuild after bulk inserts.

        // Checkpoint if needed
        if self.index_dirty 
            && self.checkpoint_interval.as_secs() > 0
            && self.last_checkpoint.elapsed() >= self.checkpoint_interval 
        {
            self.checkpoint_index()?;
        }

        Ok(inserted)
    }

    /// Inserts multiple vectors with option to skip index updates.
    /// 
    /// This is the fastest way to bulk-load vectors. When `update_index` is false,
    /// vectors are written to storage but not added to the HNSW index. Call
    /// `build_index()` after all inserts to rebuild the index.
    ///
    /// # Arguments
    ///
    /// * `keys` - Slice of key strings
    /// * `vectors` - Slice of vector slices
    /// * `update_index` - If false, skip index updates for maximum write speed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// 
    /// // Bulk load without index updates (150K+/sec)
    /// let keys = vec!["doc1", "doc2"];
    /// let v1 = vec![0.1f32; 768];
    /// let v2 = vec![0.2f32; 768];
    /// let vectors: Vec<&[f32]> = vec![&v1, &v2];
    /// store.insert_batch_fast(&keys, &vectors, false).unwrap();
    /// 
    /// // Rebuild index once at the end
    /// store.build_index().unwrap();
    /// ```
    pub fn insert_batch_fast(&mut self, keys: &[&str], vectors: &[&[f32]], update_index: bool) -> Result<usize> {
        if keys.len() != vectors.len() {
            return Err(SynaError::ShapeMismatch {
                data_size: vectors.len(),
                expected_size: keys.len(),
            });
        }

        let mut inserted = 0;

        // Phase 1: Insert all vectors to storage (fast, sequential I/O)
        for (key, vector) in keys.iter().zip(vectors.iter()) {
            if vector.len() != self.config.dimensions as usize {
                return Err(SynaError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vector.len() as u16,
                });
            }

            let full_key = format!("{}{}", self.config.key_prefix, key);
            let atom = Atom::Vector(vector.to_vec(), self.config.dimensions);

            self.db.append(&full_key, atom)?;

            let is_new_key = self.vector_keys.insert(full_key.clone());
            if is_new_key {
                self.vector_keys_ordered.push(full_key);
                inserted += 1;
            }
        }

        // Phase 2: Optionally update index
        if update_index {
            let should_build = self.hnsw_index.is_none()
                && matches!(self.config.backend, IndexBackend::Hnsw(_))
                && self.config.index_threshold > 0
                && self.vector_keys.len() >= self.config.index_threshold;

            if should_build {
                self.build_index()?;
            }
        }

        Ok(inserted)
    }

    /// Inserts a vector into the HNSW index using search-based neighbor finding.
    /// This is O(log N) instead of O(N) because it uses the HNSW search to find neighbors.
    fn insert_to_hnsw_incremental(&mut self, key: &str, vector: &[f32]) {
        use crate::hnsw::HnswNode;

        let index = match self.hnsw_index.as_mut() {
            Some(idx) => idx,
            None => return,
        };

        // Skip if key already exists in index
        if index.key_to_id.contains_key(key) {
            return;
        }

        // Generate a random level for this node
        let level = index.random_level();

        // Create the node
        let node = HnswNode::new(key.to_string(), vector.to_vec(), level);
        let node_id = index.nodes.len();

        // Add to index structures
        index.nodes.push(node);
        index.key_to_id.insert(key.to_string(), node_id);

        // If this is the first node, just set it as entry point
        if node_id == 0 {
            index.entry_point = Some(node_id);
            return;
        }

        // Get HNSW config
        let m = index.config().m;
        let m_max = index.config().m_max;
        let ef_construction = index.config().ef_construction;

        // Start from entry point
        let mut ep = index.entry_point.unwrap_or(0);

        // Descend from top level to level+1, finding closest node at each level
        let current_max_level = index.max_level();
        for lc in ((level + 1)..=current_max_level).rev() {
            let results = index.search_layer(vector, ep, 1, lc);
            if !results.is_empty() {
                ep = results[0].0;
            }
        }

        // For each level from min(level, max_level) down to 0, find and connect neighbors
        let start_level = level.min(current_max_level);
        for l in (0..=start_level).rev() {
            // Use HNSW search to find neighbors at this level - O(log N)!
            let candidates = index.search_layer(vector, ep, ef_construction, l);
            
            // Select M best neighbors
            let max_neighbors = if l == 0 { m } else { m_max };
            let neighbors: Vec<(usize, f32)> = candidates
                .into_iter()
                .take(max_neighbors)
                .collect();

            // Update entry point for next level
            if !neighbors.is_empty() {
                ep = neighbors[0].0;
            }

            // Add connections to this node
            if l < index.nodes[node_id].neighbors.len() {
                index.nodes[node_id].neighbors[l] = neighbors.clone();
            }

            // Add bidirectional connections
            for (neighbor_id, dist) in neighbors {
                if l < index.nodes[neighbor_id].neighbors.len() {
                    index.nodes[neighbor_id].neighbors[l].push((node_id, dist));
                    
                    // Prune if too many neighbors
                    if index.nodes[neighbor_id].neighbors[l].len() > max_neighbors {
                        index.nodes[neighbor_id].neighbors[l].sort_by(|a, b| {
                            a.1.total_cmp(&b.1)
                        });
                        index.nodes[neighbor_id].neighbors[l].truncate(max_neighbors);
                    }
                }
            }
        }

        // Update entry point if this node has a higher level
        if level > current_max_level || index.entry_point.is_none() {
            index.entry_point = Some(node_id);
        }
    }

    /// Checkpoints the HNSW index to disk.
    /// Called automatically based on checkpoint_interval, or manually.
    pub fn checkpoint_index(&mut self) -> Result<()> {
        if !self.index_dirty {
            return Ok(());
        }

        if let Some(ref index) = self.hnsw_index {
            let hnsw_path = Self::hnsw_index_path(&self.db_path);
            index.save(&hnsw_path)?;
        }

        self.index_dirty = false;
        self.last_checkpoint = Instant::now();
        Ok(())
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// The search method is determined by the configured backend:
    /// - `IndexBackend::Hnsw` - Uses HNSW index if available and vector count exceeds threshold
    /// - `IndexBackend::Faiss` - Uses FAISS index (requires 'faiss' feature)
    /// - `IndexBackend::None` - Always uses brute-force search
    ///
    /// Results are sorted by score (ascending), so lower scores indicate
    /// more similar vectors.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match configured dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// * `SynaError::DimensionMismatch` - If query length doesn't match configured dimensions
    /// * `SynaError::Io` - If reading vectors fails
    /// * `SynaError::IndexError` - If FAISS search fails (when using FAISS backend)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// let query = vec![0.1f32; 768];
    /// let results = store.search(&query, 5).unwrap();
    /// ```
    ///
    /// **Requirements:** 1.5, 1.7
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Validate query dimensions
        if query.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len() as u16,
            });
        }

        // Select search method based on backend configuration
        match &self.config.backend {
            // FAISS backend (feature-gated)
            #[cfg(feature = "faiss")]
            IndexBackend::Faiss(_) => {
                if self.faiss_index.is_some() {
                    return self.search_faiss(query, k);
                }
                // Fall back to brute force if FAISS index not initialized
                self.search_brute_force(query, k)
            }

            // HNSW backend (default)
            IndexBackend::Hnsw(_) => {
                // Use HNSW if we have an index and enough vectors
                if self.hnsw_index.is_some()
                    && self.config.index_threshold > 0
                    && self.vector_keys.len() >= self.config.index_threshold
                {
                    return self.search_hnsw(query, k);
                }
                // Fall back to brute force
                self.search_brute_force(query, k)
            }

            // No index - always brute force
            IndexBackend::None => self.search_brute_force(query, k),
        }
    }

    /// Searches using brute-force comparison against all vectors.
    ///
    /// This is O(N) where N is the number of vectors, but provides exact results.
    /// Used when the vector count is below the index threshold.
    fn search_brute_force(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut results: Vec<SearchResult> = Vec::new();

        // Use ordered keys for iteration
        let keys = self.vector_keys_ordered.clone();

        for full_key in &keys {
            if let Some(Atom::Vector(vec, _)) = self.db.get(full_key)? {
                let score = self.config.metric.distance(query, &vec);

                // Strip prefix from key for result
                let key = full_key
                    .strip_prefix(&self.config.key_prefix)
                    .unwrap_or(full_key)
                    .to_string();

                results.push(SearchResult {
                    key,
                    score,
                    vector: vec,
                });
            }
        }

        // Sort by score (ascending = most similar first)
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top k
        results.truncate(k);
        Ok(results)
    }

    /// Searches using the HNSW index for approximate nearest neighbors.
    ///
    /// This is O(log N) where N is the number of vectors, providing fast
    /// approximate results. Used when the vector count exceeds the index threshold.
    ///
    /// **Requirements:** 1.5
    fn search_hnsw(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self
            .hnsw_index
            .as_ref()
            .ok_or_else(|| SynaError::CorruptedIndex("HNSW index not available".to_string()))?;

        // Search the HNSW index
        let hnsw_results = index.search(query, k);

        // Convert HNSW results to SearchResult, fetching vectors from DB
        let mut results = Vec::with_capacity(hnsw_results.len());
        for (full_key, score) in hnsw_results {
            // Strip prefix from key for result
            let key = full_key
                .strip_prefix(&self.config.key_prefix)
                .unwrap_or(&full_key)
                .to_string();

            // Fetch the vector from the database
            let vector = if let Some(Atom::Vector(vec, _)) = self.db.get(&full_key)? {
                vec
            } else {
                // Vector not found in DB, skip it
                continue;
            };

            results.push(SearchResult { key, score, vector });
        }

        Ok(results)
    }

    /// Searches using the FAISS index for approximate nearest neighbors.
    ///
    /// FAISS provides high-performance vector search with support for:
    /// - Billion-scale datasets
    /// - GPU acceleration
    /// - Various index types (IVF, PQ, HNSW, etc.)
    ///
    /// This method is only available when the 'faiss' feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match configured dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// * `SynaError::CorruptedIndex` - If FAISS index is not available
    /// * `SynaError::IndexError` - If FAISS search fails
    ///
    /// **Requirements:** 1.5
    #[cfg(feature = "faiss")]
    fn search_faiss(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self
            .faiss_index
            .as_mut()
            .ok_or_else(|| SynaError::CorruptedIndex("FAISS index not available".to_string()))?;

        // Search the FAISS index
        let faiss_results = index.search(query, k)?;

        // Convert FAISS results to SearchResult, fetching vectors from DB
        let mut results = Vec::with_capacity(faiss_results.len());
        for (key_without_prefix, score) in faiss_results {
            // Reconstruct full key with prefix
            let full_key = format!("{}{}", self.config.key_prefix, key_without_prefix);

            // Fetch the vector from the database
            let vector = if let Some(Atom::Vector(vec, _)) = self.db.get(&full_key)? {
                vec
            } else {
                // Vector not found in DB, skip it
                continue;
            };

            results.push(SearchResult {
                key: key_without_prefix,
                score,
                vector,
            });
        }

        Ok(results)
    }

    /// Builds the HNSW index from all stored vectors.
    ///
    /// This is called automatically when the vector count exceeds `index_threshold`
    /// during insert, but can also be called manually to force index building.
    ///
    /// The index is automatically saved to a `.hnsw` sidecar file for persistence.
    ///
    /// # Errors
    ///
    /// * `SynaError::Io` - If reading vectors from the database fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::vector::{VectorStore, VectorConfig};
    ///
    /// let mut store = VectorStore::new("vectors.db", VectorConfig::default()).unwrap();
    /// // ... insert many vectors ...
    /// store.build_index().unwrap();
    /// ```
    ///
    /// **Requirements:** 1.5, 1.7
    pub fn build_index(&mut self) -> Result<()> {
        let mut index = HnswIndex::new(
            self.config.dimensions,
            self.config.metric,
            HnswConfig::default(),
        );

        // Use ordered keys for consistent index building
        let keys = self.vector_keys_ordered.clone();

        for full_key in &keys {
            if let Some(Atom::Vector(vec, _)) = self.db.get(full_key)? {
                // Add node to HNSW index with proper graph connections
                self.add_node_to_index(&mut index, full_key, &vec);
            }
        }

        // Save index to disk for persistence
        let hnsw_path = Self::hnsw_index_path(&self.db_path);
        index.save(&hnsw_path)?;

        self.hnsw_index = Some(index);
        self.index_dirty = false;
        self.last_checkpoint = Instant::now();
        Ok(())
    }

    /// Saves the current HNSW index to disk.
    ///
    /// This is called automatically after `build_index()`, but can be called
    /// manually to persist incremental changes.
    ///
    /// # Errors
    ///
    /// * `SynaError::Io` - If writing the index file fails
    /// * `SynaError::CorruptedIndex` - If no index exists to save
    pub fn save_index(&self) -> Result<()> {
        let index = self.hnsw_index.as_ref()
            .ok_or_else(|| SynaError::CorruptedIndex("No HNSW index to save".to_string()))?;
        
        let hnsw_path = Self::hnsw_index_path(&self.db_path);
        index.save(&hnsw_path)?;
        Ok(())
    }

    /// Helper to add a node to the HNSW index during build_index().
    ///
    /// This builds proper HNSW graph connections with multiple levels
    /// for efficient O(log N) search.
    fn add_node_to_index(&self, index: &mut HnswIndex, key: &str, vector: &[f32]) {
        use crate::hnsw::HnswNode;

        // Skip if key already exists
        if index.key_to_id.contains_key(key) {
            return;
        }

        // Generate a random level for this node (exponential distribution)
        let level = index.random_level();

        // Create the node with neighbor lists for each level
        let node = HnswNode::new(key.to_string(), vector.to_vec(), level);
        let node_id = index.nodes.len();

        // Add to index structures
        index.nodes.push(node);
        index.key_to_id.insert(key.to_string(), node_id);

        // Update entry point if this is the first node or has higher level
        let current_max_level = index.max_level();
        if index.entry_point.is_none() || level > current_max_level {
            index.entry_point = Some(node_id);
            index.set_max_level(level);
        }

        // Connect to existing nodes at each level this node exists at
        if node_id > 0 {
            let m = index.config().m;
            let m_max = index.config().m_max;

            for l in 0..=level {
                let max_neighbors = if l == 0 { m } else { m_max };
                let mut neighbors = Vec::new();

                // Find closest nodes at this level
                let mut distances: Vec<(usize, f32)> = index.nodes.iter()
                    .enumerate()
                    .filter(|(id, n)| *id != node_id && n.neighbors.len() > l)
                    .map(|(id, n)| (id, index.metric().distance(vector, &n.vector)))
                    .collect();
                distances.sort_by(|a, b| a.1.total_cmp(&b.1));

                for (neighbor_id, dist) in distances.into_iter().take(max_neighbors) {
                    neighbors.push((neighbor_id, dist));
                    
                    // Add bidirectional connection
                    if l < index.nodes[neighbor_id].neighbors.len() {
                        index.nodes[neighbor_id].neighbors[l].push((node_id, dist));
                        
                        // Prune neighbor's connections if exceeding limit
                        if index.nodes[neighbor_id].neighbors[l].len() > max_neighbors {
                            index.nodes[neighbor_id].neighbors[l]
                                .sort_by(|a, b| a.1.total_cmp(&b.1));
                            index.nodes[neighbor_id].neighbors[l].truncate(max_neighbors);
                        }
                    }
                }

                if l < index.nodes[node_id].neighbors.len() {
                    index.nodes[node_id].neighbors[l] = neighbors;
                }
            }
        }
    }

    /// Returns whether an HNSW index is currently built.
    pub fn has_index(&self) -> bool {
        self.hnsw_index.is_some()
    }

    /// Returns the index threshold configuration.
    pub fn index_threshold(&self) -> usize {
        self.config.index_threshold
    }

    /// Retrieves a vector by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up (without prefix)
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Vec<f32>))` - The vector data if found
    /// * `Ok(None)` - If the key doesn't exist
    /// * `Err(SynaError)` - If reading fails
    pub fn get(&mut self, key: &str) -> Result<Option<Vec<f32>>> {
        let full_key = format!("{}{}", self.config.key_prefix, key);
        match self.db.get(&full_key)? {
            Some(Atom::Vector(vec, _)) => Ok(Some(vec)),
            _ => Ok(None),
        }
    }

    /// Deletes a vector by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to delete (without prefix)
    ///
    /// # Errors
    ///
    /// * `SynaError::Io` - If the delete fails
    pub fn delete(&mut self, key: &str) -> Result<()> {
        let full_key = format!("{}{}", self.config.key_prefix, key);
        self.db.delete(&full_key)?;
        self.vector_keys.remove(&full_key);
        self.vector_keys_ordered.retain(|k| k != &full_key);
        Ok(())
    }

    /// Returns the number of vectors stored.
    pub fn len(&self) -> usize {
        self.vector_keys.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vector_keys.is_empty()
    }

    /// Returns whether the index has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.index_dirty
    }

    /// Returns the configured dimensions.
    pub fn dimensions(&self) -> u16 {
        self.config.dimensions
    }

    /// Returns the configured distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.config.metric
    }

    /// Flushes any pending changes to disk.
    /// This is called automatically when the VectorStore is dropped.
    pub fn flush(&mut self) -> Result<()> {
        self.checkpoint_index()
    }
}

impl Drop for VectorStore {
    fn drop(&mut self) {
        // Save index if dirty
        if self.index_dirty {
            if let Err(e) = self.checkpoint_index() {
                eprintln!("Warning: Failed to save HNSW index on drop: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vector_store_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 128,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert a vector
        let vec1: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        store.insert("v1", &vec1).unwrap();

        // Retrieve it
        let retrieved = store.get("v1").unwrap().unwrap();
        assert_eq!(retrieved.len(), 128);

        // Check length
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_vector_store_search() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert some vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // Search
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.001).collect();
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be v0 (identical to query)
        assert_eq!(results[0].key, "v0");
        assert!(results[0].score < 0.001);
    }

    #[test]
    fn test_dimension_validation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Invalid dimensions (too small)
        let config = VectorConfig {
            dimensions: 32,
            ..Default::default()
        };
        assert!(VectorStore::new(&db_path, config).is_err());

        // Invalid dimensions (too large - above 8192 limit)
        let config = VectorConfig {
            dimensions: 9000,
            ..Default::default()
        };
        assert!(VectorStore::new(&db_path, config).is_err());

        // Valid dimensions
        let config = VectorConfig {
            dimensions: 128,
            ..Default::default()
        };
        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Wrong vector size
        let wrong_vec = vec![0.1f32; 64];
        assert!(store.insert("v1", &wrong_vec).is_err());
    }

    #[test]
    fn test_vector_store_delete() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        let vec1: Vec<f32> = vec![0.1; 64];
        store.insert("v1", &vec1).unwrap();
        assert_eq!(store.len(), 1);

        store.delete("v1").unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.get("v1").unwrap().is_none());
    }

    #[test]
    fn test_hnsw_integration_build_index() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            index_threshold: 0, // Disable auto-build for this test
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // No index because auto-build is disabled (threshold = 0)
        assert!(!store.has_index());

        // Build index manually
        store.build_index().unwrap();

        // Now we have an index
        assert!(store.has_index());
    }

    #[test]
    fn test_hnsw_integration_auto_build() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            index_threshold: 5, // Auto-build after 5 vectors
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert 4 vectors - below threshold
        for i in 0..4 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // No index yet (below threshold)
        assert!(!store.has_index());

        // Insert 5th vector - triggers auto-build
        let vec: Vec<f32> = (0..64).map(|j| (4 * 64 + j) as f32 * 0.001).collect();
        store.insert("v4", &vec).unwrap();

        // Now we have an index (auto-built at threshold)
        assert!(store.has_index());
    }

    #[test]
    fn test_hnsw_integration_search_with_index() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            index_threshold: 5, // Low threshold for testing
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // Build index
        store.build_index().unwrap();

        // Search should use HNSW (since we have index and >= threshold)
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.001).collect();
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be v0 (identical to query)
        assert_eq!(results[0].key, "v0");
        assert!(results[0].score < 0.001);
    }

    #[test]
    fn test_hnsw_integration_search_below_threshold() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            index_threshold: 100, // High threshold - won't use HNSW
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert vectors (below threshold)
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // Build index anyway
        store.build_index().unwrap();
        assert!(store.has_index());

        // Search should still use brute force (below threshold)
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.001).collect();
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be v0 (identical to query)
        assert_eq!(results[0].key, "v0");
    }

    #[test]
    fn test_index_threshold_config() {
        let config = VectorConfig::default();
        assert_eq!(config.index_threshold, 10000);

        let custom_config = VectorConfig {
            index_threshold: 5000,
            ..Default::default()
        };
        assert_eq!(custom_config.index_threshold, 5000);
    }

    #[test]
    fn test_backend_selection_default() {
        let config = VectorConfig::default();
        // Default backend should be HNSW
        match config.backend {
            IndexBackend::Hnsw(_) => {} // Expected
            _ => panic!("Default backend should be HNSW"),
        }
    }

    #[test]
    fn test_backend_selection_none() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Configure with no index (brute-force only)
        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            backend: IndexBackend::None,
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // Search should use brute force (IndexBackend::None)
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.001).collect();
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be v0 (identical to query)
        assert_eq!(results[0].key, "v0");
        assert!(results[0].score < 0.001);
    }

    #[test]
    fn test_backend_selection_hnsw_custom() {
        use crate::hnsw::HnswConfig;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Configure with custom HNSW settings
        let config = VectorConfig {
            dimensions: 64,
            metric: DistanceMetric::Euclidean,
            index_threshold: 5,
            backend: IndexBackend::Hnsw(HnswConfig {
                m: 32,
                ef_construction: 400,
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut store = VectorStore::new(&db_path, config).unwrap();

        // Insert vectors
        for i in 0..10 {
            let vec: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 * 0.001).collect();
            store.insert(&format!("v{}", i), &vec).unwrap();
        }

        // Build index
        store.build_index().unwrap();
        assert!(store.has_index());

        // Search should work
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.001).collect();
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "v0");
    }
}

//! Memory-mapped Cascade Index
//!
//! A complete redesign following SynaDB physics principles:
//!
//! - **Arrow of Time**: All writes are append-only (vectors, edges, buckets)
//! - **The Delta**: Graph edges stored incrementally, not full neighbor lists
//! - **The Observer**: Vectors accessed via mmap for zero-copy reads
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ MmapCascadeIndex                                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │ vectors.cmvs (mmap)     - Append-only vector storage         │
//! │ graph.edges (append)    - Append-only edge log               │
//! │ lsh (in-memory)         - LSH with append-only buckets       │
//! │ index.meta              - Configuration and LSH state        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | Insert    | O(1) amortized | Append to mmap + LSH buckets |
//! | Search    | O(log N) | LSH candidates + graph traversal |
//! | Get       | O(1) | Direct mmap access |

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::distance::DistanceMetric;
use crate::error::{Result, SynaError};

use super::append_graph::{AppendGraph, AppendGraphConfig};
use super::mmap_store::{MmapStoreConfig, MmapVectorStorage};
use super::simple_lsh::{SimpleLSH, SimpleLSHConfig};

/// Configuration for MmapCascadeIndex
#[derive(Clone, Debug)]
pub struct MmapCascadeConfig {
    /// Vector dimensions
    pub dimensions: u16,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Initial vector capacity
    pub initial_capacity: usize,

    // LSH parameters
    /// Number of hash bits per table (2^num_bits buckets)
    pub num_bits: usize,
    /// Number of hash tables
    pub num_tables: usize,
    /// Number of probes during search
    pub num_probes: usize,

    // Graph parameters
    /// Target neighbors per node
    pub m: usize,
    /// Maximum neighbors per node
    pub m_max: usize,
    /// Search expansion factor
    pub ef_search: usize,
}

impl Default for MmapCascadeConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            initial_capacity: 100_000,
            num_bits: 6,
            num_tables: 10,
            num_probes: 16,
            m: 20,
            m_max: 40,
            ef_search: 80,
        }
    }
}

impl MmapCascadeConfig {
    /// Preset for small datasets (< 10K vectors)
    pub fn small() -> Self {
        Self {
            num_bits: 5,
            num_tables: 8,
            num_probes: 12,
            m: 16,
            m_max: 32,
            ef_search: 100,
            initial_capacity: 10_000,
            ..Default::default()
        }
    }

    /// Preset for large datasets (> 100K vectors)
    pub fn large() -> Self {
        Self {
            num_bits: 8,
            num_tables: 16,
            num_probes: 24,
            m: 32,
            m_max: 64,
            ef_search: 200,
            initial_capacity: 1_000_000,
            ..Default::default()
        }
    }

    /// Preset for high recall (> 95%)
    pub fn high_recall() -> Self {
        Self {
            num_bits: 4,
            num_tables: 20,
            num_probes: 32,
            m: 48,
            m_max: 96,
            ef_search: 400,
            ..Default::default()
        }
    }

    /// Preset for fast search
    pub fn fast_search() -> Self {
        Self {
            num_bits: 8,
            num_tables: 6,
            num_probes: 8,
            m: 16,
            m_max: 32,
            ef_search: 50,
            ..Default::default()
        }
    }
}

/// Search result
#[derive(Clone, Debug)]
pub struct MmapSearchResult {
    /// User-provided key
    pub key: String,
    /// Distance/similarity score
    pub score: f32,
    /// Vector data (optional)
    pub vector: Option<Vec<f32>>,
}

/// Memory-mapped Cascade Index
///
/// Combines mmap vector storage, append-only graph, and LSH for
/// fast approximate nearest neighbor search.
pub struct MmapCascadeIndex {
    /// Base path for all files
    base_path: PathBuf,
    /// Configuration
    config: MmapCascadeConfig,
    /// Memory-mapped vector storage
    storage: MmapVectorStorage,
    /// Append-only graph
    graph: AppendGraph,
    /// LSH for candidate generation
    lsh: SimpleLSH,
}

impl MmapCascadeIndex {
    /// Create or open a MmapCascadeIndex
    pub fn new<P: AsRef<Path>>(path: P, config: MmapCascadeConfig) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Create directory if needed
        if let Some(parent) = base_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let vectors_path = Self::vectors_path(&base_path);
        let graph_path = Self::graph_path(&base_path);
        let meta_path = Self::meta_path(&base_path);

        // Check if existing index
        let exists = meta_path.exists();

        // Load or create LSH
        let lsh = if exists {
            Self::load_lsh(&meta_path, &config)?
        } else {
            let lsh_config = SimpleLSHConfig {
                num_bits: config.num_bits,
                num_tables: config.num_tables,
                dimensions: config.dimensions as usize,
            };
            SimpleLSH::new(lsh_config)
        };

        // Create storage
        let storage_config = MmapStoreConfig {
            dimensions: config.dimensions,
            metric: config.metric,
            initial_capacity: config.initial_capacity,
        };
        let storage = MmapVectorStorage::new(&vectors_path, storage_config)?;

        // Create graph
        let graph_config = AppendGraphConfig {
            m: config.m,
            m_max: config.m_max,
            metric: config.metric,
        };
        let graph = AppendGraph::new(&graph_path, graph_config)?;

        let mut index = Self {
            base_path,
            config,
            storage,
            graph,
            lsh,
        };

        // Rebuild LSH buckets from storage if loading existing
        if exists {
            index.rebuild_lsh_buckets()?;
        }

        Ok(index)
    }

    /// Create in-memory index (no persistence)
    pub fn in_memory(config: MmapCascadeConfig) -> Result<Self> {
        let temp_dir = std::env::temp_dir().join(format!("cascade_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        Self::new(temp_dir.join("index"), config)
    }

    fn vectors_path(base: &Path) -> PathBuf {
        base.with_extension("cmvs")
    }

    fn graph_path(base: &Path) -> PathBuf {
        base.with_extension("edges")
    }

    fn meta_path(base: &Path) -> PathBuf {
        base.with_extension("meta")
    }

    fn load_lsh(meta_path: &Path, config: &MmapCascadeConfig) -> Result<SimpleLSH> {
        let file = File::open(meta_path)?;
        let mut reader = BufReader::new(file);

        // Read LSH bytes length
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let lsh_len = u64::from_le_bytes(len_buf) as usize;

        // Read LSH bytes
        let mut lsh_bytes = vec![0u8; lsh_len];
        reader.read_exact(&mut lsh_bytes)?;

        SimpleLSH::from_bytes(&lsh_bytes)
            .ok_or_else(|| SynaError::CorruptedIndex("Invalid LSH data".into()))
            .or_else(|_| {
                // Fall back to creating new LSH
                let lsh_config = SimpleLSHConfig {
                    num_bits: config.num_bits,
                    num_tables: config.num_tables,
                    dimensions: config.dimensions as usize,
                };
                Ok(SimpleLSH::new(lsh_config))
            })
    }

    fn rebuild_lsh_buckets(&mut self) -> Result<()> {
        for id in self.storage.ids() {
            if let Some(vector) = self.storage.get_vector_by_id(id) {
                self.lsh.insert(id, &vector);
            }
        }
        Ok(())
    }

    /// Insert a vector
    pub fn insert(&mut self, key: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatchUsize {
                expected: self.config.dimensions as usize,
                got: vector.len(),
            });
        }

        // Skip if exists
        if self.storage.contains(key) {
            return Ok(());
        }

        // Append to storage (Arrow of Time!)
        let id = self.storage.append(key, vector)?;

        // Get LSH candidates for graph connections
        let candidates = self.lsh.query_multiprobe(vector, self.config.num_probes);

        // Limit candidates to avoid O(N) behavior
        let max_candidates = 200;
        let limited_candidates: Vec<u32> = candidates.into_iter().take(max_candidates).collect();

        // Insert to LSH buckets (append-only!)
        self.lsh.insert(id, vector);

        // Connect in graph (append-only edges!)
        self.graph
            .insert_with_candidates(&self.storage, id, &limited_candidates)?;

        Ok(())
    }

    /// Insert batch of vectors
    pub fn insert_batch(&mut self, keys: &[&str], vectors: &[&[f32]]) -> Result<usize> {
        if keys.len() != vectors.len() {
            return Err(SynaError::InvalidInput(
                "Keys and vectors must have same length".into(),
            ));
        }

        let mut inserted = 0;

        for (key, vector) in keys.iter().zip(vectors.iter()) {
            if vector.len() != self.config.dimensions as usize {
                return Err(SynaError::DimensionMismatchUsize {
                    expected: self.config.dimensions as usize,
                    got: vector.len(),
                });
            }

            if self.storage.contains(key) {
                continue;
            }

            // Append to storage
            let id = self.storage.append(key, vector)?;

            // Insert to LSH
            self.lsh.insert(id, vector);

            inserted += 1;
        }

        // Build graph connections in second pass
        // This allows batch vectors to find each other
        let start_id = (self.storage.len() - inserted) as u32;
        for id in start_id..(start_id + inserted as u32) {
            if let Some(vector) = self.storage.get_vector_by_id(id) {
                let candidates = self.lsh.query_multiprobe(&vector, self.config.num_probes);
                let limited: Vec<u32> = candidates
                    .into_iter()
                    .filter(|&c| c != id)
                    .take(200)
                    .collect();

                self.graph
                    .insert_with_candidates(&self.storage, id, &limited)?;
            }
        }

        Ok(inserted)
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<MmapSearchResult>> {
        self.search_with_params(query, k, self.config.num_probes, self.config.ef_search)
    }

    /// Search with custom parameters
    pub fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        num_probes: usize,
        ef: usize,
    ) -> Result<Vec<MmapSearchResult>> {
        if query.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatchUsize {
                expected: self.config.dimensions as usize,
                got: query.len(),
            });
        }

        if self.storage.is_empty() {
            return Ok(Vec::new());
        }

        // Get entry points from LSH
        let mut entry_points: HashSet<u32> = self
            .lsh
            .query_multiprobe(query, num_probes)
            .into_iter()
            .collect();

        // Fallback: sample uniformly if LSH gives few candidates
        let min_entry_points = k.max(20) * 2;
        if entry_points.len() < min_entry_points {
            let sample_size = min_entry_points * 2;
            let step = (self.storage.len() / sample_size).max(1);

            for id in (0..self.storage.len() as u32).step_by(step) {
                entry_points.insert(id);
                if entry_points.len() >= min_entry_points * 2 {
                    break;
                }
            }
        }

        let entry_vec: Vec<u32> = entry_points.into_iter().collect();

        // Graph search
        let graph_results = self.graph.search(&self.storage, query, &entry_vec, k, ef);

        // Convert to results
        let results = graph_results
            .into_iter()
            .filter_map(|(id, score)| {
                let key = self.storage.get_key_by_id(id)?;
                let vector = self.storage.get_vector_by_id(id);
                Some(MmapSearchResult { key, score, vector })
            })
            .collect();

        Ok(results)
    }

    /// Get vector by key
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.storage.get_vector(key).map(|v| v.to_vec())
    }

    /// Check if key exists
    pub fn contains(&self, key: &str) -> bool {
        self.storage.contains(key)
    }

    /// Number of vectors
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Dimensions
    pub fn dimensions(&self) -> u16 {
        self.config.dimensions
    }

    /// Save index state
    pub fn save(&mut self) -> Result<()> {
        // Save LSH state
        let meta_path = Self::meta_path(&self.base_path);
        let file = File::create(&meta_path)?;
        let mut writer = BufWriter::new(file);

        let lsh_bytes = self.lsh.to_bytes();
        writer.write_all(&(lsh_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&lsh_bytes)?;
        writer.flush()?;

        // Flush storage and graph
        self.storage.flush()?;
        self.graph.flush()?;

        Ok(())
    }

    /// Flush to disk
    pub fn flush(&mut self) -> Result<()> {
        self.save()
    }

    /// Get statistics
    pub fn stats(&self) -> MmapCascadeStats {
        MmapCascadeStats {
            vector_count: self.storage.len(),
            bucket_count: self.lsh.bucket_count(),
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
        }
    }
}

impl Drop for MmapCascadeIndex {
    fn drop(&mut self) {
        let _ = self.save();
    }
}

/// Index statistics
#[derive(Clone, Debug, Default)]
pub struct MmapCascadeStats {
    pub vector_count: usize,
    pub bucket_count: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_cascade_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_index");

        let config = MmapCascadeConfig {
            dimensions: 64,
            initial_capacity: 1000,
            num_bits: 4,
            num_tables: 4,
            ..Default::default()
        };

        let mut index = MmapCascadeIndex::new(&path, config).unwrap();

        // Insert vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..64).map(|j| ((i + j) % 10) as f32 / 10.0).collect();
            index.insert(&format!("v{}", i), &vec).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query: Vec<f32> = (0..64).map(|j| (j % 10) as f32 / 10.0).collect();
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_mmap_cascade_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_index");

        let config = MmapCascadeConfig {
            dimensions: 32,
            initial_capacity: 100,
            num_bits: 4,
            num_tables: 2,
            ..Default::default()
        };

        // Create and populate
        {
            let mut index = MmapCascadeIndex::new(&path, config.clone()).unwrap();

            for i in 0..20 {
                let vec: Vec<f32> = (0..32).map(|j| ((i + j) % 5) as f32).collect();
                index.insert(&format!("k{}", i), &vec).unwrap();
            }

            index.save().unwrap();
        }

        // Reopen and verify
        {
            let index = MmapCascadeIndex::new(&path, config).unwrap();
            assert_eq!(index.len(), 20);

            let vec = index.get("k5");
            assert!(vec.is_some());
        }
    }

    #[test]
    fn test_mmap_cascade_batch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_index");

        let config = MmapCascadeConfig {
            dimensions: 64,
            initial_capacity: 1000,
            ..Default::default()
        };

        let mut index = MmapCascadeIndex::new(&path, config).unwrap();

        let keys: Vec<String> = (0..50).map(|i| format!("v{}", i)).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..64).map(|j| ((i + j) % 10) as f32 / 10.0).collect())
            .collect();
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let inserted = index.insert_batch(&key_refs, &vec_refs).unwrap();
        assert_eq!(inserted, 50);
        assert_eq!(index.len(), 50);
    }
}

//! Main Cascade Index implementation
//!
//! Combines LSH, adaptive buckets, and sparse graph for fast vector search.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::distance::DistanceMetric;
use crate::error::SynaError;

use super::bucket::{BucketConfig, BucketForest};
use super::config::CascadeConfig;
use super::graph::{CascadeGraph, GraphConfig};
use super::lsh::{HyperplaneLSH, LSHConfig};

/// Search result from Cascade Index
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// User-provided key
    pub key: String,

    /// Similarity score (lower is more similar for distance metrics)
    pub score: f32,

    /// Vector data (optional, for convenience)
    pub vector: Option<Vec<f32>>,
}

/// Cascade Index statistics
#[derive(Clone, Debug, Default)]
pub struct CascadeStats {
    pub vector_count: usize,
    pub bucket_count: usize,
    pub avg_bucket_size: f32,
    pub graph_edges: usize,
    pub avg_neighbors: f32,
}

/// Cascade Index: LSH + Adaptive Buckets + Sparse Graph
///
/// A novel vector index that achieves O(N) build time without requiring
/// initialization samples (unlike GWI) or quadratic neighbor search (unlike HNSW).
pub struct CascadeIndex {
    /// Configuration
    config: CascadeConfig,

    /// LSH hash functions
    lsh: HyperplaneLSH,

    /// Adaptive bucket forest
    buckets: BucketForest,

    /// Sparse graph for neighbor refinement
    graph: CascadeGraph,

    /// All vectors (needed for bucket splits)
    vectors: Vec<Vec<f32>>,

    /// File path for persistence
    path: Option<String>,
}

impl CascadeIndex {
    /// Create a new Cascade Index
    pub fn new(path: impl AsRef<Path>, config: CascadeConfig) -> Result<Self, SynaError> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Try to load existing index
        if path.as_ref().exists() {
            return Self::load(&path_str);
        }

        let lsh_config = LSHConfig {
            num_bits: config.num_bits,
            num_tables: config.num_tables,
            dimensions: config.dimensions as usize,
        };

        let bucket_config = BucketConfig {
            split_threshold: config.split_threshold,
            max_depth: config.max_bucket_depth,
            min_split_size: 10,
        };

        let graph_config = GraphConfig {
            m: config.m,
            m_max: config.m_max,
            bidirectional: config.bidirectional,
            metric: config.metric,
        };

        let num_tables = config.num_tables;

        Ok(Self {
            config,
            lsh: HyperplaneLSH::new(lsh_config),
            buckets: BucketForest::new(num_tables, bucket_config),
            graph: CascadeGraph::new(graph_config),
            vectors: Vec::new(),
            path: Some(path_str),
        })
    }

    /// Create an in-memory index (no persistence)
    pub fn in_memory(config: CascadeConfig) -> Self {
        let lsh_config = LSHConfig {
            num_bits: config.num_bits,
            num_tables: config.num_tables,
            dimensions: config.dimensions as usize,
        };

        let bucket_config = BucketConfig {
            split_threshold: config.split_threshold,
            max_depth: config.max_bucket_depth,
            min_split_size: 10,
        };

        let graph_config = GraphConfig {
            m: config.m,
            m_max: config.m_max,
            bidirectional: config.bidirectional,
            metric: config.metric,
        };

        Self {
            lsh: HyperplaneLSH::new(lsh_config),
            buckets: BucketForest::new(config.num_tables, bucket_config),
            graph: CascadeGraph::new(graph_config),
            vectors: Vec::new(),
            path: None,
            config,
        }
    }

    /// Insert a single vector
    pub fn insert(&mut self, key: &str, vector: &[f32]) -> Result<(), SynaError> {
        if vector.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatchUsize {
                expected: self.config.dimensions as usize,
                got: vector.len(),
            });
        }

        let id = self.vectors.len();
        let vector_owned = vector.to_vec();

        // Compute LSH hashes
        let hashes = self.lsh.hash(vector);

        // Collect candidates from buckets - LIMIT to avoid O(N) behavior
        // Only sample from a few tables and limit candidates per table
        let max_candidates_per_table = 50;
        let max_total_candidates = 200;
        let mut candidates = HashSet::new();

        for (table, &hash) in hashes.iter().enumerate() {
            if candidates.len() >= max_total_candidates {
                break;
            }

            let bucket_candidates = self.buckets.query(table, hash, vector);
            for cid in bucket_candidates.into_iter().take(max_candidates_per_table) {
                candidates.insert(cid);
                if candidates.len() >= max_total_candidates {
                    break;
                }
            }
        }

        // Add to graph with candidates
        let candidate_vec: Vec<usize> = candidates.into_iter().collect();
        self.graph
            .insert(key.to_string(), vector_owned.clone(), &candidate_vec);

        // Add to buckets AFTER graph insertion
        self.vectors.push(vector_owned);
        for (table, &hash) in hashes.iter().enumerate() {
            self.buckets.insert(table, hash, id, vector, &self.vectors);
        }

        Ok(())
    }

    /// Insert multiple vectors efficiently (batch mode)
    pub fn insert_batch(&mut self, keys: &[&str], vectors: &[&[f32]]) -> Result<(), SynaError> {
        if keys.len() != vectors.len() {
            return Err(SynaError::InvalidInput(
                "Keys and vectors must have same length".to_string(),
            ));
        }

        // For batch insert, we can be smarter about candidate selection
        // First pass: add all vectors to buckets without graph connections
        let start_id = self.vectors.len();
        let mut all_hashes = Vec::with_capacity(keys.len());

        for (_key, vector) in keys.iter().zip(vectors.iter()) {
            if vector.len() != self.config.dimensions as usize {
                return Err(SynaError::DimensionMismatchUsize {
                    expected: self.config.dimensions as usize,
                    got: vector.len(),
                });
            }

            let id = self.vectors.len(); // Use current length as ID
            let vector_owned = vector.to_vec();
            let hashes = self.lsh.hash(vector);

            // Add to vectors storage
            self.vectors.push(vector_owned.clone());

            // Add to buckets
            for (table, &hash) in hashes.iter().enumerate() {
                self.buckets.insert(table, hash, id, vector, &self.vectors);
            }

            all_hashes.push(hashes);
        }

        // Second pass: build graph connections using nearby vectors
        for (i, (key, vector)) in keys.iter().zip(vectors.iter()).enumerate() {
            let hashes = &all_hashes[i];

            // Collect candidates from same buckets
            let max_candidates = 100;
            let mut candidates = HashSet::new();

            for (table, &hash) in hashes.iter().enumerate() {
                if candidates.len() >= max_candidates {
                    break;
                }
                for cid in self.buckets.query(table, hash, vector) {
                    if cid != start_id + i {
                        // Don't include self
                        candidates.insert(cid);
                    }
                    if candidates.len() >= max_candidates {
                        break;
                    }
                }
            }

            let candidate_vec: Vec<usize> = candidates.into_iter().collect();
            self.graph
                .insert(key.to_string(), vector.to_vec(), &candidate_vec);
        }

        Ok(())
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, SynaError> {
        self.search_with_params(query, k, self.config.num_probes, self.config.ef_search)
    }

    /// Search with custom parameters
    pub fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        num_probes: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>, SynaError> {
        if query.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatchUsize {
                expected: self.config.dimensions as usize,
                got: query.len(),
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Collect entry points from LSH buckets with multi-probe
        let mut entry_points = HashSet::new();

        // Phase 1: Get candidates from LSH probes
        for table in 0..self.config.num_tables {
            let probes = self.lsh.get_probe_sequence(query, table, num_probes);
            for hash in probes {
                for id in self.buckets.query_with_neighbors(table, hash, query) {
                    entry_points.insert(id);
                }
            }
        }

        // Phase 2: If we have very few entry points, sample uniformly
        // This is a fallback for when LSH fails
        let min_entry_points = std::cmp::max(k * 2, 20);
        if entry_points.len() < min_entry_points {
            let sample_size = std::cmp::min(self.vectors.len(), min_entry_points * 2);
            let step = std::cmp::max(1, self.vectors.len() / sample_size);

            for i in (0..self.vectors.len()).step_by(step) {
                entry_points.insert(i);
                if entry_points.len() >= min_entry_points * 2 {
                    break;
                }
            }
        }

        // Convert to vec for graph search
        let entry_vec: Vec<usize> = entry_points.into_iter().collect();

        // Graph search from entry points
        let graph_results = self.graph.search(query, &entry_vec, k, ef);

        // Convert to SearchResult
        let results = graph_results
            .into_iter()
            .filter_map(|(id, score)| {
                self.graph.get_by_id(id).map(|node| SearchResult {
                    key: node.key.clone(),
                    score,
                    vector: Some(node.vector.clone()),
                })
            })
            .collect();

        Ok(results)
    }

    /// Compute distance between query and vector
    #[allow(dead_code)]
    fn compute_distance(&self, query: &[f32], vector: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = query.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_v: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_q < 1e-10 || norm_v < 1e-10 {
                    1.0
                } else {
                    1.0 - (dot / (norm_q * norm_v))
                }
            }
            DistanceMetric::Euclidean => query
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => -query
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>(),
        }
    }

    /// Get a vector by key
    pub fn get(&self, key: &str) -> Result<Option<Vec<f32>>, SynaError> {
        Ok(self.graph.get(key).map(|node| node.vector.clone()))
    }

    /// Delete a vector by key
    pub fn delete(&mut self, key: &str) -> Result<bool, SynaError> {
        Ok(self.graph.delete(key))
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Get index statistics
    pub fn stats(&self) -> CascadeStats {
        let graph_stats = self.graph.stats();
        let bucket_count = self.buckets.bucket_count();
        let avg_bucket_size = if bucket_count > 0 {
            self.vectors.len() as f32 * self.config.num_tables as f32 / bucket_count as f32
        } else {
            0.0
        };

        CascadeStats {
            vector_count: self.vectors.len(),
            bucket_count,
            avg_bucket_size,
            graph_edges: graph_stats.edge_count,
            avg_neighbors: graph_stats.avg_neighbors,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CascadeConfig {
        &self.config
    }

    /// Get dimensions
    pub fn dimensions(&self) -> u16 {
        self.config.dimensions
    }

    /// Save index to file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SynaError> {
        let file = File::create(path.as_ref()).map_err(|e| SynaError::IoError(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        // Magic number and version
        writer
            .write_all(b"CASC")
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&1u32.to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        // Config
        writer
            .write_all(&self.config.dimensions.to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.metric as u8).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.num_bits as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.num_tables as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.split_threshold as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.m as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.m_max as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.num_probes as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&(self.config.ef_search as u32).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        // LSH state
        let lsh_bytes = self.lsh.to_bytes();
        writer
            .write_all(&(lsh_bytes.len() as u64).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        writer
            .write_all(&lsh_bytes)
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        // Vectors and keys
        writer
            .write_all(&(self.vectors.len() as u64).to_le_bytes())
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        for id in self.graph.node_ids() {
            if let Some(node) = self.graph.get_by_id(id) {
                // Key length and key
                let key_bytes = node.key.as_bytes();
                writer
                    .write_all(&(key_bytes.len() as u32).to_le_bytes())
                    .map_err(|e| SynaError::IoError(e.to_string()))?;
                writer
                    .write_all(key_bytes)
                    .map_err(|e| SynaError::IoError(e.to_string()))?;

                // Vector
                for &f in &node.vector {
                    writer
                        .write_all(&f.to_le_bytes())
                        .map_err(|e| SynaError::IoError(e.to_string()))?;
                }

                // Neighbors
                writer
                    .write_all(&(node.neighbors.len() as u32).to_le_bytes())
                    .map_err(|e| SynaError::IoError(e.to_string()))?;
                for &(nid, dist) in &node.neighbors {
                    writer
                        .write_all(&(nid as u64).to_le_bytes())
                        .map_err(|e| SynaError::IoError(e.to_string()))?;
                    writer
                        .write_all(&dist.to_le_bytes())
                        .map_err(|e| SynaError::IoError(e.to_string()))?;
                }
            }
        }

        writer
            .flush()
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load index from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SynaError> {
        let file = File::open(path.as_ref()).map_err(|e| SynaError::IoError(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Magic number
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        if &magic != b"CASC" {
            return Err(SynaError::InvalidInput(
                "Invalid Cascade Index file".to_string(),
            ));
        }

        // Version
        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let _version = u32::from_le_bytes(version_bytes);

        // Config
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];

        reader
            .read_exact(&mut buf2)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let dimensions = u16::from_le_bytes(buf2);

        let mut buf1 = [0u8; 1];
        reader
            .read_exact(&mut buf1)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let metric = match buf1[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let num_bits = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let num_tables = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let split_threshold = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let m = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let m_max = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let num_probes = u32::from_le_bytes(buf4) as usize;

        reader
            .read_exact(&mut buf4)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let ef_search = u32::from_le_bytes(buf4) as usize;

        let config = CascadeConfig {
            dimensions,
            metric,
            num_bits,
            num_tables,
            split_threshold,
            max_bucket_depth: 10,
            m,
            m_max,
            bidirectional: true,
            num_probes,
            ef_search,
        };

        // LSH state
        let mut buf8 = [0u8; 8];
        reader
            .read_exact(&mut buf8)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let lsh_len = u64::from_le_bytes(buf8) as usize;

        let mut lsh_bytes = vec![0u8; lsh_len];
        reader
            .read_exact(&mut lsh_bytes)
            .map_err(|e| SynaError::IoError(e.to_string()))?;

        let lsh = HyperplaneLSH::from_bytes(&lsh_bytes)
            .ok_or_else(|| SynaError::InvalidInput("Invalid LSH data".to_string()))?;

        // Create index structure
        let bucket_config = BucketConfig {
            split_threshold,
            max_depth: 10,
            min_split_size: 10,
        };

        let graph_config = GraphConfig {
            m,
            m_max,
            bidirectional: true,
            metric,
        };

        let mut index = Self {
            config,
            lsh,
            buckets: BucketForest::new(num_tables, bucket_config),
            graph: CascadeGraph::new(graph_config),
            vectors: Vec::new(),
            path: Some(path.as_ref().to_string_lossy().to_string()),
        };

        // Read vectors
        reader
            .read_exact(&mut buf8)
            .map_err(|e| SynaError::IoError(e.to_string()))?;
        let vector_count = u64::from_le_bytes(buf8) as usize;

        for _ in 0..vector_count {
            // Key
            reader
                .read_exact(&mut buf4)
                .map_err(|e| SynaError::IoError(e.to_string()))?;
            let key_len = u32::from_le_bytes(buf4) as usize;

            let mut key_bytes = vec![0u8; key_len];
            reader
                .read_exact(&mut key_bytes)
                .map_err(|e| SynaError::IoError(e.to_string()))?;
            let key =
                String::from_utf8(key_bytes).map_err(|e| SynaError::InvalidInput(e.to_string()))?;

            // Vector
            let mut vector = Vec::with_capacity(dimensions as usize);
            for _ in 0..dimensions {
                reader
                    .read_exact(&mut buf4)
                    .map_err(|e| SynaError::IoError(e.to_string()))?;
                vector.push(f32::from_le_bytes(buf4));
            }

            // Insert (rebuilds buckets and graph)
            index.insert(&key, &vector)?;

            // Skip neighbors (they'll be rebuilt)
            reader
                .read_exact(&mut buf4)
                .map_err(|e| SynaError::IoError(e.to_string()))?;
            let neighbor_count = u32::from_le_bytes(buf4) as usize;

            let skip_bytes = neighbor_count * 12; // 8 bytes id + 4 bytes dist
            let mut skip_buf = vec![0u8; skip_bytes];
            reader
                .read_exact(&mut skip_buf)
                .map_err(|e| SynaError::IoError(e.to_string()))?;
        }

        Ok(index)
    }

    /// Save to the original path
    pub fn flush(&self) -> Result<(), SynaError> {
        if let Some(ref path) = self.path {
            self.save(path)
        } else {
            Ok(())
        }
    }
}

impl Drop for CascadeIndex {
    fn drop(&mut self) {
        // Auto-save on drop if path is set
        if self.path.is_some() {
            let _ = self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cascade_insert_search() {
        let config = CascadeConfig {
            dimensions: 64,
            num_bits: 8,
            num_tables: 4,
            m: 8,
            ..Default::default()
        };

        let mut index = CascadeIndex::in_memory(config);

        // Insert some vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..64).map(|j| ((i + j) % 10) as f32 / 10.0).collect();
            index.insert(&format!("v{}", i), &vector).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query: Vec<f32> = (0..64).map(|j| (j % 10) as f32 / 10.0).collect();
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_cascade_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cascade");

        let config = CascadeConfig {
            dimensions: 32,
            num_bits: 6,
            num_tables: 2,
            m: 4,
            ..Default::default()
        };

        // Create and populate
        {
            let mut index = CascadeIndex::new(&path, config.clone()).unwrap();
            for i in 0..20 {
                let vector: Vec<f32> = (0..32).map(|j| ((i + j) % 5) as f32).collect();
                index.insert(&format!("k{}", i), &vector).unwrap();
            }
            index.save(&path).unwrap();
        }

        // Reload and verify
        {
            let index = CascadeIndex::load(&path).unwrap();
            assert_eq!(index.len(), 20);

            let vec = index.get("k5").unwrap();
            assert!(vec.is_some());
        }
    }
}

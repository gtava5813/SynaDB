//! Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search
//!
//! HNSW provides O(log N) search time vs O(N) for brute force, enabling
//! million-scale vector search with <10ms latency.
//!
//! # Algorithm Overview
//!
//! HNSW builds a multi-layer graph where:
//! - Layer 0 contains all nodes with M connections each
//! - Higher layers contain exponentially fewer nodes with M_max connections
//! - Search starts at the top layer and greedily descends
//!
//! # References
//!
//! - Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate
//!   nearest neighbor search using Hierarchical Navigable Small World graphs.

use crate::distance::DistanceMetric;
use crate::error::{Result, SynaError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// =============================================================================
// Configuration
// =============================================================================

/// HNSW index configuration
///
/// These parameters control the trade-off between index quality, build time,
/// and search performance.
///
/// # Default Values
///
/// The defaults are tuned for a good balance of recall and performance:
/// - `m = 16`: Good for most use cases
/// - `m_max = 32`: 2x M for higher layers
/// - `ef_construction = 200`: High quality index
/// - `ef_search = 100`: Good recall with fast search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max connections per node at layer 0.
    ///
    /// Higher values improve recall but increase memory and build time.
    /// Typical range: 8-64, default: 16
    pub m: usize,

    /// Max connections per node at higher layers.
    ///
    /// Usually set to 2*M. Controls the "highway" connectivity.
    pub m_max: usize,

    /// Size of dynamic candidate list during construction.
    ///
    /// Higher values build a better index but take longer.
    /// Typical range: 100-500, default: 200
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search.
    ///
    /// Higher values improve recall but slow down search.
    /// Typical range: 50-500, default: 100
    pub ef_search: usize,

    /// Normalization factor for level generation.
    ///
    /// Controls the probability distribution of node levels.
    /// Default: 1/ln(M)
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 100,
            ml: 1.0 / (16.0_f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Create a new configuration with custom M value.
    ///
    /// Other parameters are derived from M:
    /// - m_max = 2 * m
    /// - ml = 1 / ln(m)
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m_max: 2 * m,
            ml: 1.0 / (m as f64).ln(),
            ..Default::default()
        }
    }

    /// Set ef_construction (build quality).
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search (search quality).
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

// =============================================================================
// Helper Types
// =============================================================================

/// A candidate node during search, ordered by distance.
///
/// Used in the priority queue for greedy search.
/// Ordered so that BinaryHeap (max-heap) returns the *farthest* node first,
/// which is useful for maintaining a bounded candidate set.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Node ID in the index
    pub node_id: usize,
    /// Distance from query vector
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: larger distance = higher priority (for max-heap)
        // This lets us efficiently pop the farthest candidate
        match self.distance.partial_cmp(&other.distance) {
            Some(ord) => ord,
            None => Ordering::Equal, // Handle NaN
        }
    }
}

/// A candidate ordered for min-heap (closest first).
///
/// Used when we want to iterate from closest to farthest.
#[derive(Debug, Clone)]
pub struct MinCandidate {
    /// Node ID in the index
    pub node_id: usize,
    /// Distance from query vector
    pub distance: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
    }
}

impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse of Candidate: smaller distance = higher priority
        match other.distance.partial_cmp(&self.distance) {
            Some(ord) => ord,
            None => Ordering::Equal,
        }
    }
}

// =============================================================================
// Node Structure
// =============================================================================

/// A node in the HNSW graph.
///
/// Each node stores:
/// - The key (for mapping back to the database)
/// - The vector data (cached for fast distance computation)
/// - Neighbor lists for each level the node exists in
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Key in the database (e.g., "vec/doc1")
    pub key: String,

    /// Vector data (cached for search performance)
    pub vector: Vec<f32>,

    /// Neighbors at each level: level -> [(neighbor_id, distance)]
    ///
    /// Level 0 has up to M neighbors, higher levels have up to M_max.
    pub neighbors: Vec<Vec<(usize, f32)>>,
}

impl HnswNode {
    /// Create a new node with the given key, vector, and level.
    pub fn new(key: String, vector: Vec<f32>, level: usize) -> Self {
        Self {
            key,
            vector,
            neighbors: vec![Vec::new(); level + 1],
        }
    }

    /// Get the maximum level this node exists at.
    pub fn level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }
}

// =============================================================================
// HNSW Index
// =============================================================================

/// HNSW index for approximate nearest neighbor search.
///
/// This is the main data structure for vector similarity search. It maintains
/// a multi-layer graph where each node is connected to its nearest neighbors.
///
/// # Example
///
/// ```rust,ignore
/// use synadb::hnsw::{HnswIndex, HnswConfig};
/// use synadb::distance::DistanceMetric;
///
/// // Create an index for 128-dimensional vectors
/// let config = HnswConfig::default();
/// let mut index = HnswIndex::new(128, DistanceMetric::Cosine, config);
///
/// // Insert vectors (implemented in task 6.2)
/// index.insert("doc1", &[0.1; 128]).unwrap();
/// index.insert("doc2", &[0.2; 128]).unwrap();
///
/// // Search for nearest neighbors (implemented in task 6.3)
/// let results = index.search(&[0.15; 128], 5).unwrap();
/// ```
pub struct HnswIndex {
    /// Index configuration
    config: HnswConfig,

    /// Distance metric for similarity computation
    metric: DistanceMetric,

    /// Number of dimensions for vectors in this index
    dimensions: u16,

    /// All nodes in the index.
    /// Public to allow VectorStore to build the index incrementally.
    pub nodes: Vec<HnswNode>,

    /// Entry point node ID (top of the graph).
    /// Public to allow VectorStore to set the entry point.
    pub entry_point: Option<usize>,

    /// Maximum level in the current graph
    max_level: usize,

    /// Map from key to node ID for fast lookup.
    /// Public to allow VectorStore to register keys.
    pub key_to_id: HashMap<String, usize>,

    /// Random number generator state for level selection
    rng_state: u64,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of dimensions for vectors (64-4096)
    /// * `metric` - Distance metric for similarity computation
    /// * `config` - Index configuration parameters
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use synadb::hnsw::{HnswIndex, HnswConfig};
    /// use synadb::distance::DistanceMetric;
    ///
    /// let index = HnswIndex::new(768, DistanceMetric::Cosine, HnswConfig::default());
    /// ```
    pub fn new(dimensions: u16, metric: DistanceMetric, config: HnswConfig) -> Self {
        Self {
            config,
            metric,
            dimensions,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            key_to_id: HashMap::new(),
            // Initialize RNG with a simple seed based on current time
            rng_state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42),
        }
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the configured dimensions.
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    /// Get the distance metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get the configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the entry point node ID.
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Get the maximum level in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Check if a key exists in the index.
    pub fn contains_key(&self, key: &str) -> bool {
        self.key_to_id.contains_key(key)
    }

    /// Get the node ID for a key.
    pub fn get_node_id(&self, key: &str) -> Option<usize> {
        self.key_to_id.get(key).copied()
    }

    /// Get a node by ID.
    pub fn get_node(&self, node_id: usize) -> Option<&HnswNode> {
        self.nodes.get(node_id)
    }

    /// Get all keys in the index.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.key_to_id.keys().map(|s| s.as_str())
    }

    /// Generate a random level for a new node.
    ///
    /// Uses the formula: floor(-ln(uniform(0,1)) * ml)
    /// This gives an exponential distribution where most nodes are at level 0.
    pub fn random_level(&mut self) -> usize {
        // Simple xorshift64 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;

        // Convert to uniform [0, 1)
        let uniform = (self.rng_state as f64) / (u64::MAX as f64);

        // Compute level using inverse transform sampling
        let level = (-uniform.ln() * self.config.ml).floor() as usize;

        // Cap at a reasonable maximum to prevent pathological cases
        level.min(32)
    }

    /// Compute distance between a query vector and a node.
    pub fn distance_to_node(&self, query: &[f32], node_id: usize) -> f32 {
        self.metric.distance(query, &self.nodes[node_id].vector)
    }

    /// Get all node IDs at a given level.
    pub fn nodes_at_level(&self, level: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.level() >= level)
            .map(|(id, _)| id)
            .collect()
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> HnswStats {
        let mut level_counts = vec![0usize; self.max_level + 1];
        let mut total_edges = 0usize;

        for node in &self.nodes {
            for (level, neighbors) in node.neighbors.iter().enumerate() {
                if level < level_counts.len() {
                    level_counts[level] += 1;
                }
                total_edges += neighbors.len();
            }
        }

        HnswStats {
            num_nodes: self.nodes.len(),
            max_level: self.max_level,
            level_counts,
            total_edges,
            avg_edges_per_node: if self.nodes.is_empty() {
                0.0
            } else {
                total_edges as f64 / self.nodes.len() as f64
            },
        }
    }
}

/// Statistics about an HNSW index.
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Maximum level in the graph
    pub max_level: usize,
    /// Number of nodes at each level
    pub level_counts: Vec<usize>,
    /// Total number of edges (neighbor connections)
    pub total_edges: usize,
    /// Average edges per node
    pub avg_edges_per_node: f64,
}

// =============================================================================
// Search Implementation
// =============================================================================

/// Helper struct for min-heap ordering (closest first).
/// Used in search_layer to process candidates from closest to farthest.
struct MinHeapEntry(f32, usize);

impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl Eq for MinHeapEntry {}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller distance = higher priority (for min-heap via BinaryHeap)
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Helper struct for max-heap ordering (farthest first).
/// Used to maintain the worst result in the candidate set.
struct MaxHeapEntry(f32, usize);

impl PartialEq for MaxHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl Eq for MaxHeapEntry {}

impl PartialOrd for MaxHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal ordering: larger distance = higher priority (for max-heap)
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl HnswIndex {
    /// Search for k nearest neighbors.
    ///
    /// This is the main search method for the HNSW index. It performs a greedy
    /// search starting from the entry point at the top level, descending through
    /// layers until reaching level 0 where the final candidates are selected.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector to search for
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// A vector of (key, distance) pairs, sorted by distance (closest first).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = index.search(&query_vector, 10);
    /// for (key, distance) in results {
    ///     println!("{}: {:.4}", key, distance);
    /// }
    /// ```
    ///
    /// **Requirements:** 1.4, 1.5
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        // Return empty if no entry point
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut ep = self.entry_point.unwrap();

        // Traverse from top level to level 1, finding the closest node at each level
        for lc in (1..=self.max_level).rev() {
            let results = self.search_layer(query, ep, 1, lc);
            if !results.is_empty() {
                ep = results[0].0;
            }
        }

        // Search at level 0 with ef_search candidates for better recall
        let candidates = self.search_layer(query, ep, self.config.ef_search, 0);

        // Return top k results with keys
        candidates
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.nodes[id].key.clone(), dist))
            .collect()
    }

    /// Search within a single layer of the HNSW graph.
    ///
    /// This implements the greedy search algorithm that explores neighbors
    /// starting from an entry point, maintaining a dynamic candidate list.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `entry_point` - Starting node ID for the search
    /// * `ef` - Size of the dynamic candidate list (controls recall vs speed)
    /// * `level` - The layer to search in
    ///
    /// # Returns
    ///
    /// A vector of (node_id, distance) pairs, sorted by distance (closest first).
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::{BinaryHeap, HashSet};

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // min-heap by distance (closest first)
        let mut results = BinaryHeap::new(); // max-heap for tracking worst result

        // Initialize with entry point
        let ep_dist = self.metric.distance(query, &self.nodes[entry_point].vector);
        visited.insert(entry_point);
        candidates.push(MinHeapEntry(ep_dist, entry_point));
        results.push(MaxHeapEntry(ep_dist, entry_point));

        // Greedy search: explore candidates until no improvement possible
        while let Some(MinHeapEntry(c_dist, c_id)) = candidates.pop() {
            // Get the worst distance in our result set
            let worst_dist = results.peek().map(|e| e.0).unwrap_or(f32::MAX);

            // If current candidate is farther than our worst result, we're done
            if c_dist > worst_dist {
                break;
            }

            // Explore neighbors at this level
            if level < self.nodes[c_id].neighbors.len() {
                for (neighbor_id, _) in &self.nodes[c_id].neighbors[level] {
                    // Skip already visited nodes
                    if visited.insert(*neighbor_id) {
                        let dist = self
                            .metric
                            .distance(query, &self.nodes[*neighbor_id].vector);
                        let worst_dist = results.peek().map(|e| e.0).unwrap_or(f32::MAX);

                        // Add to candidates if better than worst or we haven't filled ef yet
                        if results.len() < ef || dist < worst_dist {
                            candidates.push(MinHeapEntry(dist, *neighbor_id));
                            results.push(MaxHeapEntry(dist, *neighbor_id));

                            // Maintain ef size by removing worst
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vec (closest first)
        let mut result_vec: Vec<_> = results
            .into_iter()
            .map(|MaxHeapEntry(d, id)| (id, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Search for k nearest neighbors with a custom ef_search value.
    ///
    /// This allows overriding the configured ef_search for specific queries
    /// where you need higher recall or faster search.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector to search for
    /// * `k` - Number of nearest neighbors to return
    /// * `ef_search` - Size of dynamic candidate list (higher = better recall, slower)
    ///
    /// # Returns
    ///
    /// A vector of (key, distance) pairs, sorted by distance (closest first).
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(String, f32)> {
        // Return empty if no entry point
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let mut ep = self.entry_point.unwrap();

        // Traverse from top level to level 1
        for lc in (1..=self.max_level).rev() {
            let results = self.search_layer(query, ep, 1, lc);
            if !results.is_empty() {
                ep = results[0].0;
            }
        }

        // Search at level 0 with custom ef_search
        let candidates = self.search_layer(query, ep, ef_search, 0);

        // Return top k results with keys
        candidates
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.nodes[id].key.clone(), dist))
            .collect()
    }
}

// =============================================================================
// Persistence
// =============================================================================

/// Magic bytes for HNSW index files
const HNSW_MAGIC: &[u8; 4] = b"HNSW";

/// Current version of the HNSW file format
const HNSW_VERSION: u16 = 1;

impl HnswIndex {
    /// Save the HNSW index to a file.
    ///
    /// The index is saved in a binary format that can be loaded later with
    /// [`HnswIndex::load`]. The format includes:
    /// - Magic bytes ("HNSW")
    /// - Version number (u16)
    /// - Configuration
    /// - Dimensions (u16)
    /// - Metric (u8)
    /// - All nodes with their vectors and neighbor lists
    /// - Entry point and max level
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
    /// // ... insert vectors ...
    /// index.save("vectors.hnsw")?;
    /// ```
    ///
    /// **Requirements:** 1.5
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(HNSW_MAGIC)?;

        // Write version
        writer.write_all(&HNSW_VERSION.to_le_bytes())?;

        // Write dimensions
        writer.write_all(&self.dimensions.to_le_bytes())?;

        // Write metric as u8
        let metric_byte: u8 = match self.metric {
            DistanceMetric::Cosine => 0,
            DistanceMetric::Euclidean => 1,
            DistanceMetric::DotProduct => 2,
        };
        writer.write_all(&[metric_byte])?;

        // Serialize config
        bincode::serialize_into(&mut writer, &self.config)?;

        // Serialize nodes
        bincode::serialize_into(&mut writer, &self.nodes)?;

        // Serialize entry point
        bincode::serialize_into(&mut writer, &self.entry_point)?;

        // Serialize max level
        bincode::serialize_into(&mut writer, &self.max_level)?;

        // Serialize RNG state for reproducibility
        bincode::serialize_into(&mut writer, &self.rng_state)?;

        // Flush to ensure all data is written
        writer.flush()?;

        Ok(())
    }

    /// Load an HNSW index from a file.
    ///
    /// Loads an index that was previously saved with [`HnswIndex::save`].
    /// The metric and dimensions are read from the file and must match
    /// the expected values if provided.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    ///
    /// # Returns
    ///
    /// The loaded HNSW index, or an error if the file is corrupted or
    /// incompatible.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = HnswIndex::load("vectors.hnsw")?;
    /// let results = index.search(&query, 10);
    /// ```
    ///
    /// **Requirements:** 1.5
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != HNSW_MAGIC {
            return Err(SynaError::CorruptedIndex(
                "Invalid magic bytes - not an HNSW index file".to_string(),
            ));
        }

        // Read and verify version
        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version != HNSW_VERSION {
            return Err(SynaError::CorruptedIndex(format!(
                "Unsupported version: {} (expected {})",
                version, HNSW_VERSION
            )));
        }

        // Read dimensions
        let mut dims_bytes = [0u8; 2];
        reader.read_exact(&mut dims_bytes)?;
        let dimensions = u16::from_le_bytes(dims_bytes);

        // Read metric
        let mut metric_byte = [0u8; 1];
        reader.read_exact(&mut metric_byte)?;
        let metric = match metric_byte[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => {
                return Err(SynaError::CorruptedIndex(format!(
                    "Invalid metric byte: {}",
                    metric_byte[0]
                )))
            }
        };

        // Deserialize config
        let config: HnswConfig = bincode::deserialize_from(&mut reader)?;

        // Deserialize nodes
        let nodes: Vec<HnswNode> = bincode::deserialize_from(&mut reader)?;

        // Deserialize entry point
        let entry_point: Option<usize> = bincode::deserialize_from(&mut reader)?;

        // Deserialize max level
        let max_level: usize = bincode::deserialize_from(&mut reader)?;

        // Deserialize RNG state
        let rng_state: u64 = bincode::deserialize_from(&mut reader)?;

        // Rebuild key_to_id map from nodes
        let mut key_to_id = HashMap::new();
        for (id, node) in nodes.iter().enumerate() {
            key_to_id.insert(node.key.clone(), id);
        }

        Ok(Self {
            config,
            metric,
            dimensions,
            nodes,
            entry_point,
            max_level,
            key_to_id,
            rng_state,
        })
    }

    /// Load an HNSW index from a file with validation.
    ///
    /// Like [`HnswIndex::load`], but validates that the loaded index has
    /// the expected dimensions and metric.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    /// * `expected_dims` - Expected number of dimensions
    /// * `expected_metric` - Expected distance metric
    ///
    /// # Returns
    ///
    /// The loaded HNSW index, or an error if validation fails.
    ///
    /// **Requirements:** 1.5
    pub fn load_validated<P: AsRef<Path>>(
        path: P,
        expected_dims: u16,
        expected_metric: DistanceMetric,
    ) -> Result<Self> {
        let index = Self::load(path)?;

        if index.dimensions != expected_dims {
            return Err(SynaError::DimensionMismatch {
                expected: expected_dims,
                got: index.dimensions,
            });
        }

        if index.metric != expected_metric {
            return Err(SynaError::CorruptedIndex(format!(
                "Metric mismatch: expected {:?}, got {:?}",
                expected_metric, index.metric
            )));
        }

        Ok(index)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 100);
        assert!((config.ml - 1.0 / 16.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_hnsw_config_with_m() {
        let config = HnswConfig::with_m(32);
        assert_eq!(config.m, 32);
        assert_eq!(config.m_max, 64);
        assert!((config.ml - 1.0 / 32.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_hnsw_config_builder() {
        let config = HnswConfig::with_m(8).ef_construction(100).ef_search(50);
        assert_eq!(config.m, 8);
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn test_candidate_ordering() {
        // Max-heap: larger distance = higher priority
        let mut heap = BinaryHeap::new();
        heap.push(Candidate {
            node_id: 0,
            distance: 1.0,
        });
        heap.push(Candidate {
            node_id: 1,
            distance: 3.0,
        });
        heap.push(Candidate {
            node_id: 2,
            distance: 2.0,
        });

        // Should pop in order: 3.0, 2.0, 1.0 (farthest first)
        assert_eq!(heap.pop().unwrap().distance, 3.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 1.0);
    }

    #[test]
    fn test_min_candidate_ordering() {
        // Min-heap: smaller distance = higher priority
        let mut heap = BinaryHeap::new();
        heap.push(MinCandidate {
            node_id: 0,
            distance: 1.0,
        });
        heap.push(MinCandidate {
            node_id: 1,
            distance: 3.0,
        });
        heap.push(MinCandidate {
            node_id: 2,
            distance: 2.0,
        });

        // Should pop in order: 1.0, 2.0, 3.0 (closest first)
        assert_eq!(heap.pop().unwrap().distance, 1.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 3.0);
    }

    #[test]
    fn test_hnsw_node_creation() {
        let node = HnswNode::new("test_key".to_string(), vec![1.0, 2.0, 3.0], 2);
        assert_eq!(node.key, "test_key");
        assert_eq!(node.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(node.level(), 2);
        assert_eq!(node.neighbors.len(), 3); // Levels 0, 1, 2
    }

    #[test]
    fn test_hnsw_index_creation() {
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        assert_eq!(index.dimensions(), 128);
        assert_eq!(index.metric(), DistanceMetric::Cosine);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.entry_point().is_none());
    }

    #[test]
    fn test_random_level_distribution() {
        let mut index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());

        // Generate many levels and check distribution
        let mut level_counts = [0usize; 10];
        for _ in 0..10000 {
            let level = index.random_level();
            if level < 10 {
                level_counts[level] += 1;
            }
        }

        // Most nodes should be at level 0 (exponential distribution)
        assert!(level_counts[0] > level_counts[1]);
        assert!(level_counts[1] > level_counts[2]);

        // Level 0 should have the majority
        assert!(level_counts[0] > 5000);
    }

    #[test]
    fn test_hnsw_stats_empty() {
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        let stats = index.stats();
        assert_eq!(stats.num_nodes, 0);
        assert_eq!(stats.total_edges, 0);
        assert_eq!(stats.avg_edges_per_node, 0.0);
    }

    #[test]
    fn test_search_empty_index() {
        let index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());
        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single_node() {
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Manually add a single node
        let node = HnswNode::new("node1".to_string(), vec![1.0, 0.0, 0.0], 0);
        index.nodes.push(node);
        index.key_to_id.insert("node1".to_string(), 0);
        index.entry_point = Some(0);

        // Search should return the single node
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "node1");
        assert!(results[0].1 < 0.001); // Distance should be ~0
    }

    #[test]
    fn test_search_multiple_nodes_sorted() {
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Add three nodes at different distances from origin
        let node1 = HnswNode::new("close".to_string(), vec![1.0, 0.0, 0.0], 0);
        let node2 = HnswNode::new("medium".to_string(), vec![2.0, 0.0, 0.0], 0);
        let node3 = HnswNode::new("far".to_string(), vec![3.0, 0.0, 0.0], 0);

        index.nodes.push(node1);
        index.nodes.push(node2);
        index.nodes.push(node3);

        index.key_to_id.insert("close".to_string(), 0);
        index.key_to_id.insert("medium".to_string(), 1);
        index.key_to_id.insert("far".to_string(), 2);

        // Connect nodes at level 0 (all connected to each other)
        index.nodes[0].neighbors[0] = vec![(1, 1.0), (2, 2.0)];
        index.nodes[1].neighbors[0] = vec![(0, 1.0), (2, 1.0)];
        index.nodes[2].neighbors[0] = vec![(0, 2.0), (1, 1.0)];

        index.entry_point = Some(0);

        // Query from origin - should return nodes sorted by distance
        let query = vec![0.0, 0.0, 0.0];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "close"); // Distance 1.0
        assert_eq!(results[1].0, "medium"); // Distance 2.0
        assert_eq!(results[2].0, "far"); // Distance 3.0

        // Verify distances are sorted
        assert!(results[0].1 < results[1].1);
        assert!(results[1].1 < results[2].1);
    }

    #[test]
    fn test_search_k_limit() {
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Add 5 nodes
        for i in 0..5 {
            let node = HnswNode::new(format!("node{}", i), vec![i as f32, 0.0, 0.0], 0);
            index.nodes.push(node);
            index.key_to_id.insert(format!("node{}", i), i);
        }

        // Connect all nodes
        for i in 0..5 {
            let mut neighbors = Vec::new();
            for j in 0..5 {
                if i != j {
                    neighbors.push((j, (i as f32 - j as f32).abs()));
                }
            }
            index.nodes[i].neighbors[0] = neighbors;
        }

        index.entry_point = Some(0);

        // Request only 2 results
        let query = vec![0.0, 0.0, 0.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_with_ef() {
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Add a single node
        let node = HnswNode::new("node1".to_string(), vec![1.0, 0.0, 0.0], 0);
        index.nodes.push(node);
        index.key_to_id.insert("node1".to_string(), 0);
        index.entry_point = Some(0);

        // Search with custom ef_search
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search_with_ef(&query, 5, 50);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "node1");
    }

    #[test]
    fn test_min_heap_entry_ordering() {
        // Test the MinHeapEntry ordering for search_layer
        let mut heap = BinaryHeap::new();
        heap.push(MinHeapEntry(3.0, 0));
        heap.push(MinHeapEntry(1.0, 1));
        heap.push(MinHeapEntry(2.0, 2));

        // Should pop in order: 1.0, 2.0, 3.0 (closest first)
        assert_eq!(heap.pop().unwrap().0, 1.0);
        assert_eq!(heap.pop().unwrap().0, 2.0);
        assert_eq!(heap.pop().unwrap().0, 3.0);
    }

    #[test]
    fn test_max_heap_entry_ordering() {
        // Test the MaxHeapEntry ordering for search_layer
        let mut heap = BinaryHeap::new();
        heap.push(MaxHeapEntry(1.0, 0));
        heap.push(MaxHeapEntry(3.0, 1));
        heap.push(MaxHeapEntry(2.0, 2));

        // Should pop in order: 3.0, 2.0, 1.0 (farthest first)
        assert_eq!(heap.pop().unwrap().0, 3.0);
        assert_eq!(heap.pop().unwrap().0, 2.0);
        assert_eq!(heap.pop().unwrap().0, 1.0);
    }

    // =========================================================================
    // Persistence Tests
    // =========================================================================

    #[test]
    fn test_save_load_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.hnsw");

        // Create and save empty index
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        index.save(&path).unwrap();

        // Load and verify
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.dimensions(), 128);
        assert_eq!(loaded.metric(), DistanceMetric::Cosine);
        assert!(loaded.is_empty());
        assert!(loaded.entry_point().is_none());
    }

    #[test]
    fn test_save_load_with_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.hnsw");

        // Create index with nodes
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Add nodes manually
        let node1 = HnswNode::new("node1".to_string(), vec![1.0, 0.0, 0.0], 0);
        let node2 = HnswNode::new("node2".to_string(), vec![0.0, 1.0, 0.0], 0);
        let node3 = HnswNode::new("node3".to_string(), vec![0.0, 0.0, 1.0], 0);

        index.nodes.push(node1);
        index.nodes.push(node2);
        index.nodes.push(node3);

        index.key_to_id.insert("node1".to_string(), 0);
        index.key_to_id.insert("node2".to_string(), 1);
        index.key_to_id.insert("node3".to_string(), 2);

        // Connect nodes
        index.nodes[0].neighbors[0] = vec![(1, 1.414), (2, 1.414)];
        index.nodes[1].neighbors[0] = vec![(0, 1.414), (2, 1.414)];
        index.nodes[2].neighbors[0] = vec![(0, 1.414), (1, 1.414)];

        index.entry_point = Some(0);

        // Save
        index.save(&path).unwrap();

        // Load and verify
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.metric(), DistanceMetric::Euclidean);
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.entry_point(), Some(0));

        // Verify nodes
        assert!(loaded.contains_key("node1"));
        assert!(loaded.contains_key("node2"));
        assert!(loaded.contains_key("node3"));

        // Verify vectors
        let node1 = loaded.get_node(0).unwrap();
        assert_eq!(node1.vector, vec![1.0, 0.0, 0.0]);

        // Verify neighbors
        assert_eq!(node1.neighbors[0].len(), 2);
    }

    #[test]
    fn test_save_load_search_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("search.hnsw");

        // Create index with nodes
        let mut index = HnswIndex::new(3, DistanceMetric::Euclidean, HnswConfig::default());

        // Add nodes
        let node1 = HnswNode::new("close".to_string(), vec![1.0, 0.0, 0.0], 0);
        let node2 = HnswNode::new("medium".to_string(), vec![2.0, 0.0, 0.0], 0);
        let node3 = HnswNode::new("far".to_string(), vec![3.0, 0.0, 0.0], 0);

        index.nodes.push(node1);
        index.nodes.push(node2);
        index.nodes.push(node3);

        index.key_to_id.insert("close".to_string(), 0);
        index.key_to_id.insert("medium".to_string(), 1);
        index.key_to_id.insert("far".to_string(), 2);

        // Connect nodes
        index.nodes[0].neighbors[0] = vec![(1, 1.0), (2, 2.0)];
        index.nodes[1].neighbors[0] = vec![(0, 1.0), (2, 1.0)];
        index.nodes[2].neighbors[0] = vec![(0, 2.0), (1, 1.0)];

        index.entry_point = Some(0);

        // Search before save
        let query = vec![0.0, 0.0, 0.0];
        let results_before = index.search(&query, 3);

        // Save and load
        index.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();

        // Search after load
        let results_after = loaded.search(&query, 3);

        // Results should be identical
        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.0, after.0); // Same keys
            assert!((before.1 - after.1).abs() < 1e-6); // Same distances
        }
    }

    #[test]
    fn test_load_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.hnsw");

        // Write invalid magic bytes
        std::fs::write(&path, b"XXXX").unwrap();

        // Load should fail
        let result = HnswIndex::load(&path);
        assert!(result.is_err());
        match result {
            Err(crate::error::SynaError::CorruptedIndex(msg)) => {
                assert!(msg.contains("Invalid magic bytes"));
            }
            _ => panic!("Expected CorruptedIndex error"),
        }
    }

    #[test]
    fn test_load_invalid_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid_version.hnsw");

        // Write valid magic but invalid version
        let mut data = Vec::new();
        data.extend_from_slice(b"HNSW");
        data.extend_from_slice(&99u16.to_le_bytes()); // Invalid version

        std::fs::write(&path, &data).unwrap();

        // Load should fail
        let result = HnswIndex::load(&path);
        assert!(result.is_err());
        match result {
            Err(crate::error::SynaError::CorruptedIndex(msg)) => {
                assert!(msg.contains("Unsupported version"));
            }
            _ => panic!("Expected CorruptedIndex error"),
        }
    }

    #[test]
    fn test_load_validated_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("validated.hnsw");

        // Create and save index
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        index.save(&path).unwrap();

        // Load with correct validation
        let loaded = HnswIndex::load_validated(&path, 128, DistanceMetric::Cosine).unwrap();
        assert_eq!(loaded.dimensions(), 128);
        assert_eq!(loaded.metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_load_validated_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dim_mismatch.hnsw");

        // Create and save index with 128 dimensions
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        index.save(&path).unwrap();

        // Load with wrong dimensions
        let result = HnswIndex::load_validated(&path, 256, DistanceMetric::Cosine);
        assert!(result.is_err());
        match result {
            Err(crate::error::SynaError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 256);
                assert_eq!(got, 128);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_load_validated_metric_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metric_mismatch.hnsw");

        // Create and save index with Cosine metric
        let index = HnswIndex::new(128, DistanceMetric::Cosine, HnswConfig::default());
        index.save(&path).unwrap();

        // Load with wrong metric
        let result = HnswIndex::load_validated(&path, 128, DistanceMetric::Euclidean);
        assert!(result.is_err());
        match result {
            Err(crate::error::SynaError::CorruptedIndex(msg)) => {
                assert!(msg.contains("Metric mismatch"));
            }
            _ => panic!("Expected CorruptedIndex error"),
        }
    }

    #[test]
    fn test_save_load_preserves_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.hnsw");

        // Create index with custom config
        let config = HnswConfig::with_m(32).ef_construction(300).ef_search(150);
        let index = HnswIndex::new(64, DistanceMetric::DotProduct, config);
        index.save(&path).unwrap();

        // Load and verify config
        let loaded = HnswIndex::load(&path).unwrap();
        assert_eq!(loaded.config().m, 32);
        assert_eq!(loaded.config().m_max, 64);
        assert_eq!(loaded.config().ef_construction, 300);
        assert_eq!(loaded.config().ef_search, 150);
        assert_eq!(loaded.metric(), DistanceMetric::DotProduct);
    }

    #[test]
    fn test_save_load_all_metrics() {
        let dir = tempfile::tempdir().unwrap();

        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ] {
            let path = dir.path().join(format!("{:?}.hnsw", metric));
            let index = HnswIndex::new(64, metric, HnswConfig::default());
            index.save(&path).unwrap();

            let loaded = HnswIndex::load(&path).unwrap();
            assert_eq!(loaded.metric(), metric);
        }
    }
}

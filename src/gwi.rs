//! Gravity Well Index (GWI) - Append-Only Vector Indexing
//!
//! A novel vector indexing algorithm designed for SynaDB's append-only,
//! mmap-friendly architecture. Vectors "fall" into gravity wells (attractors)
//! rather than being connected in a mutable graph.
//!
//! # Architecture
//!
//! ```text
//! Level 0:           ★                    (1 attractor - root)
//!                   /|\
//! Level 1:        ●  ●  ●  ●              (B attractors)
//!                /|\ |  |\ |\
//! Level 2:     ○ ○ ○ ○ ○ ○ ○ ○            (B² attractors)
//!              : : : : : : : :
//! Level 3:     · · · · · · · · · ·        (B³ attractors - leaves)
//!              vectors cluster here
//! ```
//!
//! # Performance
//!
//! - Insert: O(L × B) distance calculations (L=levels, B=branching)
//! - Search: O(L × B + nprobe × cluster_size)
//! - Build: O(N × L × B) - linear in N, not O(N log N) like HNSW
//!
//! # Example
//!
//! ```rust,ignore
//! use synadb::gwi::{GravityWellIndex, GwiConfig};
//!
//! let config = GwiConfig::default();
//! let mut gwi = GravityWellIndex::new("vectors.gwi", config)?;
//!
//! // Initialize attractors from sample data
//! gwi.initialize_attractors(&sample_vectors)?;
//!
//! // Insert vectors (O(log M) each)
//! gwi.insert("doc1", &embedding)?;
//!
//! // Search (O(log M + cluster_size))
//! let results = gwi.search(&query, 10)?;
//! ```

use crate::distance::DistanceMetric;
use crate::error::SynaError;
use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes for GWI file format
const GWI_MAGIC: [u8; 4] = [0x47, 0x57, 0x49, 0x58]; // "GWIX"

/// Current file format version
const GWI_VERSION: u32 = 1;

/// Header size in bytes
const HEADER_SIZE: u64 = 64;

/// Default branching factor
const DEFAULT_BRANCHING_FACTOR: u16 = 16;

/// Default number of levels
const DEFAULT_NUM_LEVELS: u8 = 3;

/// Default number of clusters to probe during search
const DEFAULT_NPROBE: usize = 3;

/// Configuration for Gravity Well Index
#[derive(Clone, Debug)]
pub struct GwiConfig {
    /// Vector dimensions (64-8192)
    pub dimensions: u16,
    
    /// Branching factor at each level (default: 16)
    pub branching_factor: u16,
    
    /// Number of hierarchy levels (default: 3)
    pub num_levels: u8,
    
    /// Distance metric
    pub metric: DistanceMetric,
    
    /// Number of clusters to probe during search (default: 3)
    pub nprobe: usize,
    
    /// Initial capacity (number of vectors)
    pub initial_capacity: usize,
    
    /// K-means iterations for attractor selection
    pub kmeans_iterations: usize,
}

impl Default for GwiConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            branching_factor: DEFAULT_BRANCHING_FACTOR,
            num_levels: DEFAULT_NUM_LEVELS,
            metric: DistanceMetric::Cosine,
            nprobe: DEFAULT_NPROBE,
            initial_capacity: 10_000,
            kmeans_iterations: 10,
        }
    }
}

impl GwiConfig {
    /// Calculate total number of leaf attractors
    pub fn num_leaf_attractors(&self) -> usize {
        (self.branching_factor as usize).pow(self.num_levels as u32)
    }
    
    /// Calculate total number of attractors across all levels
    pub fn total_attractors(&self) -> usize {
        let b = self.branching_factor as usize;
        let mut total = 1; // root
        let mut level_size = 1;
        for _ in 0..self.num_levels {
            level_size *= b;
            total += level_size;
        }
        total
    }
}

/// Search result from GWI
#[derive(Debug, Clone)]
pub struct GwiSearchResult {
    /// Key of the vector
    pub key: String,
    /// Distance/score (lower is better for distance metrics)
    pub score: f32,
    /// The vector data
    pub vector: Vec<f32>,
}

/// File header for GWI format
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct GwiHeader {
    magic: [u8; 4],
    version: u32,
    dimensions: u16,
    branching_factor: u16,
    num_levels: u8,
    metric: u8,
    flags: u8,
    _reserved1: u8,
    vector_count: u64,
    write_offset: u64,
    attractor_table_offset: u64,
    cluster_index_offset: u64,
    data_offset: u64,
    _reserved2: [u8; 8],
}

/// The Gravity Well Index
pub struct GravityWellIndex {
    /// Configuration
    config: GwiConfig,
    
    /// Memory-mapped file
    mmap: Option<MmapMut>,
    
    /// File handle
    file: File,
    
    /// Path to the index file
    #[allow(dead_code)]
    path: PathBuf,
    
    /// Attractor hierarchy: attractors[level] = Vec of attractor vectors
    /// Level 0 has 1 attractor (root), Level L has B^L attractors (leaves)
    attractors: Vec<Vec<Vec<f32>>>,
    
    /// Cluster boundaries: leaf_attractor_id -> (start_offset, count)
    cluster_info: Vec<(u64, u64)>,
    
    /// Key to (cluster_id, offset) mapping for lookups
    key_to_location: HashMap<String, (usize, u64)>,
    
    /// Keys in each cluster for iteration
    cluster_keys: Vec<Vec<String>>,
    
    /// Current write offset in data section
    write_offset: u64,
    
    /// Number of vectors stored
    vector_count: u64,
    
    /// Whether attractors have been initialized
    attractors_initialized: bool,
    
    /// Data section start offset
    data_offset: u64,
}

impl GravityWellIndex {
    /// Create a new Gravity Well Index
    pub fn new<P: AsRef<Path>>(path: P, config: GwiConfig) -> Result<Self, SynaError> {
        let path = path.as_ref().to_path_buf();
        
        // Validate config
        if config.dimensions < 64 || config.dimensions > 8192 {
            return Err(SynaError::DimensionMismatch {
                expected: 64,
                got: config.dimensions,
            });
        }
        
        // Calculate sizes
        let num_leaves = config.num_leaf_attractors();
        let attractor_table_size = Self::calculate_attractor_table_size(&config);
        let cluster_index_size = num_leaves * 16; // 2 × u64 per cluster
        let initial_data_size = config.initial_capacity * Self::entry_size_estimate(&config);
        
        let data_offset = HEADER_SIZE + attractor_table_size as u64 + cluster_index_size as u64;
        let file_size = data_offset + initial_data_size as u64;
        
        // Create file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        file.set_len(file_size)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        let mut index = Self {
            config: config.clone(),
            mmap: None,
            file,
            path,
            attractors: Vec::new(),
            cluster_info: vec![(0, 0); num_leaves],
            key_to_location: HashMap::new(),
            cluster_keys: vec![Vec::new(); num_leaves],
            write_offset: data_offset,
            vector_count: 0,
            attractors_initialized: false,
            data_offset,
        };
        
        // Write initial header
        index.write_header()?;
        
        // Memory map the file
        index.mmap = Some(unsafe {
            MmapMut::map_mut(&index.file)
                .map_err(|e| SynaError::InvalidPath(e.to_string()))?
        });
        
        Ok(index)
    }

    /// Open an existing Gravity Well Index
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SynaError> {
        let path = path.as_ref().to_path_buf();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE as usize];
        let mut file_reader = &file;
        file_reader.read_exact(&mut header_bytes)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        let header: GwiHeader = unsafe { std::ptr::read(header_bytes.as_ptr() as *const GwiHeader) };
        
        // Validate magic
        if header.magic != GWI_MAGIC {
            return Err(SynaError::InvalidPath("Invalid GWI file format".to_string()));
        }
        
        // Reconstruct config
        let config = GwiConfig {
            dimensions: header.dimensions,
            branching_factor: header.branching_factor,
            num_levels: header.num_levels,
            metric: DistanceMetric::from_u8(header.metric),
            nprobe: DEFAULT_NPROBE,
            initial_capacity: 10_000,
            kmeans_iterations: 10,
        };
        
        let num_leaves = config.num_leaf_attractors();
        
        // Memory map
        let mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| SynaError::InvalidPath(e.to_string()))?
        };
        
        let mut index = Self {
            config: config.clone(),
            mmap: Some(mmap),
            file,
            path,
            attractors: Vec::new(),
            cluster_info: vec![(0, 0); num_leaves],
            key_to_location: HashMap::new(),
            cluster_keys: vec![Vec::new(); num_leaves],
            write_offset: header.write_offset,
            vector_count: header.vector_count,
            attractors_initialized: header.attractor_table_offset > 0,
            data_offset: header.data_offset,
        };
        
        // Load attractors if initialized
        if index.attractors_initialized {
            index.load_attractors()?;
        }
        
        // Rebuild key index by scanning data
        index.rebuild_key_index()?;
        
        Ok(index)
    }

    /// Initialize attractors from sample vectors using hierarchical K-means
    pub fn initialize_attractors(&mut self, sample_vectors: &[&[f32]]) -> Result<(), SynaError> {
        if sample_vectors.is_empty() {
            return Err(SynaError::InvalidPath("No sample vectors provided".to_string()));
        }
        
        // Validate dimensions
        for v in sample_vectors {
            if v.len() != self.config.dimensions as usize {
                return Err(SynaError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: v.len() as u16,
                });
            }
        }
        
        // Build hierarchical attractors
        self.attractors = self.build_attractor_hierarchy(sample_vectors)?;
        self.attractors_initialized = true;
        
        // Write attractors to file
        self.write_attractors()?;
        
        Ok(())
    }
    
    /// Build attractor hierarchy using hierarchical K-means
    fn build_attractor_hierarchy(&self, vectors: &[&[f32]]) -> Result<Vec<Vec<Vec<f32>>>, SynaError> {
        let dims = self.config.dimensions as usize;
        let b = self.config.branching_factor as usize;
        let levels = self.config.num_levels as usize;
        
        let mut hierarchy: Vec<Vec<Vec<f32>>> = Vec::with_capacity(levels + 1);
        
        // Level 0: Single root attractor (centroid of all vectors)
        let root = Self::compute_centroid(vectors, dims);
        hierarchy.push(vec![root]);
        
        // Build each level
        let mut current_assignments: Vec<usize> = vec![0; vectors.len()];
        
        for level in 1..=levels {
            let mut level_attractors: Vec<Vec<f32>> = Vec::new();
            
            for (parent_id, parent_attractor) in hierarchy[level - 1].iter().enumerate() {
                // Get vectors assigned to this parent
                let parent_vectors: Vec<&[f32]> = vectors
                    .iter()
                    .zip(current_assignments.iter())
                    .filter(|(_, &a)| a == parent_id)
                    .map(|(v, _)| *v)
                    .collect();
                
                if parent_vectors.is_empty() {
                    // No vectors assigned, replicate parent as children
                    for _ in 0..b {
                        level_attractors.push(parent_attractor.clone());
                    }
                } else {
                    // Run K-means to get B children
                    let children = self.kmeans(&parent_vectors, b, dims);
                    level_attractors.extend(children);
                }
            }
            
            // Update assignments for next level
            current_assignments = vectors
                .iter()
                .map(|v| self.find_nearest_attractor(v, &level_attractors))
                .collect();
            
            hierarchy.push(level_attractors);
        }
        
        Ok(hierarchy)
    }

    /// Simple K-means clustering
    fn kmeans(&self, vectors: &[&[f32]], k: usize, dims: usize) -> Vec<Vec<f32>> {
        if vectors.len() <= k {
            // Not enough vectors, just use them directly
            let mut centroids: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
            while centroids.len() < k {
                centroids.push(centroids[0].clone());
            }
            return centroids;
        }
        
        // Initialize centroids by sampling
        let step = vectors.len() / k;
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|i| vectors[i * step].to_vec())
            .collect();
        
        // K-means iterations
        for _ in 0..self.config.kmeans_iterations {
            // Assign vectors to nearest centroid
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| self.find_nearest_attractor(v, &centroids))
                .collect();
            
            // Update centroids
            let mut new_centroids = vec![vec![0.0f32; dims]; k];
            let mut counts = vec![0usize; k];
            
            for (v, &a) in vectors.iter().zip(assignments.iter()) {
                for (i, &val) in v.iter().enumerate() {
                    new_centroids[a][i] += val;
                }
                counts[a] += 1;
            }
            
            for (c, &count) in new_centroids.iter_mut().zip(counts.iter()) {
                if count > 0 {
                    for val in c.iter_mut() {
                        *val /= count as f32;
                    }
                }
            }
            
            // Handle empty clusters
            for (i, &count) in counts.iter().enumerate() {
                if count == 0 {
                    new_centroids[i] = centroids[i].clone();
                }
            }
            
            centroids = new_centroids;
        }
        
        centroids
    }
    
    /// Compute centroid of vectors
    fn compute_centroid(vectors: &[&[f32]], dims: usize) -> Vec<f32> {
        let mut centroid = vec![0.0f32; dims];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        let n = vectors.len() as f32;
        for val in centroid.iter_mut() {
            *val /= n;
        }
        centroid
    }
    
    /// Find nearest attractor in a list
    fn find_nearest_attractor(&self, vector: &[f32], attractors: &[Vec<f32>]) -> usize {
        let mut best_id = 0;
        let mut best_dist = f32::MAX;
        
        for (i, attractor) in attractors.iter().enumerate() {
            let dist = self.distance(vector, attractor);
            if dist < best_dist {
                best_dist = dist;
                best_id = i;
            }
        }
        
        best_id
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, key: &str, vector: &[f32]) -> Result<(), SynaError> {
        if !self.attractors_initialized {
            return Err(SynaError::InvalidPath(
                "Attractors not initialized. Call initialize_attractors() first.".to_string()
            ));
        }
        
        // Validate dimensions
        if vector.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len() as u16,
            });
        }
        
        // Find leaf cluster via hierarchical descent
        let cluster_id = self.find_cluster(vector);
        
        // Calculate entry size
        let entry_size = 2 + 2 + key.len() + vector.len() * 4; // key_len + cluster_id + key + vector
        
        // Ensure capacity
        self.ensure_capacity(entry_size as u64)?;
        
        let mmap = self.mmap.as_mut()
            .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
        
        // Write entry
        let offset = self.write_offset;
        unsafe {
            let ptr = mmap.as_mut_ptr().add(offset as usize);
            
            // Write key length (u16)
            std::ptr::write(ptr as *mut u16, key.len() as u16);
            
            // Write cluster ID (u16)
            std::ptr::write(ptr.add(2) as *mut u16, cluster_id as u16);
            
            // Write key bytes
            std::ptr::copy_nonoverlapping(key.as_ptr(), ptr.add(4), key.len());
            
            // Write vector
            std::ptr::copy_nonoverlapping(
                vector.as_ptr() as *const u8,
                ptr.add(4 + key.len()),
                vector.len() * 4,
            );
        }
        
        // Update metadata
        self.key_to_location.insert(key.to_string(), (cluster_id, offset));
        self.cluster_keys[cluster_id].push(key.to_string());
        self.cluster_info[cluster_id].1 += 1;
        self.write_offset += entry_size as u64;
        self.vector_count += 1;
        
        Ok(())
    }
    
    /// Find the leaf cluster for a vector via hierarchical descent
    fn find_cluster(&self, vector: &[f32]) -> usize {
        let b = self.config.branching_factor as usize;
        let mut current_id = 0;
        
        for level in 1..=self.config.num_levels as usize {
            // Get children of current attractor
            let start_child = current_id * b;
            let end_child = (start_child + b).min(self.attractors[level].len());
            
            // Find nearest child
            let mut best_id = start_child;
            let mut best_dist = f32::MAX;
            
            for i in start_child..end_child {
                let dist = self.distance(vector, &self.attractors[level][i]);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = i;
                }
            }
            
            current_id = best_id;
        }
        
        current_id
    }

    /// Batch insert for maximum throughput
    pub fn insert_batch(&mut self, keys: &[&str], vectors: &[&[f32]]) -> Result<usize, SynaError> {
        if keys.len() != vectors.len() {
            return Err(SynaError::ShapeMismatch {
                data_size: vectors.len(),
                expected_size: keys.len(),
            });
        }
        
        let mut inserted = 0;
        for (key, vector) in keys.iter().zip(vectors.iter()) {
            self.insert(key, vector)?;
            inserted += 1;
        }
        
        Ok(inserted)
    }
    
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<GwiSearchResult>, SynaError> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }
    
    /// Search for k nearest neighbors with custom nprobe
    /// 
    /// Higher nprobe = better recall but slower search.
    /// - nprobe=3: Fast, ~5-15% recall
    /// - nprobe=10: Balanced, ~30-50% recall
    /// - nprobe=30: High quality, ~70-90% recall
    /// - nprobe=100: Near-exact, ~95%+ recall
    pub fn search_with_nprobe(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<GwiSearchResult>, SynaError> {
        if !self.attractors_initialized {
            return Err(SynaError::InvalidPath(
                "Attractors not initialized".to_string()
            ));
        }
        
        // Validate dimensions
        if query.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len() as u16,
            });
        }
        
        // Find primary cluster
        let primary_cluster = self.find_cluster(query);
        
        // Find clusters to probe (primary + nearest neighbors)
        let clusters_to_probe = self.find_probe_clusters_n(query, primary_cluster, nprobe);
        
        // Collect candidates from all probed clusters
        let mut candidates: Vec<(String, f32, Vec<f32>)> = Vec::new();
        
        for cluster_id in clusters_to_probe {
            let cluster_vectors = self.get_cluster_vectors(cluster_id)?;
            for (key, vector) in cluster_vectors {
                let dist = self.distance(query, &vector);
                candidates.push((key, dist, vector));
            }
        }
        
        // Sort by distance and return top-k
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(key, score, vector)| GwiSearchResult { key, score, vector })
            .collect())
    }
    
    /// Find clusters to probe using hierarchical descent (much faster than brute force)
    /// 
    /// Instead of comparing to all 4096 leaf attractors, we descend through the
    /// hierarchy keeping only the most promising candidates at each level.
    /// 
    /// Comparisons: ~200 instead of 4096 (20x fewer!)
    #[allow(dead_code)]
    fn find_probe_clusters(&self, query: &[f32], primary: usize) -> Vec<usize> {
        self.find_probe_clusters_n(query, primary, self.config.nprobe)
    }
    
    /// Find clusters to probe with custom nprobe
    fn find_probe_clusters_n(&self, query: &[f32], _primary: usize, nprobe: usize) -> Vec<usize> {
        if nprobe <= 1 {
            // Just use hierarchical descent for single probe
            return vec![self.find_cluster(query)];
        }
        
        let b = self.config.branching_factor as usize;
        let num_levels = self.config.num_levels as usize;
        
        // Start with root's children (level 1)
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(b * 2);
        
        // Level 1: Compare to all B children of root
        for i in 0..self.attractors[1].len().min(b) {
            let dist = self.distance(query, &self.attractors[1][i]);
            candidates.push((i, dist));
        }
        
        // Sort and keep top candidates
        candidates.sort_by(|a, c| a.1.total_cmp(&c.1));
        // Keep sqrt(nprobe) * branching_factor at intermediate levels for better coverage
        let keep_per_level = ((nprobe as f32).sqrt().ceil() as usize * b).max(nprobe).max(4);
        candidates.truncate(keep_per_level);
        
        // Descend through remaining levels
        for level in 2..=num_levels {
            let mut next_candidates: Vec<(usize, f32)> = Vec::with_capacity(candidates.len() * b);
            
            for (parent_id, _) in &candidates {
                let start_child = parent_id * b;
                let end_child = (start_child + b).min(self.attractors[level].len());
                
                for child_id in start_child..end_child {
                    let dist = self.distance(query, &self.attractors[level][child_id]);
                    next_candidates.push((child_id, dist));
                }
            }
            
            // Sort and keep top candidates
            next_candidates.sort_by(|a, c| a.1.total_cmp(&c.1));
            
            // At leaf level, keep exactly nprobe; otherwise keep more for coverage
            let keep = if level == num_levels {
                nprobe
            } else {
                keep_per_level
            };
            next_candidates.truncate(keep);
            
            candidates = next_candidates;
        }
        
        // Return cluster IDs
        candidates.into_iter().map(|(id, _)| id).collect()
    }

    /// Get all vectors in a cluster
    fn get_cluster_vectors(&self, cluster_id: usize) -> Result<Vec<(String, Vec<f32>)>, SynaError> {
        let mut vectors = Vec::new();
        
        for key in &self.cluster_keys[cluster_id] {
            if let Some(&(_, offset)) = self.key_to_location.get(key) {
                let (read_key, vector) = self.read_entry_at(offset)?;
                if read_key == *key {
                    vectors.push((read_key, vector));
                }
            }
        }
        
        Ok(vectors)
    }
    
    /// Read an entry at a given offset
    fn read_entry_at(&self, offset: u64) -> Result<(String, Vec<f32>), SynaError> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
        
        let dims = self.config.dimensions as usize;
        
        unsafe {
            let ptr = mmap.as_ptr().add(offset as usize);
            
            // Read key length
            let key_len = std::ptr::read(ptr as *const u16) as usize;
            
            // Skip cluster_id (2 bytes)
            // Read key
            let key_bytes = std::slice::from_raw_parts(ptr.add(4), key_len);
            let key = String::from_utf8_lossy(key_bytes).to_string();
            
            // Read vector
            let vector_ptr = ptr.add(4 + key_len) as *const f32;
            let vector: Vec<f32> = std::slice::from_raw_parts(vector_ptr, dims).to_vec();
            
            Ok((key, vector))
        }
    }
    
    /// Calculate distance between two vectors
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => self.cosine_distance(a, b),
            DistanceMetric::Euclidean => self.euclidean_distance(a, b),
            DistanceMetric::DotProduct => -self.dot_product(a, b), // Negate for "lower is better"
        }
    }
    
    // ==========================================================================
    // SIMD-Optimized Distance Functions
    // ==========================================================================
    // These use manual loop unrolling and compiler auto-vectorization hints
    // to achieve near-optimal SIMD performance without unsafe intrinsics.
    
    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let (dot, norm_a_sq, norm_b_sq) = self.dot_and_norms_simd(a, b);
        
        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        
        1.0 - (dot / (norm_a * norm_b))
    }
    
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.euclidean_squared_simd(a, b).sqrt()
    }
    
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        self.dot_product_simd(a, b)
    }
    
    /// SIMD-optimized dot product using 8-way unrolling
    #[inline(always)]
    fn dot_product_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Process 8 elements at a time (fits in AVX register)
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut sum4 = 0.0f32;
        let mut sum5 = 0.0f32;
        let mut sum6 = 0.0f32;
        let mut sum7 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 8;
            // Compiler will auto-vectorize this with AVX2
            sum0 += a[base] * b[base];
            sum1 += a[base + 1] * b[base + 1];
            sum2 += a[base + 2] * b[base + 2];
            sum3 += a[base + 3] * b[base + 3];
            sum4 += a[base + 4] * b[base + 4];
            sum5 += a[base + 5] * b[base + 5];
            sum6 += a[base + 6] * b[base + 6];
            sum7 += a[base + 7] * b[base + 7];
        }
        
        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            sum0 += a[base + i] * b[base + i];
        }
        
        // Reduce
        (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7)
    }
    
    /// SIMD-optimized squared Euclidean distance
    #[inline(always)]
    fn euclidean_squared_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;
        
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut sum4 = 0.0f32;
        let mut sum5 = 0.0f32;
        let mut sum6 = 0.0f32;
        let mut sum7 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 8;
            let d0 = a[base] - b[base];
            let d1 = a[base + 1] - b[base + 1];
            let d2 = a[base + 2] - b[base + 2];
            let d3 = a[base + 3] - b[base + 3];
            let d4 = a[base + 4] - b[base + 4];
            let d5 = a[base + 5] - b[base + 5];
            let d6 = a[base + 6] - b[base + 6];
            let d7 = a[base + 7] - b[base + 7];
            
            sum0 += d0 * d0;
            sum1 += d1 * d1;
            sum2 += d2 * d2;
            sum3 += d3 * d3;
            sum4 += d4 * d4;
            sum5 += d5 * d5;
            sum6 += d6 * d6;
            sum7 += d7 * d7;
        }
        
        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let d = a[base + i] - b[base + i];
            sum0 += d * d;
        }
        
        (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7)
    }
    
    /// SIMD-optimized computation of dot product and both norms in one pass
    /// This is more cache-efficient than computing them separately
    #[inline(always)]
    fn dot_and_norms_simd(&self, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Accumulators for dot product
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut dot4 = 0.0f32;
        let mut dot5 = 0.0f32;
        let mut dot6 = 0.0f32;
        let mut dot7 = 0.0f32;
        
        // Accumulators for norm_a squared
        let mut na0 = 0.0f32;
        let mut na1 = 0.0f32;
        let mut na2 = 0.0f32;
        let mut na3 = 0.0f32;
        let mut na4 = 0.0f32;
        let mut na5 = 0.0f32;
        let mut na6 = 0.0f32;
        let mut na7 = 0.0f32;
        
        // Accumulators for norm_b squared
        let mut nb0 = 0.0f32;
        let mut nb1 = 0.0f32;
        let mut nb2 = 0.0f32;
        let mut nb3 = 0.0f32;
        let mut nb4 = 0.0f32;
        let mut nb5 = 0.0f32;
        let mut nb6 = 0.0f32;
        let mut nb7 = 0.0f32;
        
        for i in 0..chunks {
            let base = i * 8;
            
            let a0 = a[base];
            let a1 = a[base + 1];
            let a2 = a[base + 2];
            let a3 = a[base + 3];
            let a4 = a[base + 4];
            let a5 = a[base + 5];
            let a6 = a[base + 6];
            let a7 = a[base + 7];
            
            let b0 = b[base];
            let b1 = b[base + 1];
            let b2 = b[base + 2];
            let b3 = b[base + 3];
            let b4 = b[base + 4];
            let b5 = b[base + 5];
            let b6 = b[base + 6];
            let b7 = b[base + 7];
            
            dot0 += a0 * b0;
            dot1 += a1 * b1;
            dot2 += a2 * b2;
            dot3 += a3 * b3;
            dot4 += a4 * b4;
            dot5 += a5 * b5;
            dot6 += a6 * b6;
            dot7 += a7 * b7;
            
            na0 += a0 * a0;
            na1 += a1 * a1;
            na2 += a2 * a2;
            na3 += a3 * a3;
            na4 += a4 * a4;
            na5 += a5 * a5;
            na6 += a6 * a6;
            na7 += a7 * a7;
            
            nb0 += b0 * b0;
            nb1 += b1 * b1;
            nb2 += b2 * b2;
            nb3 += b3 * b3;
            nb4 += b4 * b4;
            nb5 += b5 * b5;
            nb6 += b6 * b6;
            nb7 += b7 * b7;
        }
        
        // Handle remainder
        let base = chunks * 8;
        for i in 0..remainder {
            let ai = a[base + i];
            let bi = b[base + i];
            dot0 += ai * bi;
            na0 += ai * ai;
            nb0 += bi * bi;
        }
        
        let dot = (dot0 + dot1) + (dot2 + dot3) + (dot4 + dot5) + (dot6 + dot7);
        let norm_a_sq = (na0 + na1) + (na2 + na3) + (na4 + na5) + (na6 + na7);
        let norm_b_sq = (nb0 + nb1) + (nb2 + nb3) + (nb4 + nb5) + (nb6 + nb7);
        
        (dot, norm_a_sq, norm_b_sq)
    }

    /// Ensure file has enough capacity
    fn ensure_capacity(&mut self, additional: u64) -> Result<(), SynaError> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
        
        let required = self.write_offset + additional;
        if required <= mmap.len() as u64 {
            return Ok(());
        }
        
        // Need to grow file
        let new_size = (required * 2).max(mmap.len() as u64 * 2);
        
        // Drop mmap before resizing
        self.mmap = None;
        
        self.file.set_len(new_size)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        // Re-map
        self.mmap = Some(unsafe {
            MmapMut::map_mut(&self.file)
                .map_err(|e| SynaError::InvalidPath(e.to_string()))?
        });
        
        Ok(())
    }
    
    /// Write header to file
    fn write_header(&mut self) -> Result<(), SynaError> {
        let attractor_table_size = Self::calculate_attractor_table_size(&self.config);
        let num_leaves = self.config.num_leaf_attractors();
        let _cluster_index_size = num_leaves * 16;
        
        let header = GwiHeader {
            magic: GWI_MAGIC,
            version: GWI_VERSION,
            dimensions: self.config.dimensions,
            branching_factor: self.config.branching_factor,
            num_levels: self.config.num_levels,
            metric: self.config.metric.to_u8(),
            flags: 0,
            _reserved1: 0,
            vector_count: self.vector_count,
            write_offset: self.write_offset,
            attractor_table_offset: HEADER_SIZE,
            cluster_index_offset: HEADER_SIZE + attractor_table_size as u64,
            data_offset: self.data_offset,
            _reserved2: [0; 8],
        };
        
        self.file.seek(SeekFrom::Start(0))
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        let header_bytes: [u8; HEADER_SIZE as usize] = unsafe {
            std::mem::transmute(header)
        };
        
        self.file.write_all(&header_bytes)
            .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        
        Ok(())
    }
    
    /// Write attractors to file
    fn write_attractors(&mut self) -> Result<(), SynaError> {
        let mmap = self.mmap.as_mut()
            .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
        
        let mut offset = HEADER_SIZE as usize;
        
        for level_attractors in &self.attractors {
            for attractor in level_attractors {
                unsafe {
                    let ptr = mmap.as_mut_ptr().add(offset);
                    std::ptr::copy_nonoverlapping(
                        attractor.as_ptr() as *const u8,
                        ptr,
                        attractor.len() * 4,
                    );
                }
                offset += attractor.len() * 4;
            }
        }
        
        // Update header
        self.write_header()?;
        
        Ok(())
    }

    /// Load attractors from file
    fn load_attractors(&mut self) -> Result<(), SynaError> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
        
        let dims = self.config.dimensions as usize;
        let b = self.config.branching_factor as usize;
        let levels = self.config.num_levels as usize;
        
        let mut offset = HEADER_SIZE as usize;
        self.attractors = Vec::with_capacity(levels + 1);
        
        // Level 0: 1 attractor
        let mut level_size = 1;
        for _level in 0..=levels {
            let mut level_attractors = Vec::with_capacity(level_size);
            
            for _ in 0..level_size {
                unsafe {
                    let ptr = mmap.as_ptr().add(offset) as *const f32;
                    let attractor: Vec<f32> = std::slice::from_raw_parts(ptr, dims).to_vec();
                    level_attractors.push(attractor);
                }
                offset += dims * 4;
            }
            
            self.attractors.push(level_attractors);
            level_size *= b;
        }
        
        Ok(())
    }
    
    /// Rebuild key index by scanning data section
    fn rebuild_key_index(&mut self) -> Result<(), SynaError> {
        self.key_to_location.clear();
        for keys in &mut self.cluster_keys {
            keys.clear();
        }
        for info in &mut self.cluster_info {
            *info = (0, 0);
        }
        
        let dims = self.config.dimensions as usize;
        let mut offset = self.data_offset;
        
        while offset < self.write_offset {
            let mmap = self.mmap.as_ref()
                .ok_or_else(|| SynaError::InvalidPath("mmap not initialized".to_string()))?;
            
            unsafe {
                let ptr = mmap.as_ptr().add(offset as usize);
                
                // Read key length
                let key_len = std::ptr::read(ptr as *const u16) as usize;
                
                // Read cluster ID
                let cluster_id = std::ptr::read(ptr.add(2) as *const u16) as usize;
                
                // Read key
                let key_bytes = std::slice::from_raw_parts(ptr.add(4), key_len);
                let key = String::from_utf8_lossy(key_bytes).to_string();
                
                // Update index
                self.key_to_location.insert(key.clone(), (cluster_id, offset));
                if cluster_id < self.cluster_keys.len() {
                    self.cluster_keys[cluster_id].push(key);
                    self.cluster_info[cluster_id].1 += 1;
                }
                
                // Move to next entry
                let entry_size = 4 + key_len + dims * 4;
                offset += entry_size as u64;
            }
        }
        
        Ok(())
    }
    
    /// Calculate attractor table size in bytes
    fn calculate_attractor_table_size(config: &GwiConfig) -> usize {
        let dims = config.dimensions as usize;
        let total = config.total_attractors();
        total * dims * 4
    }
    
    /// Estimate entry size for capacity planning
    fn entry_size_estimate(config: &GwiConfig) -> usize {
        4 + 16 + config.dimensions as usize * 4 // key_len + cluster_id + avg_key + vector
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<(), SynaError> {
        if let Some(ref mmap) = self.mmap {
            mmap.flush()
                .map_err(|e| SynaError::InvalidPath(e.to_string()))?;
        }
        Ok(())
    }
    
    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.vector_count as usize
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }
    
    /// Get dimensions
    pub fn dimensions(&self) -> u16 {
        self.config.dimensions
    }
    
    /// Get number of leaf clusters
    pub fn num_clusters(&self) -> usize {
        self.config.num_leaf_attractors()
    }
    
    /// Check if a key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.key_to_location.contains_key(key)
    }
    
    /// Get a vector by key
    pub fn get(&self, key: &str) -> Result<Option<Vec<f32>>, SynaError> {
        match self.key_to_location.get(key) {
            Some(&(_, offset)) => {
                let (_, vector) = self.read_entry_at(offset)?;
                Ok(Some(vector))
            }
            None => Ok(None),
        }
    }
    
    /// Get cluster statistics
    pub fn cluster_stats(&self) -> Vec<(usize, usize)> {
        self.cluster_keys
            .iter()
            .enumerate()
            .map(|(id, keys)| (id, keys.len()))
            .collect()
    }
    
    /// Close the index (flush and release resources)
    pub fn close(&mut self) -> Result<(), SynaError> {
        // Update header with final counts
        self.write_header()?;
        self.flush()?;
        self.mmap = None;
        Ok(())
    }
}

impl Drop for GravityWellIndex {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

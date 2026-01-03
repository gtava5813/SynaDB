//! Configuration for Cascade Index

use crate::distance::DistanceMetric;

/// Configuration for Cascade Index
#[derive(Clone, Debug)]
pub struct CascadeConfig {
    /// Vector dimensions
    pub dimensions: u16,

    /// Distance metric for similarity
    pub metric: DistanceMetric,

    // LSH parameters
    /// Number of hash bits per table (2^num_bits buckets per table)
    pub num_bits: usize,

    /// Number of hash tables (more tables = better recall, more memory)
    pub num_tables: usize,

    // Bucket parameters
    /// Threshold for bucket splitting
    pub split_threshold: usize,

    /// Maximum bucket tree depth
    pub max_bucket_depth: usize,

    // Graph parameters
    /// Target number of neighbors per node
    pub m: usize,

    /// Maximum neighbors per node (hard limit)
    pub m_max: usize,

    /// Whether to add bidirectional edges
    pub bidirectional: bool,

    // Search parameters
    /// Number of LSH probes (flip bits to check nearby buckets)
    pub num_probes: usize,

    /// Search expansion factor (like HNSW ef_search)
    pub ef_search: usize,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            // Balanced defaults for good recall and reasonable speed
            num_bits: 6,          // 64 buckets per table
            num_tables: 10,       // Good coverage
            split_threshold: 400, // Moderate bucket size
            max_bucket_depth: 4,
            m: 20, // Moderate neighbors
            m_max: 40,
            bidirectional: true,
            num_probes: 16, // More probes for better recall
            ef_search: 80,  // Balance between speed and recall
        }
    }
}

impl CascadeConfig {
    /// Create config for small datasets (< 10K vectors)
    pub fn small() -> Self {
        Self {
            num_bits: 5, // Only 32 buckets - ensures collisions
            num_tables: 8,
            split_threshold: 200,
            max_bucket_depth: 3,
            m: 16,
            m_max: 32,
            num_probes: 12,
            ef_search: 100,
            ..Default::default()
        }
    }

    /// Create config for large datasets (> 100K vectors)
    pub fn large() -> Self {
        Self {
            num_bits: 8,    // 256 buckets per table
            num_tables: 16, // More tables
            split_threshold: 1000,
            max_bucket_depth: 5,
            m: 32,
            m_max: 64,
            num_probes: 24,
            ef_search: 300,
            ..Default::default()
        }
    }

    /// Create config optimized for high recall (> 99%)
    pub fn high_recall() -> Self {
        Self {
            num_bits: 4,    // Only 16 buckets - maximum collisions
            num_tables: 20, // Many tables
            split_threshold: 2000,
            max_bucket_depth: 3,
            num_probes: 32, // Probe almost all buckets
            m: 48,
            m_max: 96,
            ef_search: 500,
            ..Default::default()
        }
    }

    /// Create config optimized for fast search
    pub fn fast_search() -> Self {
        Self {
            num_bits: 8,
            num_tables: 6,
            num_probes: 8,
            m: 16,
            m_max: 32,
            ef_search: 50,
            split_threshold: 300,
            max_bucket_depth: 5,
            ..Default::default()
        }
    }
}

//! Adaptive bucket tree for Cascade Index
//!
//! Buckets automatically split when they exceed a threshold,
//! maintaining O(1) amortized lookup time.

use std::collections::HashMap;

/// Configuration for adaptive buckets
#[derive(Clone, Debug)]
pub struct BucketConfig {
    /// Threshold for splitting a bucket
    pub split_threshold: usize,

    /// Maximum depth of bucket tree
    pub max_depth: usize,

    /// Minimum vectors required to split
    pub min_split_size: usize,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            split_threshold: 100,
            max_depth: 10,
            min_split_size: 10,
        }
    }
}

/// A node in the adaptive bucket tree
pub struct BucketNode {
    /// Vector IDs in this bucket (leaf only)
    vectors: Vec<usize>,

    /// Split hyperplane (internal nodes only)
    split_plane: Option<Vec<f32>>,

    /// Left child (dot product <= 0)
    left: Option<Box<BucketNode>>,

    /// Right child (dot product > 0)
    right: Option<Box<BucketNode>>,

    /// Depth in the tree
    depth: usize,
}

impl BucketNode {
    /// Create a new leaf bucket
    pub fn new(depth: usize) -> Self {
        Self {
            vectors: Vec::new(),
            split_plane: None,
            left: None,
            right: None,
            depth,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Insert a vector ID into the bucket
    pub fn insert(
        &mut self,
        id: usize,
        vector: &[f32],
        config: &BucketConfig,
        all_vectors: &[Vec<f32>],
    ) {
        if self.is_leaf() {
            self.vectors.push(id);

            // Check if we need to split
            if self.vectors.len() > config.split_threshold
                && self.depth < config.max_depth
                && self.vectors.len() >= config.min_split_size * 2
            {
                self.split(all_vectors, config);
            }
        } else {
            // Route to appropriate child
            let dot = self.dot_with_split_plane(vector);
            if dot > 0.0 {
                if let Some(ref mut right) = self.right {
                    right.insert(id, vector, config, all_vectors);
                }
            } else if let Some(ref mut left) = self.left {
                left.insert(id, vector, config, all_vectors);
            }
        }
    }

    /// Query vectors in this bucket
    pub fn query(&self, vector: &[f32]) -> Vec<usize> {
        if self.is_leaf() {
            self.vectors.clone()
        } else {
            let dot = self.dot_with_split_plane(vector);
            if dot > 0.0 {
                self.right
                    .as_ref()
                    .map(|r| r.query(vector))
                    .unwrap_or_default()
            } else {
                self.left
                    .as_ref()
                    .map(|l| l.query(vector))
                    .unwrap_or_default()
            }
        }
    }

    /// Query with neighboring buckets for better recall
    pub fn query_with_neighbors(&self, vector: &[f32], include_sibling: bool) -> Vec<usize> {
        if self.is_leaf() {
            self.vectors.clone()
        } else {
            let dot = self.dot_with_split_plane(vector);
            let mut results = Vec::new();

            // Always include the primary path
            if dot > 0.0 {
                if let Some(ref right) = self.right {
                    results.extend(right.query_with_neighbors(vector, include_sibling));
                }
                // Include sibling if close to boundary OR if include_sibling is true
                // Use a larger threshold for better recall
                if include_sibling || dot.abs() < 0.3 {
                    if let Some(ref left) = self.left {
                        results.extend(left.query(vector));
                    }
                }
            } else {
                if let Some(ref left) = self.left {
                    results.extend(left.query_with_neighbors(vector, include_sibling));
                }
                if include_sibling || dot.abs() < 0.3 {
                    if let Some(ref right) = self.right {
                        results.extend(right.query(vector));
                    }
                }
            }

            results
        }
    }

    /// Split this bucket into two children
    fn split(&mut self, all_vectors: &[Vec<f32>], config: &BucketConfig) {
        if self.vectors.is_empty() {
            return;
        }

        // Compute centroid
        let dims = all_vectors
            .get(self.vectors[0])
            .map(|v| v.len())
            .unwrap_or(0);
        if dims == 0 {
            return;
        }

        let mut centroid = vec![0.0f32; dims];
        let mut count = 0;

        for &id in &self.vectors {
            if let Some(vec) = all_vectors.get(id) {
                for (c, v) in centroid.iter_mut().zip(vec.iter()) {
                    *c += v;
                }
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        for c in centroid.iter_mut() {
            *c /= count as f32;
        }

        // Find vector furthest from centroid
        let mut max_dist = 0.0f32;
        let mut furthest_id = self.vectors[0];

        for &id in &self.vectors {
            if let Some(vec) = all_vectors.get(id) {
                let dist: f32 = vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(v, c)| (v - c).powi(2))
                    .sum();
                if dist > max_dist {
                    max_dist = dist;
                    furthest_id = id;
                }
            }
        }

        // Create split plane: direction from centroid to furthest point
        let mut split_plane = vec![0.0f32; dims];
        if let Some(furthest) = all_vectors.get(furthest_id) {
            for (s, (f, c)) in split_plane
                .iter_mut()
                .zip(furthest.iter().zip(centroid.iter()))
            {
                *s = f - c;
            }

            // Normalize
            let norm: f32 = split_plane.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for s in split_plane.iter_mut() {
                    *s /= norm;
                }
            }
        }

        // Partition vectors
        let mut left_ids = Vec::new();
        let mut right_ids = Vec::new();

        for &id in &self.vectors {
            if let Some(vec) = all_vectors.get(id) {
                let dot: f32 = vec.iter().zip(split_plane.iter()).map(|(v, s)| v * s).sum();

                if dot > 0.0 {
                    right_ids.push(id);
                } else {
                    left_ids.push(id);
                }
            }
        }

        // Only split if both sides have vectors
        if left_ids.is_empty() || right_ids.is_empty() {
            return;
        }

        // Create children
        let mut left = Box::new(BucketNode::new(self.depth + 1));
        left.vectors = left_ids;

        let mut right = Box::new(BucketNode::new(self.depth + 1));
        right.vectors = right_ids;

        // Recursively split if needed
        if left.vectors.len() > config.split_threshold && left.depth < config.max_depth {
            left.split(all_vectors, config);
        }
        if right.vectors.len() > config.split_threshold && right.depth < config.max_depth {
            right.split(all_vectors, config);
        }

        self.split_plane = Some(split_plane);
        self.left = Some(left);
        self.right = Some(right);
        self.vectors.clear();
    }

    /// Compute dot product with split plane
    fn dot_with_split_plane(&self, vector: &[f32]) -> f32 {
        self.split_plane
            .as_ref()
            .map(|plane| vector.iter().zip(plane.iter()).map(|(v, p)| v * p).sum())
            .unwrap_or(0.0)
    }

    /// Get total vector count in this subtree
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        if self.is_leaf() {
            self.vectors.len()
        } else {
            let left_count = self.left.as_ref().map(|l| l.len()).unwrap_or(0);
            let right_count = self.right.as_ref().map(|r| r.len()).unwrap_or(0);
            left_count + right_count
        }
    }

    /// Check if empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Collection of bucket trees (one per LSH hash value per table)
pub struct BucketForest {
    /// Buckets indexed by (table_index, hash_value)
    trees: Vec<HashMap<u64, BucketNode>>,

    /// Configuration
    config: BucketConfig,
}

impl BucketForest {
    /// Create a new bucket forest
    pub fn new(num_tables: usize, config: BucketConfig) -> Self {
        let trees = (0..num_tables).map(|_| HashMap::new()).collect();
        Self { trees, config }
    }

    /// Insert a vector into the appropriate bucket
    pub fn insert(
        &mut self,
        table: usize,
        hash: u64,
        id: usize,
        vector: &[f32],
        all_vectors: &[Vec<f32>],
    ) {
        if table >= self.trees.len() {
            return;
        }

        let bucket = self.trees[table]
            .entry(hash)
            .or_insert_with(|| BucketNode::new(0));

        bucket.insert(id, vector, &self.config, all_vectors);
    }

    /// Query vectors from a bucket
    pub fn query(&self, table: usize, hash: u64, vector: &[f32]) -> Vec<usize> {
        self.trees
            .get(table)
            .and_then(|t| t.get(&hash))
            .map(|b| b.query(vector))
            .unwrap_or_default()
    }

    /// Query with neighboring buckets
    pub fn query_with_neighbors(&self, table: usize, hash: u64, vector: &[f32]) -> Vec<usize> {
        self.trees
            .get(table)
            .and_then(|t| t.get(&hash))
            .map(|b| b.query_with_neighbors(vector, true))
            .unwrap_or_default()
    }

    /// Get total bucket count
    pub fn bucket_count(&self) -> usize {
        self.trees.iter().map(|t| t.len()).sum()
    }

    /// Get number of tables
    pub fn num_tables(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_insert_query() {
        let config = BucketConfig {
            split_threshold: 10,
            max_depth: 5,
            min_split_size: 2,
        };

        let mut forest = BucketForest::new(1, config);
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();

        for (id, vec) in vectors.iter().enumerate() {
            forest.insert(0, 42, id, vec, &vectors);
        }

        let results = forest.query(0, 42, &vectors[0]);
        assert!(results.contains(&0));
    }

    #[test]
    fn test_bucket_split() {
        let config = BucketConfig {
            split_threshold: 5,
            max_depth: 3,
            min_split_size: 2,
        };

        let mut forest = BucketForest::new(1, config);
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32, (i % 5) as f32, ((i * 7) % 13) as f32])
            .collect();

        for (id, vec) in vectors.iter().enumerate() {
            forest.insert(0, 42, id, vec, &vectors);
        }

        // All vectors should still be queryable
        for (id, vec) in vectors.iter().enumerate() {
            let results = forest.query(0, 42, vec);
            assert!(results.contains(&id), "Vector {} not found", id);
        }
    }
}

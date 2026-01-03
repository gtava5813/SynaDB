//! Sparse graph layer for Cascade Index
//!
//! Maintains neighbor connections for graph-based search refinement.

use crate::distance::DistanceMetric;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for the sparse graph
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Target number of neighbors per node
    pub m: usize,

    /// Maximum neighbors per node (hard limit)
    pub m_max: usize,

    /// Whether to add bidirectional edges
    pub bidirectional: bool,

    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            bidirectional: true,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// A node in the sparse graph
#[derive(Clone, Debug)]
pub struct CascadeNode {
    /// Node ID (index in nodes vector)
    pub id: usize,

    /// User-provided key
    pub key: String,

    /// Vector data
    pub vector: Vec<f32>,

    /// Neighbors: (node_id, distance)
    pub neighbors: Vec<(usize, f32)>,
}

impl CascadeNode {
    /// Create a new node
    pub fn new(id: usize, key: String, vector: Vec<f32>) -> Self {
        Self {
            id,
            key,
            vector,
            neighbors: Vec::new(),
        }
    }
}

/// Ordered candidate for priority queue (min-heap by distance)
#[derive(Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
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
        // Reverse for min-heap (smaller distance = higher priority)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Graph statistics
#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_neighbors: f32,
    pub max_neighbors: usize,
    pub min_neighbors: usize,
}

/// Sparse graph for neighbor refinement
pub struct CascadeGraph {
    /// All nodes
    nodes: Vec<CascadeNode>,

    /// Key to node ID mapping
    key_to_id: HashMap<String, usize>,

    /// Configuration
    config: GraphConfig,
}

impl CascadeGraph {
    /// Create a new graph
    pub fn new(config: GraphConfig) -> Self {
        Self {
            nodes: Vec::new(),
            key_to_id: HashMap::new(),
            config,
        }
    }

    /// Insert a new node with candidate neighbors
    pub fn insert(&mut self, key: String, vector: Vec<f32>, candidates: &[usize]) {
        let id = self.nodes.len();

        // Check for duplicate key
        if self.key_to_id.contains_key(&key) {
            return;
        }

        // Create node
        let mut node = CascadeNode::new(id, key.clone(), vector.clone());

        // Collect all valid candidates
        let mut neighbor_candidates: Vec<(usize, f32)> = candidates
            .iter()
            .filter(|&&cid| cid < self.nodes.len() && !self.nodes[cid].key.is_empty())
            .map(|&cid| {
                let dist = self.compute_distance(&vector, &self.nodes[cid].vector);
                (cid, dist)
            })
            .collect();

        // If we have few candidates, sample from existing nodes
        // This ensures good connectivity even for early insertions
        if neighbor_candidates.len() < self.config.m && !self.nodes.is_empty() {
            let sample_size = std::cmp::min(self.config.m * 4, self.nodes.len());
            let step = std::cmp::max(1, self.nodes.len() / sample_size);

            for i in (0..self.nodes.len()).step_by(step) {
                if self.nodes[i].key.is_empty() {
                    continue;
                }
                if neighbor_candidates.iter().any(|(cid, _)| *cid == i) {
                    continue;
                }
                let dist = self.compute_distance(&vector, &self.nodes[i].vector);
                neighbor_candidates.push((i, dist));

                if neighbor_candidates.len() >= self.config.m * 4 {
                    break;
                }
            }
        }

        // Sort by distance
        neighbor_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Take top M neighbors
        node.neighbors = neighbor_candidates
            .into_iter()
            .take(self.config.m)
            .collect();

        // Add bidirectional edges
        if self.config.bidirectional {
            for &(neighbor_id, dist) in &node.neighbors {
                if neighbor_id < self.nodes.len() {
                    self.nodes[neighbor_id].neighbors.push((id, dist));
                    self.prune_neighbors(neighbor_id);
                }
            }
        }

        self.key_to_id.insert(key, id);
        self.nodes.push(node);
    }

    /// Search for nearest neighbors using graph traversal
    pub fn search(
        &self,
        query: &[f32],
        entry_points: &[usize],
        k: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() || entry_points.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::with_capacity(ef * 2);
        let mut candidates = BinaryHeap::new(); // min-heap
        let mut results = BinaryHeap::new(); // max-heap for top-k

        // Initialize with entry points (limit to avoid too many initial candidates)
        let max_initial = std::cmp::min(entry_points.len(), ef);
        for &ep in entry_points.iter().take(max_initial) {
            if ep < self.nodes.len() && !visited.contains(&ep) {
                visited.insert(ep);
                let dist = self.compute_distance(query, &self.nodes[ep].vector);
                candidates.push(Candidate {
                    id: ep,
                    distance: dist,
                });
                results.push(std::cmp::Reverse(Candidate {
                    id: ep,
                    distance: dist,
                }));
            }
        }

        // Graph traversal with early termination
        let mut stale_count = 0;
        let max_stale = 10; // Stop after 10 iterations without improvement

        while let Some(current) = candidates.pop() {
            // Get worst distance in results
            let worst_dist = if results.len() >= ef {
                results.peek().map(|r| r.0.distance).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            // Stop if current is worse than worst result
            if current.distance > worst_dist {
                stale_count += 1;
                if stale_count >= max_stale {
                    break;
                }
                continue;
            }

            stale_count = 0; // Reset on improvement

            // Explore neighbors
            for &(neighbor_id, _) in &self.nodes[current.id].neighbors {
                if neighbor_id < self.nodes.len() && !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    let dist = self.compute_distance(query, &self.nodes[neighbor_id].vector);

                    let should_add = results.len() < ef || dist < worst_dist;

                    if should_add {
                        candidates.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push(std::cmp::Reverse(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        }));

                        // Keep only top ef results
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Extract top-k results
        let mut final_results: Vec<(usize, f32)> = results
            .into_iter()
            .map(|r| (r.0.id, r.0.distance))
            .collect();

        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        final_results.truncate(k);

        final_results
    }

    /// Get a node by key
    pub fn get(&self, key: &str) -> Option<&CascadeNode> {
        self.key_to_id.get(key).and_then(|&id| self.nodes.get(id))
    }

    /// Get a node by ID
    pub fn get_by_id(&self, id: usize) -> Option<&CascadeNode> {
        self.nodes.get(id)
    }

    /// Delete a node by key
    pub fn delete(&mut self, key: &str) -> bool {
        if let Some(&id) = self.key_to_id.get(key) {
            // Remove from neighbors' lists
            for node in &mut self.nodes {
                node.neighbors.retain(|(nid, _)| *nid != id);
            }

            // Mark as deleted (we don't actually remove to preserve IDs)
            self.nodes[id].neighbors.clear();
            self.nodes[id].key.clear();
            self.key_to_id.remove(key);

            true
        } else {
            false
        }
    }

    /// Get number of nodes
    pub fn len(&self) -> usize {
        self.key_to_id.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.key_to_id.is_empty()
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        if self.nodes.is_empty() {
            return GraphStats::default();
        }

        let active_nodes: Vec<_> = self.nodes.iter().filter(|n| !n.key.is_empty()).collect();

        if active_nodes.is_empty() {
            return GraphStats::default();
        }

        let edge_count: usize = active_nodes.iter().map(|n| n.neighbors.len()).sum();
        let max_neighbors = active_nodes
            .iter()
            .map(|n| n.neighbors.len())
            .max()
            .unwrap_or(0);
        let min_neighbors = active_nodes
            .iter()
            .map(|n| n.neighbors.len())
            .min()
            .unwrap_or(0);
        let avg_neighbors = edge_count as f32 / active_nodes.len() as f32;

        GraphStats {
            node_count: active_nodes.len(),
            edge_count,
            avg_neighbors,
            max_neighbors,
            min_neighbors,
        }
    }

    /// Prune neighbors to stay within m_max limit
    fn prune_neighbors(&mut self, node_id: usize) {
        if node_id >= self.nodes.len() {
            return;
        }

        let neighbors = &mut self.nodes[node_id].neighbors;
        if neighbors.len() <= self.config.m_max {
            return;
        }

        // Sort by distance and keep closest
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbors.truncate(self.config.m_max);
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a < 1e-10 || norm_b < 1e-10 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
        }
    }

    /// Get all node IDs (for iteration)
    pub fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.key.is_empty())
            .map(|(id, _)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_insert_creates_connections() {
        let config = GraphConfig::default();
        let mut graph = CascadeGraph::new(config);

        // Insert first node (no candidates)
        graph.insert("a".to_string(), vec![1.0, 0.0, 0.0], &[]);

        // Insert second node with first as candidate
        graph.insert("b".to_string(), vec![0.9, 0.1, 0.0], &[0]);

        assert_eq!(graph.len(), 2);

        // Check bidirectional connection
        let node_a = graph.get("a").unwrap();
        let node_b = graph.get("b").unwrap();

        assert!(!node_a.neighbors.is_empty() || !node_b.neighbors.is_empty());
    }

    #[test]
    fn test_graph_search() {
        let config = GraphConfig {
            m: 4,
            m_max: 8,
            bidirectional: true,
            metric: DistanceMetric::Cosine,
        };
        let mut graph = CascadeGraph::new(config);

        // Insert some vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
        ];

        for (i, v) in vectors.iter().enumerate() {
            let candidates: Vec<usize> = (0..i).collect();
            graph.insert(format!("v{}", i), v.clone(), &candidates);
        }

        // Search for vector similar to first
        let query = vec![0.95, 0.05, 0.0];
        let results = graph.search(&query, &[0, 1, 2], 3, 10);

        assert!(!results.is_empty());
        // First result should be one of the similar vectors
        assert!(results[0].0 < 3);
    }
}

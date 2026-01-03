//! Append-only graph for Cascade Index
//!
//! Follows SynaDB physics principles:
//! - Arrow of Time: Neighbor lists are append-only
//! - The Delta: Only store new connections, not full rewrites
//!
//! # Design
//!
//! Instead of storing mutable neighbor lists per node, we use an append-only
//! log of edges. This trades some read efficiency for much faster writes.
//!
//! ```text
//! Edge Log (append-only):
//! [from_id: u32][to_id: u32][distance: f32]
//! [from_id: u32][to_id: u32][distance: f32]
//! ...
//! ```
//!
//! In-memory index rebuilt on load for fast neighbor lookups.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::distance::DistanceMetric;
use crate::error::Result;

use super::mmap_store::MmapVectorStorage;

// Edge format in file: [from: u32][to: u32][distance: f32] = 12 bytes

/// Graph configuration
#[derive(Clone, Debug)]
pub struct AppendGraphConfig {
    /// Target neighbors per node
    pub m: usize,
    /// Maximum neighbors (soft limit for search)
    pub m_max: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for AppendGraphConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Candidate for priority queue
#[derive(Clone)]
struct Candidate {
    id: u32,
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
        // Min-heap: smaller distance = higher priority
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Append-only graph for neighbor connections
pub struct AppendGraph {
    /// Path to edge log file
    path: PathBuf,
    /// Edge log file (append-only)
    edge_file: Option<BufWriter<File>>,
    /// In-memory neighbor index: node_id -> [(neighbor_id, distance)]
    neighbors: HashMap<u32, Vec<(u32, f32)>>,
    /// Configuration
    config: AppendGraphConfig,
    /// Number of edges
    edge_count: usize,
}

impl AppendGraph {
    /// Create or open append-only graph
    pub fn new<P: AsRef<Path>>(path: P, config: AppendGraphConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let exists = path.exists();

        let mut graph = Self {
            path: path.clone(),
            edge_file: None,
            neighbors: HashMap::new(),
            config,
            edge_count: 0,
        };

        if exists {
            graph.load_edges()?;
        }

        // Open file for appending
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        graph.edge_file = Some(BufWriter::new(file));

        Ok(graph)
    }

    /// Create in-memory graph (no persistence)
    pub fn in_memory(config: AppendGraphConfig) -> Self {
        Self {
            path: PathBuf::new(),
            edge_file: None,
            neighbors: HashMap::new(),
            config,
            edge_count: 0,
        }
    }

    /// Load edges from file
    fn load_edges(&mut self) -> Result<()> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut buf = [0u8; 12]; // 4 + 4 + 4 bytes per edge

        while reader.read_exact(&mut buf).is_ok() {
            let from = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let to = u32::from_le_bytes(buf[4..8].try_into().unwrap());
            let distance = f32::from_le_bytes(buf[8..12].try_into().unwrap());

            // Add to in-memory index
            self.neighbors.entry(from).or_default().push((to, distance));
            self.edge_count += 1;
        }

        // Sort neighbors by distance for each node
        for neighbors in self.neighbors.values_mut() {
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            // Keep only m_max closest
            neighbors.truncate(self.config.m_max);
        }

        Ok(())
    }

    /// Add edge (Arrow of Time - append only!)
    pub fn add_edge(&mut self, from: u32, to: u32, distance: f32) -> Result<()> {
        // Append to file
        if let Some(ref mut file) = self.edge_file {
            file.write_all(&from.to_le_bytes())?;
            file.write_all(&to.to_le_bytes())?;
            file.write_all(&distance.to_le_bytes())?;
        }

        // Update in-memory index
        let neighbors = self.neighbors.entry(from).or_default();

        // Check if edge already exists
        if !neighbors.iter().any(|(n, _)| *n == to) {
            neighbors.push((to, distance));

            // Keep sorted and pruned
            if neighbors.len() > self.config.m_max {
                neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                neighbors.truncate(self.config.m_max);
            }
        }

        self.edge_count += 1;
        Ok(())
    }

    /// Add bidirectional edge
    pub fn add_bidirectional_edge(&mut self, a: u32, b: u32, distance: f32) -> Result<()> {
        self.add_edge(a, b, distance)?;
        self.add_edge(b, a, distance)?;
        Ok(())
    }

    /// Get neighbors of a node
    #[inline]
    pub fn get_neighbors(&self, node: u32) -> &[(u32, f32)] {
        self.neighbors
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Search for nearest neighbors using graph traversal
    pub fn search(
        &self,
        storage: &MmapVectorStorage,
        query: &[f32],
        entry_points: &[u32],
        k: usize,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        if entry_points.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::with_capacity(ef * 2);
        let mut candidates = BinaryHeap::new(); // min-heap
        let mut results = BinaryHeap::new(); // max-heap for top-k

        // Initialize with entry points
        let max_initial = entry_points.len().min(ef);
        for &ep in entry_points.iter().take(max_initial) {
            if !visited.contains(&ep) {
                visited.insert(ep);
                if let Some(dist) = storage.distance_to_id(query, ep) {
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
        }

        // Graph traversal with early termination
        let mut stale_count = 0;
        let max_stale = 10;

        while let Some(current) = candidates.pop() {
            let worst_dist = if results.len() >= ef {
                results.peek().map(|r| r.0.distance).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            if current.distance > worst_dist {
                stale_count += 1;
                if stale_count >= max_stale {
                    break;
                }
                continue;
            }

            stale_count = 0;

            // Explore neighbors
            for &(neighbor_id, _) in self.get_neighbors(current.id) {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);

                    if let Some(dist) = storage.distance_to_id(query, neighbor_id) {
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

                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Extract top-k
        let mut final_results: Vec<(u32, f32)> = results
            .into_iter()
            .map(|r| (r.0.id, r.0.distance))
            .collect();

        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        final_results.truncate(k);

        final_results
    }

    /// Insert a node and connect to candidates
    pub fn insert_with_candidates(
        &mut self,
        storage: &MmapVectorStorage,
        node_id: u32,
        candidates: &[u32],
    ) -> Result<()> {
        let vector = match storage.get_vector_by_id(node_id) {
            Some(v) => v,
            None => return Ok(()),
        };

        // Compute distances to candidates
        let mut neighbor_candidates: Vec<(u32, f32)> = candidates
            .iter()
            .filter_map(|&cid| storage.distance_to_id(&vector, cid).map(|d| (cid, d)))
            .collect();

        // If few candidates, sample from existing nodes
        if neighbor_candidates.len() < self.config.m && !self.neighbors.is_empty() {
            let sample_size = (self.config.m * 4).min(storage.len());
            let step = (storage.len() / sample_size).max(1);

            for id in (0..storage.len() as u32).step_by(step) {
                if id == node_id {
                    continue;
                }
                if neighbor_candidates.iter().any(|(c, _)| *c == id) {
                    continue;
                }
                if let Some(dist) = storage.distance_to_id(&vector, id) {
                    neighbor_candidates.push((id, dist));
                }
                if neighbor_candidates.len() >= self.config.m * 4 {
                    break;
                }
            }
        }

        // Sort and take top M
        neighbor_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        neighbor_candidates.truncate(self.config.m);

        // Add bidirectional edges
        for (neighbor_id, dist) in neighbor_candidates {
            self.add_bidirectional_edge(node_id, neighbor_id, dist)?;
        }

        Ok(())
    }

    /// Number of nodes with edges
    pub fn node_count(&self) -> usize {
        self.neighbors.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Flush to disk
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.edge_file {
            file.flush()?;
        }
        Ok(())
    }
}

impl Drop for AppendGraph {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_append_graph_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.edges");

        let config = AppendGraphConfig::default();
        let mut graph = AppendGraph::new(&path, config).unwrap();

        graph.add_bidirectional_edge(0, 1, 0.1).unwrap();
        graph.add_bidirectional_edge(0, 2, 0.2).unwrap();

        let neighbors = graph.get_neighbors(0);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_append_graph_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.edges");

        // Create and populate
        {
            let config = AppendGraphConfig::default();
            let mut graph = AppendGraph::new(&path, config).unwrap();

            for i in 0..10 {
                graph
                    .add_bidirectional_edge(i, i + 1, i as f32 * 0.1)
                    .unwrap();
            }

            graph.flush().unwrap();
        }

        // Reopen and verify
        {
            let config = AppendGraphConfig::default();
            let graph = AppendGraph::new(&path, config).unwrap();

            assert!(graph.node_count() > 0);
            let neighbors = graph.get_neighbors(5);
            assert!(!neighbors.is_empty());
        }
    }
}

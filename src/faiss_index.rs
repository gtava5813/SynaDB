// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! FAISS index integration for high-performance vector search.
//!
//! This module provides FAISS-backed indexing as an alternative to HNSW
//! for scenarios requiring billion-scale search or GPU acceleration.
//!
//! _Requirements: 1.5, 9.2_

use faiss::{index::IndexImpl, index_factory, Idx, Index, MetricType};

use crate::distance::DistanceMetric;
use crate::error::{Result, SynaError};
use std::collections::HashMap;

/// FAISS index configuration
#[derive(Debug, Clone)]
pub struct FaissConfig {
    /// Index factory string (e.g., "Flat", "IVF1024,PQ32", "HNSW32")
    pub index_type: String,
    /// Number of vectors to train on (for IVF indexes)
    pub train_size: usize,
    /// Number of probes for IVF search (higher = more accurate, slower)
    pub nprobe: usize,
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for FaissConfig {
    fn default() -> Self {
        Self {
            index_type: "IVF1024,Flat".to_string(),
            train_size: 10000,
            nprobe: 10,
            use_gpu: false,
        }
    }
}

/// FAISS-backed vector index
pub struct FaissIndex {
    index: IndexImpl,
    config: FaissConfig,
    dimensions: u16,
    key_to_id: HashMap<String, i64>,
    id_to_key: HashMap<i64, String>,
    next_id: i64,
    is_trained: bool,
    training_vectors: Vec<f32>,
}

impl FaissIndex {
    /// Creates a new FAISS index
    pub fn new(dimensions: u16, metric: DistanceMetric, config: FaissConfig) -> Result<Self> {
        let metric_type = match metric {
            DistanceMetric::Euclidean => MetricType::L2,
            DistanceMetric::Cosine | DistanceMetric::DotProduct => MetricType::InnerProduct,
        };

        let index = index_factory(dimensions as u32, &config.index_type, metric_type)
            .map_err(|e| SynaError::IndexError(format!("FAISS index creation failed: {}", e)))?;

        Ok(Self {
            index,
            config,
            dimensions,
            key_to_id: HashMap::new(),
            id_to_key: HashMap::new(),
            next_id: 0,
            is_trained: false,
            training_vectors: Vec::new(),
        })
    }

    /// Adds a vector to the index
    pub fn insert(&mut self, key: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len() as u16,
            });
        }

        // If index requires training and isn't trained yet, accumulate vectors
        if !self.is_trained && self.requires_training() {
            self.training_vectors.extend_from_slice(vector);

            // Train once we have enough vectors
            if self.training_vectors.len() / self.dimensions as usize >= self.config.train_size {
                self.train()?;
            }
        }

        // Add to index - convert i64 to Idx
        let id = self.next_id;
        let idx = Idx::new(id as u64);
        self.index
            .add_with_ids(vector, &[idx])
            .map_err(|e| SynaError::IndexError(format!("FAISS insert failed: {}", e)))?;

        self.key_to_id.insert(key.to_string(), id);
        self.id_to_key.insert(id, key.to_string());
        self.next_id += 1;

        Ok(())
    }

    /// Searches for k nearest neighbors
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let search_result = self
            .index
            .search(query, k)
            .map_err(|e| SynaError::IndexError(format!("FAISS search failed: {}", e)))?;

        let mut results = Vec::with_capacity(k);
        for (dist, idx) in search_result
            .distances
            .iter()
            .zip(search_result.labels.iter())
        {
            // Convert Idx back to i64 for lookup
            if let Some(id_val) = idx.get() {
                let id = id_val as i64;
                if let Some(key) = self.id_to_key.get(&id) {
                    results.push((key.clone(), *dist));
                }
            }
        }

        Ok(results)
    }

    /// Trains the index on accumulated vectors
    fn train(&mut self) -> Result<()> {
        self.index
            .train(&self.training_vectors)
            .map_err(|e| SynaError::IndexError(format!("FAISS training failed: {}", e)))?;
        self.is_trained = true;
        self.training_vectors.clear();
        Ok(())
    }

    /// Returns whether this index type requires training
    fn requires_training(&self) -> bool {
        self.config.index_type.contains("IVF") || self.config.index_type.contains("PQ")
    }

    /// Returns the number of vectors in the index
    pub fn len(&self) -> usize {
        self.key_to_id.len()
    }

    /// Returns true if the index is empty
    pub fn is_empty(&self) -> bool {
        self.key_to_id.is_empty()
    }

    /// Returns the dimensions of vectors in this index
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }
}

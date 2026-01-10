//! Hybrid Hot/Cold Vector Architecture
//!
//! Combines GWI (high-throughput ingestion) with Cascade (fast search).
//!
//! # Architecture
//!
//! | Layer | Index | Role | Write | Read |
//! |-------|-------|------|-------|------|
//! | Hot | GWI | Real-time buffer | Sync | Fallback |
//! | Cold | Cascade | Historical storage | Batch | Primary |
//!
//! # Example
//!
//! ```rust,ignore
//! use synadb::arch::{HybridVectorStore, HybridConfig};
//!
//! let store = HybridVectorStore::new("hot.gwi", "cold.cascade", config)?;
//! store.initialize_hot(&sample_vectors)?;
//!
//! // Ingest to hot layer
//! store.ingest("doc1", &embedding)?;
//!
//! // Search both layers
//! let results = store.search(&query, 10)?;
//! ```

use crate::cascade::{CascadeConfig, CascadeIndex, SearchResult as CascadeSearchResult};
use crate::error::SynaError;
use crate::gwi::{GravityWellIndex, GwiConfig, GwiSearchResult};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;

/// Unified search result from hybrid store
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Key of the vector
    pub key: String,
    /// Distance score (lower is better)
    pub score: f32,
    /// Which layer the result came from
    pub source: ResultSource,
}

/// Source layer for a search result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    /// Result from hot layer (GWI)
    Hot,
    /// Result from cold layer (Cascade)
    Cold,
}

/// Configuration for hybrid store
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// GWI configuration for hot layer
    pub hot: GwiConfig,
    /// Cascade configuration for cold layer
    pub cold: CascadeConfig,
}

/// Hybrid Vector Store: GWI (Hot) + Cascade (Cold)
///
/// - **Hot Layer (GWI):** Real-time ingestion with O(1) appends
/// - **Cold Layer (Cascade):** Historical data with fast search
///
/// Data flows from Hot → Cold via `promote_to_cold()`.
pub struct HybridVectorStore {
    hot_index: GravityWellIndex,
    cold_index: CascadeIndex,
    cold_path: String,
}

impl HybridVectorStore {
    /// Create a new hybrid store
    pub fn new<P: AsRef<Path>>(
        hot_path: P,
        cold_path: P,
        config: HybridConfig,
    ) -> Result<Self, SynaError> {
        let cold_path_str = cold_path.as_ref().to_string_lossy().to_string();
        let hot = GravityWellIndex::new(&hot_path, config.hot)?;
        let cold = CascadeIndex::new(&cold_path, config.cold)?;

        Ok(Self {
            hot_index: hot,
            cold_index: cold,
            cold_path: cold_path_str,
        })
    }

    /// Open existing hybrid store
    pub fn open<P: AsRef<Path>>(
        hot_path: P,
        cold_path: P,
        cold_config: CascadeConfig,
    ) -> Result<Self, SynaError> {
        let cold_path_str = cold_path.as_ref().to_string_lossy().to_string();
        let hot = GravityWellIndex::open(&hot_path)?;
        let cold = CascadeIndex::new(&cold_path, cold_config)?;

        Ok(Self {
            hot_index: hot,
            cold_index: cold,
            cold_path: cold_path_str,
        })
    }

    /// Initialize GWI attractors from sample vectors
    ///
    /// Must be called before `ingest()` on a new store.
    pub fn initialize_hot(&mut self, sample_vectors: &[&[f32]]) -> Result<(), SynaError> {
        self.hot_index.initialize_attractors(sample_vectors)
    }

    /// Ingest a vector into the hot layer (GWI)
    ///
    /// Optimized for throughput. Data is immediately searchable.
    pub fn ingest(&mut self, key: &str, vector: &[f32]) -> Result<(), SynaError> {
        self.hot_index.insert(key, vector)
    }

    /// Batch ingest vectors into the hot layer
    pub fn ingest_batch(&mut self, keys: &[&str], vectors: &[&[f32]]) -> Result<usize, SynaError> {
        self.hot_index.insert_batch(keys, vectors)
    }

    /// Search both hot and cold layers
    ///
    /// Results are merged, deduplicated (keeping best score), and sorted.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<HybridSearchResult>, SynaError> {
        // Search both indices
        let hot_results = self.hot_index.search(query, k).unwrap_or_default();
        let cold_results = self.cold_index.search(query, k).unwrap_or_default();

        // Merge and deduplicate (keep best score per key)
        let mut combined: HashMap<String, (f32, ResultSource)> = HashMap::new();

        for res in hot_results {
            combined.insert(res.key, (res.score, ResultSource::Hot));
        }

        for res in cold_results {
            combined
                .entry(res.key)
                .and_modify(|(score, source)| {
                    if res.score < *score {
                        *score = res.score;
                        *source = ResultSource::Cold;
                    }
                })
                .or_insert((res.score, ResultSource::Cold));
        }

        // Sort by score (ascending = closer)
        let mut results: Vec<HybridSearchResult> = combined
            .into_iter()
            .map(|(key, (score, source))| HybridSearchResult { key, score, source })
            .collect();

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Search only the hot layer
    pub fn search_hot(&self, query: &[f32], k: usize) -> Result<Vec<GwiSearchResult>, SynaError> {
        self.hot_index.search(query, k)
    }

    /// Search only the cold layer
    pub fn search_cold(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<CascadeSearchResult>, SynaError> {
        self.cold_index.search(query, k)
    }

    /// Number of vectors in hot layer
    pub fn hot_count(&self) -> usize {
        self.hot_index.len()
    }

    /// Number of vectors in cold layer
    pub fn cold_count(&self) -> usize {
        self.cold_index.len()
    }

    /// Total vectors across both layers
    pub fn len(&self) -> usize {
        self.hot_count() + self.cold_count()
    }

    /// Check if both layers are empty
    pub fn is_empty(&self) -> bool {
        self.hot_index.is_empty() && self.cold_index.is_empty()
    }

    /// Flush hot layer to disk
    pub fn flush_hot(&self) -> Result<(), SynaError> {
        self.hot_index.flush()
    }

    /// Save cold layer to disk
    pub fn save_cold(&self) -> Result<(), SynaError> {
        self.cold_index.save(&self.cold_path)
    }

    /// Promote data from hot to cold layer
    ///
    /// This is a maintenance operation that:
    /// 1. Reads all vectors from GWI
    /// 2. Batch inserts them into Cascade
    /// 3. Does NOT clear the hot layer (caller should create new GWI if needed)
    ///
    /// Returns the number of vectors promoted.
    pub fn promote_to_cold(&mut self) -> Result<usize, SynaError> {
        let keys = self.hot_index.keys();
        let mut promoted = 0;

        for key in keys {
            if let Some(vector) = self.hot_index.get(&key)? {
                self.cold_index.insert(&key, &vector)?;
                promoted += 1;
            }
        }

        Ok(promoted)
    }
}

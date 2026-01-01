use crate::hnsw: :{HnswConfig, HnswIndex};
#[cfg(feature = "faiss")]
use crate::faiss_index: :{FaissConfig, FaissIndex};

/// Index backend selection
#[derive(Debug, Clone)]
pub enum IndexBackend {
    /// Built-in HNSW implementation
    Hnsw(HnswConfig),
    /// FAISS-backed index (requires 'faiss' feature)
    #[cfg(feature = "faiss")]
    Faiss(FaissConfig),
    /// No index (brute-force only)
    None,
}

impl Default for IndexBackend {
    fn default() -> Self {
        IndexBackend:: Hnsw(HnswConfig::default())
    }
}

/// Extended VectorConfig with backend selection
#[derive(Debug, Clone)]
pub struct VectorConfig {
    pub dimensions: u16,
    pub metric: DistanceMetric,
    pub key_prefix: String,
    pub index_threshold: usize,
    /// Index backend to use
    pub backend: IndexBackend,
}

pub struct VectorStore {
    db:  SynaDB,
    config:  VectorConfig,
    vector_keys: Vec<String>,
    hnsw_index: Option<HnswIndex>,
    #[cfg(feature = "faiss")]
    faiss_index: Option<FaissIndex>,
}

impl VectorStore {
    /// Search implementation that delegates to the appropriate backend
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Validate dimensions
        if query.len() != self.config.dimensions as usize {
            return Err(SynaError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len() as u16,
            });
        }

        // Check if we should use an index
        if self.vector_keys.len() >= self.config.index_threshold {
            #[cfg(feature = "faiss")]
            if let Some(ref faiss_index) = self.faiss_index {
                return self.search_faiss(query, k);
            }
            
            if let Some(ref _hnsw_index) = self.hnsw_index {
                return self. search_hnsw(query, k);
            }
        }

        // Fall back to brute force
        self.search_brute_force(query, k)
    }

    #[cfg(feature = "faiss")]
    fn search_faiss(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let faiss_index = self.faiss_index. as_ref()
            .ok_or_else(|| SynaError::IndexError("FAISS index not available".to_string()))?;

        let results = faiss_index.search(query, k)?;
        
        results.into_iter()
            .filter_map(|(key, score)| {
                let full_key = format!("{}{}", self.config.key_prefix, key);
                self.db.get(&full_key).ok().flatten().and_then(|atom| {
                    if let Atom::Vector(vec, _) = atom {
                        Some(SearchResult { key, score, vector: vec })
                    } else {
                        None
                    }
                })
            })
            .collect: :<Vec<_>>()
            .pipe(Ok)
    }
}
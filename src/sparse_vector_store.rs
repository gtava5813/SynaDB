//! Sparse Vector Store with Inverted Index
//!
//! A storage and retrieval system for sparse vectors using an inverted index.
//! Works with any sparse encoder (FLES-1, SPLADE, BM25, TF-IDF, etc.).
//!
//! # Architecture
//!
//! - **Inverted Index:** Maps term_id → list of (doc_id, weight) for fast retrieval
//! - **Document Store:** Maps doc_id → SparseVector for full vector access
//! - **Key Index:** Maps string key → doc_id for user-friendly access
//!
//! # Example
//!
//! ```rust,ignore
//! use synadb::{SparseVectorStore, SparseVector};
//!
//! let mut store = SparseVectorStore::new();
//!
//! // Index a document
//! let mut vec = SparseVector::new();
//! vec.add(100, 1.5);
//! vec.add(200, 0.8);
//! store.index_with_key("doc1", vec);
//!
//! // Search
//! let mut query = SparseVector::new();
//! query.add(100, 1.0);
//! let results = store.search(&query, 10);
//!
//! // Save to disk
//! store.save("index.svs")?;
//!
//! // Load from disk
//! let loaded = SparseVectorStore::load("index.svs")?;
//! ```

use crate::error::SynaError;
use crate::sparse_vector::SparseVector;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Search result from sparse vector store
#[derive(Debug, Clone)]
pub struct SparseSearchResult {
    /// Document key
    pub key: String,
    /// Document ID (internal)
    pub doc_id: u64,
    /// Relevance score (dot product)
    pub score: f32,
}

/// Statistics about the sparse index
#[derive(Debug, Clone, Default)]
pub struct SparseIndexStats {
    /// Number of indexed documents
    pub num_documents: usize,
    /// Number of unique terms in the index
    pub num_terms: usize,
    /// Total number of postings (term-document pairs)
    pub num_postings: usize,
    /// Average document length (non-zero terms)
    pub avg_doc_length: f32,
}

/// Sparse Vector Store with Inverted Index
///
/// Provides efficient storage and retrieval of sparse vectors using an
/// inverted index structure. Optimized for lexical embeddings where
/// documents have 100-500 non-zero terms out of 30K+ vocabulary.
#[derive(Debug)]
pub struct SparseVectorStore {
    /// Inverted index: term_id → [(doc_id, weight), ...]
    postings: HashMap<u32, Vec<(u64, f32)>>,
    /// Document store: doc_id → SparseVector
    vectors: HashMap<u64, SparseVector>,
    /// Key index: string key → doc_id
    keys: HashMap<String, u64>,
    /// Reverse key index: doc_id → string key
    reverse_keys: HashMap<u64, String>,
    /// Next document ID
    next_doc_id: u64,
}

impl Default for SparseVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseVectorStore {
    /// Create a new empty sparse vector store.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVectorStore;
    /// let store = SparseVectorStore::new();
    /// assert_eq!(store.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            vectors: HashMap::new(),
            keys: HashMap::new(),
            reverse_keys: HashMap::new(),
            next_doc_id: 0,
        }
    }

    /// Index a sparse vector with auto-generated document ID.
    ///
    /// Returns the assigned document ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    ///
    /// let doc_id = store.index(vec);
    /// assert_eq!(store.len(), 1);
    /// ```
    pub fn index(&mut self, vector: SparseVector) -> u64 {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Add to posting lists
        for (term_id, weight) in vector.iter() {
            self.postings
                .entry(*term_id)
                .or_default()
                .push((doc_id, *weight));
        }

        // Store the vector
        self.vectors.insert(doc_id, vector);

        doc_id
    }

    /// Index a sparse vector with a user-specified key.
    ///
    /// If the key already exists, the old document is replaced.
    /// Returns the document ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    ///
    /// let doc_id = store.index_with_key("doc1", vec);
    /// assert!(store.get_by_key("doc1").is_some());
    /// ```
    pub fn index_with_key(&mut self, key: &str, vector: SparseVector) -> u64 {
        // If key exists, delete old document first
        if let Some(&old_doc_id) = self.keys.get(key) {
            self.delete_by_id(old_doc_id);
        }

        let doc_id = self.index(vector);

        // Store key mapping
        self.keys.insert(key.to_string(), doc_id);
        self.reverse_keys.insert(doc_id, key.to_string());

        doc_id
    }

    /// Search for similar documents using dot product scoring.
    ///
    /// Returns top-k results sorted by score (descending).
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    ///
    /// // Index documents
    /// let mut doc1 = SparseVector::new();
    /// doc1.add(100, 2.0);
    /// doc1.add(200, 1.0);
    /// store.index_with_key("doc1", doc1);
    ///
    /// let mut doc2 = SparseVector::new();
    /// doc2.add(100, 1.0);
    /// doc2.add(300, 3.0);
    /// store.index_with_key("doc2", doc2);
    ///
    /// // Search
    /// let mut query = SparseVector::new();
    /// query.add(100, 1.0);
    /// let results = store.search(&query, 10);
    ///
    /// assert_eq!(results.len(), 2);
    /// assert_eq!(results[0].key, "doc1"); // Higher score (2.0 vs 1.0)
    /// ```
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<SparseSearchResult> {
        if query.is_empty() || self.vectors.is_empty() {
            return Vec::new();
        }

        // Accumulate scores from posting lists
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for (term_id, query_weight) in query.iter() {
            if let Some(postings) = self.postings.get(term_id) {
                for (doc_id, doc_weight) in postings {
                    *scores.entry(*doc_id).or_default() += query_weight * doc_weight;
                }
            }
        }

        // Sort by score descending and take top-k
        let mut results: Vec<_> = scores
            .into_iter()
            .map(|(doc_id, score)| {
                let key = self
                    .reverse_keys
                    .get(&doc_id)
                    .cloned()
                    .unwrap_or_else(|| format!("_doc_{}", doc_id));
                SparseSearchResult { key, doc_id, score }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        results
    }

    /// Delete a document by its ID.
    ///
    /// Returns true if the document was found and deleted.
    pub fn delete_by_id(&mut self, doc_id: u64) -> bool {
        if let Some(vector) = self.vectors.remove(&doc_id) {
            // Remove from posting lists
            for (term_id, _) in vector.iter() {
                if let Some(postings) = self.postings.get_mut(term_id) {
                    postings.retain(|(id, _)| *id != doc_id);
                    if postings.is_empty() {
                        self.postings.remove(term_id);
                    }
                }
            }

            // Remove key mapping
            if let Some(key) = self.reverse_keys.remove(&doc_id) {
                self.keys.remove(&key);
            }

            true
        } else {
            false
        }
    }

    /// Delete a document by its key.
    ///
    /// Returns true if the document was found and deleted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// store.index_with_key("doc1", vec);
    ///
    /// assert!(store.delete("doc1"));
    /// assert!(store.get_by_key("doc1").is_none());
    /// ```
    pub fn delete(&mut self, key: &str) -> bool {
        if let Some(&doc_id) = self.keys.get(key) {
            self.delete_by_id(doc_id)
        } else {
            false
        }
    }

    /// Get a document by its ID.
    pub fn get_by_id(&self, doc_id: u64) -> Option<&SparseVector> {
        self.vectors.get(&doc_id)
    }

    /// Get a document by its key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// store.index_with_key("doc1", vec);
    ///
    /// let retrieved = store.get_by_key("doc1").unwrap();
    /// assert_eq!(retrieved.get(100), 1.5);
    /// ```
    pub fn get_by_key(&self, key: &str) -> Option<&SparseVector> {
        self.keys.get(key).and_then(|id| self.vectors.get(id))
    }

    /// Get the document ID for a key.
    pub fn get_doc_id(&self, key: &str) -> Option<u64> {
        self.keys.get(key).copied()
    }

    /// Get statistics about the index.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// vec.add(200, 0.8);
    /// store.index_with_key("doc1", vec);
    ///
    /// let stats = store.stats();
    /// assert_eq!(stats.num_documents, 1);
    /// assert_eq!(stats.num_terms, 2);
    /// assert_eq!(stats.num_postings, 2);
    /// ```
    pub fn stats(&self) -> SparseIndexStats {
        let num_documents = self.vectors.len();
        let num_terms = self.postings.len();
        let num_postings: usize = self.postings.values().map(|v| v.len()).sum();

        let total_doc_length: usize = self.vectors.values().map(|v| v.nnz()).sum();
        let avg_doc_length = if num_documents > 0 {
            total_doc_length as f32 / num_documents as f32
        } else {
            0.0
        };

        SparseIndexStats {
            num_documents,
            num_terms,
            num_postings,
            avg_doc_length,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get all document keys.
    pub fn keys(&self) -> Vec<String> {
        self.keys.keys().cloned().collect()
    }

    /// Clear all documents from the store.
    pub fn clear(&mut self) {
        self.postings.clear();
        self.vectors.clear();
        self.keys.clear();
        self.reverse_keys.clear();
        self.next_doc_id = 0;
    }

    /// Save the index to a file.
    ///
    /// # File Format
    ///
    /// ```text
    /// [magic: 4 bytes "SVS\0"]
    /// [version: u32]
    /// [next_doc_id: u64]
    /// [num_keys: u32]
    /// for each key:
    ///   [key_len: u32][key: bytes][doc_id: u64]
    /// [num_vectors: u32]
    /// for each vector:
    ///   [doc_id: u64][vector_bytes_len: u32][vector_bytes: bytes]
    /// ```
    ///
    /// Note: Postings are rebuilt from vectors on load for consistency.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use synadb::{SparseVectorStore, SparseVector};
    ///
    /// let mut store = SparseVectorStore::new();
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// store.index_with_key("doc1", vec);
    ///
    /// store.save("index.svs")?;
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), SynaError> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);

        // Magic number "SVS\0"
        writer.write_all(b"SVS\0")?;

        // Version (1)
        writer.write_all(&1u32.to_le_bytes())?;

        // next_doc_id
        writer.write_all(&self.next_doc_id.to_le_bytes())?;

        // Keys: [num_keys][key_len, key, doc_id]...
        let num_keys = self.keys.len() as u32;
        writer.write_all(&num_keys.to_le_bytes())?;

        for (key, doc_id) in &self.keys {
            let key_bytes = key.as_bytes();
            let key_len = key_bytes.len() as u32;
            writer.write_all(&key_len.to_le_bytes())?;
            writer.write_all(key_bytes)?;
            writer.write_all(&doc_id.to_le_bytes())?;
        }

        // Vectors: [num_vectors][doc_id, vec_len, vec_bytes]...
        let num_vectors = self.vectors.len() as u32;
        writer.write_all(&num_vectors.to_le_bytes())?;

        for (doc_id, vector) in &self.vectors {
            writer.write_all(&doc_id.to_le_bytes())?;

            let vec_bytes = vector.to_bytes();
            let vec_len = vec_bytes.len() as u32;
            writer.write_all(&vec_len.to_le_bytes())?;
            writer.write_all(&vec_bytes)?;
        }

        writer.flush()?;

        Ok(())
    }

    /// Load an index from a file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use synadb::SparseVectorStore;
    ///
    /// let store = SparseVectorStore::load("index.svs")?;
    /// let results = store.search(&query, 10);
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SynaError> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"SVS\0" {
            return Err(SynaError::IoError("Invalid SVS file magic".to_string()));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(SynaError::IoError(format!(
                "Unsupported SVS version: {}",
                version
            )));
        }

        // Read next_doc_id
        let mut next_doc_id_bytes = [0u8; 8];
        reader.read_exact(&mut next_doc_id_bytes)?;
        let next_doc_id = u64::from_le_bytes(next_doc_id_bytes);

        // Read keys
        let mut num_keys_bytes = [0u8; 4];
        reader.read_exact(&mut num_keys_bytes)?;
        let num_keys = u32::from_le_bytes(num_keys_bytes) as usize;

        let mut keys = HashMap::with_capacity(num_keys);
        let mut reverse_keys = HashMap::with_capacity(num_keys);

        for _ in 0..num_keys {
            let mut key_len_bytes = [0u8; 4];
            reader.read_exact(&mut key_len_bytes)?;
            let key_len = u32::from_le_bytes(key_len_bytes) as usize;

            let mut key_bytes = vec![0u8; key_len];
            reader.read_exact(&mut key_bytes)?;
            let key =
                String::from_utf8(key_bytes).map_err(|e| SynaError::IoError(e.to_string()))?;

            let mut doc_id_bytes = [0u8; 8];
            reader.read_exact(&mut doc_id_bytes)?;
            let doc_id = u64::from_le_bytes(doc_id_bytes);

            keys.insert(key.clone(), doc_id);
            reverse_keys.insert(doc_id, key);
        }

        // Read vectors
        let mut num_vectors_bytes = [0u8; 4];
        reader.read_exact(&mut num_vectors_bytes)?;
        let num_vectors = u32::from_le_bytes(num_vectors_bytes) as usize;

        let mut vectors = HashMap::with_capacity(num_vectors);
        let mut postings: HashMap<u32, Vec<(u64, f32)>> = HashMap::new();

        for _ in 0..num_vectors {
            let mut doc_id_bytes = [0u8; 8];
            reader.read_exact(&mut doc_id_bytes)?;
            let doc_id = u64::from_le_bytes(doc_id_bytes);

            let mut vec_len_bytes = [0u8; 4];
            reader.read_exact(&mut vec_len_bytes)?;
            let vec_len = u32::from_le_bytes(vec_len_bytes) as usize;

            let mut vec_bytes = vec![0u8; vec_len];
            reader.read_exact(&mut vec_bytes)?;

            let vector = SparseVector::from_bytes(&vec_bytes)
                .ok_or_else(|| SynaError::IoError("Failed to deserialize vector".to_string()))?;

            // Rebuild postings from vector
            for (term_id, weight) in vector.iter() {
                postings
                    .entry(*term_id)
                    .or_default()
                    .push((doc_id, *weight));
            }

            vectors.insert(doc_id, vector);
        }

        Ok(Self {
            postings,
            vectors,
            keys,
            reverse_keys,
            next_doc_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let store = SparseVectorStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_index_auto_id() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);

        let doc_id = store.index(vec);
        assert_eq!(doc_id, 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_index_with_key() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);

        store.index_with_key("doc1", vec);
        assert!(store.get_by_key("doc1").is_some());
        assert_eq!(store.get_by_key("doc1").unwrap().get(100), 1.5);
    }

    #[test]
    fn test_index_replace_key() {
        let mut store = SparseVectorStore::new();

        let mut vec1 = SparseVector::new();
        vec1.add(100, 1.0);
        store.index_with_key("doc1", vec1);

        let mut vec2 = SparseVector::new();
        vec2.add(200, 2.0);
        store.index_with_key("doc1", vec2);

        assert_eq!(store.len(), 1);
        assert_eq!(store.get_by_key("doc1").unwrap().get(100), 0.0);
        assert_eq!(store.get_by_key("doc1").unwrap().get(200), 2.0);
    }

    #[test]
    fn test_search_basic() {
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 2.0);
        doc1.add(200, 1.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(100, 1.0);
        doc2.add(300, 3.0);
        store.index_with_key("doc2", doc2);

        let mut query = SparseVector::new();
        query.add(100, 1.0);

        let results = store.search(&query, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "doc1"); // score 2.0
        assert_eq!(results[1].key, "doc2"); // score 1.0
    }

    #[test]
    fn test_search_empty_query() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        store.index_with_key("doc1", vec);

        let query = SparseVector::new();
        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_store() {
        let store = SparseVectorStore::new();
        let mut query = SparseVector::new();
        query.add(100, 1.0);

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_no_overlap() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        store.index_with_key("doc1", vec);

        let mut query = SparseVector::new();
        query.add(999, 1.0);

        let results = store.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete_by_key() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        store.index_with_key("doc1", vec);

        assert!(store.delete("doc1"));
        assert!(store.get_by_key("doc1").is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_delete_removes_from_search() {
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 2.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(100, 1.0);
        store.index_with_key("doc2", doc2);

        store.delete("doc1");

        let mut query = SparseVector::new();
        query.add(100, 1.0);

        let results = store.search(&query, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "doc2");
    }

    #[test]
    fn test_stats() {
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 1.0);
        doc1.add(200, 2.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(100, 1.0);
        doc2.add(300, 3.0);
        store.index_with_key("doc2", doc2);

        let stats = store.stats();
        assert_eq!(stats.num_documents, 2);
        assert_eq!(stats.num_terms, 3); // 100, 200, 300
        assert_eq!(stats.num_postings, 4); // 2 docs × 2 terms each
        assert_eq!(stats.avg_doc_length, 2.0);
    }

    #[test]
    fn test_clear() {
        let mut store = SparseVectorStore::new();
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        store.index_with_key("doc1", vec);

        store.clear();
        assert!(store.is_empty());
        assert!(store.get_by_key("doc1").is_none());
    }

    #[test]
    fn test_keys() {
        let mut store = SparseVectorStore::new();

        let mut vec1 = SparseVector::new();
        vec1.add(100, 1.0);
        store.index_with_key("doc1", vec1);

        let mut vec2 = SparseVector::new();
        vec2.add(200, 2.0);
        store.index_with_key("doc2", vec2);

        let keys = store.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"doc1".to_string()));
        assert!(keys.contains(&"doc2".to_string()));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svs");

        // Create and populate store
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 1.5);
        doc1.add(200, 2.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(100, 0.5);
        doc2.add(300, 3.0);
        store.index_with_key("doc2", doc2);

        // Save
        store.save(&path).unwrap();

        // Load
        let loaded = SparseVectorStore::load(&path).unwrap();

        // Verify
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get_by_key("doc1").unwrap().get(100), 1.5);
        assert_eq!(loaded.get_by_key("doc1").unwrap().get(200), 2.0);
        assert_eq!(loaded.get_by_key("doc2").unwrap().get(100), 0.5);
        assert_eq!(loaded.get_by_key("doc2").unwrap().get(300), 3.0);
    }

    #[test]
    fn test_save_load_search_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svs");

        // Create and populate store
        let mut store = SparseVectorStore::new();

        let mut doc1 = SparseVector::new();
        doc1.add(100, 2.0);
        store.index_with_key("doc1", doc1);

        let mut doc2 = SparseVector::new();
        doc2.add(100, 1.0);
        store.index_with_key("doc2", doc2);

        // Save and load
        store.save(&path).unwrap();
        let loaded = SparseVectorStore::load(&path).unwrap();

        // Search should work
        let mut query = SparseVector::new();
        query.add(100, 1.0);
        let results = loaded.search(&query, 10);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "doc1"); // Higher score
        assert_eq!(results[1].key, "doc2");
    }

    #[test]
    fn test_save_load_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svs");

        let store = SparseVectorStore::new();
        store.save(&path).unwrap();

        let loaded = SparseVectorStore::load(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_load_invalid_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svs");

        // Write invalid file
        std::fs::write(&path, b"XXXX").unwrap();

        let result = SparseVectorStore::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load_preserves_next_doc_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svs");

        let mut store = SparseVectorStore::new();

        // Index some docs to advance next_doc_id
        let mut vec = SparseVector::new();
        vec.add(100, 1.0);
        store.index_with_key("doc1", vec.clone());
        store.index_with_key("doc2", vec.clone());
        store.index_with_key("doc3", vec);

        store.save(&path).unwrap();
        let mut loaded = SparseVectorStore::load(&path).unwrap();

        // New doc should get id 3
        let mut new_vec = SparseVector::new();
        new_vec.add(200, 2.0);
        let new_id = loaded.index_with_key("doc4", new_vec);
        assert_eq!(new_id, 3);
    }
}

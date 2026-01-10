//! Sparse Vector for Lexical Embeddings
//!
//! A sparse vector representation optimized for lexical embeddings where most
//! dimensions are zero. Used with sparse encoders like FLES-1, SPLADE, BM25, TF-IDF.
//!
//! # Example
//!
//! ```rust
//! use synadb::SparseVector;
//!
//! let mut vec = SparseVector::new();
//! vec.add(100, 1.5);  // term_id 100 with weight 1.5
//! vec.add(200, 0.8);  // term_id 200 with weight 0.8
//!
//! assert_eq!(vec.nnz(), 2);
//! assert_eq!(vec.get(100), 1.5);
//! ```

use std::collections::HashMap;

/// Sparse vector representation for lexical embeddings.
///
/// Each dimension corresponds to a vocabulary term (e.g., 0-30521 for BERT).
/// Only non-zero positive weights are stored, making this efficient for
/// sparse representations where typically 100-500 terms are active out of 30K+.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    /// Term index → weight (only positive weights stored)
    weights: HashMap<u32, f32>,
}

impl Default for SparseVector {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseVector {
    /// Create an empty sparse vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    /// let vec = SparseVector::new();
    /// assert_eq!(vec.nnz(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Create a sparse vector from a weights map.
    ///
    /// Zero and negative weights are filtered out.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    /// use std::collections::HashMap;
    ///
    /// let mut weights = HashMap::new();
    /// weights.insert(100, 1.5);
    /// weights.insert(200, 0.0);  // Will be filtered
    /// weights.insert(300, -0.5); // Will be filtered
    ///
    /// let vec = SparseVector::from_weights(weights);
    /// assert_eq!(vec.nnz(), 1);
    /// ```
    pub fn from_weights(weights: HashMap<u32, f32>) -> Self {
        let filtered: HashMap<u32, f32> = weights.into_iter().filter(|(_, w)| *w > 0.0).collect();
        Self { weights: filtered }
    }

    /// Add a term with weight.
    ///
    /// If weight is zero or negative, the term is ignored.
    /// If the term already exists, the weight is replaced.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// vec.add(200, 0.0);   // Ignored
    /// vec.add(300, -0.5);  // Ignored
    ///
    /// assert_eq!(vec.nnz(), 1);
    /// ```
    pub fn add(&mut self, term_id: u32, weight: f32) {
        if weight > 0.0 {
            self.weights.insert(term_id, weight);
        }
    }

    /// Compute dot product with another sparse vector.
    ///
    /// Complexity: O(min(nnz₁, nnz₂))
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut a = SparseVector::new();
    /// a.add(100, 1.0);
    /// a.add(200, 2.0);
    ///
    /// let mut b = SparseVector::new();
    /// b.add(100, 3.0);
    /// b.add(300, 4.0);
    ///
    /// // Dot product: 1.0 * 3.0 = 3.0 (only term 100 overlaps)
    /// assert_eq!(a.dot(&b), 3.0);
    /// ```
    pub fn dot(&self, other: &SparseVector) -> f32 {
        // Iterate over the smaller vector for efficiency
        let (smaller, larger) = if self.weights.len() <= other.weights.len() {
            (&self.weights, &other.weights)
        } else {
            (&other.weights, &self.weights)
        };

        smaller
            .iter()
            .filter_map(|(term_id, w1)| larger.get(term_id).map(|w2| w1 * w2))
            .sum()
    }

    /// Compute L2 norm: sqrt(sum(w²))
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 3.0);
    /// vec.add(200, 4.0);
    ///
    /// assert_eq!(vec.norm(), 5.0);  // sqrt(9 + 16) = 5
    /// ```
    pub fn norm(&self) -> f32 {
        self.weights.values().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Compute L1 norm: sum(|w|)
    ///
    /// Since we only store positive weights, this is just sum(w).
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// vec.add(200, 2.5);
    ///
    /// assert_eq!(vec.l1_norm(), 4.0);
    /// ```
    pub fn l1_norm(&self) -> f32 {
        self.weights.values().sum()
    }

    /// Number of non-zero terms.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.0);
    /// vec.add(200, 2.0);
    ///
    /// assert_eq!(vec.nnz(), 2);
    /// ```
    pub fn nnz(&self) -> usize {
        self.weights.len()
    }

    /// Get weight for a term (0.0 if not present).
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    ///
    /// assert_eq!(vec.get(100), 1.5);
    /// assert_eq!(vec.get(999), 0.0);
    /// ```
    pub fn get(&self, term_id: u32) -> f32 {
        *self.weights.get(&term_id).unwrap_or(&0.0)
    }

    /// Iterate over non-zero terms.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.0);
    /// vec.add(200, 2.0);
    ///
    /// for (term_id, weight) in vec.iter() {
    ///     println!("Term {}: {}", term_id, weight);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &f32)> {
        self.weights.iter()
    }

    /// Serialize to bytes.
    ///
    /// Format: [count: u32][term_id: u32, weight: f32]...
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    ///
    /// let bytes = vec.to_bytes();
    /// let restored = SparseVector::from_bytes(&bytes).unwrap();
    ///
    /// assert_eq!(vec, restored);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.weights.len() as u32;
        let mut bytes = Vec::with_capacity(4 + self.weights.len() * 8);

        // Write count
        bytes.extend_from_slice(&count.to_le_bytes());

        // Write term_id, weight pairs (sorted for deterministic output)
        let mut pairs: Vec<_> = self.weights.iter().collect();
        pairs.sort_by_key(|(k, _)| *k);

        for (term_id, weight) in pairs {
            bytes.extend_from_slice(&term_id.to_le_bytes());
            bytes.extend_from_slice(&weight.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes.
    ///
    /// Returns None if the data is invalid or truncated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use synadb::SparseVector;
    ///
    /// let mut vec = SparseVector::new();
    /// vec.add(100, 1.5);
    /// vec.add(200, 2.5);
    ///
    /// let bytes = vec.to_bytes();
    /// let restored = SparseVector::from_bytes(&bytes).unwrap();
    ///
    /// assert_eq!(vec.nnz(), restored.nnz());
    /// assert_eq!(vec.get(100), restored.get(100));
    /// ```
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let expected_len = 4 + count * 8;

        if data.len() < expected_len {
            return None;
        }

        let mut weights = HashMap::with_capacity(count);
        let mut offset = 4;

        for _ in 0..count {
            let term_id = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            let weight = f32::from_le_bytes([
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);

            if weight > 0.0 {
                weights.insert(term_id, weight);
            }

            offset += 8;
        }

        Some(Self { weights })
    }

    /// Get the underlying weights map.
    pub fn weights(&self) -> &HashMap<u32, f32> {
        &self.weights
    }

    /// Check if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let vec = SparseVector::new();
        assert_eq!(vec.nnz(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_add_positive_weight() {
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        assert_eq!(vec.nnz(), 1);
        assert_eq!(vec.get(100), 1.5);
    }

    #[test]
    fn test_add_zero_weight_ignored() {
        let mut vec = SparseVector::new();
        vec.add(100, 0.0);
        assert_eq!(vec.nnz(), 0);
    }

    #[test]
    fn test_add_negative_weight_ignored() {
        let mut vec = SparseVector::new();
        vec.add(100, -1.0);
        assert_eq!(vec.nnz(), 0);
    }

    #[test]
    fn test_from_weights_filters() {
        let mut weights = HashMap::new();
        weights.insert(100, 1.5);
        weights.insert(200, 0.0);
        weights.insert(300, -0.5);

        let vec = SparseVector::from_weights(weights);
        assert_eq!(vec.nnz(), 1);
        assert_eq!(vec.get(100), 1.5);
    }

    #[test]
    fn test_dot_product() {
        let mut a = SparseVector::new();
        a.add(100, 1.0);
        a.add(200, 2.0);

        let mut b = SparseVector::new();
        b.add(100, 3.0);
        b.add(300, 4.0);

        assert_eq!(a.dot(&b), 3.0);
    }

    #[test]
    fn test_dot_product_no_overlap() {
        let mut a = SparseVector::new();
        a.add(100, 1.0);

        let mut b = SparseVector::new();
        b.add(200, 2.0);

        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn test_dot_product_commutative() {
        let mut a = SparseVector::new();
        a.add(100, 1.5);
        a.add(200, 2.5);

        let mut b = SparseVector::new();
        b.add(100, 3.0);
        b.add(200, 1.0);
        b.add(300, 5.0);

        assert_eq!(a.dot(&b), b.dot(&a));
    }

    #[test]
    fn test_norm() {
        let mut vec = SparseVector::new();
        vec.add(100, 3.0);
        vec.add(200, 4.0);

        assert_eq!(vec.norm(), 5.0);
    }

    #[test]
    fn test_l1_norm() {
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        vec.add(200, 2.5);

        assert_eq!(vec.l1_norm(), 4.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut vec = SparseVector::new();
        vec.add(100, 1.5);
        vec.add(200, 2.5);
        vec.add(300, 3.5);

        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();

        assert_eq!(vec.nnz(), restored.nnz());
        assert_eq!(vec.get(100), restored.get(100));
        assert_eq!(vec.get(200), restored.get(200));
        assert_eq!(vec.get(300), restored.get(300));
    }

    #[test]
    fn test_serialization_empty() {
        let vec = SparseVector::new();
        let bytes = vec.to_bytes();
        let restored = SparseVector::from_bytes(&bytes).unwrap();

        assert_eq!(restored.nnz(), 0);
    }

    #[test]
    fn test_from_bytes_invalid() {
        assert!(SparseVector::from_bytes(&[]).is_none());
        assert!(SparseVector::from_bytes(&[1, 0, 0, 0]).is_none()); // count=1 but no data
    }
}

// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Distance metrics for vector similarity search.
//!
//! This module provides distance functions for comparing vectors in similarity search.
//! All metrics are designed so that lower values indicate more similar vectors.
//!
//! # Supported Metrics
//!
//! | Metric | Range | Best For |
//! |--------|-------|----------|
//! | Cosine | [0, 2] | Text embeddings, normalized vectors |
//! | Euclidean | [0, ∞) | Image features, spatial data |
//! | DotProduct | (-∞, ∞) | Pre-normalized vectors, recommendation |
//!
//! # Performance
//!
//! All distance functions are O(n) where n is the vector dimension.
//! For 768-dimensional vectors (BERT), expect ~1μs per distance computation.
//!
//! # Examples
//!
//! ```rust
//! use synadb::distance::DistanceMetric;
//!
//! let a = vec![1.0f32, 0.0, 0.0];
//! let b = vec![0.0f32, 1.0, 0.0];
//!
//! // Cosine distance: orthogonal vectors have distance 1.0
//! let cosine_dist = DistanceMetric::Cosine.distance(&a, &b);
//! assert!((cosine_dist - 1.0).abs() < 1e-6);
//!
//! // Euclidean distance: sqrt(2) for unit orthogonal vectors
//! let euclidean_dist = DistanceMetric::Euclidean.distance(&a, &b);
//! assert!((euclidean_dist - std::f32::consts::SQRT_2).abs() < 1e-6);
//! ```
//!
//! _Requirements: 1.4_

/// Distance metric for similarity search.
///
/// Determines how vector similarity is computed. All metrics return
/// lower values for more similar vectors.
///
/// # Choosing a Metric
///
/// - **Cosine**: Best for text embeddings where magnitude doesn't matter.
///   Measures the angle between vectors. Range: [0, 2].
/// - **Euclidean**: Best for spatial data where absolute positions matter.
///   Measures straight-line distance. Range: [0, ∞).
/// - **DotProduct**: Best for pre-normalized vectors or when magnitude
///   should influence similarity. Range: (-∞, ∞), negated so lower = more similar.
///
/// # Examples
///
/// ```rust
/// use synadb::distance::DistanceMetric;
///
/// // Identical vectors have distance 0 for all metrics
/// let v = vec![1.0f32, 2.0, 3.0];
/// assert!(DistanceMetric::Cosine.distance(&v, &v) < 1e-6);
/// assert!(DistanceMetric::Euclidean.distance(&v, &v) < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine distance: `1 - cos(θ)` where θ is the angle between vectors.
    ///
    /// Range: [0, 2] where 0 = identical direction, 1 = orthogonal, 2 = opposite.
    /// Invariant to vector magnitude - only measures direction.
    Cosine,

    /// Euclidean (L2) distance: `sqrt(Σ(a_i - b_i)²)`.
    ///
    /// Range: [0, ∞) where 0 = identical vectors.
    /// Sensitive to both direction and magnitude.
    Euclidean,

    /// Negative dot product: `-Σ(a_i * b_i)`.
    ///
    /// Range: (-∞, ∞), negated so lower values indicate higher similarity.
    /// For normalized vectors, equivalent to cosine similarity.
    DotProduct,
}

impl DistanceMetric {
    /// Compute distance between two vectors.
    ///
    /// Lower values indicate more similar vectors for all metrics.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector (must have same length as `a`)
    ///
    /// # Returns
    ///
    /// The distance between the vectors according to this metric.
    ///
    /// # Panics
    ///
    /// Debug builds will panic if vectors have different lengths.
    /// Release builds have undefined behavior for mismatched lengths.
    ///
    /// # Performance
    ///
    /// O(n) where n is the vector dimension. Approximately 1μs for 768-dim vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::distance::DistanceMetric;
    ///
    /// let query = vec![0.1f32; 128];
    /// let candidate = vec![0.2f32; 128];
    ///
    /// let dist = DistanceMetric::Cosine.distance(&query, &candidate);
    /// println!("Distance: {:.4}", dist);
    /// ```
    ///
    /// _Requirements: 1.4_
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
        }
    }
}

/// Cosine distance: `1 - (a·b)/(|a||b|)`.
///
/// Measures the angle between two vectors, ignoring magnitude.
/// Returns 0 for identical directions, 1 for orthogonal, 2 for opposite.
///
/// # Special Cases
///
/// - Zero vectors return distance 1.0 (treated as orthogonal)
/// - Identical vectors return ~0.0 (floating point precision)
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        return 1.0; // Handle zero vectors
    }
    1.0 - (dot / denom)
}

/// Euclidean (L2) distance: `sqrt(Σ(a_i - b_i)²)`.
///
/// Measures the straight-line distance between two points in n-dimensional space.
/// Sensitive to both direction and magnitude of vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Negative dot product: `-Σ(a_i * b_i)`.
///
/// Returns the negated dot product so that lower values indicate higher similarity.
/// For normalized vectors, this is equivalent to negative cosine similarity.
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    -dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = DistanceMetric::Cosine.distance(&v, &v);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = DistanceMetric::Euclidean.distance(&v, &v);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_identical() {
        let v = vec![1.0, 0.0, 0.0]; // unit vector
        let d = DistanceMetric::DotProduct.distance(&v, &v);
        assert!((d + 1.0).abs() < 1e-6); // -1.0 for identical unit vectors
    }
}

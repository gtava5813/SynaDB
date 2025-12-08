//! Distance metrics for vector similarity search
//!
//! This module provides distance functions for comparing vectors in similarity search.
//! All metrics are designed so that lower values indicate more similar vectors.

/// Distance metric for similarity search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_sim)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product (negative for max similarity)
    DotProduct,
}

impl DistanceMetric {
    /// Compute distance between two vectors
    /// Lower is more similar for all metrics
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
        }
    }
}

/// Cosine distance: 1 - (a·b)/(|a||b|)
/// Range: [0, 2], 0 = identical, 2 = opposite
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

/// Euclidean (L2) distance: sqrt(sum((a-b)^2))
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Negative dot product (so lower = more similar)
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

//! Vector Similarity Queries — SIMILAR_TO integration with VectorStore/GWI/Cascade.
//!
//! Enables queries like:
//! ```sql
//! SELECT * FROM vectors WHERE SIMILAR_TO([0.1, 0.2, ...], 10)
//! ```

use crate::query::ast::SimilarityCondition;
use crate::query::error::QueryError;
use crate::query::ResultRow;
use crate::types::Atom;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Result of a vector similarity search within the query engine.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// Key of the matching vector.
    pub key: String,
    /// Similarity score (higher = more similar for cosine/dot, lower for euclidean).
    pub score: f64,
    /// The matched vector (if requested).
    pub vector: Option<Vec<f32>>,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Execute a vector similarity search using the appropriate index.
///
/// This is a standalone function that can be called from the executor
/// when a `SIMILAR_TO` condition is encountered in a WHERE clause.
///
/// In a full integration, this would dispatch to VectorStore, MmapVectorStore,
/// GWI, or CascadeIndex based on the `index_hint` and available indexes.
/// For now, it performs brute-force cosine similarity over the provided rows.
pub fn execute_similarity(
    condition: &SimilarityCondition,
    candidate_rows: &[ResultRow],
) -> Result<Vec<SimilarityResult>, QueryError> {
    let query = &condition.query_vector;
    let k = condition.k;

    let mut scored: Vec<SimilarityResult> = candidate_rows
        .iter()
        .filter_map(|row| {
            // Extract vector from the row value
            match &row.value {
                Atom::Vector(vec_data, _dims) => {
                    let score = cosine_similarity(query, vec_data);
                    Some(SimilarityResult {
                        key: row.key.clone(),
                        score,
                        vector: Some(vec_data.clone()),
                    })
                }
                _ => None, // Skip non-vector rows
            }
        })
        .collect();

    // Sort by score descending (higher = more similar)
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top-k
    scored.truncate(k);

    Ok(scored)
}

/// Filter rows to only those that pass the similarity threshold.
pub fn filter_by_similarity(
    condition: &SimilarityCondition,
    rows: Vec<ResultRow>,
    min_score: f64,
) -> Vec<ResultRow> {
    let query = &condition.query_vector;

    rows.into_iter()
        .filter(|row| match &row.value {
            Atom::Vector(vec_data, _) => cosine_similarity(query, vec_data) >= min_score,
            _ => false,
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Similarity computation
// ═══════════════════════════════════════════════════════════════════════

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 = identical direction.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector_row(key: &str, vec: Vec<f32>) -> ResultRow {
        let dims = vec.len() as u16;
        ResultRow {
            key: key.to_string(),
            value: Atom::Vector(vec, dims),
            timestamp: 0,
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_execute_similarity_top_k() {
        let rows = vec![
            make_vector_row("close", vec![0.9, 0.1, 0.0]),
            make_vector_row("medium", vec![0.5, 0.5, 0.0]),
            make_vector_row("far", vec![0.0, 0.0, 1.0]),
        ];

        let condition = SimilarityCondition {
            query_vector: vec![1.0, 0.0, 0.0],
            k: 2,
            index_hint: None,
        };

        let results = execute_similarity(&condition, &rows).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "close"); // highest similarity
        assert_eq!(results[1].key, "medium");
    }

    #[test]
    fn test_filter_by_similarity() {
        let rows = vec![
            make_vector_row("close", vec![0.9, 0.1, 0.0]),
            make_vector_row("far", vec![0.0, 0.0, 1.0]),
        ];

        let condition = SimilarityCondition {
            query_vector: vec![1.0, 0.0, 0.0],
            k: 10,
            index_hint: None,
        };

        let filtered = filter_by_similarity(&condition, rows, 0.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].key, "close");
    }

    #[test]
    fn test_non_vector_rows_skipped() {
        let rows = vec![
            ResultRow {
                key: "text".into(),
                value: Atom::Text("hello".into()),
                timestamp: 0,
            },
            make_vector_row("vec", vec![1.0, 0.0, 0.0]),
        ];

        let condition = SimilarityCondition {
            query_vector: vec![1.0, 0.0, 0.0],
            k: 10,
            index_hint: None,
        };

        let results = execute_similarity(&condition, &rows).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "vec");
    }
}

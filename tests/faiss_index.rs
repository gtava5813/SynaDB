//! FAISS index integration tests
//! Only run when faiss feature is enabled
//!
//! **Validates: Requirements 1.5**

#![cfg(feature = "faiss")]

use synadb::distance::DistanceMetric;
use synadb::faiss_index::{FaissConfig, FaissIndex};

#[test]
fn test_faiss_config_default() {
    let config = FaissConfig::default();
    assert_eq!(config.index_type, "IVF1024,Flat");
    assert_eq!(config.train_size, 10000);
    assert_eq!(config.nprobe, 10);
    assert!(!config.use_gpu);
}

#[test]
fn test_faiss_index_creation() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let index = FaissIndex::new(768, DistanceMetric::Cosine, config);
    assert!(index.is_ok());
}

#[test]
fn test_faiss_index_creation_euclidean() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let index = FaissIndex::new(128, DistanceMetric::Euclidean, config);
    assert!(index.is_ok());
}

#[test]
fn test_faiss_insert_search() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let mut index = FaissIndex::new(128, DistanceMetric::Cosine, config).unwrap();

    // Insert vectors
    let v1: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let v2: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.01).collect();

    index.insert("vec1", &v1).unwrap();
    index.insert("vec2", &v2).unwrap();

    // Search
    let results = index.search(&v1, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "vec1"); // Most similar to itself
}

#[test]
fn test_faiss_index_len() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let mut index = FaissIndex::new(64, DistanceMetric::Cosine, config).unwrap();

    assert!(index.is_empty());
    assert_eq!(index.len(), 0);

    let v1: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    index.insert("vec1", &v1).unwrap();

    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);
}

#[test]
fn test_faiss_index_dimensions() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let index = FaissIndex::new(256, DistanceMetric::Cosine, config).unwrap();
    assert_eq!(index.dimensions(), 256);
}

#[test]
fn test_faiss_dimension_mismatch() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let mut index = FaissIndex::new(128, DistanceMetric::Cosine, config).unwrap();

    // Try to insert a vector with wrong dimensions
    let wrong_dims: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
    let result = index.insert("wrong", &wrong_dims);

    assert!(result.is_err());
}

#[test]
fn test_faiss_multiple_inserts() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let mut index = FaissIndex::new(32, DistanceMetric::Euclidean, config).unwrap();

    // Insert multiple vectors
    for i in 0..100 {
        let vec: Vec<f32> = (0..32).map(|j| (i * 32 + j) as f32 * 0.001).collect();
        index.insert(&format!("vec{}", i), &vec).unwrap();
    }

    assert_eq!(index.len(), 100);

    // Search should return k results
    let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.001).collect();
    let results = index.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_faiss_search_k_larger_than_index() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let mut index = FaissIndex::new(32, DistanceMetric::Cosine, config).unwrap();

    // Insert only 3 vectors
    for i in 0..3 {
        let vec: Vec<f32> = (0..32).map(|j| (i * 32 + j) as f32 * 0.01).collect();
        index.insert(&format!("vec{}", i), &vec).unwrap();
    }

    // Search for more than available
    let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let results = index.search(&query, 10).unwrap();

    // Should return at most the number of vectors in the index
    assert!(results.len() <= 3);
}

#[test]
fn test_faiss_dot_product_metric() {
    let config = FaissConfig {
        index_type: "Flat".to_string(),
        ..Default::default()
    };
    let index = FaissIndex::new(64, DistanceMetric::DotProduct, config);
    assert!(index.is_ok());
}

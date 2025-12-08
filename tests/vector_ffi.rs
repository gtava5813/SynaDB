//! Integration tests for vector storage and retrieval.
//!
//! Tests that vectors can be stored and retrieved via the database API.
//! This validates the underlying functionality that the FFI functions use.
//!
//! **Validates: Requirements 1.1**

use synadb::{Atom, SynaDB};
use tempfile::tempdir;

#[test]
fn test_vector_roundtrip() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector.db");

    // Create test vector data (768 dimensions like BERT)
    let dimensions: u16 = 768;
    let vector_data: Vec<f32> = (0..dimensions).map(|i| i as f32 * 0.001).collect();

    let mut db = SynaDB::new(&db_path).expect("failed to create db");

    // Store vector
    let _offset = db
        .append("test_vector", Atom::Vector(vector_data.clone(), dimensions))
        .expect("append should succeed");

    // Retrieve vector
    let result = db.get("test_vector").expect("get should succeed");
    assert!(result.is_some(), "Key should exist after write");

    match result.unwrap() {
        Atom::Vector(data, dims) => {
            assert_eq!(dims, dimensions, "Dimensions mismatch");
            assert_eq!(data.len(), dimensions as usize, "Data length mismatch");

            // Verify data matches
            for (i, (original, retrieved)) in vector_data.iter().zip(data.iter()).enumerate() {
                assert!(
                    (original - retrieved).abs() < 1e-6,
                    "Data mismatch at index {}: expected {}, got {}",
                    i,
                    original,
                    retrieved
                );
            }
        }
        other => panic!("Expected Vector atom, got {:?}", other.type_name()),
    }
}

#[test]
fn test_vector_small_dimensions() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector_small.db");

    // Test with small vector (64 dimensions - minimum common size)
    let dimensions: u16 = 64;
    let vector_data: Vec<f32> = (0..dimensions).map(|i| (i as f32).sin()).collect();

    let mut db = SynaDB::new(&db_path).expect("failed to create db");

    db.append("small_vec", Atom::Vector(vector_data.clone(), dimensions))
        .expect("append should succeed");

    let result = db.get("small_vec").expect("get should succeed");
    match result.unwrap() {
        Atom::Vector(data, dims) => {
            assert_eq!(dims, dimensions);
            for (original, retrieved) in vector_data.iter().zip(data.iter()) {
                assert!((original - retrieved).abs() < 1e-6);
            }
        }
        _ => panic!("Expected Vector atom"),
    }
}

#[test]
fn test_vector_large_dimensions() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector_large.db");

    // Test with large vector (4096 dimensions - maximum common size)
    let dimensions: u16 = 4096;
    let vector_data: Vec<f32> = (0..dimensions).map(|i| (i as f32 * 0.0001).cos()).collect();

    let mut db = SynaDB::new(&db_path).expect("failed to create db");

    db.append("large_vec", Atom::Vector(vector_data.clone(), dimensions))
        .expect("append should succeed");

    let result = db.get("large_vec").expect("get should succeed");
    match result.unwrap() {
        Atom::Vector(data, dims) => {
            assert_eq!(dims, dimensions);
            for (original, retrieved) in vector_data.iter().zip(data.iter()) {
                assert!((original - retrieved).abs() < 1e-6);
            }
        }
        _ => panic!("Expected Vector atom"),
    }
}

#[test]
fn test_vector_key_not_found() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector_notfound.db");

    let mut db = SynaDB::new(&db_path).expect("failed to create db");

    let result = db.get("nonexistent").expect("get should succeed");
    assert!(result.is_none(), "Key should not exist");
}

#[test]
fn test_vector_type_mismatch() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector_type.db");

    let mut db = SynaDB::new(&db_path).expect("failed to create db");

    // Store a float, not a vector
    db.append("not_a_vector", Atom::Float(3.14))
        .expect("append should succeed");

    // Retrieve and check it's not a vector
    let result = db.get("not_a_vector").expect("get should succeed");
    match result.unwrap() {
        Atom::Vector(_, _) => panic!("Should not be a vector"),
        Atom::Float(f) => assert!((f - 3.14).abs() < 1e-6),
        _ => panic!("Expected Float atom"),
    }
}

#[test]
fn test_vector_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("test_vector_persist.db");

    let dimensions: u16 = 128;
    let vector_data: Vec<f32> = (0..dimensions).map(|i| i as f32 * 0.01).collect();

    // Write and close
    {
        let mut db = SynaDB::new(&db_path).expect("failed to create db");
        db.append("persist_vec", Atom::Vector(vector_data.clone(), dimensions))
            .expect("append should succeed");
    }

    // Reopen and verify
    {
        let mut db = SynaDB::new(&db_path).expect("failed to reopen db");
        let result = db.get("persist_vec").expect("get should succeed");
        match result.unwrap() {
            Atom::Vector(data, dims) => {
                assert_eq!(dims, dimensions);
                for (original, retrieved) in vector_data.iter().zip(data.iter()) {
                    assert!((original - retrieved).abs() < 1e-6);
                }
            }
            _ => panic!("Expected Vector atom"),
        }
    }
}

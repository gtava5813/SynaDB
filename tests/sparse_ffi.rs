//! FFI Integration Tests for Sparse Vector Store
//!
//! Tests the C-ABI interface for the Sparse Vector Store.
//! Note: svs_* functions are declared safe (not `unsafe extern "C"`),
//! so no `unsafe` blocks are needed for calling them.

use std::ffi::CString;
use std::ptr;

// Import FFI functions and constants
use synadb::ffi_sparse::*;

#[test]
fn test_svs_create_close_cycle() {
    let path = CString::new("test_svs_create_close.db").unwrap();

    let result = svs_new(path.as_ptr());
    assert_eq!(result, SVS_SUCCESS, "Failed to create store");

    let exists = svs_exists(path.as_ptr());
    assert_eq!(exists, 1, "Store should exist");

    let result = svs_close(path.as_ptr());
    assert_eq!(result, SVS_SUCCESS, "Failed to close store");

    let exists = svs_exists(path.as_ptr());
    assert_eq!(exists, 0, "Store should not exist after close");
}

#[test]
fn test_svs_create_already_exists() {
    let path = CString::new("test_svs_already_exists.db").unwrap();

    let result = svs_new(path.as_ptr());
    assert_eq!(result, SVS_SUCCESS);

    let result = svs_new(path.as_ptr());
    assert_eq!(result, SVS_ERR_ALREADY_EXISTS);

    svs_close(path.as_ptr());
}

#[test]
fn test_svs_close_not_found() {
    let path = CString::new("test_svs_close_not_found.db").unwrap();

    let result = svs_close(path.as_ptr());
    assert_eq!(result, SVS_ERR_NOT_FOUND);
}

#[test]
fn test_svs_null_pointer_handling() {
    let result = svs_new(ptr::null());
    assert_eq!(result, SVS_ERR_NULL_PTR);

    let result = svs_close(ptr::null());
    assert_eq!(result, SVS_ERR_NULL_PTR);

    let result = svs_exists(ptr::null());
    assert_eq!(result, SVS_ERR_NULL_PTR);
}

#[test]
fn test_svs_index_and_search() {
    let path = CString::new("test_svs_index_search.db").unwrap();
    let key1 = CString::new("doc1").unwrap();
    let key2 = CString::new("doc2").unwrap();

    svs_new(path.as_ptr());

    // Index document 1: terms 100, 200 with weights 2.0, 1.0
    let term_ids1: [u32; 2] = [100, 200];
    let weights1: [f32; 2] = [2.0, 1.0];
    let doc_id1 = svs_index(
        path.as_ptr(),
        key1.as_ptr(),
        term_ids1.as_ptr(),
        weights1.as_ptr(),
        2,
    );
    assert!(doc_id1 >= 0, "Failed to index doc1");

    // Index document 2: terms 100, 300 with weights 1.0, 3.0
    let term_ids2: [u32; 2] = [100, 300];
    let weights2: [f32; 2] = [1.0, 3.0];
    let doc_id2 = svs_index(
        path.as_ptr(),
        key2.as_ptr(),
        term_ids2.as_ptr(),
        weights2.as_ptr(),
        2,
    );
    assert!(doc_id2 >= 0, "Failed to index doc2");

    let len = svs_len(path.as_ptr());
    assert_eq!(len, 2, "Should have 2 documents");

    // Search for term 100
    let query_ids: [u32; 1] = [100];
    let query_weights: [f32; 1] = [1.0];
    let mut out_keys: [*mut i8; 10] = [ptr::null_mut(); 10];
    let mut out_scores: [f32; 10] = [0.0; 10];
    let mut out_count: u32 = 0;

    let result = svs_search(
        path.as_ptr(),
        query_ids.as_ptr(),
        query_weights.as_ptr(),
        1,
        10,
        out_keys.as_mut_ptr(),
        out_scores.as_mut_ptr(),
        &mut out_count,
    );
    assert_eq!(result, SVS_SUCCESS, "Search failed");
    assert_eq!(out_count, 2, "Should find 2 results");

    assert!(
        out_scores[0] > out_scores[1],
        "Results should be sorted by score"
    );

    // Free keys
    for i in 0..out_count as usize {
        svs_free_key(out_keys[i]);
    }

    svs_close(path.as_ptr());
}


#[test]
fn test_svs_delete() {
    let path = CString::new("test_svs_delete.db").unwrap();
    let key = CString::new("doc_to_delete").unwrap();

    svs_new(path.as_ptr());

    let term_ids: [u32; 1] = [100];
    let weights: [f32; 1] = [1.0];
    svs_index(
        path.as_ptr(),
        key.as_ptr(),
        term_ids.as_ptr(),
        weights.as_ptr(),
        1,
    );

    assert_eq!(svs_len(path.as_ptr()), 1);

    let result = svs_delete(path.as_ptr(), key.as_ptr());
    assert_eq!(result, SVS_SUCCESS);

    assert_eq!(svs_len(path.as_ptr()), 0);

    // Try to delete again - should fail
    let result = svs_delete(path.as_ptr(), key.as_ptr());
    assert_eq!(result, SVS_ERR_NOT_FOUND);

    svs_close(path.as_ptr());
}

#[test]
fn test_svs_stats() {
    let path = CString::new("test_svs_stats.db").unwrap();
    let key1 = CString::new("doc1").unwrap();
    let key2 = CString::new("doc2").unwrap();

    svs_new(path.as_ptr());

    let term_ids1: [u32; 2] = [100, 200];
    let weights1: [f32; 2] = [1.0, 2.0];
    svs_index(
        path.as_ptr(),
        key1.as_ptr(),
        term_ids1.as_ptr(),
        weights1.as_ptr(),
        2,
    );

    let term_ids2: [u32; 2] = [100, 300];
    let weights2: [f32; 2] = [1.0, 3.0];
    svs_index(
        path.as_ptr(),
        key2.as_ptr(),
        term_ids2.as_ptr(),
        weights2.as_ptr(),
        2,
    );

    let mut num_docs: u32 = 0;
    let mut num_terms: u32 = 0;
    let mut num_postings: u32 = 0;
    let mut avg_doc_len: f32 = 0.0;

    let result = svs_stats(
        path.as_ptr(),
        &mut num_docs,
        &mut num_terms,
        &mut num_postings,
        &mut avg_doc_len,
    );

    assert_eq!(result, SVS_SUCCESS);
    assert_eq!(num_docs, 2);
    assert_eq!(num_terms, 3); // 100, 200, 300
    assert_eq!(num_postings, 4); // 2 docs × 2 terms each
    assert!((avg_doc_len - 2.0).abs() < 0.001);

    svs_close(path.as_ptr());
}

#[test]
fn test_svs_index_not_found() {
    let path = CString::new("test_svs_index_not_found.db").unwrap();
    let key = CString::new("doc1").unwrap();

    let term_ids: [u32; 1] = [100];
    let weights: [f32; 1] = [1.0];
    let result = svs_index(
        path.as_ptr(),
        key.as_ptr(),
        term_ids.as_ptr(),
        weights.as_ptr(),
        1,
    );
    assert_eq!(result, SVS_ERR_NOT_FOUND as i64);
}

#[test]
fn test_svs_save_and_open() {
    let path = CString::new("test_svs_save_open.db").unwrap();
    let file_path = CString::new("target/test_svs_save_open.svs").unwrap();
    let key1 = CString::new("doc1").unwrap();
    let key2 = CString::new("doc2").unwrap();

    svs_new(path.as_ptr());

    let term_ids1: [u32; 2] = [100, 200];
    let weights1: [f32; 2] = [2.0, 1.0];
    svs_index(
        path.as_ptr(),
        key1.as_ptr(),
        term_ids1.as_ptr(),
        weights1.as_ptr(),
        2,
    );

    let term_ids2: [u32; 2] = [100, 300];
    let weights2: [f32; 2] = [1.0, 3.0];
    svs_index(
        path.as_ptr(),
        key2.as_ptr(),
        term_ids2.as_ptr(),
        weights2.as_ptr(),
        2,
    );

    let result = svs_save(path.as_ptr(), file_path.as_ptr());
    assert_eq!(result, SVS_SUCCESS, "Failed to save");

    svs_close(path.as_ptr());

    // Open from file with a new registry key
    let path2 = CString::new("test_svs_save_open_loaded.db").unwrap();
    let result = svs_open(path2.as_ptr(), file_path.as_ptr());
    assert_eq!(result, SVS_SUCCESS, "Failed to open");

    let len = svs_len(path2.as_ptr());
    assert_eq!(len, 2, "Should have 2 documents after load");

    // Search should work
    let query_ids: [u32; 1] = [100];
    let query_weights: [f32; 1] = [1.0];
    let mut out_keys: [*mut i8; 10] = [ptr::null_mut(); 10];
    let mut out_scores: [f32; 10] = [0.0; 10];
    let mut out_count: u32 = 0;

    let result = svs_search(
        path2.as_ptr(),
        query_ids.as_ptr(),
        query_weights.as_ptr(),
        1,
        10,
        out_keys.as_mut_ptr(),
        out_scores.as_mut_ptr(),
        &mut out_count,
    );
    assert_eq!(result, SVS_SUCCESS, "Search failed after load");
    assert_eq!(out_count, 2, "Should find 2 results after load");

    for i in 0..out_count as usize {
        svs_free_key(out_keys[i]);
    }

    svs_close(path2.as_ptr());
    let _ = std::fs::remove_file("target/test_svs_save_open.svs");
}

#[test]
fn test_svs_save_not_found() {
    let path = CString::new("test_svs_save_not_found.db").unwrap();
    let file_path = CString::new("target/test_svs_save_not_found.svs").unwrap();

    let result = svs_save(path.as_ptr(), file_path.as_ptr());
    assert_eq!(result, SVS_ERR_NOT_FOUND);
}

#[test]
fn test_svs_open_already_exists() {
    let path = CString::new("test_svs_open_exists.db").unwrap();
    let file_path = CString::new("target/test_svs_open_exists.svs").unwrap();

    svs_new(path.as_ptr());
    svs_save(path.as_ptr(), file_path.as_ptr());

    let result = svs_open(path.as_ptr(), file_path.as_ptr());
    assert_eq!(result, SVS_ERR_ALREADY_EXISTS);

    svs_close(path.as_ptr());
    let _ = std::fs::remove_file("target/test_svs_open_exists.svs");
}

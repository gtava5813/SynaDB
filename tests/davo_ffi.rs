//! FFI integration tests for the DAVO module.
//!
//! Tests the C-ABI functions for FreshnessIndexV2 and DecayPredictor.

#![cfg(feature = "davo")]

use std::ffi::CString;
use std::os::raw::c_char;
use synadb::ffi_davo::*;

/// Helper to create a C string from a Rust string.
fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

// ── FreshnessIndex FFI tests ─────────────────────────────────────────

#[test]
fn test_freshness_index_lifecycle() {
    let path = cstr("test_fi_lifecycle");

    // Create
    let rc = SYNA_davo_freshness_index_new(path.as_ptr(), 0.5);
    assert_eq!(rc, DAVO_SUCCESS);

    // Duplicate create should fail
    let rc = SYNA_davo_freshness_index_new(path.as_ptr(), 0.5);
    assert_eq!(rc, DAVO_ERR_ALREADY_EXISTS);

    // Insert a key
    let key = cstr("sensor/temp");
    let rc = SYNA_davo_freshness_index_insert(path.as_ptr(), key.as_ptr(), 0.001);
    assert_eq!(rc, DAVO_SUCCESS);

    // Get freshness (should be ~1.0 since just inserted)
    let mut freshness: f64 = 0.0;
    let rc = SYNA_davo_freshness_index_get_freshness(
        path.as_ptr(),
        key.as_ptr(),
        &mut freshness as *mut f64,
    );
    assert_eq!(rc, DAVO_SUCCESS);
    assert!(
        freshness > 0.99,
        "freshness should be ~1.0, got {}",
        freshness
    );

    // Len should be 1
    let len = SYNA_davo_freshness_index_len(path.as_ptr());
    assert_eq!(len, 1);

    // Close
    let rc = SYNA_davo_freshness_index_close(path.as_ptr());
    assert_eq!(rc, DAVO_SUCCESS);

    // Close again should fail
    let rc = SYNA_davo_freshness_index_close(path.as_ptr());
    assert_eq!(rc, DAVO_ERR_NOT_FOUND);
}

#[test]
fn test_freshness_index_not_found_key() {
    let path = cstr("test_fi_notfound_key");
    let rc = SYNA_davo_freshness_index_new(path.as_ptr(), 0.0);
    assert_eq!(rc, DAVO_SUCCESS);

    let key = cstr("nonexistent");
    let mut freshness: f64 = 0.0;
    let rc = SYNA_davo_freshness_index_get_freshness(
        path.as_ptr(),
        key.as_ptr(),
        &mut freshness as *mut f64,
    );
    assert_eq!(rc, DAVO_ERR_NOT_FOUND);

    SYNA_davo_freshness_index_close(path.as_ptr());
}

#[test]
fn test_freshness_index_null_ptr() {
    let null: *const c_char = std::ptr::null();

    assert_eq!(SYNA_davo_freshness_index_new(null, 0.5), DAVO_ERR_NULL_PTR);
    assert_eq!(
        SYNA_davo_freshness_index_insert(null, null, 0.0),
        DAVO_ERR_NULL_PTR
    );
    assert_eq!(SYNA_davo_freshness_index_close(null), DAVO_ERR_NULL_PTR);
}

#[test]
fn test_freshness_index_evict_stale() {
    let path = cstr("test_fi_evict");
    SYNA_davo_freshness_index_new(path.as_ptr(), 0.5);

    // Insert with very fast decay
    let fast = cstr("fast_key");
    SYNA_davo_freshness_index_insert(path.as_ptr(), fast.as_ptr(), 10000.0);

    // Insert with no decay
    let slow = cstr("slow_key");
    SYNA_davo_freshness_index_insert(path.as_ptr(), slow.as_ptr(), 0.0);

    // Wait a tiny bit for fast key to become stale
    std::thread::sleep(std::time::Duration::from_millis(2));

    // Evict
    let mut evicted: usize = 0;
    let rc = SYNA_davo_freshness_index_evict_stale(path.as_ptr(), &mut evicted as *mut usize);
    assert_eq!(rc, DAVO_SUCCESS);
    assert_eq!(evicted, 1, "fast_key should have been evicted");

    // Only slow_key should remain
    let len = SYNA_davo_freshness_index_len(path.as_ptr());
    assert_eq!(len, 1);

    SYNA_davo_freshness_index_close(path.as_ptr());
}

#[test]
fn test_freshness_index_query_stale() {
    let path = cstr("test_fi_query_stale");
    SYNA_davo_freshness_index_new(path.as_ptr(), 0.5);

    // Insert with very fast decay
    let k1 = cstr("stale1");
    let k2 = cstr("stale2");
    SYNA_davo_freshness_index_insert(path.as_ptr(), k1.as_ptr(), 50000.0);
    SYNA_davo_freshness_index_insert(path.as_ptr(), k2.as_ptr(), 50000.0);

    // Insert with no decay
    let k3 = cstr("fresh");
    SYNA_davo_freshness_index_insert(path.as_ptr(), k3.as_ptr(), 0.0);

    std::thread::sleep(std::time::Duration::from_millis(2));

    let mut out_keys: *mut *mut c_char = std::ptr::null_mut();
    let mut out_count: usize = 0;
    let rc = SYNA_davo_freshness_index_query_stale(
        path.as_ptr(),
        &mut out_keys as *mut *mut *mut c_char,
        &mut out_count as *mut usize,
    );
    assert_eq!(rc, DAVO_SUCCESS);
    assert_eq!(out_count, 2, "two stale keys expected");

    // Free the returned keys
    SYNA_davo_free_keys(out_keys, out_count);

    SYNA_davo_freshness_index_close(path.as_ptr());
}

// ── DecayPredictor FFI tests ─────────────────────────────────────────

#[test]
fn test_predictor_lifecycle() {
    let path = cstr("test_pred_lifecycle");

    // Create
    let rc = SYNA_davo_predictor_new(path.as_ptr());
    assert_eq!(rc, DAVO_SUCCESS);

    // Duplicate create should fail
    let rc = SYNA_davo_predictor_new(path.as_ptr());
    assert_eq!(rc, DAVO_ERR_ALREADY_EXISTS);

    // Observe
    for _ in 0..50 {
        let rc = SYNA_davo_predictor_observe(path.as_ptr(), 0.05);
        assert_eq!(rc, DAVO_SUCCESS);
    }

    // Predict
    let mut prediction: f64 = 0.0;
    let rc = SYNA_davo_predictor_predict(path.as_ptr(), &mut prediction as *mut f64);
    assert_eq!(rc, DAVO_SUCCESS);
    assert!(
        (prediction - 0.05).abs() < 0.02,
        "prediction should be ~0.05, got {}",
        prediction
    );

    // Sample
    let mut sample: f64 = 0.0;
    let rc = SYNA_davo_predictor_sample(path.as_ptr(), &mut sample as *mut f64);
    assert_eq!(rc, DAVO_SUCCESS);
    assert!(sample > 0.0, "sample must be positive");

    // Uncertainty
    let mut unc: f64 = 0.0;
    let rc = SYNA_davo_predictor_uncertainty(path.as_ptr(), &mut unc as *mut f64);
    assert_eq!(rc, DAVO_SUCCESS);
    assert!(unc > 0.0, "uncertainty must be positive");

    // Close
    let rc = SYNA_davo_predictor_close(path.as_ptr());
    assert_eq!(rc, DAVO_SUCCESS);

    // Close again should fail
    let rc = SYNA_davo_predictor_close(path.as_ptr());
    assert_eq!(rc, DAVO_ERR_NOT_FOUND);
}

#[test]
fn test_predictor_null_ptr() {
    let null: *const c_char = std::ptr::null();

    assert_eq!(SYNA_davo_predictor_new(null), DAVO_ERR_NULL_PTR);
    assert_eq!(SYNA_davo_predictor_observe(null, 0.0), DAVO_ERR_NULL_PTR);
    assert_eq!(SYNA_davo_predictor_close(null), DAVO_ERR_NULL_PTR);
}

#[test]
fn test_predictor_not_found() {
    let path = cstr("nonexistent_predictor");
    let mut val: f64 = 0.0;

    assert_eq!(
        SYNA_davo_predictor_predict(path.as_ptr(), &mut val as *mut f64),
        DAVO_ERR_NOT_FOUND
    );
    assert_eq!(
        SYNA_davo_predictor_observe(path.as_ptr(), 0.01),
        DAVO_ERR_NOT_FOUND
    );
    assert_eq!(SYNA_davo_predictor_close(path.as_ptr()), DAVO_ERR_NOT_FOUND);
}

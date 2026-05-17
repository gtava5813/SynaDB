// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! FFI integration tests for the Feature Store.
//!
//! Tests all SYNA_fs_* functions through the C-ABI.

use std::ffi::CString;
use std::ptr;

use synadb::feature_store::ffi::*;

fn c_str(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[test]
fn test_ffi_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_test.db");
    let path_c = c_str(path.to_str().unwrap());

    // Open
    let result = SYNA_fs_new(path_c.as_ptr());
    assert_eq!(result, 1); // ERR_SUCCESS

    // Double open is OK
    let result = SYNA_fs_new(path_c.as_ptr());
    assert_eq!(result, 1);

    // Close
    let result = SYNA_fs_close(path_c.as_ptr());
    assert_eq!(result, 1);
}

#[test]
fn test_ffi_null_path() {
    let result = SYNA_fs_new(ptr::null());
    assert_eq!(result, -2); // ERR_INVALID_PATH

    let result = SYNA_fs_close(ptr::null());
    assert_eq!(result, -2);
}

#[test]
fn test_ffi_ingest_and_serve_float() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_ingest.db");
    let path_c = c_str(path.to_str().unwrap());
    let group_c = c_str("users");
    let entity_c = c_str("u1");
    let feature_c = c_str("score");

    SYNA_fs_new(path_c.as_ptr());

    // Ingest
    let result = SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        0.95,
        1000,
    );
    assert_eq!(result, 1);

    // Serve
    let mut out: f64 = 0.0;
    let result = SYNA_fs_serve_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        &mut out,
    );
    assert_eq!(result, 1);
    assert!((out - 0.95).abs() < 1e-10);

    SYNA_fs_close(path_c.as_ptr());
}

#[test]
fn test_ffi_ingest_int() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_int.db");
    let path_c = c_str(path.to_str().unwrap());
    let group_c = c_str("g");
    let entity_c = c_str("e");
    let feature_c = c_str("count");

    SYNA_fs_new(path_c.as_ptr());

    let result = SYNA_fs_ingest_int(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        42,
        1000,
    );
    assert_eq!(result, 1);

    SYNA_fs_close(path_c.as_ptr());
}

#[test]
fn test_ffi_get_at_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_version.db");
    let path_c = c_str(path.to_str().unwrap());
    let group_c = c_str("g");
    let entity_c = c_str("e");
    let feature_c = c_str("val");

    SYNA_fs_new(path_c.as_ptr());

    // Ingest 3 values
    SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        1.0,
        1000,
    );
    SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        2.0,
        2000,
    );
    SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        3.0,
        3000,
    );

    // Get latest (version 0)
    let mut out: f64 = 0.0;
    let result = SYNA_fs_get_at_version(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        0,
        &mut out,
    );
    assert_eq!(result, 1);
    assert!((out - 3.0).abs() < 1e-10);

    // Get version -2
    let result = SYNA_fs_get_at_version(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        -2,
        &mut out,
    );
    assert_eq!(result, 1);
    assert!((out - 2.0).abs() < 1e-10);

    // Get non-existent version
    let result = SYNA_fs_get_at_version(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        -5,
        &mut out,
    );
    assert_eq!(result, 0); // ERR_GENERIC (not found)

    SYNA_fs_close(path_c.as_ptr());
}

#[test]
fn test_ffi_get_at_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_ts.db");
    let path_c = c_str(path.to_str().unwrap());
    let group_c = c_str("g");
    let entity_c = c_str("e");
    let feature_c = c_str("val");

    SYNA_fs_new(path_c.as_ptr());

    SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        1.0,
        1000,
    );
    SYNA_fs_ingest_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        2.0,
        2000,
    );

    let mut out: f64 = 0.0;
    let result = SYNA_fs_get_at_timestamp(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        1500,
        &mut out,
    );
    assert_eq!(result, 1);
    assert!((out - 1.0).abs() < 1e-10);

    SYNA_fs_close(path_c.as_ptr());
}

#[test]
fn test_ffi_flush() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_flush.db");
    let path_c = c_str(path.to_str().unwrap());

    SYNA_fs_new(path_c.as_ptr());
    let result = SYNA_fs_flush(path_c.as_ptr());
    assert_eq!(result, 1);
    SYNA_fs_close(path_c.as_ptr());
}

#[test]
fn test_ffi_serve_null_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ffi_null.db");
    let path_c = c_str(path.to_str().unwrap());
    let group_c = c_str("g");
    let entity_c = c_str("e");
    let feature_c = c_str("f");

    SYNA_fs_new(path_c.as_ptr());

    // Null out pointer
    let result = SYNA_fs_serve_float(
        path_c.as_ptr(),
        group_c.as_ptr(),
        entity_c.as_ptr(),
        feature_c.as_ptr(),
        ptr::null_mut(),
    );
    assert_eq!(result, -2); // ERR_INVALID_PATH (null pointer)

    SYNA_fs_close(path_c.as_ptr());
}

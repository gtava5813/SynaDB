// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! C-ABI FFI layer for the Feature Store.
//!
//! All functions use `catch_unwind` to prevent panics from crossing the FFI boundary.
//! Functions follow the `SYNA_fs_*` naming convention.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::error::{ERR_GENERIC, ERR_INTERNAL_PANIC, ERR_INVALID_PATH, ERR_SUCCESS};

use super::schema::FeatureValue;
use super::{FeatureStore, FeatureStoreConfig};

// =============================================================================
// Global Feature Store Registry
// =============================================================================

static FS_REGISTRY: Lazy<Mutex<HashMap<String, FeatureStore>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// =============================================================================
// Lifecycle Functions
// =============================================================================

/// Create or open a feature store at the given path.
///
/// # Safety
/// `path` must be a valid null-terminated C string or null.
#[no_mangle]
pub extern "C" fn SYNA_fs_new(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        if registry.contains_key(path_str) {
            return ERR_SUCCESS;
        }

        let config = FeatureStoreConfig::default();
        match FeatureStore::new(path_str, config) {
            Ok(store) => {
                registry.insert(path_str.to_string(), store);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Close a feature store and remove it from the registry.
///
/// # Safety
/// `path` must be a valid null-terminated C string or null.
#[no_mangle]
pub extern "C" fn SYNA_fs_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        match registry.remove(path_str) {
            Some(store) => match store.close() {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

// =============================================================================
// Ingestion Functions
// =============================================================================

/// Ingest a single float feature value.
///
/// # Safety
/// All pointer parameters must be valid null-terminated C strings or null.
#[no_mangle]
pub extern "C" fn SYNA_fs_ingest_float(
    path: *const c_char,
    group: *const c_char,
    entity_key: *const c_char,
    feature: *const c_char,
    value: f64,
    event_ts: u64,
) -> i32 {
    std::panic::catch_unwind(|| {
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let group_str = match ptr_to_str(group) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let entity_str = match ptr_to_str(entity_key) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let feature_str = match ptr_to_str(feature) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        let store = match registry.get_mut(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.ingest(
            group_str,
            entity_str,
            event_ts,
            &[(feature_str, FeatureValue::Float64(value))],
        ) {
            Ok(_) => ERR_SUCCESS,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Ingest a single integer feature value.
///
/// # Safety
/// All pointer parameters must be valid null-terminated C strings or null.
#[no_mangle]
pub extern "C" fn SYNA_fs_ingest_int(
    path: *const c_char,
    group: *const c_char,
    entity_key: *const c_char,
    feature: *const c_char,
    value: i64,
    event_ts: u64,
) -> i32 {
    std::panic::catch_unwind(|| {
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let group_str = match ptr_to_str(group) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let entity_str = match ptr_to_str(entity_key) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let feature_str = match ptr_to_str(feature) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        let store = match registry.get_mut(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.ingest(
            group_str,
            entity_str,
            event_ts,
            &[(feature_str, FeatureValue::Int64(value))],
        ) {
            Ok(_) => ERR_SUCCESS,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

// =============================================================================
// Serving Functions
// =============================================================================

/// Serve the latest float value for a feature.
///
/// Returns ERR_SUCCESS and writes to `out` on success.
///
/// # Safety
/// All pointer parameters must be valid. `out` must point to writable f64.
#[no_mangle]
pub extern "C" fn SYNA_fs_serve_float(
    path: *const c_char,
    group: *const c_char,
    entity_key: *const c_char,
    feature: *const c_char,
    out: *mut f64,
) -> i32 {
    std::panic::catch_unwind(|| {
        if out.is_null() {
            return ERR_INVALID_PATH;
        }
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let group_str = match ptr_to_str(group) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let entity_str = match ptr_to_str(entity_key) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let feature_str = match ptr_to_str(feature) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        let store = match registry.get_mut(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.serve(group_str, entity_str, &[feature_str]) {
            Ok(vector) => {
                if let Some((_, FeatureValue::Float64(v))) = vector.values.first() {
                    unsafe { *out = *v };
                    ERR_SUCCESS
                } else {
                    ERR_GENERIC
                }
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Get a float value at a specific version.
///
/// version = 0 or -1 returns latest. version = -N returns Nth-most-recent.
///
/// # Safety
/// All pointer parameters must be valid. `out` must point to writable f64.
#[no_mangle]
pub extern "C" fn SYNA_fs_get_at_version(
    path: *const c_char,
    group: *const c_char,
    entity_key: *const c_char,
    feature: *const c_char,
    version: i64,
    out: *mut f64,
) -> i32 {
    std::panic::catch_unwind(|| {
        if out.is_null() {
            return ERR_INVALID_PATH;
        }
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let group_str = match ptr_to_str(group) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let entity_str = match ptr_to_str(entity_key) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let feature_str = match ptr_to_str(feature) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let registry = FS_REGISTRY.lock();
        let store = match registry.get(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.get_at_version(group_str, entity_str, feature_str, version) {
            Some(FeatureValue::Float64(v)) => {
                unsafe { *out = v };
                ERR_SUCCESS
            }
            _ => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Get a float value at a specific timestamp.
///
/// # Safety
/// All pointer parameters must be valid. `out` must point to writable f64.
#[no_mangle]
pub extern "C" fn SYNA_fs_get_at_timestamp(
    path: *const c_char,
    group: *const c_char,
    entity_key: *const c_char,
    feature: *const c_char,
    timestamp: u64,
    out: *mut f64,
) -> i32 {
    std::panic::catch_unwind(|| {
        if out.is_null() {
            return ERR_INVALID_PATH;
        }
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let group_str = match ptr_to_str(group) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let entity_str = match ptr_to_str(entity_key) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };
        let feature_str = match ptr_to_str(feature) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let registry = FS_REGISTRY.lock();
        let store = match registry.get(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.get_at_timestamp(group_str, entity_str, feature_str, timestamp) {
            Some(FeatureValue::Float64(v)) => {
                unsafe { *out = v };
                ERR_SUCCESS
            }
            _ => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Flush the write buffer for a feature store.
///
/// # Safety
/// `path` must be a valid null-terminated C string or null.
#[no_mangle]
pub extern "C" fn SYNA_fs_flush(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        let path_str = match ptr_to_str(path) {
            Some(s) => s,
            None => return ERR_INVALID_PATH,
        };

        let mut registry = FS_REGISTRY.lock();
        let store = match registry.get_mut(path_str) {
            Some(s) => s,
            None => return ERR_GENERIC,
        };

        match store.flush() {
            Ok(_) => ERR_SUCCESS,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert a C string pointer to a Rust &str, returning None on null or invalid UTF-8.
fn ptr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

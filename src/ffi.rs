// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! C-ABI Foreign Function Interface for Syna database.
//!
//! This module provides extern "C" functions for cross-language access.
//! All functions use `catch_unwind` to prevent Rust panics from unwinding
//! into foreign code.

// FFI functions intentionally take raw pointers without being marked unsafe
// because they handle null checks and use catch_unwind for safety
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Using slice::from_raw_parts_mut with Box::from_raw is intentional for FFI memory management
#![allow(clippy::cast_slice_from_raw_parts)]

use std::ffi::CStr;
use std::os::raw::c_char;

use crate::engine::{close_db, free_tensor, open_db, open_db_with_config, with_db, DbConfig};
use crate::error::{
    ERR_GENERIC, ERR_INTERNAL_PANIC, ERR_INVALID_PATH, ERR_KEY_NOT_FOUND, ERR_SUCCESS,
    ERR_TYPE_MISMATCH,
};
use crate::types::Atom;

/// Opens a database at the given path and registers it in the global registry.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Database opened successfully or was already open
/// * `0` (ERR_GENERIC) - Generic error during database open
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 6.2, 6.3, 9.4_
#[no_mangle]
pub extern "C" fn SYNA_open(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Open the database
        match open_db(path_str) {
            Ok(_) => ERR_SUCCESS,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Opens a database with sync_on_write disabled for high-throughput writes.
///
/// This is optimized for bulk ingestion scenarios where durability can be
/// traded for speed. Data is still written to disk but not fsynced after
/// each write, achieving 100K+ ops/sec instead of ~100 ops/sec.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `sync_on_write` - 1 for sync after each write (durable), 0 for no sync (fast)
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Database opened successfully or was already open
/// * `0` (ERR_GENERIC) - Generic error during database open
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
#[no_mangle]
pub extern "C" fn SYNA_open_with_config(path: *const c_char, sync_on_write: i32) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Create config with sync_on_write setting
        let config = DbConfig {
            sync_on_write: sync_on_write != 0,
            ..DbConfig::default()
        };

        // Open the database with config
        match open_db_with_config(path_str, config) {
            Ok(_) => ERR_SUCCESS,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Closes a database and removes it from the global registry.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Database closed successfully
/// * `0` (ERR_GENERIC) - Generic error during database close
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 6.2_
#[no_mangle]
pub extern "C" fn SYNA_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Close the database
        match close_db(path_str) {
            Ok(_) => ERR_SUCCESS,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Writes a float value to the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `value` - The f64 value to store
///
/// # Returns
/// * Positive value - The byte offset where the entry was written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path or key is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
///
/// _Requirements: 6.2, 6.3_
#[no_mangle]
pub extern "C" fn SYNA_put_float(path: *const c_char, key: *const c_char, value: f64) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Call with_db to append the value
        match with_db(path_str, |db| db.append(key_str, Atom::Float(value))) {
            Ok(offset) => offset as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Writes an integer value to the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `value` - The i64 value to store
///
/// # Returns
/// * Positive value - The byte offset where the entry was written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path or key is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
///
/// _Requirements: 6.2, 6.3_
#[no_mangle]
pub extern "C" fn SYNA_put_int(path: *const c_char, key: *const c_char, value: i64) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Call with_db to append the value
        match with_db(path_str, |db| db.append(key_str, Atom::Int(value))) {
            Ok(offset) => offset as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Writes a text value to the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `value` - Null-terminated C string containing the text value to store
///
/// # Returns
/// * Positive value - The byte offset where the entry was written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or value is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path`, `key`, and `value` must be valid null-terminated C strings or null
///
/// _Requirements: 6.2, 6.3_
#[no_mangle]
pub extern "C" fn SYNA_put_text(
    path: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || value.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let value_str = match unsafe { CStr::from_ptr(value) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Call with_db to append the value
        match with_db(path_str, |db| {
            db.append(key_str, Atom::Text(value_str.to_string()))
        }) {
            Ok(offset) => offset as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Writes a byte array to the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `data` - Pointer to the byte array to store
/// * `len` - Length of the byte array
///
/// # Returns
/// * Positive value - The byte offset where the entry was written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or data is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `data` must be a valid pointer to at least `len` bytes, or null
///
/// _Requirements: 6.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_put_bytes(
    path: *const c_char,
    key: *const c_char,
    data: *const u8,
    len: usize,
) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || (data.is_null() && len > 0) {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Create Vec<u8> from raw pointer and length
        let bytes = if len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(data, len) }.to_vec()
        };

        // Call with_db to append the value
        match with_db(path_str, |db| db.append(key_str, Atom::Bytes(bytes))) {
            Ok(offset) => offset as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Writes multiple float values to the database in a single batch operation.
///
/// This is optimized for high-throughput ingestion scenarios. All values are
/// written under the same key, building up a history that can be extracted
/// as a tensor with `SYNA_get_history_tensor()`.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `values` - Pointer to array of f64 values
/// * `count` - Number of values in the array
///
/// # Returns
/// * Positive value - Number of values written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or values is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `values` must be a valid pointer to at least `count` f64 values, or null if count is 0
///
/// # Performance
/// This function is significantly faster than calling `SYNA_put_float()` in a loop:
/// - Single FFI boundary crossing
/// - Single mutex lock for all writes
/// - Single fsync at the end (if sync_on_write is enabled)
#[no_mangle]
pub extern "C" fn SYNA_put_floats_batch(
    path: *const c_char,
    key: *const c_char,
    values: *const f64,
    count: usize,
) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || (values.is_null() && count > 0) {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Create slice from raw pointer
        let slice = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(values, count) }
        };

        // Call with_db to batch append
        match with_db(path_str, |db| db.append_floats_batch(key_str, slice)) {
            Ok(n) => n as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Reads a float value from the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out` - Pointer to write the f64 value to
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Value read successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or out is null or invalid UTF-8
/// * `-5` (ERR_KEY_NOT_FOUND) - Key not found in database
/// * `-6` (ERR_TYPE_MISMATCH) - Value exists but is not a Float
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out` must be a valid pointer to an f64
///
/// _Requirements: 6.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_get_float(path: *const c_char, key: *const c_char, out: *mut f64) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to get the value
        match with_db(path_str, |db| db.get(key_str)) {
            Ok(Some(Atom::Float(f))) => {
                // Write value to out pointer
                unsafe { *out = f };
                ERR_SUCCESS
            }
            Ok(Some(_)) => {
                // Value exists but is not a Float
                ERR_TYPE_MISMATCH
            }
            Ok(None) => {
                // Key not found
                ERR_KEY_NOT_FOUND
            }
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Reads an integer value from the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out` - Pointer to write the i64 value to
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Value read successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or out is null or invalid UTF-8
/// * `-5` (ERR_KEY_NOT_FOUND) - Key not found in database
/// * `-6` (ERR_TYPE_MISMATCH) - Value exists but is not an Int
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out` must be a valid pointer to an i64
///
/// _Requirements: 6.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_get_int(path: *const c_char, key: *const c_char, out: *mut i64) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to get the value
        match with_db(path_str, |db| db.get(key_str)) {
            Ok(Some(Atom::Int(i))) => {
                // Write value to out pointer
                unsafe { *out = i };
                ERR_SUCCESS
            }
            Ok(Some(_)) => {
                // Value exists but is not an Int
                ERR_TYPE_MISMATCH
            }
            Ok(None) => {
                // Key not found
                ERR_KEY_NOT_FOUND
            }
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Retrieves the complete history of float values for a key as a contiguous array.
///
/// This function is designed for AI/ML workloads where you need to feed time-series
/// data directly to frameworks like PyTorch or TensorFlow.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out_len` - Pointer to write the array length to
///
/// # Returns
/// * Non-null pointer to contiguous f64 array on success
/// * Null pointer on error (check out_len for error code)
///
/// # Error Codes (written to out_len on error)
/// * `0` - Empty history (no float values for this key)
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out_len` must be a valid pointer to a usize
/// * The returned pointer MUST be freed using `SYNA_free_tensor()` to avoid memory leaks
///
/// _Requirements: 4.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_get_history_tensor(
    path: *const c_char,
    key: *const c_char,
    out_len: *mut usize,
) -> *mut f64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // Call with_db to get the history tensor
        match with_db(path_str, |db| db.get_history_tensor(key_str)) {
            Ok((ptr, len)) => {
                // Write length to out_len pointer
                unsafe { *out_len = len };
                ptr
            }
            Err(_) => {
                // Set out_len to 0 on error
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees memory allocated by `SYNA_get_history_tensor()`.
///
/// # Arguments
/// * `ptr` - Pointer returned by `SYNA_get_history_tensor()`
/// * `len` - Length returned by `SYNA_get_history_tensor()`
///
/// # Safety
/// * `ptr` must have been returned by `SYNA_get_history_tensor()`
/// * `len` must be the length returned alongside the pointer
/// * This function must only be called once per pointer
/// * Calling with a null pointer or zero length is safe (no-op)
///
/// _Requirements: 4.3, 6.5_
#[no_mangle]
pub extern "C" fn SYNA_free_tensor(ptr: *mut f64, len: usize) {
    std::panic::catch_unwind(|| {
        // Call internal free_tensor (safe wrapper)
        unsafe { free_tensor(ptr, len) };
    })
    .ok(); // Ignore panic result - we don't want to propagate panics from free
}

/// Deletes a key from the database by appending a tombstone entry.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key to delete
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Key deleted successfully
/// * `0` (ERR_GENERIC) - Generic error during delete
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path or key is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
///
/// _Requirements: 10.1_
#[no_mangle]
pub extern "C" fn SYNA_delete(path: *const c_char, key: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to delete the key
        match with_db(path_str, |db| db.delete(key_str)) {
            Ok(_) => ERR_SUCCESS,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Checks if a key exists in the database and is not deleted.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key to check
///
/// # Returns
/// * `1` - Key exists and is not deleted
/// * `0` - Key does not exist or is deleted
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path or key is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
///
/// _Requirements: 10.2_
#[no_mangle]
pub extern "C" fn SYNA_exists(path: *const c_char, key: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to check if key exists
        match with_db(path_str, |db| Ok(db.exists(key_str))) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Compacts the database by rewriting only the latest non-deleted entries.
///
/// This operation reclaims disk space by removing deleted entries and old versions.
/// After compaction, `get_history()` will only return the latest value for each key.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Compaction completed successfully
/// * `0` (ERR_GENERIC) - Generic error during compaction
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 11.1_
#[no_mangle]
pub extern "C" fn SYNA_compact(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to compact the database
        match with_db(path_str, |db| db.compact()) {
            Ok(_) => ERR_SUCCESS,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND,
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Returns a list of all non-deleted keys in the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `out_len` - Pointer to write the number of keys to
///
/// # Returns
/// * Non-null pointer to array of null-terminated C strings on success
/// * Null pointer on error
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
/// * `out_len` must be a valid pointer to a usize
/// * The returned pointer MUST be freed using `SYNA_free_keys()` to avoid memory leaks
///
/// _Requirements: 10.5_
#[no_mangle]
pub extern "C" fn SYNA_keys(path: *const c_char, out_len: *mut usize) -> *mut *mut c_char {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }

        // Convert C string to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // Call with_db to get keys
        match with_db(path_str, |db| Ok(db.keys())) {
            Ok(keys) => {
                let len = keys.len();

                if len == 0 {
                    // No keys - return null with length 0
                    unsafe { *out_len = 0 };
                    return std::ptr::null_mut();
                }

                // Allocate array of C string pointers
                let mut c_strings: Vec<*mut c_char> = Vec::with_capacity(len);

                for key in keys {
                    // Convert each key to a C string
                    // Add null terminator
                    let mut bytes = key.into_bytes();
                    bytes.push(0); // null terminator

                    // Allocate memory for the C string
                    let c_str = bytes.into_boxed_slice();
                    let ptr = Box::into_raw(c_str) as *mut c_char;
                    c_strings.push(ptr);
                }

                // Write length to out_len
                unsafe { *out_len = len };

                // Convert Vec to boxed slice and leak for FFI
                let boxed = c_strings.into_boxed_slice();
                Box::into_raw(boxed) as *mut *mut c_char
            }
            Err(_) => {
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees memory allocated by `SYNA_keys()`.
///
/// # Arguments
/// * `keys` - Pointer returned by `SYNA_keys()`
/// * `len` - Length returned by `SYNA_keys()`
///
/// # Safety
/// * `keys` must have been returned by `SYNA_keys()`
/// * `len` must be the length returned alongside the pointer
/// * This function must only be called once per pointer
/// * Calling with a null pointer or zero length is safe (no-op)
///
/// _Requirements: 6.5_
#[no_mangle]
pub extern "C" fn SYNA_free_keys(keys: *mut *mut c_char, len: usize) {
    std::panic::catch_unwind(|| {
        if keys.is_null() || len == 0 {
            return;
        }

        unsafe {
            // Reconstruct the slice of pointers
            let key_slice = std::slice::from_raw_parts_mut(keys, len);

            // Free each individual string
            for key_ptr in key_slice.iter() {
                if !key_ptr.is_null() {
                    // Find the length of the C string (including null terminator)
                    let c_str = CStr::from_ptr(*key_ptr);
                    let str_len = c_str.to_bytes_with_nul().len();

                    // Reconstruct the box and drop it
                    let _ =
                        Box::from_raw(std::slice::from_raw_parts_mut(*key_ptr as *mut u8, str_len));
                }
            }

            // Free the array itself
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(keys, len));
        }
    })
    .ok(); // Ignore panic result - we don't want to propagate panics from free
}

/// Reads a text value from the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out_len` - Pointer to write the string length to (excluding null terminator)
///
/// # Returns
/// * Non-null pointer to null-terminated C string on success
/// * Null pointer on error
///
/// # Error Codes (check return value)
/// * Null with out_len = 0 - Key not found or type mismatch
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out_len` must be a valid pointer to a usize
/// * The returned pointer MUST be freed using `SYNA_free_text()` to avoid memory leaks
///
/// _Requirements: 6.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_get_text(
    path: *const c_char,
    key: *const c_char,
    out_len: *mut usize,
) -> *mut c_char {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // Call with_db to get the value
        match with_db(path_str, |db| db.get(key_str)) {
            Ok(Some(Atom::Text(s))) => {
                let len = s.len();
                unsafe { *out_len = len };

                // Convert to C string with null terminator
                let mut bytes = s.into_bytes();
                bytes.push(0); // null terminator

                let c_str = bytes.into_boxed_slice();
                Box::into_raw(c_str) as *mut c_char
            }
            Ok(Some(_)) => {
                // Value exists but is not Text
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
            Ok(None) => {
                // Key not found
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
            Err(_) => {
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees memory allocated by `SYNA_get_text()`.
///
/// # Arguments
/// * `ptr` - Pointer returned by `SYNA_get_text()`
/// * `len` - Length returned by `SYNA_get_text()` (excluding null terminator)
///
/// # Safety
/// * `ptr` must have been returned by `SYNA_get_text()`
/// * `len` must be the length returned alongside the pointer
/// * This function must only be called once per pointer
/// * Calling with a null pointer is safe (no-op)
///
/// _Requirements: 6.5_
#[no_mangle]
pub extern "C" fn SYNA_free_text(ptr: *mut c_char, len: usize) {
    std::panic::catch_unwind(|| {
        if ptr.is_null() {
            return;
        }

        unsafe {
            // Reconstruct the box (len + 1 for null terminator)
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr as *mut u8, len + 1));
        }
    })
    .ok();
}

/// Reads a byte array from the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out_len` - Pointer to write the array length to
///
/// # Returns
/// * Non-null pointer to byte array on success
/// * Null pointer on error
///
/// # Error Codes (check return value)
/// * Null with out_len = 0 - Key not found or type mismatch
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out_len` must be a valid pointer to a usize
/// * The returned pointer MUST be freed using `SYNA_free_bytes()` to avoid memory leaks
///
/// _Requirements: 6.2, 6.4_
#[no_mangle]
pub extern "C" fn SYNA_get_bytes(
    path: *const c_char,
    key: *const c_char,
    out_len: *mut usize,
) -> *mut u8 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // Call with_db to get the value
        match with_db(path_str, |db| db.get(key_str)) {
            Ok(Some(Atom::Bytes(bytes))) => {
                let len = bytes.len();

                if len == 0 {
                    unsafe { *out_len = 0 };
                    return std::ptr::null_mut();
                }

                unsafe { *out_len = len };

                // Convert to boxed slice and leak for FFI
                let boxed = bytes.into_boxed_slice();
                Box::into_raw(boxed) as *mut u8
            }
            Ok(Some(_)) => {
                // Value exists but is not Bytes
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
            Ok(None) => {
                // Key not found
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
            Err(_) => {
                unsafe { *out_len = 0 };
                std::ptr::null_mut()
            }
        }
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees memory allocated by `SYNA_get_bytes()`.
///
/// # Arguments
/// * `ptr` - Pointer returned by `SYNA_get_bytes()`
/// * `len` - Length returned by `SYNA_get_bytes()`
///
/// # Safety
/// * `ptr` must have been returned by `SYNA_get_bytes()`
/// * `len` must be the length returned alongside the pointer
/// * This function must only be called once per pointer
/// * Calling with a null pointer or zero length is safe (no-op)
///
/// _Requirements: 6.5_
#[no_mangle]
pub extern "C" fn SYNA_free_bytes(ptr: *mut u8, len: usize) {
    std::panic::catch_unwind(|| {
        if ptr.is_null() || len == 0 {
            return;
        }

        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        }
    })
    .ok();
}

/* ============================================================================
 * Vector Functions (AI/ML Embeddings)
 * ============================================================================ */

/// Stores a vector (embedding) in the database.
///
/// Vectors are stored as `Atom::Vector(Vec<f32>, u16)` where the second
/// element is the dimensionality. This is optimized for AI/ML embedding
/// storage and similarity search.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `data` - Pointer to f32 array containing the vector data
/// * `dimensions` - Number of dimensions (elements) in the vector
///
/// # Returns
/// * Positive value - The byte offset where the entry was written
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or data is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `data` must be a valid pointer to at least `dimensions` f32 values, or null
///
/// _Requirements: 1.1_
#[no_mangle]
pub extern "C" fn SYNA_put_vector(
    path: *const c_char,
    key: *const c_char,
    data: *const f32,
    dimensions: u16,
) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || (data.is_null() && dimensions > 0) {
            return ERR_INVALID_PATH as i64;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Create Vec<f32> from raw pointer and dimensions
        let vector = if dimensions == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(data, dimensions as usize) }.to_vec()
        };

        // Call with_db to append the vector
        match with_db(path_str, |db| {
            db.append(key_str, Atom::Vector(vector, dimensions))
        }) {
            Ok(offset) => offset as i64,
            Err(crate::error::SynaError::NotFound(_)) => crate::error::ERR_DB_NOT_FOUND as i64,
            Err(_) => ERR_GENERIC as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Retrieves a vector (embedding) from the database.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `out_data` - Pointer to store the allocated f32 array pointer
/// * `out_dimensions` - Pointer to store the number of dimensions
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector retrieved successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Database not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or output pointers are null or invalid UTF-8
/// * `-5` (ERR_KEY_NOT_FOUND) - Key not found in database
/// * `-6` (ERR_TYPE_MISMATCH) - Value exists but is not a Vector
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `out_data` must be a valid pointer to a `*mut f32`
/// * `out_dimensions` must be a valid pointer to a `u16`
/// * The returned data pointer MUST be freed using `SYNA_free_vector()` to avoid memory leaks
///
/// _Requirements: 1.1_
#[no_mangle]
pub extern "C" fn SYNA_get_vector(
    path: *const c_char,
    key: *const c_char,
    out_data: *mut *mut f32,
    out_dimensions: *mut u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || out_data.is_null() || out_dimensions.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Call with_db to get the value
        match with_db(path_str, |db| db.get(key_str)) {
            Ok(Some(Atom::Vector(vec_data, dims))) => {
                // Write dimensions to output
                unsafe { *out_dimensions = dims };

                if vec_data.is_empty() {
                    // Empty vector - return null pointer with 0 dimensions
                    unsafe { *out_data = std::ptr::null_mut() };
                    return ERR_SUCCESS;
                }

                // Convert to boxed slice and leak for FFI
                let boxed = vec_data.into_boxed_slice();
                unsafe { *out_data = Box::into_raw(boxed) as *mut f32 };

                ERR_SUCCESS
            }
            Ok(Some(_)) => {
                // Value exists but is not a Vector
                unsafe {
                    *out_data = std::ptr::null_mut();
                    *out_dimensions = 0;
                }
                ERR_TYPE_MISMATCH
            }
            Ok(None) => {
                // Key not found
                unsafe {
                    *out_data = std::ptr::null_mut();
                    *out_dimensions = 0;
                }
                ERR_KEY_NOT_FOUND
            }
            Err(crate::error::SynaError::NotFound(_)) => {
                unsafe {
                    *out_data = std::ptr::null_mut();
                    *out_dimensions = 0;
                }
                crate::error::ERR_DB_NOT_FOUND
            }
            Err(_) => {
                unsafe {
                    *out_data = std::ptr::null_mut();
                    *out_dimensions = 0;
                }
                ERR_GENERIC
            }
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Frees memory allocated by `SYNA_get_vector()`.
///
/// # Arguments
/// * `data` - Pointer returned by `SYNA_get_vector()` in `out_data`
/// * `dimensions` - Dimensions returned by `SYNA_get_vector()` in `out_dimensions`
///
/// # Safety
/// * `data` must have been returned by `SYNA_get_vector()`
/// * `dimensions` must be the dimensions returned alongside the pointer
/// * This function must only be called once per pointer
/// * Calling with a null pointer or zero dimensions is safe (no-op)
///
/// _Requirements: 1.1_
#[no_mangle]
pub extern "C" fn SYNA_free_vector(data: *mut f32, dimensions: u16) {
    std::panic::catch_unwind(|| {
        if data.is_null() || dimensions == 0 {
            return;
        }

        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(data, dimensions as usize));
        }
    })
    .ok(); // Ignore panic result - we don't want to propagate panics from free
}

// =============================================================================
// VectorStore FFI Functions
// =============================================================================

use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::distance::DistanceMetric;
use crate::vector::{VectorConfig, VectorStore};

/// Thread-safe global registry for managing open VectorStore instances.
/// Uses canonicalized paths as keys to ensure uniqueness.
static VECTOR_STORE_REGISTRY: Lazy<Mutex<HashMap<String, VectorStore>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Canonicalizes a path to an absolute path string for consistent registry keys.
///
/// If the path doesn't exist yet (for new databases), we use the parent directory's
/// canonical path combined with the filename.
fn canonicalize_vector_path(path: &str) -> Option<String> {
    let path_buf = PathBuf::from(path);

    // Try to canonicalize directly (works if file exists)
    if let Ok(canonical) = std::fs::canonicalize(&path_buf) {
        return Some(canonical.to_string_lossy().to_string());
    }

    // File doesn't exist yet - canonicalize parent and append filename
    let parent = path_buf.parent().unwrap_or(Path::new("."));
    let filename = path_buf.file_name()?;

    // Canonicalize parent directory
    let canonical_parent = if parent.as_os_str().is_empty() || parent == Path::new(".") {
        std::env::current_dir().ok()?
    } else {
        std::fs::canonicalize(parent).ok()?
    };

    let canonical = canonical_parent.join(filename);
    Some(canonical.to_string_lossy().to_string())
}

/// Creates a new vector store at the given path.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `dimensions` - Number of dimensions for vectors (64-8192)
/// * `metric` - Distance metric: 0=Cosine, 1=Euclidean, 2=DotProduct
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector store created successfully
/// * `0` (ERR_GENERIC) - Generic error during creation
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_new(path: *const c_char, dimensions: u16, metric: i32) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Convert metric integer to DistanceMetric enum
        let distance_metric = match metric {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine, // Default to Cosine for invalid values
        };

        // Create config
        let config = VectorConfig {
            dimensions,
            metric: distance_metric,
            ..Default::default()
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Create the vector store
        match VectorStore::new(path_str, config) {
            Ok(store) => {
                // Register in global registry with canonicalized path
                let mut registry = VECTOR_STORE_REGISTRY.lock();
                registry.insert(canonical_path, store);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Creates a new vector store with sync_on_write configuration.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `dimensions` - Number of dimensions for vectors (64-8192)
/// * `metric` - Distance metric: 0=Cosine, 1=Euclidean, 2=DotProduct
/// * `sync_on_write` - 1 for sync after each write (durable), 0 for no sync (fast)
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector store created successfully
/// * `0` (ERR_GENERIC) - Generic error during creation
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
#[no_mangle]
pub extern "C" fn SYNA_vector_store_new_with_config(
    path: *const c_char,
    dimensions: u16,
    metric: i32,
    sync_on_write: i32,
) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Convert metric integer to DistanceMetric enum
        let distance_metric = match metric {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };

        // Create config with sync_on_write setting
        let config = VectorConfig {
            dimensions,
            metric: distance_metric,
            sync_on_write: sync_on_write != 0,
            ..Default::default()
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Create the vector store
        match VectorStore::new(path_str, config) {
            Ok(store) => {
                // Register in global registry with canonicalized path
                let mut registry = VECTOR_STORE_REGISTRY.lock();
                registry.insert(canonical_path, store);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts a vector into the store.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `key` - Null-terminated C string containing the key
/// * `data` - Pointer to the f32 vector data
/// * `dimensions` - Number of dimensions in the vector
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector inserted successfully
/// * `0` (ERR_GENERIC) - Generic error during insertion
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, key, or data is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` and `key` must be valid null-terminated C strings or null
/// * `data` must be a valid pointer to at least `dimensions` f32 values
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_insert(
    path: *const c_char,
    key: *const c_char,
    data: *const f32,
    dimensions: u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || key.is_null() || (data.is_null() && dimensions > 0) {
            return ERR_INVALID_PATH;
        }

        // Convert C strings to Rust strings
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Create vector from raw pointer
        let vector = if dimensions == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(data, dimensions as usize) }.to_vec()
        };

        // Get the vector store from registry using canonicalized path
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.insert(key_str, &vector) {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts multiple vectors in a single batch operation.
///
/// This is significantly faster than calling `SYNA_vector_store_insert()` in a loop:
/// - Single FFI boundary crossing for all vectors
/// - Deferred index building until after all vectors are inserted
/// - Reduced lock contention
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `keys` - Array of null-terminated C strings (keys for each vector)
/// * `data` - Pointer to contiguous f32 array containing all vectors (row-major)
/// * `dimensions` - Number of dimensions per vector
/// * `count` - Number of vectors to insert
///
/// # Returns
/// * Non-negative value - Number of vectors successfully inserted
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, keys, or data is null or invalid
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string
/// * `keys` must be a valid pointer to `count` null-terminated C strings
/// * `data` must be a valid pointer to `count * dimensions` f32 values
///
/// # Example (C)
/// ```c
/// const char* keys[] = {"doc1", "doc2", "doc3"};
/// float data[3 * 768] = { ... };  // 3 vectors of 768 dimensions
/// int32_t inserted = SYNA_vector_store_insert_batch("vectors.db", keys, data, 768, 3);
/// ```
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_insert_batch(
    path: *const c_char,
    keys: *const *const c_char,
    data: *const f32,
    dimensions: u16,
    count: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || keys.is_null() || (data.is_null() && count > 0 && dimensions > 0) {
            return ERR_INVALID_PATH;
        }

        // Convert path
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Convert keys array
        let keys_slice = unsafe { std::slice::from_raw_parts(keys, count) };
        let mut key_strings: Vec<&str> = Vec::with_capacity(count);
        for key_ptr in keys_slice {
            if key_ptr.is_null() {
                return ERR_INVALID_PATH;
            }
            match unsafe { CStr::from_ptr(*key_ptr) }.to_str() {
                Ok(s) => key_strings.push(s),
                Err(_) => return ERR_INVALID_PATH,
            }
        }

        // Create vector slices from contiguous data
        let total_floats = count * dimensions as usize;
        let data_slice = if total_floats == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data, total_floats) }
        };

        // Split into individual vectors
        let vectors: Vec<&[f32]> = data_slice.chunks(dimensions as usize).collect();

        // Get the vector store from registry
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.insert_batch(&key_strings, &vectors) {
                Ok(n) => n as i32,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts multiple vectors without updating the index (maximum write speed).
///
/// This is the fastest way to bulk-load vectors. Vectors are written to storage
/// but NOT added to the HNSW index. Call `SYNA_vector_store_build_index()` after
/// all inserts to build the index.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `keys` - Array of null-terminated C strings (keys for each vector)
/// * `data` - Pointer to contiguous f32 array containing all vectors (row-major)
/// * `dimensions` - Number of dimensions per vector
/// * `count` - Number of vectors to insert
///
/// # Returns
/// * Non-negative value - Number of vectors successfully inserted
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, keys, or data is null or invalid
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Performance
/// This function achieves 100K+ inserts/sec by skipping index updates.
/// After bulk loading, call `SYNA_vector_store_build_index()` to enable fast search.
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_insert_batch_fast(
    path: *const c_char,
    keys: *const *const c_char,
    data: *const f32,
    dimensions: u16,
    count: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || keys.is_null() || (data.is_null() && count > 0 && dimensions > 0) {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        let keys_slice = unsafe { std::slice::from_raw_parts(keys, count) };
        let mut key_strings: Vec<&str> = Vec::with_capacity(count);
        for key_ptr in keys_slice {
            if key_ptr.is_null() {
                return ERR_INVALID_PATH;
            }
            match unsafe { CStr::from_ptr(*key_ptr) }.to_str() {
                Ok(s) => key_strings.push(s),
                Err(_) => return ERR_INVALID_PATH,
            }
        }

        let total_floats = count * dimensions as usize;
        let data_slice = if total_floats == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data, total_floats) }
        };

        let vectors: Vec<&[f32]> = data_slice.chunks(dimensions as usize).collect();

        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.insert_batch_fast(&key_strings, &vectors, false) {
                Ok(n) => n as i32,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Searches for k nearest neighbors in the vector store.
///
/// Returns a JSON array of results with the following structure:
/// ```json
/// [
///   {"key": "doc1", "score": 0.123, "vector": [0.1, 0.2, ...]},
///   {"key": "doc2", "score": 0.456, "vector": [0.3, 0.4, ...]}
/// ]
/// ```
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
/// * `query` - Pointer to the f32 query vector
/// * `dimensions` - Number of dimensions in the query vector
/// * `k` - Number of nearest neighbors to return
/// * `out_json` - Pointer to write the JSON result string to
///
/// # Returns
/// * Non-negative value - Number of results found
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path, query, or out_json is null or invalid
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
/// * `query` must be a valid pointer to at least `dimensions` f32 values
/// * `out_json` must be a valid pointer to a `*mut c_char`
/// * The returned JSON string MUST be freed using `SYNA_free_json()` to avoid memory leaks
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_search(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    std::panic::catch_unwind(|| {
        // Validate pointers
        if path.is_null() || query.is_null() || out_json.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Create query vector from raw pointer
        let query_vec: Vec<f32> =
            unsafe { std::slice::from_raw_parts(query, dimensions as usize) }.to_vec();

        // Get the vector store from registry using canonicalized path
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.search(&query_vec, k) {
                Ok(results) => {
                    // Convert results to JSON
                    let json_results: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "key": r.key,
                                "score": r.score,
                                "vector": r.vector
                            })
                        })
                        .collect();

                    let json_str =
                        serde_json::to_string(&json_results).unwrap_or_else(|_| "[]".to_string());
                    let result_count = results.len() as i32;

                    // Convert to C string
                    match CString::new(json_str) {
                        Ok(c_string) => {
                            unsafe { *out_json = c_string.into_raw() };
                            result_count
                        }
                        Err(_) => ERR_GENERIC,
                    }
                }
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Builds the HNSW index for a vector store.
///
/// This function manually triggers HNSW index construction for faster search.
/// The index is built automatically when vector count exceeds the threshold,
/// but this function allows explicit control.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the vector store
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Index built successfully
/// * `0` (ERR_GENERIC) - Generic error during build
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_build_index(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert path to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Get the vector store from registry
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.build_index() {
                Ok(()) => crate::error::ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Returns whether a vector store has an HNSW index built.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the vector store
///
/// # Returns
/// * `1` - Index is built
/// * `0` - Index is not built
/// * `-1` (ERR_DB_NOT_FOUND) - Vector store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_has_index(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert path to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent registry keys
        let canonical_path = match canonicalize_vector_path(path_str) {
            Some(p) => p,
            None => return ERR_INVALID_PATH,
        };

        // Get the vector store from registry
        let registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get(&canonical_path) {
            Some(store) => {
                if store.has_index() {
                    1
                } else {
                    0
                }
            }
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Closes a vector store and saves any pending changes.
///
/// This function removes the vector store from the global registry and
/// triggers the Drop implementation, which saves any dirty index to disk.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Store closed successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent lookup
        let canonical_path = match std::fs::canonicalize(path_str) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => path_str.to_string(),
        };

        // Remove from registry (this triggers Drop which saves the index)
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.remove(&canonical_path) {
            Some(_store) => {
                // Store is dropped here, triggering index save
                ERR_SUCCESS
            }
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Flushes any pending changes to disk without closing the store.
///
/// This saves the HNSW index if it has unsaved changes.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Flush successful
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found in registry
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `0` (ERR_GENERIC) - Flush failed
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_vector_store_flush(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        // Check for null pointer
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        // Convert C string to Rust string
        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Canonicalize path for consistent lookup
        let canonical_path = match std::fs::canonicalize(path_str) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => path_str.to_string(),
        };

        // Get store from registry
        let mut registry = VECTOR_STORE_REGISTRY.lock();
        match registry.get_mut(&canonical_path) {
            Some(store) => match store.flush() {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Frees a JSON string allocated by `SYNA_vector_store_search()`.
///
/// # Arguments
/// * `json` - Pointer returned by `SYNA_vector_store_search()` in `out_json`
///
/// # Safety
/// * `json` must have been returned by `SYNA_vector_store_search()`
/// * This function must only be called once per pointer
/// * Calling with a null pointer is safe (no-op)
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_free_json(json: *mut c_char) {
    std::panic::catch_unwind(|| {
        if json.is_null() {
            return;
        }

        unsafe {
            // Reconstruct the CString and drop it
            let _ = CString::from_raw(json);
        }
    })
    .ok(); // Ignore panic result - we don't want to propagate panics from free
}

// =============================================================================
// Model Registry FFI Functions
// =============================================================================

use crate::model_registry::{ModelRegistry, ModelStage};

/// Thread-safe global registry for managing open ModelRegistry instances.
static MODEL_REGISTRY: Lazy<Mutex<HashMap<String, ModelRegistry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Opens or creates a model registry at the given path.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Registry opened successfully
/// * `0` (ERR_GENERIC) - Generic error during open
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// # Safety
/// * `path` must be a valid null-terminated C string or null
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_model_registry_open(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        match ModelRegistry::new(path_str) {
            Ok(registry) => {
                let mut reg = MODEL_REGISTRY.lock();
                reg.insert(path_str.to_string(), registry);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Saves a model to the registry with automatic versioning.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the registry
/// * `name` - Null-terminated C string containing the model name
/// * `data` - Pointer to the model data bytes
/// * `data_len` - Length of the model data
/// * `metadata_json` - Null-terminated JSON string with metadata (can be null for empty)
/// * `out_version` - Pointer to write the assigned version number
/// * `out_checksum` - Pointer to write the checksum string (caller must free with SYNA_free_text)
/// * `out_checksum_len` - Pointer to write the checksum string length
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Model saved successfully
/// * `0` (ERR_GENERIC) - Generic error during save
/// * `-1` (ERR_DB_NOT_FOUND) - Registry not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_model_save(
    path: *const c_char,
    name: *const c_char,
    data: *const u8,
    data_len: usize,
    metadata_json: *const c_char,
    out_version: *mut u32,
    out_checksum: *mut *mut c_char,
    out_checksum_len: *mut usize,
) -> i64 {
    std::panic::catch_unwind(|| {
        // Validate required pointers
        if path.is_null() || name.is_null() || (data.is_null() && data_len > 0) {
            return ERR_INVALID_PATH as i64;
        }
        if out_version.is_null() || out_checksum.is_null() || out_checksum_len.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        // Parse metadata JSON if provided
        let metadata: std::collections::HashMap<String, String> = if metadata_json.is_null() {
            std::collections::HashMap::new()
        } else {
            match unsafe { CStr::from_ptr(metadata_json) }.to_str() {
                Ok(json_str) => serde_json::from_str(json_str).unwrap_or_default(),
                Err(_) => std::collections::HashMap::new(),
            }
        };

        // Create data slice
        let model_data = if data_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data, data_len) }
        };

        // Get registry and save model
        let mut reg = MODEL_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(registry) => {
                match registry.save_model(name_str, model_data, metadata) {
                    Ok(version) => {
                        unsafe { *out_version = version.version };

                        // Return checksum as C string
                        let checksum_len = version.checksum.len();
                        unsafe { *out_checksum_len = checksum_len };

                        let mut bytes = version.checksum.into_bytes();
                        bytes.push(0);
                        let c_str = bytes.into_boxed_slice();
                        unsafe { *out_checksum = Box::into_raw(c_str) as *mut c_char };

                        ERR_SUCCESS as i64
                    }
                    Err(_) => ERR_GENERIC as i64,
                }
            }
            None => crate::error::ERR_DB_NOT_FOUND as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

/// Loads a model from the registry with checksum verification.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the registry
/// * `name` - Null-terminated C string containing the model name
/// * `version` - Version number to load (0 for latest)
/// * `out_data` - Pointer to write the model data pointer
/// * `out_data_len` - Pointer to write the model data length
/// * `out_meta_json` - Pointer to write the metadata JSON string
/// * `out_meta_len` - Pointer to write the metadata JSON length
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Model loaded successfully
/// * `0` (ERR_GENERIC) - Generic error during load
/// * `-1` (ERR_DB_NOT_FOUND) - Registry not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Model not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_model_load(
    path: *const c_char,
    name: *const c_char,
    version: u32,
    out_data: *mut *mut u8,
    out_data_len: *mut usize,
    out_meta_json: *mut *mut c_char,
    out_meta_len: *mut usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || name.is_null() || out_data.is_null() || out_data_len.is_null() {
            return ERR_INVALID_PATH;
        }
        if out_meta_json.is_null() || out_meta_len.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let version_opt = if version == 0 { None } else { Some(version) };

        let mut reg = MODEL_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(registry) => {
                match registry.load_model(name_str, version_opt) {
                    Ok((data, version_info)) => {
                        // Return model data
                        let data_len = data.len();
                        unsafe { *out_data_len = data_len };

                        if data_len > 0 {
                            let boxed = data.into_boxed_slice();
                            unsafe { *out_data = Box::into_raw(boxed) as *mut u8 };
                        } else {
                            unsafe { *out_data = std::ptr::null_mut() };
                        }

                        // Return metadata as JSON
                        let meta_json = serde_json::to_string(&version_info)
                            .unwrap_or_else(|_| "{}".to_string());
                        let meta_len = meta_json.len();
                        unsafe { *out_meta_len = meta_len };

                        let mut bytes = meta_json.into_bytes();
                        bytes.push(0);
                        let c_str = bytes.into_boxed_slice();
                        unsafe { *out_meta_json = Box::into_raw(c_str) as *mut c_char };

                        ERR_SUCCESS
                    }
                    Err(crate::error::SynaError::ModelNotFound(_)) => ERR_KEY_NOT_FOUND,
                    Err(crate::error::SynaError::ChecksumMismatch { .. }) => ERR_GENERIC,
                    Err(_) => ERR_GENERIC,
                }
            }
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Lists all versions of a model.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the registry
/// * `name` - Null-terminated C string containing the model name
/// * `out_json` - Pointer to write the JSON array of versions
/// * `out_len` - Pointer to write the JSON string length
///
/// # Returns
/// * Non-negative value - Number of versions found
/// * `-1` (ERR_DB_NOT_FOUND) - Registry not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_model_list(
    path: *const c_char,
    name: *const c_char,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || name.is_null() || out_json.is_null() || out_len.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = MODEL_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(registry) => match registry.list_versions(name_str) {
                Ok(versions) => {
                    let count = versions.len() as i32;
                    let json =
                        serde_json::to_string(&versions).unwrap_or_else(|_| "[]".to_string());

                    unsafe { *out_len = json.len() };
                    match CString::new(json) {
                        Ok(c_string) => {
                            unsafe { *out_json = c_string.into_raw() };
                            count
                        }
                        Err(_) => ERR_GENERIC,
                    }
                }
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Sets the deployment stage for a model version.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the registry
/// * `name` - Null-terminated C string containing the model name
/// * `version` - Version number to update
/// * `stage` - Stage: 0=Development, 1=Staging, 2=Production, 3=Archived
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Stage updated successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Registry not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Model/version not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_model_set_stage(
    path: *const c_char,
    name: *const c_char,
    version: u32,
    stage: i32,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || name.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let model_stage = match stage {
            0 => ModelStage::Development,
            1 => ModelStage::Staging,
            2 => ModelStage::Production,
            3 => ModelStage::Archived,
            _ => ModelStage::Development,
        };

        let mut reg = MODEL_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(registry) => match registry.set_stage(name_str, version, model_stage) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::ModelNotFound(_)) => ERR_KEY_NOT_FOUND,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

// =============================================================================
// Experiment Tracking FFI Functions
// =============================================================================

use crate::experiment::{ExperimentTracker, RunStatus};

/// Thread-safe global registry for managing open ExperimentTracker instances.
static EXPERIMENT_REGISTRY: Lazy<Mutex<HashMap<String, ExperimentTracker>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Opens or creates an experiment tracker at the given path.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the database file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Tracker opened successfully
/// * `0` (ERR_GENERIC) - Generic error during open
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_tracker_open(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        match ExperimentTracker::new(path_str) {
            Ok(tracker) => {
                let mut reg = EXPERIMENT_REGISTRY.lock();
                reg.insert(path_str.to_string(), tracker);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Starts a new experiment run.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the tracker
/// * `experiment` - Null-terminated C string containing the experiment name
/// * `tags_json` - Null-terminated JSON array of tags (can be null for empty)
/// * `out_run_id` - Pointer to write the run ID string
/// * `out_run_id_len` - Pointer to write the run ID length
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Run started successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Tracker not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_start_run(
    path: *const c_char,
    experiment: *const c_char,
    tags_json: *const c_char,
    out_run_id: *mut *mut c_char,
    out_run_id_len: *mut usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null()
            || experiment.is_null()
            || out_run_id.is_null()
            || out_run_id_len.is_null()
        {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let exp_str = match unsafe { CStr::from_ptr(experiment) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let tags: Vec<String> = if tags_json.is_null() {
            Vec::new()
        } else {
            match unsafe { CStr::from_ptr(tags_json) }.to_str() {
                Ok(json_str) => serde_json::from_str(json_str).unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        };

        let mut reg = EXPERIMENT_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(tracker) => match tracker.start_run(exp_str, tags) {
                Ok(run_id) => {
                    let run_id_len = run_id.len();
                    unsafe { *out_run_id_len = run_id_len };

                    let mut bytes = run_id.into_bytes();
                    bytes.push(0);
                    let c_str = bytes.into_boxed_slice();
                    unsafe { *out_run_id = Box::into_raw(c_str) as *mut c_char };

                    ERR_SUCCESS
                }
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Logs a parameter for a run.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the tracker
/// * `run_id` - Null-terminated C string containing the run ID
/// * `key` - Null-terminated C string containing the parameter name
/// * `value` - Null-terminated C string containing the parameter value
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Parameter logged successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Tracker not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Run not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_log_param(
    path: *const c_char,
    run_id: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || run_id.is_null() || key.is_null() || value.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let run_id_str = match unsafe { CStr::from_ptr(run_id) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let value_str = match unsafe { CStr::from_ptr(value) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = EXPERIMENT_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(tracker) => match tracker.log_param(run_id_str, key_str, value_str) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::RunNotFound(_)) => ERR_KEY_NOT_FOUND,
                Err(crate::error::SynaError::RunAlreadyEnded(_)) => ERR_GENERIC,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Logs a metric value for a run.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the tracker
/// * `run_id` - Null-terminated C string containing the run ID
/// * `key` - Null-terminated C string containing the metric name
/// * `value` - The metric value (f64)
/// * `step` - Step number (use -1 for auto-generated timestamp)
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Metric logged successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Tracker not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Run not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_log_metric(
    path: *const c_char,
    run_id: *const c_char,
    key: *const c_char,
    value: f64,
    step: i64,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || run_id.is_null() || key.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let run_id_str = match unsafe { CStr::from_ptr(run_id) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let step_opt = if step < 0 { None } else { Some(step as u64) };

        let mut reg = EXPERIMENT_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(tracker) => match tracker.log_metric(run_id_str, key_str, value, step_opt) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::RunNotFound(_)) => ERR_KEY_NOT_FOUND,
                Err(crate::error::SynaError::RunAlreadyEnded(_)) => ERR_GENERIC,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Logs an artifact for a run.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the tracker
/// * `run_id` - Null-terminated C string containing the run ID
/// * `name` - Null-terminated C string containing the artifact name
/// * `data` - Pointer to the artifact data
/// * `data_len` - Length of the artifact data
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Artifact logged successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Tracker not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Run not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_log_artifact(
    path: *const c_char,
    run_id: *const c_char,
    name: *const c_char,
    data: *const u8,
    data_len: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || run_id.is_null() || name.is_null() {
            return ERR_INVALID_PATH;
        }
        if data.is_null() && data_len > 0 {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let run_id_str = match unsafe { CStr::from_ptr(run_id) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let artifact_data = if data_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data, data_len) }
        };

        let mut reg = EXPERIMENT_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(tracker) => match tracker.log_artifact(run_id_str, name_str, artifact_data) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::RunNotFound(_)) => ERR_KEY_NOT_FOUND,
                Err(crate::error::SynaError::RunAlreadyEnded(_)) => ERR_GENERIC,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Ends a run with the given status.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the tracker
/// * `run_id` - Null-terminated C string containing the run ID
/// * `status` - Status: 0=Running, 1=Completed, 2=Failed, 3=Killed
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Run ended successfully
/// * `0` (ERR_GENERIC) - Generic error (e.g., run already ended)
/// * `-1` (ERR_DB_NOT_FOUND) - Tracker not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-5` (ERR_KEY_NOT_FOUND) - Run not found
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
///
/// _Requirements: 8.1_
#[no_mangle]
pub extern "C" fn SYNA_exp_end_run(path: *const c_char, run_id: *const c_char, status: i32) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || run_id.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let run_id_str = match unsafe { CStr::from_ptr(run_id) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let run_status = match status {
            0 => RunStatus::Running,
            1 => RunStatus::Completed,
            2 => RunStatus::Failed,
            3 => RunStatus::Killed,
            _ => RunStatus::Completed,
        };

        let mut reg = EXPERIMENT_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(tracker) => match tracker.end_run(run_id_str, run_status) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::RunNotFound(_)) => ERR_KEY_NOT_FOUND,
                Err(crate::error::SynaError::RunAlreadyEnded(_)) => ERR_GENERIC,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

// =============================================================================
// MmapVectorStore FFI Functions (Ultra-High-Throughput)
// =============================================================================

use crate::mmap_vector::{MmapVectorConfig, MmapVectorStore};

/// Thread-safe global registry for managing open MmapVectorStore instances.
static MMAP_VECTOR_REGISTRY: Lazy<Mutex<HashMap<String, MmapVectorStore>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Creates or opens a memory-mapped vector store at the given path.
///
/// This is an alternative to `SYNA_vector_store_new()` that uses memory-mapped I/O
/// for ultra-high-throughput writes (500K-1M vectors/sec).
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the mmap file
/// * `dimensions` - Number of dimensions (64-8192)
/// * `metric` - Distance metric: 0=Cosine, 1=Euclidean, 2=DotProduct
/// * `initial_capacity` - Pre-allocated capacity in number of vectors
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Store opened successfully
/// * `0` (ERR_GENERIC) - Generic error during open
/// * `-2` (ERR_INVALID_PATH) - Path is null or invalid UTF-8
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_new(
    path: *const c_char,
    dimensions: u16,
    metric: i32,
    initial_capacity: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let distance_metric = match metric {
            0 => crate::distance::DistanceMetric::Cosine,
            1 => crate::distance::DistanceMetric::Euclidean,
            2 => crate::distance::DistanceMetric::DotProduct,
            _ => crate::distance::DistanceMetric::Cosine,
        };

        let config = MmapVectorConfig {
            dimensions,
            metric: distance_metric,
            initial_capacity,
            ..Default::default()
        };

        match MmapVectorStore::new(path_str, config) {
            Ok(store) => {
                let mut reg = MMAP_VECTOR_REGISTRY.lock();
                reg.insert(path_str.to_string(), store);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts a vector into the mmap vector store.
///
/// This is an ultra-fast operation (no syscalls, just memcpy).
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
/// * `key` - Null-terminated C string containing the vector key
/// * `vector` - Pointer to the vector data (f32 array)
/// * `dimensions` - Number of dimensions in the vector
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector inserted successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-6` (ERR_TYPE_MISMATCH) - Dimension mismatch
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_insert(
    path: *const c_char,
    key: *const c_char,
    vector: *const f32,
    dimensions: u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || key.is_null() || vector.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let vec_slice = unsafe { std::slice::from_raw_parts(vector, dimensions as usize) };

        let mut reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(store) => match store.insert(key_str, vec_slice) {
                Ok(_) => ERR_SUCCESS,
                Err(crate::error::SynaError::DimensionMismatch { .. }) => ERR_TYPE_MISMATCH,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts multiple vectors in a batch (maximum throughput).
///
/// This achieves 500K-1M vectors/sec by writing directly to memory.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
/// * `keys` - Array of null-terminated C strings (vector keys)
/// * `vectors` - Contiguous array of vector data (count * dimensions floats)
/// * `dimensions` - Number of dimensions per vector
/// * `count` - Number of vectors to insert
///
/// # Returns
/// * Non-negative value - Number of vectors inserted
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_insert_batch(
    path: *const c_char,
    keys: *const *const c_char,
    vectors: *const f32,
    dimensions: u16,
    count: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || keys.is_null() || vectors.is_null() || count == 0 {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        // Convert keys
        let key_ptrs = unsafe { std::slice::from_raw_parts(keys, count) };
        let mut key_strings: Vec<String> = Vec::with_capacity(count);
        for &key_ptr in key_ptrs {
            if key_ptr.is_null() {
                return ERR_INVALID_PATH;
            }
            match unsafe { CStr::from_ptr(key_ptr) }.to_str() {
                Ok(s) => key_strings.push(s.to_string()),
                Err(_) => return ERR_INVALID_PATH,
            }
        }
        let key_refs: Vec<&str> = key_strings.iter().map(|s| s.as_str()).collect();

        // Convert vectors
        let dims = dimensions as usize;
        let all_vectors = unsafe { std::slice::from_raw_parts(vectors, count * dims) };
        let vec_refs: Vec<&[f32]> = (0..count)
            .map(|i| &all_vectors[i * dims..(i + 1) * dims])
            .collect();

        let mut reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(store) => match store.insert_batch(&key_refs, &vec_refs) {
                Ok(inserted) => inserted as i32,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Searches for the k nearest neighbors in the mmap vector store.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
/// * `query` - Pointer to the query vector (f32 array)
/// * `dimensions` - Number of dimensions in the query
/// * `k` - Number of results to return
/// * `out_json` - Pointer to write the JSON results string
/// * `out_len` - Pointer to write the JSON string length
///
/// # Returns
/// * Non-negative value - Number of results found
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid arguments
/// * `-6` (ERR_TYPE_MISMATCH) - Dimension mismatch
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_search(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || query.is_null() || out_json.is_null() || out_len.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let query_slice = unsafe { std::slice::from_raw_parts(query, dimensions as usize) };

        let reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get(path_str) {
            Some(store) => match store.search(query_slice, k) {
                Ok(results) => {
                    let count = results.len() as i32;

                    // Convert results to JSON
                    let json_results: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "key": r.key,
                                "score": r.score,
                            })
                        })
                        .collect();

                    let json =
                        serde_json::to_string(&json_results).unwrap_or_else(|_| "[]".to_string());

                    unsafe { *out_len = json.len() };
                    match CString::new(json) {
                        Ok(c_string) => {
                            unsafe { *out_json = c_string.into_raw() };
                            count
                        }
                        Err(_) => ERR_GENERIC,
                    }
                }
                Err(crate::error::SynaError::DimensionMismatch { .. }) => ERR_TYPE_MISMATCH,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Builds the HNSW index for the mmap vector store.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Index built successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_build_index(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(store) => match store.build_index() {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Flushes the mmap vector store to disk.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Flushed successfully
/// * `0` (ERR_GENERIC) - Generic error
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_flush(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(store) => match store.flush() {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Closes the mmap vector store and removes it from the registry.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Closed successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.remove(path_str) {
            Some(_store) => {
                // Store is dropped here, which triggers checkpoint
                ERR_SUCCESS
            }
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Returns the number of vectors in the mmap vector store.
///
/// # Arguments
/// * `path` - Null-terminated C string containing the path to the store
///
/// # Returns
/// * Non-negative value - Number of vectors
/// * `-1` (ERR_DB_NOT_FOUND) - Store not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_mmap_vector_store_len(path: *const c_char) -> i64 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let reg = MMAP_VECTOR_REGISTRY.lock();
        match reg.get(path_str) {
            Some(store) => store.len() as i64,
            None => crate::error::ERR_DB_NOT_FOUND as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

// =============================================================================
// Gravity Well Index (GWI) FFI Functions
// =============================================================================

use crate::gwi::{GravityWellIndex, GwiConfig};

/// Thread-safe global registry for managing open GravityWellIndex instances.
static GWI_REGISTRY: Lazy<Mutex<HashMap<String, GravityWellIndex>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Create a new Gravity Well Index
///
/// # Arguments
/// * `path` - Path to the index file
/// * `dimensions` - Vector dimensions (64-8192)
/// * `branching_factor` - Branching factor at each level (default: 16)
/// * `num_levels` - Number of hierarchy levels (default: 3)
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Index created successfully
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_new(
    path: *const c_char,
    dimensions: u16,
    branching_factor: u16,
    num_levels: u8,
    initial_capacity: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let config = GwiConfig {
            dimensions,
            branching_factor: if branching_factor == 0 {
                16
            } else {
                branching_factor
            },
            num_levels: if num_levels == 0 { 3 } else { num_levels },
            initial_capacity: if initial_capacity == 0 {
                10_000
            } else {
                initial_capacity
            },
            ..Default::default()
        };

        match GravityWellIndex::new(path_str, config) {
            Ok(index) => {
                let mut reg = GWI_REGISTRY.lock();
                reg.insert(path_str.to_string(), index);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Open an existing Gravity Well Index
///
/// # Arguments
/// * `path` - Path to the existing index file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Index opened successfully
/// * `-2` (ERR_INVALID_PATH) - Invalid path or file doesn't exist
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_open(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        match GravityWellIndex::open(path_str) {
            Ok(index) => {
                let mut reg = GWI_REGISTRY.lock();
                reg.insert(path_str.to_string(), index);
                ERR_SUCCESS
            }
            Err(_) => ERR_INVALID_PATH,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Initialize GWI attractors from sample vectors
///
/// # Arguments
/// * `path` - Path to the index file
/// * `vectors` - Pointer to contiguous f32 vector data
/// * `num_vectors` - Number of sample vectors
/// * `dimensions` - Dimensions per vector
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Attractors initialized
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_initialize(
    path: *const c_char,
    vectors: *const f32,
    num_vectors: usize,
    dimensions: u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || vectors.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let total_floats = num_vectors * dimensions as usize;
        let data_slice = unsafe { std::slice::from_raw_parts(vectors, total_floats) };

        let sample_vectors: Vec<&[f32]> = data_slice.chunks(dimensions as usize).collect();

        let mut reg = GWI_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(index) => match index.initialize_attractors(&sample_vectors) {
                Ok(()) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Insert a vector into the GWI
///
/// # Arguments
/// * `path` - Path to the index file
/// * `key` - Null-terminated key string
/// * `vector` - Pointer to f32 vector data
/// * `dimensions` - Vector dimensions
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector inserted
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path/key
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_insert(
    path: *const c_char,
    key: *const c_char,
    vector: *const f32,
    dimensions: u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || key.is_null() || vector.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let vector_slice = unsafe { std::slice::from_raw_parts(vector, dimensions as usize) };

        let mut reg = GWI_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(index) => match index.insert(key_str, vector_slice) {
                Ok(()) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Batch insert vectors into the GWI
///
/// # Arguments
/// * `path` - Path to the index file
/// * `keys` - Array of null-terminated key strings
/// * `vectors` - Pointer to contiguous f32 vector data
/// * `dimensions` - Dimensions per vector
/// * `count` - Number of vectors to insert
///
/// # Returns
/// * Non-negative - Number of vectors inserted
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_insert_batch(
    path: *const c_char,
    keys: *const *const c_char,
    vectors: *const f32,
    dimensions: u16,
    count: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || keys.is_null() || (vectors.is_null() && count > 0) {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let keys_slice = unsafe { std::slice::from_raw_parts(keys, count) };
        let mut key_strings: Vec<&str> = Vec::with_capacity(count);
        for key_ptr in keys_slice {
            if key_ptr.is_null() {
                return ERR_INVALID_PATH;
            }
            match unsafe { CStr::from_ptr(*key_ptr) }.to_str() {
                Ok(s) => key_strings.push(s),
                Err(_) => return ERR_INVALID_PATH,
            }
        }

        let total_floats = count * dimensions as usize;
        let data_slice = unsafe { std::slice::from_raw_parts(vectors, total_floats) };
        let vector_refs: Vec<&[f32]> = data_slice.chunks(dimensions as usize).collect();

        let mut reg = GWI_REGISTRY.lock();
        match reg.get_mut(path_str) {
            Some(index) => match index.insert_batch(&key_strings, &vector_refs) {
                Ok(n) => n as i32,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Search for k nearest neighbors in the GWI
///
/// # Arguments
/// * `path` - Path to the index file
/// * `query` - Pointer to f32 query vector
/// * `dimensions` - Query vector dimensions
/// * `k` - Number of neighbors to return
/// * `out_json` - Pointer to write JSON result string
///
/// # Returns
/// * Non-negative - Number of results found
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_search(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    // Default nprobe
    SYNA_gwi_search_nprobe(path, query, dimensions, k, 3, out_json)
}

/// Search for k nearest neighbors with custom nprobe
///
/// Higher nprobe = better recall but slower search.
/// - nprobe=3: Fast, ~5-15% recall
/// - nprobe=10: Balanced, ~30-50% recall
/// - nprobe=30: High quality, ~70-90% recall
/// - nprobe=100: Near-exact, ~95%+ recall
///
/// # Arguments
/// * `path` - Path to the index file
/// * `query` - Query vector
/// * `dimensions` - Number of dimensions
/// * `k` - Number of results to return
/// * `nprobe` - Number of clusters to probe
/// * `out_json` - Output JSON string with results
///
/// # Returns
/// * Number of results on success
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_search_nprobe(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    nprobe: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || query.is_null() || out_json.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let query_slice = unsafe { std::slice::from_raw_parts(query, dimensions as usize) };

        let reg = GWI_REGISTRY.lock();
        match reg.get(path_str) {
            Some(index) => match index.search_with_nprobe(query_slice, k, nprobe) {
                Ok(results) => {
                    let json_results: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "key": r.key,
                                "score": r.score,
                            })
                        })
                        .collect();

                    let json_str = serde_json::to_string(&json_results).unwrap_or_default();
                    let c_str = CString::new(json_str).unwrap_or_default();
                    unsafe {
                        *out_json = c_str.into_raw();
                    }
                    results.len() as i32
                }
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Flush GWI changes to disk
///
/// # Arguments
/// * `path` - Path to the index file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Flushed successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_flush(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let reg = GWI_REGISTRY.lock();
        match reg.get(path_str) {
            Some(index) => match index.flush() {
                Ok(()) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            },
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Close a GWI and remove from registry
///
/// # Arguments
/// * `path` - Path to the index file
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Closed successfully
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut reg = GWI_REGISTRY.lock();
        match reg.remove(path_str) {
            Some(mut index) => {
                let _ = index.close();
                ERR_SUCCESS
            }
            None => crate::error::ERR_DB_NOT_FOUND,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Get number of vectors in the GWI
///
/// # Arguments
/// * `path` - Path to the index file
///
/// # Returns
/// * Non-negative - Number of vectors
/// * `-1` (ERR_DB_NOT_FOUND) - Index not found
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic occurred
#[no_mangle]
pub extern "C" fn SYNA_gwi_len(path: *const c_char) -> i64 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH as i64;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH as i64,
        };

        let reg = GWI_REGISTRY.lock();
        match reg.get(path_str) {
            Some(index) => index.len() as i64,
            None => crate::error::ERR_DB_NOT_FOUND as i64,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC as i64)
}

// =============================================================================
// Cascade Index FFI Functions
// =============================================================================

use crate::cascade::{CascadeConfig, CascadeIndex};

/// Global registry for Cascade Index instances
static CASCADE_REGISTRY: Lazy<Mutex<HashMap<String, CascadeIndex>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Creates a new Cascade Index.
///
/// # Arguments
/// * `path` - Path to the index file
/// * `dimensions` - Vector dimensions (64-8192)
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Index created successfully
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic
#[no_mangle]
pub extern "C" fn SYNA_cascade_new(path: *const c_char, dimensions: u16) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let c_str = unsafe { CStr::from_ptr(path) };
        let path_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let config = CascadeConfig {
            dimensions,
            ..Default::default()
        };

        match CascadeIndex::new(path_str, config) {
            Ok(index) => {
                let mut registry = CASCADE_REGISTRY.lock();
                registry.insert(path_str.to_string(), index);
                ERR_SUCCESS
            }
            Err(_) => ERR_GENERIC,
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts a vector into the Cascade Index.
///
/// # Arguments
/// * `path` - Path to the index
/// * `key` - Key for the vector
/// * `vector` - Pointer to vector data
/// * `dimensions` - Vector dimensions
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vector inserted
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic
#[no_mangle]
pub extern "C" fn SYNA_cascade_insert(
    path: *const c_char,
    key: *const c_char,
    vector: *const f32,
    dimensions: u16,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || key.is_null() || vector.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let vec_slice = unsafe { std::slice::from_raw_parts(vector, dimensions as usize) };

        let mut registry = CASCADE_REGISTRY.lock();
        if let Some(index) = registry.get_mut(path_str) {
            match index.insert(key_str, vec_slice) {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            }
        } else {
            ERR_GENERIC
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Inserts multiple vectors into the Cascade Index.
///
/// # Arguments
/// * `path` - Path to the index
/// * `keys` - Array of key pointers
/// * `vectors` - Pointer to flattened vector data
/// * `dimensions` - Vector dimensions
/// * `count` - Number of vectors
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Vectors inserted
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic
#[no_mangle]
pub extern "C" fn SYNA_cascade_insert_batch(
    path: *const c_char,
    keys: *const *const c_char,
    vectors: *const f32,
    dimensions: u16,
    count: usize,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || keys.is_null() || vectors.is_null() || count == 0 {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let dims = dimensions as usize;
        let mut registry = CASCADE_REGISTRY.lock();

        if let Some(index) = registry.get_mut(path_str) {
            for i in 0..count {
                let key_ptr = unsafe { *keys.add(i) };
                if key_ptr.is_null() {
                    continue;
                }

                let key_str = match unsafe { CStr::from_ptr(key_ptr) }.to_str() {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let vec_start = i * dims;
                let vec_slice = unsafe { std::slice::from_raw_parts(vectors.add(vec_start), dims) };

                if index.insert(key_str, vec_slice).is_err() {
                    return ERR_GENERIC;
                }
            }
            ERR_SUCCESS
        } else {
            ERR_GENERIC
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Searches the Cascade Index for nearest neighbors.
///
/// # Arguments
/// * `path` - Path to the index
/// * `query` - Query vector
/// * `dimensions` - Vector dimensions
/// * `k` - Number of results
/// * `out_json` - Output JSON string pointer
///
/// # Returns
/// * `1` (ERR_SUCCESS) - Search completed
/// * `-2` (ERR_INVALID_PATH) - Invalid path
/// * `-100` (ERR_INTERNAL_PANIC) - Internal panic
#[no_mangle]
pub extern "C" fn SYNA_cascade_search(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    // Use good defaults matching CascadeConfig::default()
    SYNA_cascade_search_params(path, query, dimensions, k, 16, 80, out_json)
}

/// Searches with custom parameters.
#[no_mangle]
pub extern "C" fn SYNA_cascade_search_params(
    path: *const c_char,
    query: *const f32,
    dimensions: u16,
    k: usize,
    num_probes: usize,
    ef_search: usize,
    out_json: *mut *mut c_char,
) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() || query.is_null() || out_json.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let query_slice = unsafe { std::slice::from_raw_parts(query, dimensions as usize) };

        let registry = CASCADE_REGISTRY.lock();
        if let Some(index) = registry.get(path_str) {
            match index.search_with_params(query_slice, k, num_probes, ef_search) {
                Ok(results) => {
                    let json_results: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "key": r.key,
                                "score": r.score
                            })
                        })
                        .collect();

                    let json_str = serde_json::to_string(&json_results).unwrap_or_default();
                    let c_string = std::ffi::CString::new(json_str).unwrap_or_default();
                    unsafe {
                        *out_json = c_string.into_raw();
                    }
                    ERR_SUCCESS
                }
                Err(_) => ERR_GENERIC,
            }
        } else {
            ERR_GENERIC
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Flushes the Cascade Index to disk.
#[no_mangle]
pub extern "C" fn SYNA_cascade_flush(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let registry = CASCADE_REGISTRY.lock();
        if let Some(index) = registry.get(path_str) {
            match index.flush() {
                Ok(_) => ERR_SUCCESS,
                Err(_) => ERR_GENERIC,
            }
        } else {
            ERR_GENERIC
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Closes the Cascade Index.
#[no_mangle]
pub extern "C" fn SYNA_cascade_close(path: *const c_char) -> i32 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return ERR_INVALID_PATH;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return ERR_INVALID_PATH,
        };

        let mut registry = CASCADE_REGISTRY.lock();
        if registry.remove(path_str).is_some() {
            ERR_SUCCESS
        } else {
            ERR_GENERIC
        }
    })
    .unwrap_or(ERR_INTERNAL_PANIC)
}

/// Returns the number of vectors in the Cascade Index.
#[no_mangle]
pub extern "C" fn SYNA_cascade_len(path: *const c_char) -> i64 {
    std::panic::catch_unwind(|| {
        if path.is_null() {
            return -1;
        }

        let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        let registry = CASCADE_REGISTRY.lock();
        if let Some(index) = registry.get(path_str) {
            index.len() as i64
        } else {
            -1
        }
    })
    .unwrap_or(-1)
}

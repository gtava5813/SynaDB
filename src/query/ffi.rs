//! FFI Interface for Syna Query — execute EQL/EMQ queries from any language.
//!
//! Returns JSON-encoded results via C-ABI functions.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::query::executor::QueryExecutor;
use crate::query::optimizer::optimize;
use crate::query::parser::parse_eql;
use crate::query::planner::QueryPlan;
use std::ffi::{c_char, c_int, CStr};

const QUERY_ERR_INTERNAL_FFI: c_int = -100;
const QUERY_ERR_PARSE_FFI: c_int = -10;
const QUERY_SUCCESS_FFI: c_int = 1;

/// Execute an EQL query against an open database and return JSON results.
///
/// # Arguments
/// * `path` — Database path (must be open via `SYNA_open`)
/// * `query` — EQL query string
/// * `out_json` — Pointer to write the JSON result string
/// * `out_len` — Pointer to write the JSON string length
///
/// # Returns
/// `QUERY_SUCCESS_FFI` (1) on success, negative error code on failure.
/// The caller must free `out_json` with `SYNA_query_free_result`.
#[no_mangle]
pub extern "C" fn SYNA_query_eql(
    path: *const c_char,
    query: *const c_char,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return QUERY_ERR_PARSE_FFI,
        };
        let query_str = match unsafe { cstr_to_str(query) } {
            Some(q) => q,
            None => return QUERY_ERR_PARSE_FFI,
        };
        if out_json.is_null() || out_len.is_null() {
            return QUERY_ERR_PARSE_FFI;
        }

        // Parse
        let ast = match parse_eql(query_str) {
            Ok(a) => a,
            Err(_) => return QUERY_ERR_PARSE_FFI,
        };

        // Plan + optimize + execute using the global DB registry
        let result = crate::engine::with_db(path, |db| {
            let total_keys = db.keys().len() as u64;
            let mut plan = match QueryPlan::from_ast(&ast, total_keys) {
                Ok(p) => p,
                Err(_) => return Err(crate::error::SynaError::IoError("plan failed".into())),
            };
            optimize(&mut plan);
            let mut executor = QueryExecutor::new(db);
            match executor.execute(plan) {
                Ok(r) => Ok(r),
                Err(_) => Err(crate::error::SynaError::IoError("execute failed".into())),
            }
        });

        match result {
            Ok(query_result) => {
                let json = serde_json::to_string(&query_result).unwrap_or_default();
                let len = json.len();
                let cstring = match std::ffi::CString::new(json) {
                    Ok(c) => c,
                    Err(_) => return QUERY_ERR_INTERNAL_FFI,
                };
                unsafe {
                    *out_json = cstring.into_raw();
                    *out_len = len;
                }
                QUERY_SUCCESS_FFI
            }
            Err(_) => QUERY_ERR_INTERNAL_FFI,
        }
    })
    .unwrap_or(QUERY_ERR_INTERNAL_FFI)
}

/// Execute an EMQ (MongoDB-like) query and return JSON results.
#[no_mangle]
pub extern "C" fn SYNA_query_emq(
    path: *const c_char,
    document: *const c_char,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return QUERY_ERR_PARSE_FFI,
        };
        let doc_str = match unsafe { cstr_to_str(document) } {
            Some(d) => d,
            None => return QUERY_ERR_PARSE_FFI,
        };
        if out_json.is_null() || out_len.is_null() {
            return QUERY_ERR_PARSE_FFI;
        }

        // Parse EMQ
        let ast = match crate::query::emq_parser::parse_emq(doc_str) {
            Ok(a) => a,
            Err(_) => return QUERY_ERR_PARSE_FFI,
        };

        // Plan + optimize + execute
        let result = crate::engine::with_db(path, |db| {
            let total_keys = db.keys().len() as u64;
            let mut plan = match QueryPlan::from_ast(&ast, total_keys) {
                Ok(p) => p,
                Err(_) => return Err(crate::error::SynaError::IoError("plan failed".into())),
            };
            optimize(&mut plan);
            let mut executor = QueryExecutor::new(db);
            match executor.execute(plan) {
                Ok(r) => Ok(r),
                Err(_) => Err(crate::error::SynaError::IoError("execute failed".into())),
            }
        });

        match result {
            Ok(query_result) => {
                let json = serde_json::to_string(&query_result).unwrap_or_default();
                let len = json.len();
                let cstring = match std::ffi::CString::new(json) {
                    Ok(c) => c,
                    Err(_) => return QUERY_ERR_INTERNAL_FFI,
                };
                unsafe {
                    *out_json = cstring.into_raw();
                    *out_len = len;
                }
                QUERY_SUCCESS_FFI
            }
            Err(_) => QUERY_ERR_INTERNAL_FFI,
        }
    })
    .unwrap_or(QUERY_ERR_INTERNAL_FFI)
}

/// Free a JSON result string returned by `SYNA_query_eql` or `SYNA_query_emq`.
#[no_mangle]
pub extern "C" fn SYNA_query_free_result(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

/// Helper to convert C string to Rust str.
unsafe fn cstr_to_str(ptr: *const c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

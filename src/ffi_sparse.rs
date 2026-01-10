//! FFI Layer for Sparse Vector Store
//!
//! C-ABI interface for the Sparse Vector Store, enabling use from Python,
//! C++, and other FFI-capable languages.
//!
//! # Error Codes
//!
//! | Code | Constant | Meaning |
//! |------|----------|---------|
//! | 1 | SVS_SUCCESS | Operation succeeded |
//! | 0 | SVS_ERR_GENERIC | Generic error |
//! | -1 | SVS_ERR_NULL_PTR | Null pointer argument |
//! | -2 | SVS_ERR_INVALID_UTF8 | Invalid UTF-8 string |
//! | -3 | SVS_ERR_NOT_FOUND | Store or key not found |
//! | -4 | SVS_ERR_ALREADY_EXISTS | Store already exists |
//! | -100 | SVS_ERR_INTERNAL | Internal panic |

// FFI functions intentionally take raw pointers without being marked unsafe
// because they handle null checks and use catch_unwind for safety
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::sparse_vector::SparseVector;
use crate::sparse_vector_store::SparseVectorStore;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::{c_char, c_float, c_int, c_uint, CStr};
use std::ptr;

// Error codes
pub const SVS_SUCCESS: c_int = 1;
pub const SVS_ERR_GENERIC: c_int = 0;
pub const SVS_ERR_NULL_PTR: c_int = -1;
pub const SVS_ERR_INVALID_UTF8: c_int = -2;
pub const SVS_ERR_NOT_FOUND: c_int = -3;
pub const SVS_ERR_ALREADY_EXISTS: c_int = -4;
pub const SVS_ERR_INTERNAL: c_int = -100;

/// Global registry of sparse vector stores
static SVS_REGISTRY: Lazy<Mutex<HashMap<String, SparseVectorStore>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Helper to convert C string to Rust string
unsafe fn cstr_to_str(ptr: *const c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Create a new sparse vector store.
///
/// # Arguments
/// * `path` - Unique identifier for the store (used as registry key)
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if path is null
/// * `SVS_ERR_INVALID_UTF8` if path is not valid UTF-8
/// * `SVS_ERR_ALREADY_EXISTS` if store already exists
/// * `SVS_ERR_INTERNAL` on internal error
#[no_mangle]
pub extern "C" fn svs_new(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let mut registry = SVS_REGISTRY.lock();
        if registry.contains_key(path) {
            return SVS_ERR_ALREADY_EXISTS;
        }

        registry.insert(path.to_string(), SparseVectorStore::new());
        SVS_SUCCESS
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Close and remove a sparse vector store from the registry.
///
/// # Arguments
/// * `path` - Store identifier
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if path is null
/// * `SVS_ERR_NOT_FOUND` if store not found
#[no_mangle]
pub extern "C" fn svs_close(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let mut registry = SVS_REGISTRY.lock();
        if registry.remove(path).is_some() {
            SVS_SUCCESS
        } else {
            SVS_ERR_NOT_FOUND
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Index a sparse vector with a key.
///
/// # Arguments
/// * `path` - Store identifier
/// * `key` - Document key
/// * `term_ids` - Array of term IDs
/// * `weights` - Array of weights (same length as term_ids)
/// * `count` - Number of terms
///
/// # Returns
/// * Document ID (>= 0) on success
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_NOT_FOUND` if store not found
#[no_mangle]
pub extern "C" fn svs_index(
    path: *const c_char,
    key: *const c_char,
    term_ids: *const c_uint,
    weights: *const c_float,
    count: c_uint,
) -> i64 {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR as i64,
        };

        let key = match unsafe { cstr_to_str(key) } {
            Some(k) => k,
            None => return SVS_ERR_NULL_PTR as i64,
        };

        if term_ids.is_null() || weights.is_null() {
            return SVS_ERR_NULL_PTR as i64;
        }

        let mut registry = SVS_REGISTRY.lock();
        let store = match registry.get_mut(path) {
            Some(s) => s,
            None => return SVS_ERR_NOT_FOUND as i64,
        };

        // Build sparse vector from arrays
        let mut vec = SparseVector::new();
        for i in 0..count as usize {
            let term_id = unsafe { *term_ids.add(i) };
            let weight = unsafe { *weights.add(i) };
            vec.add(term_id, weight);
        }

        let doc_id = store.index_with_key(key, vec);
        doc_id as i64
    })
    .unwrap_or(SVS_ERR_INTERNAL as i64)
}

/// Search for similar documents.
///
/// # Arguments
/// * `path` - Store identifier
/// * `term_ids` - Query term IDs
/// * `weights` - Query weights
/// * `count` - Number of query terms
/// * `k` - Number of results to return
/// * `out_keys` - Output array for result keys (caller allocates, k elements)
/// * `out_scores` - Output array for result scores (caller allocates, k elements)
/// * `out_count` - Output: actual number of results
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_NOT_FOUND` if store not found
#[no_mangle]
pub extern "C" fn svs_search(
    path: *const c_char,
    term_ids: *const c_uint,
    weights: *const c_float,
    count: c_uint,
    k: c_uint,
    out_keys: *mut *mut c_char,
    out_scores: *mut c_float,
    out_count: *mut c_uint,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        if term_ids.is_null()
            || weights.is_null()
            || out_keys.is_null()
            || out_scores.is_null()
            || out_count.is_null()
        {
            return SVS_ERR_NULL_PTR;
        }

        let registry = SVS_REGISTRY.lock();
        let store = match registry.get(path) {
            Some(s) => s,
            None => return SVS_ERR_NOT_FOUND,
        };

        // Build query vector
        let mut query = SparseVector::new();
        for i in 0..count as usize {
            let term_id = unsafe { *term_ids.add(i) };
            let weight = unsafe { *weights.add(i) };
            query.add(term_id, weight);
        }

        let results = store.search(&query, k as usize);

        // Write results to output arrays
        unsafe {
            *out_count = results.len() as c_uint;
            for (i, result) in results.iter().enumerate() {
                // Allocate and copy key string
                let key_cstr = std::ffi::CString::new(result.key.clone()).unwrap();
                let key_ptr = libc::malloc(key_cstr.as_bytes_with_nul().len()) as *mut c_char;
                if !key_ptr.is_null() {
                    ptr::copy_nonoverlapping(
                        key_cstr.as_ptr(),
                        key_ptr,
                        key_cstr.as_bytes_with_nul().len(),
                    );
                }
                *out_keys.add(i) = key_ptr;
                *out_scores.add(i) = result.score;
            }
        }

        SVS_SUCCESS
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Free a key string returned by svs_search.
///
/// # Arguments
/// * `key` - Key string to free
#[no_mangle]
pub extern "C" fn svs_free_key(key: *mut c_char) {
    if !key.is_null() {
        unsafe {
            libc::free(key as *mut libc::c_void);
        }
    }
}

/// Get the number of documents in the store.
///
/// # Arguments
/// * `path` - Store identifier
///
/// # Returns
/// * Number of documents (>= 0) on success
/// * `SVS_ERR_NULL_PTR` if path is null
/// * `SVS_ERR_NOT_FOUND` if store not found
#[no_mangle]
pub extern "C" fn svs_len(path: *const c_char) -> i64 {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR as i64,
        };

        let registry = SVS_REGISTRY.lock();
        match registry.get(path) {
            Some(store) => store.len() as i64,
            None => SVS_ERR_NOT_FOUND as i64,
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL as i64)
}

/// Delete a document by key.
///
/// # Arguments
/// * `path` - Store identifier
/// * `key` - Document key to delete
///
/// # Returns
/// * `SVS_SUCCESS` if deleted
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_NOT_FOUND` if store or key not found
#[no_mangle]
pub extern "C" fn svs_delete(path: *const c_char, key: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let key = match unsafe { cstr_to_str(key) } {
            Some(k) => k,
            None => return SVS_ERR_NULL_PTR,
        };

        let mut registry = SVS_REGISTRY.lock();
        let store = match registry.get_mut(path) {
            Some(s) => s,
            None => return SVS_ERR_NOT_FOUND,
        };

        if store.delete(key) {
            SVS_SUCCESS
        } else {
            SVS_ERR_NOT_FOUND
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Get index statistics.
///
/// # Arguments
/// * `path` - Store identifier
/// * `out_num_docs` - Output: number of documents
/// * `out_num_terms` - Output: number of unique terms
/// * `out_num_postings` - Output: total postings
/// * `out_avg_doc_len` - Output: average document length
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_NOT_FOUND` if store not found
#[no_mangle]
pub extern "C" fn svs_stats(
    path: *const c_char,
    out_num_docs: *mut c_uint,
    out_num_terms: *mut c_uint,
    out_num_postings: *mut c_uint,
    out_avg_doc_len: *mut c_float,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        if out_num_docs.is_null()
            || out_num_terms.is_null()
            || out_num_postings.is_null()
            || out_avg_doc_len.is_null()
        {
            return SVS_ERR_NULL_PTR;
        }

        let registry = SVS_REGISTRY.lock();
        let store = match registry.get(path) {
            Some(s) => s,
            None => return SVS_ERR_NOT_FOUND,
        };

        let stats = store.stats();
        unsafe {
            *out_num_docs = stats.num_documents as c_uint;
            *out_num_terms = stats.num_terms as c_uint;
            *out_num_postings = stats.num_postings as c_uint;
            *out_avg_doc_len = stats.avg_doc_length;
        }

        SVS_SUCCESS
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Check if a store exists in the registry.
///
/// # Arguments
/// * `path` - Store identifier
///
/// # Returns
/// * 1 if exists, 0 if not
/// * `SVS_ERR_NULL_PTR` if path is null
#[no_mangle]
pub extern "C" fn svs_exists(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let registry = SVS_REGISTRY.lock();
        if registry.contains_key(path) {
            1
        } else {
            0
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Save the index to a file.
///
/// # Arguments
/// * `path` - Store identifier (registry key)
/// * `file_path` - File path to save to
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_NOT_FOUND` if store not found
/// * `SVS_ERR_INTERNAL` on I/O error
#[no_mangle]
pub extern "C" fn svs_save(path: *const c_char, file_path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let file_path = match unsafe { cstr_to_str(file_path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let registry = SVS_REGISTRY.lock();
        let store = match registry.get(path) {
            Some(s) => s,
            None => return SVS_ERR_NOT_FOUND,
        };

        match store.save(file_path) {
            Ok(()) => SVS_SUCCESS,
            Err(_) => SVS_ERR_INTERNAL,
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

/// Open an existing index from a file.
///
/// Loads the index from disk and registers it in the global registry.
///
/// # Arguments
/// * `path` - Store identifier (registry key)
/// * `file_path` - File path to load from
///
/// # Returns
/// * `SVS_SUCCESS` on success
/// * `SVS_ERR_NULL_PTR` if any pointer is null
/// * `SVS_ERR_ALREADY_EXISTS` if store already exists in registry
/// * `SVS_ERR_INTERNAL` on I/O or parse error
#[no_mangle]
pub extern "C" fn svs_open(path: *const c_char, file_path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let file_path = match unsafe { cstr_to_str(file_path) } {
            Some(p) => p,
            None => return SVS_ERR_NULL_PTR,
        };

        let mut registry = SVS_REGISTRY.lock();
        if registry.contains_key(path) {
            return SVS_ERR_ALREADY_EXISTS;
        }

        match SparseVectorStore::load(file_path) {
            Ok(store) => {
                registry.insert(path.to_string(), store);
                SVS_SUCCESS
            }
            Err(_) => SVS_ERR_INTERNAL,
        }
    })
    .unwrap_or(SVS_ERR_INTERNAL)
}

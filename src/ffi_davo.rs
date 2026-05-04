//! FFI Layer for DAVO (Decay-Aware Value Optimization).
//!
//! C-ABI interface for the DAVO module, enabling use from Python, C++,
//! and other FFI-capable languages.
//!
//! # Error Codes
//!
//! Reuses the standard SynaDB error codes:
//!
//! | Code | Constant | Meaning |
//! |------|----------|---------|
//! | 1 | `DAVO_SUCCESS` | Operation succeeded |
//! | -1 | `DAVO_ERR_NULL_PTR` | Null pointer argument |
//! | -2 | `DAVO_ERR_INVALID_UTF8` | Invalid UTF-8 string |
//! | -3 | `DAVO_ERR_NOT_FOUND` | Registry entry or key not found |
//! | -4 | `DAVO_ERR_ALREADY_EXISTS` | Registry entry already exists |
//! | -100 | `DAVO_ERR_INTERNAL` | Internal panic |

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::davo::freshness_v2::FreshnessIndexV2;
use crate::davo::predictor::DecayPredictor;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::{c_char, c_double, c_int, CStr};
use std::ptr;

// ── Error codes ──────────────────────────────────────────────────────

/// Operation succeeded.
pub const DAVO_SUCCESS: c_int = 1;
/// Null pointer argument.
pub const DAVO_ERR_NULL_PTR: c_int = -1;
/// Invalid UTF-8 string.
pub const DAVO_ERR_INVALID_UTF8: c_int = -2;
/// Registry entry or key not found.
pub const DAVO_ERR_NOT_FOUND: c_int = -3;
/// Registry entry already exists.
pub const DAVO_ERR_ALREADY_EXISTS: c_int = -4;
/// Internal panic caught by `catch_unwind`.
pub const DAVO_ERR_INTERNAL: c_int = -100;

// ── Global registries ────────────────────────────────────────────────

/// Global registry of FreshnessIndexV2 instances.
static FRESHNESS_REGISTRY: Lazy<Mutex<HashMap<String, FreshnessIndexV2>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global registry of DecayPredictor instances.
static PREDICTOR_REGISTRY: Lazy<Mutex<HashMap<String, DecayPredictor>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// ── Helpers ──────────────────────────────────────────────────────────

/// Convert a C string pointer to a Rust `&str`.
///
/// Returns `None` if the pointer is null or the bytes are not valid UTF-8.
unsafe fn cstr_to_str(ptr: *const c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

// ═══════════════════════════════════════════════════════════════════════
//  FreshnessIndexV2 FFI
// ═══════════════════════════════════════════════════════════════════════

/// Create a new [`FreshnessIndexV2`] and register it under `path`.
///
/// # Arguments
/// * `path` — Unique identifier (registry key).
/// * `threshold` — Staleness threshold in (0, 1). Pass 0.0 for the default (0.5).
///
/// # Returns
/// `DAVO_SUCCESS` on success, or an error code.
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_new(path: *const c_char, threshold: c_double) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = FRESHNESS_REGISTRY.lock();
        if registry.contains_key(path) {
            return DAVO_ERR_ALREADY_EXISTS;
        }

        let index = if threshold > 0.0 && threshold < 1.0 {
            FreshnessIndexV2::with_threshold(threshold as f32)
        } else {
            FreshnessIndexV2::new()
        };

        registry.insert(path.to_string(), index);
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Insert or update a key in the freshness index.
///
/// # Arguments
/// * `path` — Registry key for the index.
/// * `key` — Data key to track.
/// * `decay_rate` — Decay rate λ (per second). 0.0 = static (never stale).
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_insert(
    path: *const c_char,
    key: *const c_char,
    decay_rate: c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        let key = match unsafe { cstr_to_str(key) } {
            Some(k) => k,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = FRESHNESS_REGISTRY.lock();
        let index = match registry.get_mut(path) {
            Some(i) => i,
            None => return DAVO_ERR_NOT_FOUND,
        };

        index.insert(key, decay_rate as f32);
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Get the freshness of a key (0.0 – 1.0).
///
/// # Arguments
/// * `path` — Registry key for the index.
/// * `key` — Data key to query.
/// * `out_freshness` — Pointer to write the freshness value.
///
/// # Returns
/// `DAVO_SUCCESS` if the key was found, `DAVO_ERR_NOT_FOUND` otherwise.
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_get_freshness(
    path: *const c_char,
    key: *const c_char,
    out_freshness: *mut c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        let key = match unsafe { cstr_to_str(key) } {
            Some(k) => k,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_freshness.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let registry = FRESHNESS_REGISTRY.lock();
        let index = match registry.get(path) {
            Some(i) => i,
            None => return DAVO_ERR_NOT_FOUND,
        };

        match index.get_freshness(key) {
            Some(f) => {
                unsafe { *out_freshness = f as c_double };
                DAVO_SUCCESS
            }
            None => DAVO_ERR_NOT_FOUND,
        }
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Query all stale keys and write them to caller-allocated arrays.
///
/// # Arguments
/// * `path` — Registry key for the index.
/// * `out_keys` — Pointer to write an allocated `*mut *mut c_char` array.
/// * `out_count` — Pointer to write the number of stale keys.
///
/// The caller must free the returned array with [`SYNA_davo_free_keys`].
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_query_stale(
    path: *const c_char,
    out_keys: *mut *mut *mut c_char,
    out_count: *mut usize,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_keys.is_null() || out_count.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let registry = FRESHNESS_REGISTRY.lock();
        let index = match registry.get(path) {
            Some(i) => i,
            None => return DAVO_ERR_NOT_FOUND,
        };

        let stale = index.query_stale();
        let count = stale.len();

        // Allocate array of C strings
        let array = if count > 0 {
            let layout = std::alloc::Layout::array::<*mut c_char>(count)
                .unwrap_or(std::alloc::Layout::new::<*mut c_char>());
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut c_char;
            if ptr.is_null() {
                return DAVO_ERR_INTERNAL;
            }
            for (i, key) in stale.iter().enumerate() {
                let cstring = match std::ffi::CString::new(key.as_str()) {
                    Ok(c) => c,
                    Err(_) => {
                        // Clean up already-allocated strings
                        for j in 0..i {
                            unsafe {
                                let _ = std::ffi::CString::from_raw(*ptr.add(j));
                            }
                        }
                        unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
                        return DAVO_ERR_INTERNAL;
                    }
                };
                unsafe { *ptr.add(i) = cstring.into_raw() };
            }
            ptr
        } else {
            ptr::null_mut()
        };

        unsafe {
            *out_keys = array;
            *out_count = count;
        }
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Evict all stale entries from the index.
///
/// # Arguments
/// * `path` — Registry key for the index.
/// * `out_count` — Pointer to write the number of evicted entries.
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_evict_stale(
    path: *const c_char,
    out_count: *mut usize,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_count.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let mut registry = FRESHNESS_REGISTRY.lock();
        let index = match registry.get_mut(path) {
            Some(i) => i,
            None => return DAVO_ERR_NOT_FOUND,
        };

        let evicted = index.evict_stale();
        unsafe { *out_count = evicted.len() };
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Get the number of tracked keys in the freshness index.
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_len(path: *const c_char) -> i64 {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return -1_i64,
        };

        let registry = FRESHNESS_REGISTRY.lock();
        match registry.get(path) {
            Some(i) => i.len() as i64,
            None => -1,
        }
    })
    .unwrap_or(-1)
}

/// Close and remove a freshness index from the registry.
#[no_mangle]
pub extern "C" fn SYNA_davo_freshness_index_close(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = FRESHNESS_REGISTRY.lock();
        if registry.remove(path).is_some() {
            DAVO_SUCCESS
        } else {
            DAVO_ERR_NOT_FOUND
        }
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Free an array of C strings returned by [`SYNA_davo_freshness_index_query_stale`].
///
/// # Safety
/// `keys` must be a pointer returned by `SYNA_davo_freshness_index_query_stale`,
/// and `count` must match the `out_count` value from that call.
#[no_mangle]
pub extern "C" fn SYNA_davo_free_keys(keys: *mut *mut c_char, count: usize) {
    if keys.is_null() || count == 0 {
        return;
    }
    for i in 0..count {
        let key_ptr = unsafe { *keys.add(i) };
        if !key_ptr.is_null() {
            unsafe {
                let _ = std::ffi::CString::from_raw(key_ptr);
            }
        }
    }
    if let Ok(layout) = std::alloc::Layout::array::<*mut c_char>(count) {
        unsafe { std::alloc::dealloc(keys as *mut u8, layout) };
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  DecayPredictor FFI
// ═══════════════════════════════════════════════════════════════════════

/// Create a new [`DecayPredictor`] with default prior and register it under `path`.
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_new(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = PREDICTOR_REGISTRY.lock();
        if registry.contains_key(path) {
            return DAVO_ERR_ALREADY_EXISTS;
        }

        registry.insert(path.to_string(), DecayPredictor::new());
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Feed an observed decay rate to the predictor.
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_observe(
    path: *const c_char,
    actual_decay: c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = PREDICTOR_REGISTRY.lock();
        let predictor = match registry.get_mut(path) {
            Some(p) => p,
            None => return DAVO_ERR_NOT_FOUND,
        };

        predictor.observe(actual_decay as f32);
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Get the point-estimate prediction (posterior mean α/β).
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_predict(
    path: *const c_char,
    out_prediction: *mut c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_prediction.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let registry = PREDICTOR_REGISTRY.lock();
        let predictor = match registry.get(path) {
            Some(p) => p,
            None => return DAVO_ERR_NOT_FOUND,
        };

        unsafe { *out_prediction = predictor.predict() as c_double };
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Sample a decay rate from the posterior (Thompson Sampling).
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_sample(
    path: *const c_char,
    out_sample: *mut c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_sample.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let mut registry = PREDICTOR_REGISTRY.lock();
        let predictor = match registry.get_mut(path) {
            Some(p) => p,
            None => return DAVO_ERR_NOT_FOUND,
        };

        unsafe { *out_sample = predictor.sample() as c_double };
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Get the posterior uncertainty (variance α/β²).
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_uncertainty(
    path: *const c_char,
    out_uncertainty: *mut c_double,
) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };
        if out_uncertainty.is_null() {
            return DAVO_ERR_NULL_PTR;
        }

        let registry = PREDICTOR_REGISTRY.lock();
        let predictor = match registry.get(path) {
            Some(p) => p,
            None => return DAVO_ERR_NOT_FOUND,
        };

        unsafe { *out_uncertainty = predictor.uncertainty() as c_double };
        DAVO_SUCCESS
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

/// Close and remove a predictor from the registry.
#[no_mangle]
pub extern "C" fn SYNA_davo_predictor_close(path: *const c_char) -> c_int {
    std::panic::catch_unwind(|| {
        let path = match unsafe { cstr_to_str(path) } {
            Some(p) => p,
            None => return DAVO_ERR_NULL_PTR,
        };

        let mut registry = PREDICTOR_REGISTRY.lock();
        if registry.remove(path).is_some() {
            DAVO_SUCCESS
        } else {
            DAVO_ERR_NOT_FOUND
        }
    })
    .unwrap_or(DAVO_ERR_INTERNAL)
}

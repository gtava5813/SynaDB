// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Compression utilities for Syna database.
//!
//! This module provides two compression strategies:
//!
//! 1. **LZ4 compression** - General-purpose compression for large values
//! 2. **Delta encoding** - Specialized compression for float sequences
//!
//! Both can be enabled via [`DbConfig`](crate::DbConfig).

use crate::error::{Result, SynaError};

/// Compresses data using LZ4.
///
/// Uses `lz4_flex` with size prepended for self-describing decompression.
///
/// # Arguments
///
/// * `data` - The raw bytes to compress
///
/// # Returns
///
/// Compressed bytes with the original size prepended (4 bytes).
pub fn compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompresses LZ4 compressed data.
///
/// # Arguments
///
/// * `data` - Compressed bytes (with size prepended)
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - Decompressed data
/// * `Err(SynaError::DecompressionFailed)` - If decompression fails
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    lz4_flex::decompress_size_prepended(data).map_err(|_| SynaError::DecompressionFailed)
}

/// Returns `true` if the data is large enough to benefit from compression.
///
/// Small values (≤64 bytes) may actually grow when compressed due to
/// LZ4 overhead, so we skip compression for them.
pub fn should_compress(data: &[u8]) -> bool {
    data.len() > 64
}

/// Encodes a float value as a delta from the previous value.
///
/// Delta encoding stores the difference between consecutive values,
/// which is typically smaller for time-series data with gradual changes.
///
/// # Arguments
///
/// * `current` - The current value to encode
/// * `previous` - The previous value in the sequence
///
/// # Returns
///
/// The delta: `current - previous`
///
/// # Example
///
/// ```rust
/// use synadb::compression::{encode_delta, decode_delta};
///
/// let prev = 100.0;
/// let curr = 100.5;
/// let delta = encode_delta(curr, prev);
/// assert_eq!(delta, 0.5);
/// assert_eq!(decode_delta(delta, prev), curr);
/// ```
pub fn encode_delta(current: f64, previous: f64) -> f64 {
    current - previous
}

/// Decodes a delta-encoded float value.
///
/// # Arguments
///
/// * `delta` - The delta value (difference from previous)
/// * `previous` - The previous value in the sequence
///
/// # Returns
///
/// The reconstructed value: `previous + delta`
pub fn decode_delta(delta: f64, previous: f64) -> f64 {
    previous + delta
}

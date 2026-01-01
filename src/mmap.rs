// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Memory-mapped file access for zero-copy reads.
//!
//! This module provides memory-mapped access to database files, enabling
//! zero-copy reads for tensor data. This is particularly useful for large
//! tensors where copying data would be expensive.
//!
//! # Features
//!
//! - Zero-copy access to tensor data via memory mapping
//! - Direct slice access for f32 and f64 arrays
//! - Safe bounds checking with clear error messages
//!
//! # Safety
//!
//! The `as_f32_slice` and `as_f64_slice` methods use unsafe code to
//! reinterpret byte slices as typed slices. This is safe when:
//! - The offset and count are within bounds
//! - The data was originally written as the requested type
//! - The platform uses little-endian byte order (most common)
//!
//! # Examples
//!
//! ```rust,no_run
//! use synadb::mmap::MmapReader;
//!
//! // Open a database file for memory-mapped reading
//! let reader = MmapReader::open("data.db").unwrap();
//!
//! // Read raw bytes at an offset
//! let bytes = reader.slice(0, 100);
//!
//! // Read f32 tensor data (zero-copy)
//! let floats = reader.as_f32_slice(1024, 256);
//! ```
//!
//! _Requirements: 2.4, 9.3_

use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;

use crate::error::{Result, SynaError};

/// Memory-mapped database file for zero-copy reads.
///
/// This struct wraps a memory-mapped file and provides safe access
/// to the underlying data. It's particularly useful for reading
/// large tensor data without copying.
///
/// # Examples
///
/// ```rust,no_run
/// use synadb::mmap::MmapReader;
///
/// let reader = MmapReader::open("data.db").unwrap();
/// let data = reader.slice(0, 1024);
/// println!("Read {} bytes", data.len());
/// ```
pub struct MmapReader {
    mmap: Mmap,
}

impl MmapReader {
    /// Open a file for memory-mapped reading.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to memory-map
    ///
    /// # Returns
    ///
    /// A new `MmapReader` instance.
    ///
    /// # Errors
    ///
    /// Returns `SynaError::Io` if the file cannot be opened or mapped.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // Safety: We're only reading from the file, and the file handle
        // is kept alive by the Mmap struct internally.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    /// Get the total length of the memory-mapped file.
    ///
    /// # Returns
    ///
    /// The size of the file in bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// println!("File size: {} bytes", reader.len());
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if the memory-mapped file is empty.
    ///
    /// # Returns
    ///
    /// `true` if the file has zero length, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Get a slice of bytes at the specified offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// A byte slice referencing the memory-mapped data.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` exceeds the file size.
    /// Use [`try_slice`](Self::try_slice) for a non-panicking version.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// let header = reader.slice(0, 15); // Read 15-byte header
    /// ```
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> &[u8] {
        &self.mmap[offset..offset + len]
    }

    /// Try to get a slice of bytes at the specified offset.
    ///
    /// This is a non-panicking version of [`slice`](Self::slice).
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// `Some(&[u8])` if the range is valid, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// if let Some(data) = reader.try_slice(0, 100) {
    ///     println!("Read {} bytes", data.len());
    /// }
    /// ```
    #[inline]
    pub fn try_slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end <= self.mmap.len() {
            Some(&self.mmap[offset..end])
        } else {
            None
        }
    }

    /// Get tensor data as f32 slice (zero-copy).
    ///
    /// This method reinterprets the raw bytes as a slice of f32 values
    /// without copying the data. The data must have been written as
    /// little-endian f32 values.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset (must be 4-byte aligned for best performance)
    /// * `count` - Number of f32 elements to read
    ///
    /// # Returns
    ///
    /// A slice of f32 values referencing the memory-mapped data.
    ///
    /// # Panics
    ///
    /// Panics if the requested range exceeds the file size.
    /// Use [`try_as_f32_slice`](Self::try_as_f32_slice) for a non-panicking version.
    ///
    /// # Safety
    ///
    /// This method uses unsafe code to reinterpret bytes as f32.
    /// It is safe when:
    /// - The data was originally written as f32 values
    /// - The platform uses little-endian byte order
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("vectors.db").unwrap();
    /// let floats = reader.as_f32_slice(1024, 768); // Read 768-dim vector
    /// println!("First value: {}", floats[0]);
    /// ```
    ///
    /// _Requirements: 2.4_
    #[inline]
    pub fn as_f32_slice(&self, offset: usize, count: usize) -> &[f32] {
        let byte_len = count * std::mem::size_of::<f32>();
        let bytes = &self.mmap[offset..offset + byte_len];
        // Safety: We ensure bounds are valid above. The caller is responsible
        // for ensuring the data was written as f32 values.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) }
    }

    /// Try to get tensor data as f32 slice (zero-copy).
    ///
    /// This is a non-panicking version of [`as_f32_slice`](Self::as_f32_slice).
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `count` - Number of f32 elements to read
    ///
    /// # Returns
    ///
    /// `Ok(&[f32])` if the range is valid, `Err` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("vectors.db").unwrap();
    /// match reader.try_as_f32_slice(1024, 768) {
    ///     Ok(floats) => println!("Read {} floats", floats.len()),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// ```
    pub fn try_as_f32_slice(&self, offset: usize, count: usize) -> Result<&[f32]> {
        let byte_len = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(SynaError::ShapeMismatch {
                data_size: usize::MAX,
                expected_size: 0,
            })?;

        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| SynaError::ShapeMismatch {
                data_size: usize::MAX,
                expected_size: self.mmap.len(),
            })?;

        if end > self.mmap.len() {
            return Err(SynaError::ShapeMismatch {
                data_size: end,
                expected_size: self.mmap.len(),
            });
        }

        let bytes = &self.mmap[offset..end];
        // Safety: We've verified bounds above
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) })
    }

    /// Get tensor data as f64 slice (zero-copy).
    ///
    /// This method reinterprets the raw bytes as a slice of f64 values
    /// without copying the data. The data must have been written as
    /// little-endian f64 values.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset (must be 8-byte aligned for best performance)
    /// * `count` - Number of f64 elements to read
    ///
    /// # Returns
    ///
    /// A slice of f64 values referencing the memory-mapped data.
    ///
    /// # Panics
    ///
    /// Panics if the requested range exceeds the file size.
    /// Use [`try_as_f64_slice`](Self::try_as_f64_slice) for a non-panicking version.
    ///
    /// # Safety
    ///
    /// This method uses unsafe code to reinterpret bytes as f64.
    /// It is safe when:
    /// - The data was originally written as f64 values
    /// - The platform uses little-endian byte order
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// let doubles = reader.as_f64_slice(0, 100);
    /// println!("Sum: {}", doubles.iter().sum::<f64>());
    /// ```
    ///
    /// _Requirements: 2.4_
    #[inline]
    pub fn as_f64_slice(&self, offset: usize, count: usize) -> &[f64] {
        let byte_len = count * std::mem::size_of::<f64>();
        let bytes = &self.mmap[offset..offset + byte_len];
        // Safety: We ensure bounds are valid above. The caller is responsible
        // for ensuring the data was written as f64 values.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, count) }
    }

    /// Try to get tensor data as f64 slice (zero-copy).
    ///
    /// This is a non-panicking version of [`as_f64_slice`](Self::as_f64_slice).
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `count` - Number of f64 elements to read
    ///
    /// # Returns
    ///
    /// `Ok(&[f64])` if the range is valid, `Err` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use synadb::mmap::MmapReader;
    ///
    /// let reader = MmapReader::open("data.db").unwrap();
    /// match reader.try_as_f64_slice(0, 100) {
    ///     Ok(doubles) => println!("Read {} doubles", doubles.len()),
    ///     Err(e) => println!("Error: {}", e),
    /// }
    /// ```
    pub fn try_as_f64_slice(&self, offset: usize, count: usize) -> Result<&[f64]> {
        let byte_len = count
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(SynaError::ShapeMismatch {
                data_size: usize::MAX,
                expected_size: 0,
            })?;

        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| SynaError::ShapeMismatch {
                data_size: usize::MAX,
                expected_size: self.mmap.len(),
            })?;

        if end > self.mmap.len() {
            return Err(SynaError::ShapeMismatch {
                data_size: end,
                expected_size: self.mmap.len(),
            });
        }

        let bytes = &self.mmap[offset..end];
        // Safety: We've verified bounds above
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, count) })
    }

    /// Get tensor data as i32 slice (zero-copy).
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `count` - Number of i32 elements to read
    ///
    /// # Returns
    ///
    /// A slice of i32 values referencing the memory-mapped data.
    ///
    /// # Panics
    ///
    /// Panics if the requested range exceeds the file size.
    #[inline]
    pub fn as_i32_slice(&self, offset: usize, count: usize) -> &[i32] {
        let byte_len = count * std::mem::size_of::<i32>();
        let bytes = &self.mmap[offset..offset + byte_len];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i32, count) }
    }

    /// Get tensor data as i64 slice (zero-copy).
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `count` - Number of i64 elements to read
    ///
    /// # Returns
    ///
    /// A slice of i64 values referencing the memory-mapped data.
    ///
    /// # Panics
    ///
    /// Panics if the requested range exceeds the file size.
    #[inline]
    pub fn as_i64_slice(&self, offset: usize, count: usize) -> &[i64] {
        let byte_len = count * std::mem::size_of::<i64>();
        let bytes = &self.mmap[offset..offset + byte_len];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i64, count) }
    }

    /// Get the raw pointer to the memory-mapped data.
    ///
    /// This is useful for advanced use cases where direct pointer access
    /// is needed, such as GPU memory transfers.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as this `MmapReader`
    /// instance exists. Do not use the pointer after dropping the reader.
    ///
    /// # Returns
    ///
    /// A raw pointer to the start of the memory-mapped region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_reader_open() {
        // Create a temp file with some data
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        assert_eq!(reader.len(), 13);
        assert!(!reader.is_empty());
    }

    #[test]
    fn test_mmap_reader_slice() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let slice = reader.slice(0, 5);
        assert_eq!(slice, b"Hello");

        let slice = reader.slice(7, 5);
        assert_eq!(slice, b"World");
    }

    #[test]
    fn test_mmap_reader_try_slice() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello").unwrap();
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();

        // Valid range
        assert!(reader.try_slice(0, 5).is_some());

        // Out of bounds
        assert!(reader.try_slice(0, 100).is_none());
        assert!(reader.try_slice(10, 1).is_none());
    }

    #[test]
    fn test_mmap_reader_f32_slice() {
        let mut file = NamedTempFile::new().unwrap();

        // Write some f32 values
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let slice = reader.as_f32_slice(0, 4);

        assert_eq!(slice.len(), 4);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.0);
        assert_eq!(slice[2], 3.0);
        assert_eq!(slice[3], 4.0);
    }

    #[test]
    fn test_mmap_reader_f64_slice() {
        let mut file = NamedTempFile::new().unwrap();

        // Write some f64 values
        let values: Vec<f64> = vec![1.5, 2.5, 3.5];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let slice = reader.as_f64_slice(0, 3);

        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1.5);
        assert_eq!(slice[1], 2.5);
        assert_eq!(slice[2], 3.5);
    }

    #[test]
    fn test_mmap_reader_try_f32_slice_bounds() {
        let mut file = NamedTempFile::new().unwrap();
        let values: Vec<f32> = vec![1.0, 2.0];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();

        // Valid range
        assert!(reader.try_as_f32_slice(0, 2).is_ok());

        // Out of bounds
        assert!(reader.try_as_f32_slice(0, 100).is_err());
    }

    #[test]
    fn test_mmap_reader_try_f64_slice_bounds() {
        let mut file = NamedTempFile::new().unwrap();
        let values: Vec<f64> = vec![1.0, 2.0];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();

        // Valid range
        assert!(reader.try_as_f64_slice(0, 2).is_ok());

        // Out of bounds
        assert!(reader.try_as_f64_slice(0, 100).is_err());
    }

    #[test]
    fn test_mmap_reader_i32_slice() {
        let mut file = NamedTempFile::new().unwrap();
        let values: Vec<i32> = vec![10, 20, 30];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let slice = reader.as_i32_slice(0, 3);

        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn test_mmap_reader_i64_slice() {
        let mut file = NamedTempFile::new().unwrap();
        let values: Vec<i64> = vec![100, 200, 300];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let slice = reader.as_i64_slice(0, 3);

        assert_eq!(slice, &[100, 200, 300]);
    }

    #[test]
    fn test_mmap_reader_offset_access() {
        let mut file = NamedTempFile::new().unwrap();

        // Write header (8 bytes) + f32 data
        file.write_all(&[0u8; 8]).unwrap(); // 8-byte header
        let values: Vec<f32> = vec![1.0, 2.0, 3.0];
        for v in &values {
            file.write_all(&v.to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();

        // Read f32 data starting at offset 8
        let slice = reader.as_f32_slice(8, 3);
        assert_eq!(slice, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mmap_reader_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let reader = MmapReader::open(file.path()).unwrap();

        assert_eq!(reader.len(), 0);
        assert!(reader.is_empty());
    }

    #[test]
    fn test_mmap_reader_as_ptr() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test").unwrap();
        file.flush().unwrap();

        let reader = MmapReader::open(file.path()).unwrap();
        let ptr = reader.as_ptr();

        // Verify pointer is valid by reading through it
        unsafe {
            assert_eq!(*ptr, b't');
        }
    }
}

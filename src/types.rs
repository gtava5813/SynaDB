// Copyright (c) 2025 SynaDB Contributors
// Licensed under the SynaDB License. See LICENSE file for details.

//! Core data types for Syna database.
//!
//! This module defines the fundamental data structures used throughout Syna:
//! - [`Atom`] - The tagged union for storing heterogeneous values
//! - [`LogHeader`] - Fixed-size metadata for each log entry
//! - Flag constants for compression and deletion markers

use serde::{Deserialize, Serialize};

/// Flag indicating the value is delta-encoded from the previous value.
///
/// When set in [`LogHeader::flags`], the stored value represents the difference
/// from the previous value for the same key, rather than an absolute value.
/// This is used for float sequences to reduce storage size.
pub const IS_DELTA: u8 = 0x01;

/// Flag indicating the value is LZ4 compressed.
///
/// When set in [`LogHeader::flags`], the value bytes have been compressed
/// using LZ4 and must be decompressed before deserialization.
pub const IS_COMPRESSED: u8 = 0x02;

/// Flag indicating this entry is a tombstone (deletion marker).
///
/// When set in [`LogHeader::flags`], this entry marks the key as deleted.
/// The key will not appear in [`keys()`](crate::SynaDB::keys) and
/// [`get()`](crate::SynaDB::get) will return `None`.
pub const IS_TOMBSTONE: u8 = 0x04;

/// Size of the [`LogHeader`] in bytes (8 + 2 + 4 + 1 = 15).
pub const HEADER_SIZE: usize = 15;

/// The fundamental data unit - a tagged union that can hold various types.
///
/// Atom is the core value type in Syna, supporting six variants:
/// - `Null` - Absence of value
/// - `Float(f64)` - 64-bit floating point numbers
/// - `Int(i64)` - 64-bit signed integers
/// - `Text(String)` - UTF-8 strings
/// - `Bytes(Vec<u8>)` - Raw byte arrays
/// - `Vector(Vec<f32>, u16)` - Embedding vectors with dimensions
///
/// # Examples
///
/// ```rust
/// use synadb::Atom;
///
/// // Create atoms from various types
/// let float_atom = Atom::Float(3.14159);
/// let int_atom = Atom::Int(42);
/// let text_atom = Atom::Text("hello".to_string());
/// let bytes_atom = Atom::Bytes(vec![0x01, 0x02, 0x03]);
/// let vector_atom = Atom::Vector(vec![0.1, 0.2, 0.3], 3);
///
/// // Use From trait for convenience
/// let f: Atom = 2.718.into();
/// let i: Atom = 100i64.into();
///
/// // Check type and extract value
/// assert!(float_atom.is_float());
/// assert_eq!(float_atom.as_float(), Some(3.14159));
/// assert!(vector_atom.is_vector());
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Atom {
    /// Absence of value (similar to SQL NULL).
    Null,
    /// 64-bit floating point number.
    Float(f64),
    /// 64-bit signed integer.
    Int(i64),
    /// UTF-8 encoded string.
    Text(String),
    /// Raw byte array.
    Bytes(Vec<u8>),
    /// Embedding vector with dimensions (data, dimensions).
    /// Used for storing high-dimensional vectors for similarity search.
    /// Dimensions typically range from 64 to 8192.
    Vector(Vec<f32>, u16),
}

impl Atom {
    /// Returns the type name as a static string for debugging.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::Atom;
    ///
    /// assert_eq!(Atom::Float(1.0).type_name(), "Float");
    /// assert_eq!(Atom::Int(42).type_name(), "Int");
    /// assert_eq!(Atom::Null.type_name(), "Null");
    /// assert_eq!(Atom::Vector(vec![1.0, 2.0], 2).type_name(), "Vector");
    /// ```
    pub fn type_name(&self) -> &'static str {
        match self {
            Atom::Null => "Null",
            Atom::Float(_) => "Float",
            Atom::Int(_) => "Int",
            Atom::Text(_) => "Text",
            Atom::Bytes(_) => "Bytes",
            Atom::Vector(_, _) => "Vector",
        }
    }

    /// Returns `true` if this Atom is a [`Float`](Atom::Float) variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::Atom;
    ///
    /// assert!(Atom::Float(1.0).is_float());
    /// assert!(!Atom::Int(42).is_float());
    /// ```
    pub fn is_float(&self) -> bool {
        matches!(self, Atom::Float(_))
    }

    /// Returns the float value if this is a [`Float`](Atom::Float) variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::Atom;
    ///
    /// assert_eq!(Atom::Float(3.14).as_float(), Some(3.14));
    /// assert_eq!(Atom::Int(42).as_float(), None);
    /// ```
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Atom::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns `true` if this Atom is a [`Vector`](Atom::Vector) variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::Atom;
    ///
    /// assert!(Atom::Vector(vec![1.0, 2.0, 3.0], 3).is_vector());
    /// assert!(!Atom::Float(1.0).is_vector());
    /// ```
    pub fn is_vector(&self) -> bool {
        matches!(self, Atom::Vector(_, _))
    }

    /// Returns the vector data and dimensions if this is a [`Vector`](Atom::Vector) variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::Atom;
    ///
    /// let vec_atom = Atom::Vector(vec![1.0, 2.0, 3.0], 3);
    /// let (data, dims) = vec_atom.as_vector().unwrap();
    /// assert_eq!(data, &[1.0, 2.0, 3.0]);
    /// assert_eq!(dims, 3);
    ///
    /// assert_eq!(Atom::Float(1.0).as_vector(), None);
    /// ```
    pub fn as_vector(&self) -> Option<(&[f32], u16)> {
        match self {
            Atom::Vector(data, dims) => Some((data, *dims)),
            _ => None,
        }
    }
}

impl From<f64> for Atom {
    fn from(value: f64) -> Self {
        Atom::Float(value)
    }
}

impl From<i64> for Atom {
    fn from(value: i64) -> Self {
        Atom::Int(value)
    }
}

impl From<String> for Atom {
    fn from(value: String) -> Self {
        Atom::Text(value)
    }
}

impl From<Vec<u8>> for Atom {
    fn from(value: Vec<u8>) -> Self {
        Atom::Bytes(value)
    }
}

/// Fixed-size metadata preceding each log entry.
///
/// Each entry in the database file starts with a 15-byte header containing:
/// - `timestamp` (8 bytes) - Unix timestamp in microseconds
/// - `key_len` (2 bytes) - Length of the key in bytes
/// - `val_len` (4 bytes) - Length of the serialized value in bytes
/// - `flags` (1 byte) - Bit flags for compression and deletion
///
/// # Binary Layout
///
/// ```text
/// Offset  Size  Field
/// 0       8     timestamp (u64 little-endian)
/// 8       2     key_len (u16 little-endian)
/// 10      4     val_len (u32 little-endian)
/// 14      1     flags (u8)
/// ```
///
/// # Flag Bits
///
/// - Bit 0 ([`IS_DELTA`]): Value is delta-encoded
/// - Bit 1 ([`IS_COMPRESSED`]): Value is LZ4 compressed
/// - Bit 2 ([`IS_TOMBSTONE`]): Entry is a deletion marker
#[repr(C, packed)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogHeader {
    /// Unix timestamp in microseconds when the entry was written.
    pub timestamp: u64,
    /// Length of the key in bytes (max 65535).
    pub key_len: u16,
    /// Length of the serialized value in bytes (max ~4GB).
    pub val_len: u32,
    /// Bit flags: [`IS_DELTA`] | [`IS_COMPRESSED`] | [`IS_TOMBSTONE`].
    pub flags: u8,
}

impl LogHeader {
    /// Creates a new LogHeader with the current timestamp.
    ///
    /// # Arguments
    ///
    /// * `key_len` - Length of the key in bytes
    /// * `val_len` - Length of the serialized value in bytes
    /// * `flags` - Bit flags (see [`IS_DELTA`], [`IS_COMPRESSED`], [`IS_TOMBSTONE`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::{LogHeader, IS_COMPRESSED};
    ///
    /// let header = LogHeader::new(5, 100, IS_COMPRESSED);
    /// // Copy fields to avoid unaligned access on packed struct
    /// let key_len = header.key_len;
    /// let val_len = header.val_len;
    /// let flags = header.flags;
    /// assert_eq!(key_len, 5);
    /// assert_eq!(val_len, 100);
    /// assert!(flags & IS_COMPRESSED != 0);
    /// ```
    pub fn new(key_len: u16, val_len: u32, flags: u8) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Self {
            timestamp,
            key_len,
            val_len,
            flags,
        }
    }

    /// Serializes the header to a fixed-size byte array.
    ///
    /// The bytes are written in little-endian format for cross-platform compatibility.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::{LogHeader, HEADER_SIZE};
    ///
    /// let header = LogHeader::new(5, 100, 0);
    /// let bytes = header.to_bytes();
    /// assert_eq!(bytes.len(), HEADER_SIZE);
    /// ```
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[8..10].copy_from_slice(&self.key_len.to_le_bytes());
        buf[10..14].copy_from_slice(&self.val_len.to_le_bytes());
        buf[14] = self.flags;
        buf
    }

    /// Deserializes a header from a fixed-size byte array.
    ///
    /// # Arguments
    ///
    /// * `buf` - A 15-byte array containing the serialized header
    ///
    /// # Examples
    ///
    /// ```rust
    /// use synadb::{LogHeader, HEADER_SIZE};
    ///
    /// let original = LogHeader::new(5, 100, 0);
    /// let bytes = original.to_bytes();
    /// let restored = LogHeader::from_bytes(&bytes);
    /// // Copy fields to avoid unaligned access on packed struct
    /// let orig_key_len = original.key_len;
    /// let orig_val_len = original.val_len;
    /// let rest_key_len = restored.key_len;
    /// let rest_val_len = restored.val_len;
    /// assert_eq!(orig_key_len, rest_key_len);
    /// assert_eq!(orig_val_len, rest_val_len);
    /// ```
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        let timestamp = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let key_len = u16::from_le_bytes(buf[8..10].try_into().unwrap());
        let val_len = u32::from_le_bytes(buf[10..14].try_into().unwrap());
        let flags = buf[14];
        Self {
            timestamp,
            key_len,
            val_len,
            flags,
        }
    }

    /// Validates the header has reasonable values.
    ///
    /// Returns `true` if:
    /// - `key_len` is less than 65535 bytes
    /// - `val_len` is less than 1GB
    ///
    /// This is used during recovery to detect corrupted entries.
    pub fn is_valid(&self) -> bool {
        // Key length should be reasonable (< 64KB)
        // Value length should be reasonable (< 1GB)
        self.key_len < 65535 && self.val_len < 1_000_000_000
    }
}

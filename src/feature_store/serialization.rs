// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Bincode serialization helpers for the feature store.
//!
//! All feature store data uses bincode for persistence (not JSON).
//! This module provides versioned wrappers for forward-compatible
//! serialization of the feature registry and related structures.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SynaError};

/// Current persistence format version.
pub const CURRENT_FORMAT_VERSION: u32 = 1;

/// Versioned binary format for persisting the feature registry.
///
/// The version field enables forward-compatible deserialization:
/// older code can detect newer formats and fail gracefully.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistedRegistry {
    /// Format version (currently 1).
    pub version: u32,
    /// Serialized registry data.
    pub data: Vec<u8>,
    /// Reserved for future extension fields.
    pub _reserved: Vec<u8>,
}

/// Serialize a value to bincode bytes.
pub fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serialize(value).map_err(SynaError::Serialization)
}

/// Deserialize a value from bincode bytes.
pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T> {
    bincode::deserialize(bytes).map_err(SynaError::Serialization)
}

/// Wrap data in a versioned persistence envelope.
pub fn wrap_versioned(data: Vec<u8>) -> PersistedRegistry {
    PersistedRegistry {
        version: CURRENT_FORMAT_VERSION,
        data,
        _reserved: Vec::new(),
    }
}

/// Unwrap a versioned persistence envelope, checking the version.
pub fn unwrap_versioned(persisted: &PersistedRegistry) -> Result<&[u8]> {
    if persisted.version > CURRENT_FORMAT_VERSION {
        return Err(SynaError::InvalidInput(format!(
            "Unsupported feature store format version: {} (max supported: {})",
            persisted.version, CURRENT_FORMAT_VERSION
        )));
    }
    Ok(&persisted.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let value: Vec<i64> = vec![1, 2, 3, 4, 5];
        let bytes = serialize(&value).unwrap();
        let restored: Vec<i64> = deserialize(&bytes).unwrap();
        assert_eq!(value, restored);
    }

    #[test]
    fn test_versioned_wrap_unwrap() {
        let data = vec![1u8, 2, 3, 4];
        let wrapped = wrap_versioned(data.clone());
        assert_eq!(wrapped.version, CURRENT_FORMAT_VERSION);
        let unwrapped = unwrap_versioned(&wrapped).unwrap();
        assert_eq!(unwrapped, &data[..]);
    }

    #[test]
    fn test_versioned_future_version_rejected() {
        let persisted = PersistedRegistry {
            version: 999,
            data: vec![],
            _reserved: vec![],
        };
        assert!(unwrap_versioned(&persisted).is_err());
    }
}

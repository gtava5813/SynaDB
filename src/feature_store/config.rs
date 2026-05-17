// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Feature Store configuration.
//!
//! This module defines [`FeatureStoreConfig`] which controls the performance
//! characteristics of the feature store: cache capacity, buffer sizes,
//! freshness thresholds, and parallelism settings.

use serde::{Deserialize, Serialize};

/// Feature store configuration.
///
/// Controls the performance trade-offs between latency, throughput, and durability.
///
/// # Examples
///
/// ```rust
/// use synadb::feature_store::FeatureStoreConfig;
///
/// // High-throughput ingestion config
/// let config = FeatureStoreConfig {
///     online_cache_capacity: 200_000,
///     write_buffer_size: 8192,
///     sync_on_write: false,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStoreConfig {
    /// Maximum entities in online cache (default: 100,000).
    pub online_cache_capacity: usize,
    /// Write buffer flush threshold in entries (default: 4096).
    pub write_buffer_size: usize,
    /// Write buffer max age before flush in microseconds (default: 10,000 = 10ms).
    pub write_buffer_max_age_micros: u64,
    /// Default TTL for features without explicit TTL in seconds (default: 86400 = 1 day).
    pub default_ttl_seconds: u64,
    /// DAVO freshness threshold below which features are considered stale (default: 0.5).
    pub freshness_threshold: f64,
    /// Enable sync_on_write for durability (default: false for throughput).
    pub sync_on_write: bool,
    /// Number of threads for parallel dataset generation (default: 0 = num_cpus).
    pub parallelism: usize,
    /// Compaction threshold: ratio of superseded entries triggering compaction (default: 0.5).
    pub compaction_threshold: f64,
}

impl Default for FeatureStoreConfig {
    fn default() -> Self {
        Self {
            online_cache_capacity: 100_000,
            write_buffer_size: 4096,
            write_buffer_max_age_micros: 10_000,
            default_ttl_seconds: 86_400,
            freshness_threshold: 0.5,
            sync_on_write: false,
            parallelism: 0,
            compaction_threshold: 0.5,
        }
    }
}

impl FeatureStoreConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.online_cache_capacity == 0 {
            return Err(crate::error::SynaError::InvalidInput(
                "online_cache_capacity must be > 0".to_string(),
            ));
        }
        if self.write_buffer_size == 0 {
            return Err(crate::error::SynaError::InvalidInput(
                "write_buffer_size must be > 0".to_string(),
            ));
        }
        if self.freshness_threshold < 0.0 || self.freshness_threshold > 1.0 {
            return Err(crate::error::SynaError::InvalidInput(
                "freshness_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.compaction_threshold < 0.0 || self.compaction_threshold > 1.0 {
            return Err(crate::error::SynaError::InvalidInput(
                "compaction_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

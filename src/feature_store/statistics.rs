// Copyright (c) 2026 Mindoval, Inc
// Licensed under the SynaDB License. See LICENSE file for details.

//! Feature statistics and drift detection.
//!
//! Uses Welford's online algorithm for incremental computation of
//! mean and variance without storing all values. Also provides
//! Population Stability Index (PSI) for drift detection.

use serde::{Deserialize, Serialize};

/// Running statistics for a single feature, computed incrementally
/// using Welford's online algorithm.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeatureStatistics {
    /// Number of values observed.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running M2 (sum of squared differences from mean) for variance.
    pub m2: f64,
    /// Minimum value observed.
    pub min: f64,
    /// Maximum value observed.
    pub max: f64,
    /// Number of null values observed.
    pub null_count: u64,
}

impl FeatureStatistics {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            null_count: 0,
        }
    }

    /// Update statistics with a new numeric value (Welford's algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Record a null value.
    pub fn update_null(&mut self) {
        self.null_count += 1;
    }

    /// Get the population variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Get the sample variance.
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the standard deviation (population).
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the null rate (proportion of null values).
    pub fn null_rate(&self) -> f64 {
        let total = self.count + self.null_count;
        if total == 0 {
            0.0
        } else {
            self.null_count as f64 / total as f64
        }
    }
}

impl Default for FeatureStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Column-level statistics for training dataset results.
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub stddev: f64,
    /// Null rate (0.0 to 1.0).
    pub null_rate: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Total count.
    pub count: u64,
}

/// Compute Population Stability Index (PSI) between two distributions.
///
/// PSI measures how much a distribution has shifted. Values:
/// - PSI < 0.1: No significant change
/// - 0.1 ≤ PSI < 0.25: Moderate change
/// - PSI ≥ 0.25: Significant change (drift detected)
///
/// Both histograms must have the same number of buckets and represent
/// proportions (sum to ~1.0).
pub fn compute_psi(expected: &[f64], actual: &[f64]) -> f64 {
    if expected.len() != actual.len() || expected.is_empty() {
        return 0.0;
    }

    let epsilon = 1e-10; // Avoid log(0)
    let mut psi = 0.0;

    for (e, a) in expected.iter().zip(actual.iter()) {
        let e_safe = e.max(epsilon);
        let a_safe = a.max(epsilon);
        psi += (a_safe - e_safe) * (a_safe / e_safe).ln();
    }

    psi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let mut stats = FeatureStatistics::new();
        stats.update(10.0);
        stats.update(20.0);
        stats.update(30.0);

        assert_eq!(stats.count, 3);
        assert!((stats.mean - 20.0).abs() < 1e-10);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 30.0);
    }

    #[test]
    fn test_welford_variance() {
        let mut stats = FeatureStatistics::new();
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        for v in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(*v);
        }

        assert!((stats.mean - 5.0).abs() < 1e-10);
        // Population variance = 4.0
        assert!((stats.variance() - 4.0).abs() < 1e-10);
        // Sample variance = 32/7 ≈ 4.571
        assert!((stats.sample_variance() - 32.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_null_rate() {
        let mut stats = FeatureStatistics::new();
        stats.update(1.0);
        stats.update(2.0);
        stats.update_null();
        stats.update_null();

        assert!((stats.null_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_psi_identical() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let psi = compute_psi(&dist, &dist);
        assert!(psi.abs() < 1e-10);
    }

    #[test]
    fn test_psi_shifted() {
        let expected = vec![0.25, 0.25, 0.25, 0.25];
        let actual = vec![0.1, 0.1, 0.4, 0.4];
        let psi = compute_psi(&expected, &actual);
        // Should be > 0.1 (moderate shift)
        assert!(psi > 0.1);
    }

    #[test]
    fn test_single_value() {
        let mut stats = FeatureStatistics::new();
        stats.update(42.0);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }
}

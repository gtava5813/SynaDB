//! Anomaly Detection — Z-score, IQR, and moving-average deviation methods.
//!
//! Detects statistical outliers in time-series data directly within queries.

use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Anomaly detection method.
#[derive(Debug, Clone)]
pub enum AnomalyMethod {
    /// Flag values where |z-score| > threshold.
    ZScore { threshold: f64 },
    /// Flag values outside `[Q1 - multiplier*IQR, Q3 + multiplier*IQR]`.
    Iqr { multiplier: f64 },
    /// Flag values deviating from moving average by > threshold std-devs.
    MovingAvgDeviation { window: usize, threshold: f64 },
}

/// Result of anomaly detection for a single point.
#[derive(Debug, Clone, Serialize)]
pub struct AnomalyResult {
    /// Index in the input series.
    pub index: usize,
    /// Timestamp (if available).
    pub timestamp: u64,
    /// The value.
    pub value: f64,
    /// Anomaly score (higher = more anomalous).
    pub anomaly_score: f64,
    /// Whether this point is flagged as anomalous.
    pub is_anomaly: bool,
    /// Expected range `(lower, upper)`.
    pub expected_range: (f64, f64),
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Detect anomalies in a time-series.
///
/// `values` is a slice of `(timestamp, value)` pairs sorted by timestamp.
pub fn detect(values: &[(u64, f64)], method: &AnomalyMethod) -> Vec<AnomalyResult> {
    match method {
        AnomalyMethod::ZScore { threshold } => detect_zscore(values, *threshold),
        AnomalyMethod::Iqr { multiplier } => detect_iqr(values, *multiplier),
        AnomalyMethod::MovingAvgDeviation { window, threshold } => {
            detect_moving_avg(values, *window, *threshold)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Z-Score
// ═══════════════════════════════════════════════════════════════════════

fn detect_zscore(values: &[(u64, f64)], threshold: f64) -> Vec<AnomalyResult> {
    if values.is_empty() {
        return vec![];
    }

    let vals: Vec<f64> = values.iter().map(|(_, v)| *v).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        // All values identical — no anomalies
        return values
            .iter()
            .enumerate()
            .map(|(i, (ts, v))| AnomalyResult {
                index: i,
                timestamp: *ts,
                value: *v,
                anomaly_score: 0.0,
                is_anomaly: false,
                expected_range: (mean, mean),
            })
            .collect();
    }

    let lower = mean - threshold * std_dev;
    let upper = mean + threshold * std_dev;

    values
        .iter()
        .enumerate()
        .map(|(i, (ts, v))| {
            let z = (v - mean).abs() / std_dev;
            AnomalyResult {
                index: i,
                timestamp: *ts,
                value: *v,
                anomaly_score: z,
                is_anomaly: z > threshold,
                expected_range: (lower, upper),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  IQR
// ═══════════════════════════════════════════════════════════════════════

fn detect_iqr(values: &[(u64, f64)], multiplier: f64) -> Vec<AnomalyResult> {
    if values.is_empty() {
        return vec![];
    }

    let mut sorted_vals: Vec<f64> = values.iter().map(|(_, v)| *v).collect();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_vals.len();
    let q1 = sorted_vals[n / 4];
    let q3 = sorted_vals[3 * n / 4];
    let iqr = q3 - q1;

    let lower = q1 - multiplier * iqr;
    let upper = q3 + multiplier * iqr;

    values
        .iter()
        .enumerate()
        .map(|(i, (ts, v))| {
            let outside = *v < lower || *v > upper;
            let score = if outside {
                if *v < lower {
                    (lower - v) / iqr.max(1e-10)
                } else {
                    (v - upper) / iqr.max(1e-10)
                }
            } else {
                0.0
            };
            AnomalyResult {
                index: i,
                timestamp: *ts,
                value: *v,
                anomaly_score: score,
                is_anomaly: outside,
                expected_range: (lower, upper),
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Moving Average Deviation
// ═══════════════════════════════════════════════════════════════════════

fn detect_moving_avg(values: &[(u64, f64)], window: usize, threshold: f64) -> Vec<AnomalyResult> {
    if values.is_empty() || window == 0 {
        return vec![];
    }

    let vals: Vec<f64> = values.iter().map(|(_, v)| *v).collect();
    let n = vals.len();
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_slice = &vals[start..=i];
        let w_mean = window_slice.iter().sum::<f64>() / window_slice.len() as f64;
        let w_var = window_slice
            .iter()
            .map(|v| (v - w_mean).powi(2))
            .sum::<f64>()
            / window_slice.len() as f64;
        let w_std = w_var.sqrt();

        let deviation = if w_std > 0.0 {
            (vals[i] - w_mean).abs() / w_std
        } else {
            0.0
        };

        let lower = w_mean - threshold * w_std;
        let upper = w_mean + threshold * w_std;

        results.push(AnomalyResult {
            index: i,
            timestamp: values[i].0,
            value: vals[i],
            anomaly_score: deviation,
            is_anomaly: deviation > threshold,
            expected_range: (lower, upper),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_detects_outlier() {
        // Normal values around 10, one outlier at 100
        let values: Vec<(u64, f64)> = (0..20)
            .map(|i| (i as u64 * 1_000_000, 10.0 + (i as f64 * 0.1)))
            .chain(std::iter::once((20_000_000, 100.0)))
            .collect();

        let results = detect(&values, &AnomalyMethod::ZScore { threshold: 2.0 });
        assert_eq!(results.len(), 21);

        // The outlier (last) should be flagged
        let outlier = &results[20];
        assert!(outlier.is_anomaly);
        assert!(outlier.anomaly_score > 2.0);

        // Normal values should not be flagged
        assert!(!results[0].is_anomaly);
        assert!(!results[10].is_anomaly);
    }

    #[test]
    fn test_iqr_detects_outlier() {
        let mut values: Vec<(u64, f64)> = (0..100)
            .map(|i| (i as u64 * 1_000_000, 50.0 + (i as f64 * 0.1)))
            .collect();
        values.push((100_000_000, 500.0)); // extreme outlier

        let results = detect(&values, &AnomalyMethod::Iqr { multiplier: 1.5 });
        let outlier = results.last().unwrap();
        assert!(outlier.is_anomaly);
    }

    #[test]
    fn test_moving_avg_detects_spike() {
        let mut values: Vec<(u64, f64)> = (0..50).map(|i| (i as u64 * 1_000_000, 10.0)).collect();
        // Insert a spike
        values[25] = (25_000_000, 100.0);

        let results = detect(
            &values,
            &AnomalyMethod::MovingAvgDeviation {
                window: 10,
                threshold: 2.0,
            },
        );

        assert!(results[25].is_anomaly);
        assert!(!results[0].is_anomaly);
        assert!(!results[49].is_anomaly);
    }

    #[test]
    fn test_empty_input() {
        let results = detect(&[], &AnomalyMethod::ZScore { threshold: 3.0 });
        assert!(results.is_empty());
    }

    #[test]
    fn test_constant_values_no_anomalies() {
        let values: Vec<(u64, f64)> = (0..10).map(|i| (i as u64, 5.0)).collect();
        let results = detect(&values, &AnomalyMethod::ZScore { threshold: 2.0 });
        assert!(results.iter().all(|r| !r.is_anomaly));
    }
}

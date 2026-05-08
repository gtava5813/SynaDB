//! Correlation Analysis — Pearson correlation and cross-correlation with lag.

use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Correlation result between two time-series.
#[derive(Debug, Clone, Serialize)]
pub struct CorrelationResult {
    /// Pearson correlation coefficient (-1.0 to 1.0).
    pub correlation: f64,
    /// Optimal lag (in number of samples) for cross-correlation.
    pub lag: i64,
    /// Whether the correlation is statistically significant (|r| > threshold).
    pub is_significant: bool,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Compute Pearson correlation between two equal-length series.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum();

    let x_var: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
    let y_var: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let denominator = (x_var * y_var).sqrt();
    if denominator == 0.0 {
        return Some(0.0);
    }

    Some(numerator / denominator)
}

/// Compute cross-correlation at different lags, return the lag with maximum |correlation|.
///
/// `max_lag` is the maximum number of samples to shift.
pub fn cross_correlate(x: &[f64], y: &[f64], max_lag: usize) -> CorrelationResult {
    let mut best_corr = 0.0_f64;
    let mut best_lag: i64 = 0;

    let max_lag = max_lag.min(x.len().min(y.len()) / 2);

    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let (x_slice, y_slice) = if lag >= 0 {
            let l = lag as usize;
            let len = x.len().min(y.len() - l);
            (&x[..len], &y[l..l + len])
        } else {
            let l = (-lag) as usize;
            let len = x.len().min(y.len()).saturating_sub(l);
            (&x[l..l + len], &y[..len])
        };

        if let Some(r) = pearson(x_slice, y_slice) {
            if r.abs() > best_corr.abs() {
                best_corr = r;
                best_lag = lag;
            }
        }
    }

    CorrelationResult {
        correlation: best_corr,
        lag: best_lag,
        is_significant: best_corr.abs() > 0.5,
    }
}

/// Find all series in `candidates` that correlate with `target` above `min_correlation`.
pub fn find_correlated(
    target: &[f64],
    candidates: &[(&str, &[f64])],
    min_correlation: f64,
) -> Vec<(String, CorrelationResult)> {
    candidates
        .iter()
        .filter_map(|(key, series)| {
            let result = cross_correlate(target, series, 10);
            if result.correlation.abs() >= min_correlation {
                Some((key.to_string(), result))
            } else {
                None
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        let r = pearson(&x, &y).unwrap();
        assert!(r.abs() < 0.5); // weak or no correlation
    }

    #[test]
    fn test_cross_correlation_with_lag() {
        // y is x shifted by 2 positions
        let x = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        let result = cross_correlate(&x, &y, 5);
        // Should detect lag of ~2
        assert!(result.lag >= 1 && result.lag <= 3);
        assert!(result.correlation > 0.7);
    }

    #[test]
    fn test_find_correlated() {
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let correlated = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // r=1.0
        let uncorrelated = vec![5.0, 1.0, 4.0, 2.0, 3.0]; // r≈0

        let candidates: Vec<(&str, &[f64])> =
            vec![("corr", &correlated), ("uncorr", &uncorrelated)];

        let results = find_correlated(&target, &candidates, 0.8);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "corr");
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(pearson(&[], &[]), None);
        assert_eq!(pearson(&[1.0], &[2.0]), None); // need at least 2
    }
}

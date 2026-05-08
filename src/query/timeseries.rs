//! Time-Series Operations — DIFF, RATE, MOVING_AVG, RESAMPLE.
//!
//! These operate on ordered sequences of `(timestamp, value)` pairs
//! extracted from the database.

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Compute differences between consecutive values.
///
/// Returns `(timestamp_of_second, v2 - v1)` for each consecutive pair.
pub fn diff(values: &[(u64, f64)]) -> Vec<(u64, f64)> {
    if values.len() < 2 {
        return vec![];
    }
    values
        .windows(2)
        .map(|w| (w[1].0, w[1].1 - w[0].1))
        .collect()
}

/// Compute rate of change per second.
///
/// Returns `(timestamp, delta_value / delta_time_seconds)`.
pub fn rate(values: &[(u64, f64)]) -> Vec<(u64, f64)> {
    if values.len() < 2 {
        return vec![];
    }
    values
        .windows(2)
        .map(|w| {
            let dt_micros = w[1].0.saturating_sub(w[0].0) as f64;
            let dv = w[1].1 - w[0].1;
            let rate = if dt_micros > 0.0 {
                dv / (dt_micros / 1_000_000.0)
            } else {
                0.0
            };
            (w[1].0, rate)
        })
        .collect()
}

/// Compute a simple moving average over a sliding window.
///
/// The window is measured in number of points (not time).
/// Returns a vector of the same length as input; the first `window-1`
/// values use a shorter window (expanding window start).
pub fn moving_avg(values: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || values.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0;

    for (i, &val) in values.iter().enumerate() {
        sum += val;
        if i >= window {
            sum -= values[i - window];
        }
        let count = (i + 1).min(window);
        result.push(sum / count as f64);
    }
    result
}

/// Resample a time-series to a fixed interval using linear interpolation.
///
/// `interval_micros` is the desired spacing between output points.
/// Returns evenly-spaced `(timestamp, interpolated_value)` pairs.
pub fn resample(values: &[(u64, f64)], interval_micros: u64) -> Vec<(u64, f64)> {
    if values.len() < 2 || interval_micros == 0 {
        return values.to_vec();
    }

    let start = values[0].0;
    let end = values[values.len() - 1].0;
    let mut result = Vec::new();
    let mut t = start;
    let mut idx = 0;

    while t <= end {
        // Advance idx so that values[idx].0 <= t < values[idx+1].0
        while idx + 1 < values.len() && values[idx + 1].0 <= t {
            idx += 1;
        }

        if idx + 1 >= values.len() {
            // Past the last point — use last value
            result.push((t, values[values.len() - 1].1));
        } else {
            // Linear interpolation between idx and idx+1
            let (t0, v0) = values[idx];
            let (t1, v1) = values[idx + 1];
            let frac = if t1 > t0 {
                (t - t0) as f64 / (t1 - t0) as f64
            } else {
                0.0
            };
            let interpolated = v0 + frac * (v1 - v0);
            result.push((t, interpolated));
        }

        t += interval_micros;
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff() {
        let values = vec![(0, 10.0), (1_000_000, 13.0), (2_000_000, 11.0)];
        let d = diff(&values);
        assert_eq!(d.len(), 2);
        assert!((d[0].1 - 3.0).abs() < 1e-10);
        assert!((d[1].1 - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_empty() {
        assert!(diff(&[]).is_empty());
        assert!(diff(&[(0, 1.0)]).is_empty());
    }

    #[test]
    fn test_rate() {
        // 10 units over 1 second = rate of 10/s
        let values = vec![(0, 0.0), (1_000_000, 10.0)];
        let r = rate(&values);
        assert_eq!(r.len(), 1);
        assert!((r[0].1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_avg() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_avg(&values, 3);
        assert_eq!(ma.len(), 5);
        // First: 1/1 = 1.0
        assert!((ma[0] - 1.0).abs() < 1e-10);
        // Second: (1+2)/2 = 1.5
        assert!((ma[1] - 1.5).abs() < 1e-10);
        // Third: (1+2+3)/3 = 2.0
        assert!((ma[2] - 2.0).abs() < 1e-10);
        // Fourth: (2+3+4)/3 = 3.0
        assert!((ma[3] - 3.0).abs() < 1e-10);
        // Fifth: (3+4+5)/3 = 4.0
        assert!((ma[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_avg_window_larger_than_data() {
        let values = vec![2.0, 4.0];
        let ma = moving_avg(&values, 10);
        assert_eq!(ma.len(), 2);
        assert!((ma[0] - 2.0).abs() < 1e-10);
        assert!((ma[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_linear() {
        // Two points: (0, 0.0) and (10_000_000, 10.0)
        // Resample at 2_000_000 interval → 6 points
        let values = vec![(0u64, 0.0), (10_000_000, 10.0)];
        let resampled = resample(&values, 2_000_000);
        assert_eq!(resampled.len(), 6); // 0, 2M, 4M, 6M, 8M, 10M
        assert!((resampled[0].1 - 0.0).abs() < 1e-10);
        assert!((resampled[1].1 - 2.0).abs() < 1e-10);
        assert!((resampled[2].1 - 4.0).abs() < 1e-10);
        assert!((resampled[5].1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_empty() {
        assert!(resample(&[], 1000).is_empty());
        let single = vec![(0u64, 5.0)];
        assert_eq!(resample(&single, 1000), single);
    }
}

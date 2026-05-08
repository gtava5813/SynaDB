//! Predictive Queries — forecast future values using lightweight models.
//!
//! Supports linear extrapolation, exponential smoothing, and simple moving average.

use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// Prediction method.
#[derive(Debug, Clone)]
pub enum PredictionMethod {
    /// Linear regression extrapolation.
    Linear,
    /// Exponential smoothing with alpha parameter.
    ExponentialSmoothing { alpha: f64 },
    /// Simple moving average projection.
    MovingAverage { window: usize },
}

/// A single predicted point.
#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    /// Predicted timestamp.
    pub timestamp: u64,
    /// Predicted value.
    pub value: f64,
    /// Lower confidence bound (95%).
    pub confidence_lower: f64,
    /// Upper confidence bound (95%).
    pub confidence_upper: f64,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Predict future values from historical data.
///
/// `history` must be sorted by timestamp. `horizon` is the number of
/// future points to predict. `interval_micros` is the spacing between
/// predicted points.
pub fn predict(
    history: &[(u64, f64)],
    method: &PredictionMethod,
    horizon: usize,
    interval_micros: u64,
) -> Vec<Prediction> {
    if history.is_empty() || horizon == 0 {
        return vec![];
    }

    match method {
        PredictionMethod::Linear => predict_linear(history, horizon, interval_micros),
        PredictionMethod::ExponentialSmoothing { alpha } => {
            predict_exp_smoothing(history, *alpha, horizon, interval_micros)
        }
        PredictionMethod::MovingAverage { window } => {
            predict_moving_avg(history, *window, horizon, interval_micros)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Implementations
// ═══════════════════════════════════════════════════════════════════════

fn predict_linear(history: &[(u64, f64)], horizon: usize, interval: u64) -> Vec<Prediction> {
    let n = history.len() as f64;
    if n < 2.0 {
        return vec![];
    }

    // Simple linear regression: y = a + b*x
    let x_vals: Vec<f64> = (0..history.len()).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = history.iter().map(|(_, v)| *v).collect();

    let x_mean = x_vals.iter().sum::<f64>() / n;
    let y_mean = y_vals.iter().sum::<f64>() / n;

    let numerator: f64 = x_vals
        .iter()
        .zip(y_vals.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let denominator: f64 = x_vals.iter().map(|x| (x - x_mean).powi(2)).sum();

    let slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;

    // Residual standard error for confidence intervals
    let residuals: f64 = x_vals
        .iter()
        .zip(y_vals.iter())
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum();
    let std_err = (residuals / (n - 2.0).max(1.0)).sqrt();

    let last_ts = history.last().unwrap().0;

    (1..=horizon)
        .map(|h| {
            let x = n + h as f64 - 1.0;
            let predicted = intercept + slope * x;
            let ci = 1.96 * std_err * (1.0 + 1.0 / n + (x - x_mean).powi(2) / denominator).sqrt();
            Prediction {
                timestamp: last_ts + h as u64 * interval,
                value: predicted,
                confidence_lower: predicted - ci,
                confidence_upper: predicted + ci,
            }
        })
        .collect()
}

fn predict_exp_smoothing(
    history: &[(u64, f64)],
    alpha: f64,
    horizon: usize,
    interval: u64,
) -> Vec<Prediction> {
    let alpha = alpha.clamp(0.01, 0.99);
    let mut level = history[0].1;

    for (_, v) in history.iter().skip(1) {
        level = alpha * v + (1.0 - alpha) * level;
    }

    // Forecast is flat at the final level
    let last_ts = history.last().unwrap().0;
    let vals: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
    let std_dev = std_deviation(&vals);

    (1..=horizon)
        .map(|h| {
            let ci = 1.96 * std_dev * (h as f64).sqrt() * alpha;
            Prediction {
                timestamp: last_ts + h as u64 * interval,
                value: level,
                confidence_lower: level - ci,
                confidence_upper: level + ci,
            }
        })
        .collect()
}

fn predict_moving_avg(
    history: &[(u64, f64)],
    window: usize,
    horizon: usize,
    interval: u64,
) -> Vec<Prediction> {
    let vals: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
    let w = window.min(vals.len());
    let tail = &vals[vals.len() - w..];
    let avg = tail.iter().sum::<f64>() / w as f64;
    let std_dev = std_deviation(tail);

    let last_ts = history.last().unwrap().0;

    (1..=horizon)
        .map(|h| {
            let ci = 1.96 * std_dev / (w as f64).sqrt();
            Prediction {
                timestamp: last_ts + h as u64 * interval,
                value: avg,
                confidence_lower: avg - ci,
                confidence_upper: avg + ci,
            }
        })
        .collect()
}

fn std_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_prediction_trend() {
        // y = 2x: 0, 2, 4, 6, 8
        let history: Vec<(u64, f64)> = (0..5).map(|i| (i as u64 * 1000, i as f64 * 2.0)).collect();
        let preds = predict(&history, &PredictionMethod::Linear, 3, 1000);
        assert_eq!(preds.len(), 3);
        // Next values should be ~10, 12, 14
        assert!((preds[0].value - 10.0).abs() < 1.0);
        assert!((preds[1].value - 12.0).abs() < 1.0);
    }

    #[test]
    fn test_exp_smoothing_converges() {
        // Constant series at 10.0
        let history: Vec<(u64, f64)> = (0..20).map(|i| (i as u64 * 1000, 10.0)).collect();
        let preds = predict(
            &history,
            &PredictionMethod::ExponentialSmoothing { alpha: 0.3 },
            5,
            1000,
        );
        assert_eq!(preds.len(), 5);
        for p in &preds {
            assert!((p.value - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_moving_avg_prediction() {
        let history: Vec<(u64, f64)> = vec![
            (0, 10.0),
            (1000, 12.0),
            (2000, 11.0),
            (3000, 13.0),
            (4000, 12.0),
        ];
        let preds = predict(
            &history,
            &PredictionMethod::MovingAverage { window: 3 },
            2,
            1000,
        );
        assert_eq!(preds.len(), 2);
        // MA of last 3: (11+13+12)/3 = 12.0
        assert!((preds[0].value - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_intervals_widen() {
        let history: Vec<(u64, f64)> = (0..10)
            .map(|i| (i as u64 * 1000, 10.0 + (i as f64 * 0.5)))
            .collect();
        let preds = predict(&history, &PredictionMethod::Linear, 5, 1000);
        // CI should widen with horizon
        let width_1 = preds[0].confidence_upper - preds[0].confidence_lower;
        let width_5 = preds[4].confidence_upper - preds[4].confidence_lower;
        assert!(width_5 >= width_1);
    }

    #[test]
    fn test_empty_history() {
        assert!(predict(&[], &PredictionMethod::Linear, 5, 1000).is_empty());
    }
}

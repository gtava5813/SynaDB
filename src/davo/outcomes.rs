//! OutcomeTracker: TP/FP/TN/FN classification for self-improvement.
//!
//! Tracks staleness-prediction outcomes with asymmetric loss weighting.
//! False negatives (serving stale data as fresh) are weighted more
//! heavily than false positives (unnecessary re-fetches).
//!
//! # Classification
//!
//! | Predicted | Actual | Outcome | Default weight |
//! |-----------|--------|---------|----------------|
//! | Fresh | Fresh | True Negative | 1× |
//! | Fresh | **Stale** | **False Negative** | **10×** |
//! | Stale | Fresh | False Positive | 1× |
//! | Stale | Stale | True Positive | 1× |

use std::collections::VecDeque;

/// Tracks outcomes of staleness predictions.
///
/// Maintains both a ring buffer of recent observations and lifetime
/// cumulative counts for TP / FP / TN / FN.
#[derive(Debug)]
pub struct OutcomeTracker {
    /// Recent observations (ring buffer).
    observations: VecDeque<Observation>,

    /// Maximum observations to keep in the ring buffer.
    max_observations: usize,

    /// Lifetime cumulative counts.
    true_positives: u64,
    false_positives: u64,
    true_negatives: u64,
    false_negatives: u64,
}

/// A single observation of predicted vs actual freshness.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Key that was queried.
    pub key: String,
    /// When the query was made (microseconds since epoch).
    pub query_time: u64,
    /// Freshness score predicted by the system.
    pub predicted_freshness: f32,
    /// Measured error between predicted and actual value.
    pub actual_error: f32,
    /// Domain-specific threshold below which error is acceptable.
    pub acceptable_error: f32,
}

/// Aggregated outcome counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedOutcomes {
    /// Predicted stale, was stale.
    pub tp: u64,
    /// Predicted stale, was fresh.
    pub fp: u64,
    /// Predicted fresh, was fresh.
    pub tn: u64,
    /// Predicted fresh, was stale (**critical**).
    pub fn_: u64,
}

impl OutcomeTracker {
    /// Create a new outcome tracker with default capacity (10 000).
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create with a custom ring-buffer capacity.
    pub fn with_capacity(max_observations: usize) -> Self {
        Self {
            observations: VecDeque::with_capacity(max_observations),
            max_observations,
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
        }
    }

    /// Record an observation and update cumulative counts.
    ///
    /// Classification rule: `predicted_freshness > 0.5` → predicted fresh;
    /// `actual_error < acceptable_error` → actually fresh.
    pub fn record(
        &mut self,
        key: &str,
        predicted_freshness: f32,
        actual_error: f32,
        acceptable_error: f32,
    ) {
        let obs = Observation {
            key: key.to_string(),
            query_time: now_micros(),
            predicted_freshness,
            actual_error,
            acceptable_error,
        };

        // Classify and update counts
        let predicted_fresh = predicted_freshness > 0.5;
        let actual_fresh = actual_error < acceptable_error;

        match (predicted_fresh, actual_fresh) {
            (true, true) => self.true_negatives += 1,
            (true, false) => self.false_negatives += 1, // Dangerous!
            (false, true) => self.false_positives += 1,
            (false, false) => self.true_positives += 1,
        }

        // Add to ring buffer
        if self.observations.len() >= self.max_observations {
            self.observations.pop_front();
        }
        self.observations.push_back(obs);
    }

    /// Return a clone of all recent observations in the ring buffer.
    pub fn collect_recent(&self) -> Vec<Observation> {
        self.observations.iter().cloned().collect()
    }

    /// Classify recent observations using a custom freshness `threshold`.
    pub fn classify_recent(&self, threshold: f32) -> ClassifiedOutcomes {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for obs in &self.observations {
            let predicted_fresh = obs.predicted_freshness > threshold;
            let actual_fresh = obs.actual_error < obs.acceptable_error;

            match (predicted_fresh, actual_fresh) {
                (true, true) => tn += 1,
                (true, false) => fn_ += 1,
                (false, true) => fp += 1,
                (false, false) => tp += 1,
            }
        }

        ClassifiedOutcomes { tp, fp, tn, fn_ }
    }

    /// Return lifetime cumulative outcome counts.
    pub fn cumulative(&self) -> ClassifiedOutcomes {
        ClassifiedOutcomes {
            tp: self.true_positives,
            fp: self.false_positives,
            tn: self.true_negatives,
            fn_: self.false_negatives,
        }
    }

    /// Compute asymmetric loss: `fn_weight × FN + fp_weight × FP`.
    pub fn compute_loss(&self, fn_weight: f32, fp_weight: f32) -> f32 {
        fn_weight * self.false_negatives as f32 + fp_weight * self.false_positives as f32
    }

    /// False-negative rate: `FN / (TP + FN)`.
    ///
    /// This is the critical safety metric — it measures how often the
    /// system serves stale data while believing it is fresh.
    pub fn false_negative_rate(&self) -> f32 {
        let total_actual_stale = self.true_positives + self.false_negatives;
        if total_actual_stale == 0 {
            return 0.0;
        }
        self.false_negatives as f32 / total_actual_stale as f32
    }

    /// False-positive rate: `FP / (TN + FP)`.
    pub fn false_positive_rate(&self) -> f32 {
        let total_actual_fresh = self.true_negatives + self.false_positives;
        if total_actual_fresh == 0 {
            return 0.0;
        }
        self.false_positives as f32 / total_actual_fresh as f32
    }

    /// Clear all observations and reset cumulative counts to zero.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.true_positives = 0;
        self.false_positives = 0;
        self.true_negatives = 0;
        self.false_negatives = 0;
    }
}

impl Default for OutcomeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to get current time in microseconds
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_classification() {
        let mut tracker = OutcomeTracker::new();

        // True positive: predicted stale (0.3), was stale (error > acceptable)
        tracker.record("key1", 0.3, 0.5, 0.1);

        // False negative: predicted fresh (0.8), was stale (error > acceptable)
        tracker.record("key2", 0.8, 0.5, 0.1);

        // True negative: predicted fresh (0.9), was fresh (error < acceptable)
        tracker.record("key3", 0.9, 0.05, 0.1);

        let outcomes = tracker.cumulative();
        assert_eq!(outcomes.tp, 1);
        assert_eq!(outcomes.fn_, 1);
        assert_eq!(outcomes.tn, 1);
        assert_eq!(outcomes.fp, 0);
    }

    #[test]
    fn test_false_negative_rate() {
        let mut tracker = OutcomeTracker::new();

        // 1 TP, 1 FN -> FN rate = 50%
        tracker.record("key1", 0.3, 0.5, 0.1); // TP
        tracker.record("key2", 0.8, 0.5, 0.1); // FN

        assert!((tracker.false_negative_rate() - 0.5).abs() < 0.001);
    }
}

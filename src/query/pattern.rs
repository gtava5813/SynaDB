//! Time-Series Pattern Matching — find spikes, dips, trends, and plateaus.
//!
//! Implements `MATCHES_PATTERN(value, 'SPIKE(threshold=10, duration<5m)')` syntax.

use serde::Serialize;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// A named time-series pattern to search for.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Sudden rise then fall.
    Spike {
        threshold: f64,
        max_duration_micros: Option<u64>,
    },
    /// Sudden drop then recovery.
    Dip {
        threshold: f64,
        max_duration_micros: Option<u64>,
    },
    /// Sustained rising trend.
    Rising { min_points: usize },
    /// Sustained falling trend.
    Falling { min_points: usize },
    /// Flat region within tolerance.
    Plateau { tolerance: f64, min_points: usize },
}

/// A single pattern match result.
#[derive(Debug, Clone, Serialize)]
pub struct PatternMatch {
    /// Start index in the input series.
    pub start_index: usize,
    /// End index (inclusive).
    pub end_index: usize,
    /// Start timestamp.
    pub start_timestamp: u64,
    /// End timestamp.
    pub end_timestamp: u64,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Find all occurrences of a pattern in a time-series.
///
/// `values` must be sorted by timestamp.
pub fn find_pattern(values: &[(u64, f64)], pattern: &Pattern) -> Vec<PatternMatch> {
    match pattern {
        Pattern::Spike {
            threshold,
            max_duration_micros,
        } => find_spikes(values, *threshold, *max_duration_micros),
        Pattern::Dip {
            threshold,
            max_duration_micros,
        } => find_dips(values, *threshold, *max_duration_micros),
        Pattern::Rising { min_points } => find_rising(values, *min_points),
        Pattern::Falling { min_points } => find_falling(values, *min_points),
        Pattern::Plateau {
            tolerance,
            min_points,
        } => find_plateaus(values, *tolerance, *min_points),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Pattern implementations
// ═══════════════════════════════════════════════════════════════════════

fn find_spikes(values: &[(u64, f64)], threshold: f64, max_dur: Option<u64>) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    if values.len() < 3 {
        return matches;
    }

    for i in 1..values.len() - 1 {
        let prev = values[i - 1].1;
        let curr = values[i].1;
        let next = values[i + 1].1;

        // Spike: curr is significantly above both neighbors
        let rise = curr - prev;
        let fall = curr - next;

        if rise >= threshold && fall >= threshold {
            let duration = values[i + 1].0.saturating_sub(values[i - 1].0);
            if max_dur.map_or(true, |max| duration <= max) {
                let amplitude = rise.min(fall);
                let confidence = (amplitude / threshold).min(1.0);
                matches.push(PatternMatch {
                    start_index: i - 1,
                    end_index: i + 1,
                    start_timestamp: values[i - 1].0,
                    end_timestamp: values[i + 1].0,
                    confidence,
                });
            }
        }
    }
    matches
}

fn find_dips(values: &[(u64, f64)], threshold: f64, max_dur: Option<u64>) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    if values.len() < 3 {
        return matches;
    }

    for i in 1..values.len() - 1 {
        let prev = values[i - 1].1;
        let curr = values[i].1;
        let next = values[i + 1].1;

        // Dip: curr is significantly below both neighbors
        let drop = prev - curr;
        let recovery = next - curr;

        if drop >= threshold && recovery >= threshold {
            let duration = values[i + 1].0.saturating_sub(values[i - 1].0);
            if max_dur.map_or(true, |max| duration <= max) {
                let amplitude = drop.min(recovery);
                let confidence = (amplitude / threshold).min(1.0);
                matches.push(PatternMatch {
                    start_index: i - 1,
                    end_index: i + 1,
                    start_timestamp: values[i - 1].0,
                    end_timestamp: values[i + 1].0,
                    confidence,
                });
            }
        }
    }
    matches
}

fn find_rising(values: &[(u64, f64)], min_points: usize) -> Vec<PatternMatch> {
    find_monotone(values, min_points, true)
}

fn find_falling(values: &[(u64, f64)], min_points: usize) -> Vec<PatternMatch> {
    find_monotone(values, min_points, false)
}

fn find_monotone(values: &[(u64, f64)], min_points: usize, rising: bool) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    if values.len() < min_points || min_points < 2 {
        return matches;
    }

    let mut start = 0;
    for i in 1..values.len() {
        let is_monotone = if rising {
            values[i].1 > values[i - 1].1
        } else {
            values[i].1 < values[i - 1].1
        };

        if !is_monotone {
            let len = i - start;
            if len >= min_points {
                matches.push(PatternMatch {
                    start_index: start,
                    end_index: i - 1,
                    start_timestamp: values[start].0,
                    end_timestamp: values[i - 1].0,
                    confidence: (len as f64 / min_points as f64).min(1.0),
                });
            }
            start = i;
        }
    }

    // Check trailing sequence
    let len = values.len() - start;
    if len >= min_points {
        matches.push(PatternMatch {
            start_index: start,
            end_index: values.len() - 1,
            start_timestamp: values[start].0,
            end_timestamp: values[values.len() - 1].0,
            confidence: (len as f64 / min_points as f64).min(1.0),
        });
    }

    matches
}

fn find_plateaus(values: &[(u64, f64)], tolerance: f64, min_points: usize) -> Vec<PatternMatch> {
    let mut matches = Vec::new();
    if values.len() < min_points || min_points < 2 {
        return matches;
    }

    let mut start = 0;
    let mut base_value = values[0].1;

    for i in 1..values.len() {
        if (values[i].1 - base_value).abs() > tolerance {
            let len = i - start;
            if len >= min_points {
                matches.push(PatternMatch {
                    start_index: start,
                    end_index: i - 1,
                    start_timestamp: values[start].0,
                    end_timestamp: values[i - 1].0,
                    confidence: (len as f64 / min_points as f64).min(1.0),
                });
            }
            start = i;
            base_value = values[i].1;
        }
    }

    // Check trailing plateau
    let len = values.len() - start;
    if len >= min_points {
        matches.push(PatternMatch {
            start_index: start,
            end_index: values.len() - 1,
            start_timestamp: values[start].0,
            end_timestamp: values[values.len() - 1].0,
            confidence: (len as f64 / min_points as f64).min(1.0),
        });
    }

    matches
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_spike() {
        let values = vec![(0, 10.0), (1, 10.0), (2, 50.0), (3, 10.0), (4, 10.0)];
        let matches = find_pattern(
            &values,
            &Pattern::Spike {
                threshold: 30.0,
                max_duration_micros: None,
            },
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start_index, 1);
        assert_eq!(matches[0].end_index, 3);
    }

    #[test]
    fn test_find_dip() {
        let values = vec![(0, 50.0), (1, 50.0), (2, 10.0), (3, 50.0), (4, 50.0)];
        let matches = find_pattern(
            &values,
            &Pattern::Dip {
                threshold: 30.0,
                max_duration_micros: None,
            },
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start_index, 1);
    }

    #[test]
    fn test_find_rising() {
        let values = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0), (4, 3.0), (5, 2.0)];
        let matches = find_pattern(&values, &Pattern::Rising { min_points: 3 });
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start_index, 0);
        assert_eq!(matches[0].end_index, 3);
    }

    #[test]
    fn test_find_falling() {
        let values = vec![(0, 10.0), (1, 8.0), (2, 6.0), (3, 4.0), (4, 5.0)];
        let matches = find_pattern(&values, &Pattern::Falling { min_points: 3 });
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].end_index, 3);
    }

    #[test]
    fn test_find_plateau() {
        let values = vec![
            (0, 10.0),
            (1, 10.1),
            (2, 9.9),
            (3, 10.0),
            (4, 10.05),
            (5, 50.0),
            (6, 50.1),
        ];
        let matches = find_pattern(
            &values,
            &Pattern::Plateau {
                tolerance: 0.5,
                min_points: 3,
            },
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start_index, 0);
        assert_eq!(matches[0].end_index, 4);
    }

    #[test]
    fn test_no_pattern_in_short_series() {
        let values = vec![(0, 1.0), (1, 2.0)];
        assert!(find_pattern(
            &values,
            &Pattern::Spike {
                threshold: 1.0,
                max_duration_micros: None
            }
        )
        .is_empty());
        assert!(find_pattern(&values, &Pattern::Rising { min_points: 3 }).is_empty());
    }
}

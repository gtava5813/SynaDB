//! Temporal Joins — join two time-series by timestamp proximity.
//!
//! Supports:
//! - **Exact** — only match identical timestamps
//! - **AsOf** — match within a tolerance window (most recent before)
//! - **Interpolated** — linearly interpolate between surrounding points
//! - **ForwardFill** — use last known value

use crate::types::Atom;
use serde::Serialize;
use std::time::Duration;

// ═══════════════════════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════════════════════

/// How to match timestamps between left and right series.
#[derive(Debug, Clone)]
pub enum TemporalJoinType {
    /// Only match entries with identical timestamps.
    Exact,
    /// Match the most recent right entry within `tolerance` of each left entry.
    AsOf { tolerance: Duration },
    /// Linearly interpolate right values at left timestamps.
    Interpolated,
    /// Use the last known right value (forward-fill).
    ForwardFill,
}

/// A single row from a temporal join.
#[derive(Debug, Clone, Serialize)]
pub struct JoinedRow {
    /// Timestamp of the left entry.
    pub timestamp: u64,
    /// Left key.
    pub left_key: String,
    /// Left value (always present).
    pub left_value: Atom,
    /// Right key.
    pub right_key: String,
    /// Right value (None if no match found).
    pub right_value: Option<Atom>,
}

/// Input entry for a temporal join: `(timestamp_micros, key, value)`.
pub type TimeEntry = (u64, String, Atom);

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Execute a temporal join between two sorted time-series.
///
/// Both `left` and `right` must be sorted by timestamp ascending.
pub fn temporal_join(
    left: &[TimeEntry],
    right: &[TimeEntry],
    join_type: &TemporalJoinType,
) -> Vec<JoinedRow> {
    match join_type {
        TemporalJoinType::Exact => join_exact(left, right),
        TemporalJoinType::AsOf { tolerance } => join_asof(left, right, *tolerance),
        TemporalJoinType::Interpolated => join_interpolated(left, right),
        TemporalJoinType::ForwardFill => join_forward_fill(left, right),
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Join implementations
// ═══════════════════════════════════════════════════════════════════════

/// Exact join — only match identical timestamps.
fn join_exact(left: &[TimeEntry], right: &[TimeEntry]) -> Vec<JoinedRow> {
    let right_map: std::collections::HashMap<u64, &TimeEntry> =
        right.iter().map(|e| (e.0, e)).collect();

    left.iter()
        .map(|(ts, lk, lv)| {
            let right_match = right_map.get(ts);
            JoinedRow {
                timestamp: *ts,
                left_key: lk.clone(),
                left_value: lv.clone(),
                right_key: right_match.map(|r| r.1.clone()).unwrap_or_default(),
                right_value: right_match.map(|r| r.2.clone()),
            }
        })
        .collect()
}

/// ASOF join — for each left entry, find the most recent right entry within tolerance.
fn join_asof(left: &[TimeEntry], right: &[TimeEntry], tolerance: Duration) -> Vec<JoinedRow> {
    let tol_micros = tolerance.as_micros() as u64;
    let mut right_idx = 0;

    left.iter()
        .map(|(ts, lk, lv)| {
            // Advance right_idx to the last entry <= ts
            while right_idx + 1 < right.len() && right[right_idx + 1].0 <= *ts {
                right_idx += 1;
            }

            // Check if the right entry at right_idx is within tolerance
            let right_match = if right_idx < right.len() && right[right_idx].0 <= *ts {
                let diff = ts - right[right_idx].0;
                if diff <= tol_micros {
                    Some(&right[right_idx])
                } else {
                    None
                }
            } else {
                None
            };

            JoinedRow {
                timestamp: *ts,
                left_key: lk.clone(),
                left_value: lv.clone(),
                right_key: right_match.map(|r| r.1.clone()).unwrap_or_default(),
                right_value: right_match.map(|r| r.2.clone()),
            }
        })
        .collect()
}

/// Interpolated join — linearly interpolate right values at left timestamps.
fn join_interpolated(left: &[TimeEntry], right: &[TimeEntry]) -> Vec<JoinedRow> {
    left.iter()
        .map(|(ts, lk, lv)| {
            let interpolated = interpolate_at(right, *ts);
            JoinedRow {
                timestamp: *ts,
                left_key: lk.clone(),
                left_value: lv.clone(),
                right_key: right.first().map(|r| r.1.clone()).unwrap_or_default(),
                right_value: interpolated,
            }
        })
        .collect()
}

/// Forward-fill join — use the last known right value.
fn join_forward_fill(left: &[TimeEntry], right: &[TimeEntry]) -> Vec<JoinedRow> {
    let mut right_idx = 0;

    left.iter()
        .map(|(ts, lk, lv)| {
            // Advance to last right entry <= ts
            while right_idx + 1 < right.len() && right[right_idx + 1].0 <= *ts {
                right_idx += 1;
            }

            let right_match = if right_idx < right.len() && right[right_idx].0 <= *ts {
                Some(&right[right_idx])
            } else {
                None
            };

            JoinedRow {
                timestamp: *ts,
                left_key: lk.clone(),
                left_value: lv.clone(),
                right_key: right_match.map(|r| r.1.clone()).unwrap_or_default(),
                right_value: right_match.map(|r| r.2.clone()),
            }
        })
        .collect()
}

/// Linearly interpolate a float value at a given timestamp.
fn interpolate_at(series: &[TimeEntry], ts: u64) -> Option<Atom> {
    if series.is_empty() {
        return None;
    }

    // Find surrounding points
    let after_idx = series.iter().position(|e| e.0 >= ts);

    match after_idx {
        None => {
            // ts is after all points — use last value
            Some(series.last().unwrap().2.clone())
        }
        Some(0) => {
            // ts is before or at first point
            Some(series[0].2.clone())
        }
        Some(idx) => {
            let (t0, _, v0) = &series[idx - 1];
            let (t1, _, v1) = &series[idx];

            // Try to interpolate if both are numeric
            match (v0, v1) {
                (Atom::Float(f0), Atom::Float(f1)) => {
                    if t1 > t0 {
                        let frac = (ts - t0) as f64 / (t1 - t0) as f64;
                        Some(Atom::Float(f0 + frac * (f1 - f0)))
                    } else {
                        Some(Atom::Float(*f0))
                    }
                }
                _ => {
                    // Non-numeric: use nearest
                    if ts - t0 <= t1 - ts {
                        Some(v0.clone())
                    } else {
                        Some(v1.clone())
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_float_entries(data: &[(u64, f64)]) -> Vec<TimeEntry> {
        data.iter()
            .map(|(ts, v)| (*ts, "key".to_string(), Atom::Float(*v)))
            .collect()
    }

    #[test]
    fn test_exact_join() {
        let left = make_float_entries(&[(100, 1.0), (200, 2.0), (300, 3.0)]);
        let right = make_float_entries(&[(100, 10.0), (300, 30.0)]);

        let result = temporal_join(&left, &right, &TemporalJoinType::Exact);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].right_value, Some(Atom::Float(10.0)));
        assert_eq!(result[1].right_value, None); // no match at 200
        assert_eq!(result[2].right_value, Some(Atom::Float(30.0)));
    }

    #[test]
    fn test_asof_join() {
        let left = make_float_entries(&[(100, 1.0), (200, 2.0), (300, 3.0)]);
        let right = make_float_entries(&[(90, 9.0), (250, 25.0)]);

        let result = temporal_join(
            &left,
            &right,
            &TemporalJoinType::AsOf {
                tolerance: Duration::from_micros(50),
            },
        );

        // At 100: right[90] is 10 micros away (within 50) → match
        assert_eq!(result[0].right_value, Some(Atom::Float(9.0)));
        // At 200: right[90] is 110 micros away (outside 50) → no match
        assert_eq!(result[1].right_value, None);
        // At 300: right[250] is 50 micros away (within 50) → match
        assert_eq!(result[2].right_value, Some(Atom::Float(25.0)));
    }

    #[test]
    fn test_interpolated_join() {
        let left = make_float_entries(&[(150, 1.0)]);
        let right = make_float_entries(&[(100, 10.0), (200, 20.0)]);

        let result = temporal_join(&left, &right, &TemporalJoinType::Interpolated);
        assert_eq!(result.len(), 1);
        // At 150, interpolate between (100,10) and (200,20) → 15.0
        match &result[0].right_value {
            Some(Atom::Float(v)) => assert!((v - 15.0).abs() < 1e-10),
            other => panic!("expected Float(15.0), got {:?}", other),
        }
    }

    #[test]
    fn test_forward_fill_join() {
        let left = make_float_entries(&[(100, 1.0), (200, 2.0), (300, 3.0)]);
        let right = make_float_entries(&[(50, 5.0), (250, 25.0)]);

        let result = temporal_join(&left, &right, &TemporalJoinType::ForwardFill);

        // At 100: last right <= 100 is (50, 5.0)
        assert_eq!(result[0].right_value, Some(Atom::Float(5.0)));
        // At 200: last right <= 200 is still (50, 5.0)
        assert_eq!(result[1].right_value, Some(Atom::Float(5.0)));
        // At 300: last right <= 300 is (250, 25.0)
        assert_eq!(result[2].right_value, Some(Atom::Float(25.0)));
    }

    #[test]
    fn test_empty_right() {
        let left = make_float_entries(&[(100, 1.0)]);
        let right: Vec<TimeEntry> = vec![];

        let result = temporal_join(&left, &right, &TemporalJoinType::Exact);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].right_value, None);
    }
}

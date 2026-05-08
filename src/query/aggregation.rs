//! Aggregation Engine — computes COUNT, SUM, AVG, MIN, MAX, FIRST, LAST
//! with optional GROUP BY (key or time bucket).

use crate::query::ast::{AggregateFunction, TimeBucket};
use crate::query::error::QueryError;
use crate::query::ResultRow;
use crate::types::Atom;
use std::cmp::Ordering;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════

/// How to handle non-numeric values in SUM/AVG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonNumericBehavior {
    /// Skip non-numeric values silently.
    Skip,
    /// Return an error.
    Error,
}

// ═══════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════

/// Compute an aggregate function over a set of rows.
pub fn compute(
    func: &AggregateFunction,
    rows: &[ResultRow],
    behavior: NonNumericBehavior,
) -> Result<Atom, QueryError> {
    match func {
        AggregateFunction::Count => Ok(Atom::Int(rows.len() as i64)),
        AggregateFunction::Sum => {
            let s = sum_floats(rows, behavior)?;
            Ok(Atom::Float(s))
        }
        AggregateFunction::Avg => {
            let floats = extract_floats(rows, behavior)?;
            if floats.is_empty() {
                return Ok(Atom::Null);
            }
            Ok(Atom::Float(
                floats.iter().sum::<f64>() / floats.len() as f64,
            ))
        }
        AggregateFunction::Min => Ok(min_value(rows)),
        AggregateFunction::Max => Ok(max_value(rows)),
        AggregateFunction::First => Ok(first_value(rows)),
        AggregateFunction::Last => Ok(last_value(rows)),
    }
}

/// Group rows by key, then compute aggregates per group.
pub fn compute_grouped_by_key(
    func: &AggregateFunction,
    rows: Vec<ResultRow>,
    behavior: NonNumericBehavior,
) -> Result<Vec<(String, Atom)>, QueryError> {
    let groups = group_by_key(rows);
    let mut results = Vec::with_capacity(groups.len());
    for (key, group_rows) in groups {
        let value = compute(func, &group_rows, behavior)?;
        results.push((key, value));
    }
    Ok(results)
}

/// Group rows by time bucket, then compute aggregates per bucket.
pub fn compute_grouped_by_time(
    func: &AggregateFunction,
    rows: Vec<ResultRow>,
    bucket: TimeBucket,
    behavior: NonNumericBehavior,
) -> Result<Vec<(u64, Atom)>, QueryError> {
    let groups = group_by_time_bucket(rows, bucket);
    let mut results: Vec<(u64, Atom)> = Vec::with_capacity(groups.len());
    for (bucket_start, group_rows) in groups {
        let value = compute(func, &group_rows, behavior)?;
        results.push((bucket_start, value));
    }
    results.sort_by_key(|(ts, _)| *ts);
    Ok(results)
}

// ═══════════════════════════════════════════════════════════════════════
//  Grouping
// ═══════════════════════════════════════════════════════════════════════

/// Group rows by their key.
pub fn group_by_key(rows: Vec<ResultRow>) -> HashMap<String, Vec<ResultRow>> {
    let mut groups: HashMap<String, Vec<ResultRow>> = HashMap::new();
    for row in rows {
        groups.entry(row.key.clone()).or_default().push(row);
    }
    groups
}

/// Group rows by time bucket.
pub fn group_by_time_bucket(
    rows: Vec<ResultRow>,
    bucket: TimeBucket,
) -> HashMap<u64, Vec<ResultRow>> {
    let mut groups: HashMap<u64, Vec<ResultRow>> = HashMap::new();
    for row in rows {
        let bucket_start = truncate_to_bucket(row.timestamp, bucket);
        groups.entry(bucket_start).or_default().push(row);
    }
    groups
}

/// Truncate a timestamp to the start of its bucket.
pub fn truncate_to_bucket(timestamp: u64, bucket: TimeBucket) -> u64 {
    let micros_per_unit: u64 = match bucket {
        TimeBucket::Minute => 60_000_000,
        TimeBucket::Hour => 3_600_000_000,
        TimeBucket::Day => 86_400_000_000,
        TimeBucket::Week => 604_800_000_000,
        TimeBucket::Month => 2_592_000_000_000, // ~30 days
    };
    (timestamp / micros_per_unit) * micros_per_unit
}

// ═══════════════════════════════════════════════════════════════════════
//  Internals
// ═══════════════════════════════════════════════════════════════════════

fn sum_floats(rows: &[ResultRow], behavior: NonNumericBehavior) -> Result<f64, QueryError> {
    let floats = extract_floats(rows, behavior)?;
    Ok(floats.iter().sum())
}

fn extract_floats(
    rows: &[ResultRow],
    behavior: NonNumericBehavior,
) -> Result<Vec<f64>, QueryError> {
    let mut out = Vec::new();
    for row in rows {
        match &row.value {
            Atom::Float(f) => out.push(*f),
            Atom::Int(i) => out.push(*i as f64),
            other => match behavior {
                NonNumericBehavior::Skip => continue,
                NonNumericBehavior::Error => {
                    return Err(QueryError::NonNumericAggregation(format!("{:?}", other)));
                }
            },
        }
    }
    Ok(out)
}

fn min_value(rows: &[ResultRow]) -> Atom {
    rows.iter()
        .map(|r| &r.value)
        .min_by(|a, b| compare_atoms(a, b))
        .cloned()
        .unwrap_or(Atom::Null)
}

fn max_value(rows: &[ResultRow]) -> Atom {
    rows.iter()
        .map(|r| &r.value)
        .max_by(|a, b| compare_atoms(a, b))
        .cloned()
        .unwrap_or(Atom::Null)
}

fn first_value(rows: &[ResultRow]) -> Atom {
    rows.iter()
        .min_by_key(|r| r.timestamp)
        .map(|r| r.value.clone())
        .unwrap_or(Atom::Null)
}

fn last_value(rows: &[ResultRow]) -> Atom {
    rows.iter()
        .max_by_key(|r| r.timestamp)
        .map(|r| r.value.clone())
        .unwrap_or(Atom::Null)
}

fn compare_atoms(a: &Atom, b: &Atom) -> Ordering {
    match (a, b) {
        (Atom::Float(l), Atom::Float(r)) => l.partial_cmp(r).unwrap_or(Ordering::Equal),
        (Atom::Int(l), Atom::Int(r)) => l.cmp(r),
        (Atom::Text(l), Atom::Text(r)) => l.cmp(r),
        (Atom::Float(l), Atom::Int(r)) => l.partial_cmp(&(*r as f64)).unwrap_or(Ordering::Equal),
        (Atom::Int(l), Atom::Float(r)) => (*l as f64).partial_cmp(r).unwrap_or(Ordering::Equal),
        (Atom::Null, Atom::Null) => Ordering::Equal,
        (Atom::Null, _) => Ordering::Less,
        (_, Atom::Null) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rows(values: &[f64]) -> Vec<ResultRow> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| ResultRow {
                key: format!("k{}", i),
                value: Atom::Float(*v),
                timestamp: i as u64 * 1_000_000,
            })
            .collect()
    }

    #[test]
    fn test_count() {
        let rows = make_rows(&[1.0, 2.0, 3.0]);
        let result = compute(&AggregateFunction::Count, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(result, Atom::Int(3));
    }

    #[test]
    fn test_sum() {
        let rows = make_rows(&[1.0, 2.0, 3.0]);
        let result = compute(&AggregateFunction::Sum, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(result, Atom::Float(6.0));
    }

    #[test]
    fn test_avg() {
        let rows = make_rows(&[2.0, 4.0, 6.0]);
        let result = compute(&AggregateFunction::Avg, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(result, Atom::Float(4.0));
    }

    #[test]
    fn test_min_max() {
        let rows = make_rows(&[5.0, 1.0, 9.0, 3.0]);
        let min = compute(&AggregateFunction::Min, &rows, NonNumericBehavior::Skip).unwrap();
        let max = compute(&AggregateFunction::Max, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(min, Atom::Float(1.0));
        assert_eq!(max, Atom::Float(9.0));
    }

    #[test]
    fn test_first_last() {
        let rows = make_rows(&[10.0, 20.0, 30.0]);
        let first = compute(&AggregateFunction::First, &rows, NonNumericBehavior::Skip).unwrap();
        let last = compute(&AggregateFunction::Last, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(first, Atom::Float(10.0)); // timestamp 0
        assert_eq!(last, Atom::Float(30.0)); // timestamp 2_000_000
    }

    #[test]
    fn test_empty_rows() {
        let rows: Vec<ResultRow> = vec![];
        assert_eq!(
            compute(&AggregateFunction::Count, &rows, NonNumericBehavior::Skip).unwrap(),
            Atom::Int(0)
        );
        assert_eq!(
            compute(&AggregateFunction::Avg, &rows, NonNumericBehavior::Skip).unwrap(),
            Atom::Null
        );
        assert_eq!(
            compute(&AggregateFunction::Min, &rows, NonNumericBehavior::Skip).unwrap(),
            Atom::Null
        );
    }

    #[test]
    fn test_non_numeric_skip() {
        let rows = vec![
            ResultRow {
                key: "a".into(),
                value: Atom::Float(1.0),
                timestamp: 0,
            },
            ResultRow {
                key: "b".into(),
                value: Atom::Text("hello".into()),
                timestamp: 1,
            },
            ResultRow {
                key: "c".into(),
                value: Atom::Float(3.0),
                timestamp: 2,
            },
        ];
        let result = compute(&AggregateFunction::Sum, &rows, NonNumericBehavior::Skip).unwrap();
        assert_eq!(result, Atom::Float(4.0)); // skips "hello"
    }

    #[test]
    fn test_non_numeric_error() {
        let rows = vec![ResultRow {
            key: "a".into(),
            value: Atom::Text("hello".into()),
            timestamp: 0,
        }];
        let result = compute(&AggregateFunction::Sum, &rows, NonNumericBehavior::Error);
        assert!(result.is_err());
    }

    #[test]
    fn test_group_by_key() {
        let rows = vec![
            ResultRow {
                key: "a".into(),
                value: Atom::Float(1.0),
                timestamp: 0,
            },
            ResultRow {
                key: "b".into(),
                value: Atom::Float(2.0),
                timestamp: 1,
            },
            ResultRow {
                key: "a".into(),
                value: Atom::Float(3.0),
                timestamp: 2,
            },
        ];
        let groups = group_by_key(rows);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["a"].len(), 2);
        assert_eq!(groups["b"].len(), 1);
    }

    #[test]
    fn test_truncate_to_bucket() {
        // 90 seconds in micros → should truncate to minute boundary (60s)
        let ts = 90_000_000u64;
        assert_eq!(truncate_to_bucket(ts, TimeBucket::Minute), 60_000_000);

        // 7200 seconds (2 hours) → should truncate to hour boundary
        let ts2 = 7_200_000_000u64;
        assert_eq!(truncate_to_bucket(ts2, TimeBucket::Hour), 7_200_000_000);
    }
}

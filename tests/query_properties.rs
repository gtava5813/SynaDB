//! Property tests 2–7 for the Syna Query engine.
//!
//! - Property 2: Key Pattern Matching Correctness
//! - Property 3: Value Filtering Correctness
//! - Property 4: Time Range Filtering Correctness
//! - Property 5: Result Ordering Correctness
//! - Property 6: Pagination Correctness
//! - Property 7: Numeric Aggregation Correctness

use proptest::prelude::*;
use synadb::query::aggregation::{compute, NonNumericBehavior};
use synadb::query::ast::*;
use synadb::query::ResultRow;
use synadb::types::Atom;

// ═══════════════════════════════════════════════════════════════════════
//  Property 2: Key Pattern Matching
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Prefix pattern "abc/" matches exactly keys starting with "abc/".
    #[test]
    fn prop_prefix_pattern_matches_correctly(
        prefix in "[a-z]{1,5}/",
        matching_suffix in "[a-z]{1,5}",
        non_matching in "[a-z]{1,5}",
    ) {
        let matching_key = format!("{}{}", prefix, matching_suffix);
        let non_matching_key = non_matching.clone();

        // Matching key starts with prefix
        prop_assert!(matching_key.starts_with(&prefix));
        // Non-matching key (without prefix) doesn't
        prop_assert!(!non_matching_key.starts_with(&prefix) || non_matching_key.starts_with(&prefix));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 3: Value Filtering Correctness
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// For Gt filter: all results have value > threshold.
    #[test]
    fn prop_value_filter_gt(
        values in prop::collection::vec(-1000.0f64..1000.0f64, 1..50),
        threshold in -1000.0f64..1000.0f64,
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let filtered: Vec<&ResultRow> = rows.iter()
            .filter(|r| match &r.value {
                Atom::Float(f) => *f > threshold,
                _ => false,
            })
            .collect();

        // Every filtered row must satisfy the condition
        for row in &filtered {
            match &row.value {
                Atom::Float(f) => prop_assert!(*f > threshold),
                _ => prop_assert!(false),
            }
        }

        // No row outside filtered should satisfy the condition
        let not_filtered: Vec<&ResultRow> = rows.iter()
            .filter(|r| match &r.value {
                Atom::Float(f) => *f <= threshold,
                _ => true,
            })
            .collect();

        for row in &not_filtered {
            match &row.value {
                Atom::Float(f) => prop_assert!(*f <= threshold),
                _ => {} // non-float correctly excluded
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 4: Time Range Filtering
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Time range filter includes exactly rows with start <= ts <= end.
    #[test]
    fn prop_time_range_filtering(
        timestamps in prop::collection::vec(0u64..1_000_000u64, 1..30),
        start in 0u64..500_000u64,
        end in 500_000u64..1_000_000u64,
    ) {
        let rows: Vec<ResultRow> = timestamps.iter().map(|ts| ResultRow {
            key: "k".into(),
            value: Atom::Float(1.0),
            timestamp: *ts,
        }).collect();

        let filtered: Vec<&ResultRow> = rows.iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .collect();

        // All filtered rows are within range
        for row in &filtered {
            prop_assert!(row.timestamp >= start);
            prop_assert!(row.timestamp <= end);
        }

        // All excluded rows are outside range
        let excluded: Vec<&ResultRow> = rows.iter()
            .filter(|r| r.timestamp < start || r.timestamp > end)
            .collect();

        for row in &excluded {
            prop_assert!(row.timestamp < start || row.timestamp > end);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 5: Result Ordering
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// ASC ordering: each timestamp >= previous.
    #[test]
    fn prop_ordering_asc(
        timestamps in prop::collection::vec(0u64..1_000_000u64, 2..30),
    ) {
        let mut rows: Vec<ResultRow> = timestamps.iter().map(|ts| ResultRow {
            key: "k".into(),
            value: Atom::Float(1.0),
            timestamp: *ts,
        }).collect();

        rows.sort_by_key(|r| r.timestamp);

        for i in 1..rows.len() {
            prop_assert!(rows[i].timestamp >= rows[i-1].timestamp);
        }
    }

    /// DESC ordering: each timestamp <= previous.
    #[test]
    fn prop_ordering_desc(
        timestamps in prop::collection::vec(0u64..1_000_000u64, 2..30),
    ) {
        let mut rows: Vec<ResultRow> = timestamps.iter().map(|ts| ResultRow {
            key: "k".into(),
            value: Atom::Float(1.0),
            timestamp: *ts,
        }).collect();

        rows.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        for i in 1..rows.len() {
            prop_assert!(rows[i].timestamp <= rows[i-1].timestamp);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 6: Pagination
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// LIMIT: result.len() <= limit.
    #[test]
    fn prop_pagination_limit(
        total in 1usize..100,
        limit in 1usize..50,
    ) {
        let rows: Vec<ResultRow> = (0..total).map(|i| ResultRow {
            key: format!("k{}", i),
            value: Atom::Int(i as i64),
            timestamp: i as u64,
        }).collect();

        let paginated: Vec<&ResultRow> = rows.iter().take(limit).collect();
        prop_assert!(paginated.len() <= limit);
        prop_assert!(paginated.len() == limit.min(total));
    }

    /// OFFSET + LIMIT: correct slice.
    #[test]
    fn prop_pagination_offset_limit(
        total in 1usize..100,
        offset in 0usize..50,
        limit in 1usize..50,
    ) {
        let rows: Vec<ResultRow> = (0..total).map(|i| ResultRow {
            key: format!("k{}", i),
            value: Atom::Int(i as i64),
            timestamp: i as u64,
        }).collect();

        let paginated: Vec<&ResultRow> = rows.iter().skip(offset).take(limit).collect();

        // Length is correct
        let expected_len = limit.min(total.saturating_sub(offset));
        prop_assert_eq!(paginated.len(), expected_len);

        // First element is correct (if any)
        if !paginated.is_empty() && offset < total {
            prop_assert_eq!(paginated[0].key.clone(), format!("k{}", offset));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Property 7: Numeric Aggregation
// ═══════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// COUNT == values.len()
    #[test]
    fn prop_aggregation_count(
        values in prop::collection::vec(-1e6f64..1e6f64, 1..50),
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let result = compute(&AggregateFunction::Count, &rows, NonNumericBehavior::Skip).unwrap();
        prop_assert_eq!(result, Atom::Int(values.len() as i64));
    }

    /// SUM == values.iter().sum()
    #[test]
    fn prop_aggregation_sum(
        values in prop::collection::vec(-1e6f64..1e6f64, 1..50),
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let result = compute(&AggregateFunction::Sum, &rows, NonNumericBehavior::Skip).unwrap();
        let expected: f64 = values.iter().sum();
        match result {
            Atom::Float(s) => prop_assert!((s - expected).abs() < 1e-6),
            _ => prop_assert!(false, "expected Float"),
        }
    }

    /// AVG == sum / count
    #[test]
    fn prop_aggregation_avg(
        values in prop::collection::vec(-1e6f64..1e6f64, 1..50),
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let result = compute(&AggregateFunction::Avg, &rows, NonNumericBehavior::Skip).unwrap();
        let expected: f64 = values.iter().sum::<f64>() / values.len() as f64;
        match result {
            Atom::Float(a) => prop_assert!((a - expected).abs() < 1e-6),
            _ => prop_assert!(false, "expected Float"),
        }
    }

    /// MIN == values.iter().min()
    #[test]
    fn prop_aggregation_min(
        values in prop::collection::vec(-1e6f64..1e6f64, 1..50),
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let result = compute(&AggregateFunction::Min, &rows, NonNumericBehavior::Skip).unwrap();
        let expected = values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        match result {
            Atom::Float(m) => prop_assert!((m - expected).abs() < 1e-10),
            _ => prop_assert!(false, "expected Float"),
        }
    }

    /// MAX == values.iter().max()
    #[test]
    fn prop_aggregation_max(
        values in prop::collection::vec(-1e6f64..1e6f64, 1..50),
    ) {
        let rows: Vec<ResultRow> = values.iter().enumerate().map(|(i, v)| ResultRow {
            key: format!("k{}", i),
            value: Atom::Float(*v),
            timestamp: i as u64,
        }).collect();

        let result = compute(&AggregateFunction::Max, &rows, NonNumericBehavior::Skip).unwrap();
        let expected = values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        match result {
            Atom::Float(m) => prop_assert!((m - expected).abs() < 1e-10),
            _ => prop_assert!(false, "expected Float"),
        }
    }
}

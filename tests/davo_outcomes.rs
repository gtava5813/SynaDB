//! Property-based tests for OutcomeTracker.
//!
//! - Property 24: Outcome Classification Consistency

#![cfg(feature = "davo")]

use proptest::prelude::*;
use synadb::davo::OutcomeTracker;

// ── Property 24: Outcome Classification Consistency ──────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 24: OutcomeTracker Classification Consistency**
    ///
    /// TP + FN must equal the total number of actually-stale observations.
    /// TN + FP must equal the total number of actually-fresh observations.
    #[test]
    fn prop_classification_consistency(
        observations in prop::collection::vec(
            (0.0f32..1.0f32, 0.0f32..1.0f32, 0.01f32..0.5f32),
            1..50
        ),
    ) {
        let mut tracker = OutcomeTracker::new();

        let mut total_actual_stale: u64 = 0;
        let mut total_actual_fresh: u64 = 0;

        for (i, (predicted_freshness, actual_error, acceptable_error)) in observations.iter().enumerate() {
            let actual_fresh = *actual_error < *acceptable_error;
            if actual_fresh {
                total_actual_fresh += 1;
            } else {
                total_actual_stale += 1;
            }

            tracker.record(
                &format!("key_{}", i),
                *predicted_freshness,
                *actual_error,
                *acceptable_error,
            );
        }

        let outcomes = tracker.cumulative();

        // TP + FN = total actually stale
        prop_assert_eq!(
            outcomes.tp + outcomes.fn_,
            total_actual_stale,
            "TP({}) + FN({}) should equal total_actual_stale({})",
            outcomes.tp, outcomes.fn_, total_actual_stale
        );

        // TN + FP = total actually fresh
        prop_assert_eq!(
            outcomes.tn + outcomes.fp,
            total_actual_fresh,
            "TN({}) + FP({}) should equal total_actual_fresh({})",
            outcomes.tn, outcomes.fp, total_actual_fresh
        );

        // Total should match observation count
        let total = outcomes.tp + outcomes.fp + outcomes.tn + outcomes.fn_;
        prop_assert_eq!(
            total,
            observations.len() as u64,
            "Total outcomes should match observation count"
        );
    }
}

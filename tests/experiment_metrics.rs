//! Property-based tests for Experiment metric ordering.
//!
//! **Feature: Syna-ai-native, Property 22: Experiment Metric Monotonicity**
//! **Validates: Requirements 5.3**

use proptest::prelude::*;
use synadb::experiment::ExperimentTracker;
use tempfile::tempdir;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// **Feature: Syna-ai-native, Property 22: Experiment Metric Monotonicity**
    ///
    /// For any experiment run, logged metrics with step numbers SHALL be
    /// retrievable in step order, and step numbers SHALL be monotonically
    /// increasing per metric.
    ///
    /// This property verifies that:
    /// 1. Metrics logged with out-of-order steps are returned sorted by step
    /// 2. The step ordering is strictly monotonically increasing
    ///
    /// **Validates: Requirements 5.3**
    #[test]
    fn prop_metrics_returned_in_step_order(
        n_metrics in 10usize..100,
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("exp.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("test_exp", vec![]).unwrap();

        // Log metrics with shuffled steps (not in order)
        let mut steps: Vec<u64> = (0..n_metrics as u64).collect();
        // Shuffle steps using a deterministic algorithm
        for i in (1..steps.len()).rev() {
            let j = (i as u64 * 31337) as usize % (i + 1);
            steps.swap(i, j);
        }

        for (i, &step) in steps.iter().enumerate() {
            tracker.log_metric(&run_id, "loss", i as f64 * 0.1, Some(step)).unwrap();
        }

        // Retrieve metrics - should be in step order
        let metrics = tracker.get_metric(&run_id, "loss").unwrap();

        // Verify we got all metrics
        prop_assert_eq!(metrics.len(), n_metrics,
            "Expected {} metrics, got {}", n_metrics, metrics.len());

        // Verify metrics are returned in strictly increasing step order
        for i in 1..metrics.len() {
            prop_assert!(
                metrics[i-1].0 < metrics[i].0,
                "Metrics not in step order: step {} >= step {} at positions {} and {}",
                metrics[i-1].0, metrics[i].0, i-1, i
            );
        }
    }

    /// Test that multiple metrics maintain independent ordering.
    ///
    /// **Validates: Requirements 5.3**
    #[test]
    fn prop_multiple_metrics_independent_ordering(
        n_metrics in 5usize..30,
    ) {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("exp_multi.db");

        let mut tracker = ExperimentTracker::new(&db_path).unwrap();
        let run_id = tracker.start_run("test_exp_multi", vec![]).unwrap();

        // Log two different metrics with interleaved, out-of-order steps
        let mut loss_steps: Vec<u64> = (0..n_metrics as u64).collect();
        let mut acc_steps: Vec<u64> = (0..n_metrics as u64).collect();

        // Shuffle both independently
        for i in (1..loss_steps.len()).rev() {
            let j = (i as u64 * 12345) as usize % (i + 1);
            loss_steps.swap(i, j);
        }
        for i in (1..acc_steps.len()).rev() {
            let j = (i as u64 * 67890) as usize % (i + 1);
            acc_steps.swap(i, j);
        }

        // Log metrics interleaved
        for i in 0..n_metrics {
            tracker.log_metric(&run_id, "loss", loss_steps[i] as f64 * 0.1, Some(loss_steps[i])).unwrap();
            tracker.log_metric(&run_id, "accuracy", acc_steps[i] as f64 * 0.01, Some(acc_steps[i])).unwrap();
        }

        // Retrieve both metrics
        let loss_metrics = tracker.get_metric(&run_id, "loss").unwrap();
        let acc_metrics = tracker.get_metric(&run_id, "accuracy").unwrap();

        // Verify both are sorted independently
        for i in 1..loss_metrics.len() {
            prop_assert!(
                loss_metrics[i-1].0 < loss_metrics[i].0,
                "Loss metrics not in step order"
            );
        }

        for i in 1..acc_metrics.len() {
            prop_assert!(
                acc_metrics[i-1].0 < acc_metrics[i].0,
                "Accuracy metrics not in step order"
            );
        }
    }
}

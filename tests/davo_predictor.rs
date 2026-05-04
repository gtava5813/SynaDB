//! Property-based tests for DecayPredictor.
//!
//! - Property 22: Predictor Convergence

#![cfg(feature = "davo")]

use proptest::prelude::*;
use synadb::davo::DecayPredictor;

// ── Property 22: Predictor Convergence ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 22: DecayPredictor Convergence**
    ///
    /// After 100 observations of a constant λ, the prediction should be
    /// within ±50% of the true value.
    #[test]
    fn prop_predictor_convergence(
        true_lambda in 0.001f32..1.0f32,
    ) {
        let mut predictor = DecayPredictor::new();

        for _ in 0..100 {
            predictor.observe(true_lambda);
        }

        let prediction = predictor.predict();
        let lower = true_lambda * 0.5;
        let upper = true_lambda * 1.5;

        prop_assert!(
            prediction >= lower && prediction <= upper,
            "After 100 observations of λ={}, prediction {} should be in [{}, {}]",
            true_lambda, prediction, lower, upper
        );
    }

    /// Uncertainty should decrease with more observations.
    #[test]
    fn prop_uncertainty_decreases(
        true_lambda in 0.001f32..1.0f32,
    ) {
        let mut predictor = DecayPredictor::new();

        // Warm up with 10 observations to establish a baseline
        for _ in 0..10 {
            predictor.observe(true_lambda);
        }
        let baseline_unc = predictor.uncertainty();

        // Add 90 more observations
        for _ in 0..90 {
            predictor.observe(true_lambda);
        }

        let final_unc = predictor.uncertainty();
        prop_assert!(
            final_unc < baseline_unc,
            "Uncertainty should decrease with more data: {} -> {}",
            baseline_unc, final_unc
        );
    }
}

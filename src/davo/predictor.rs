//! DecayPredictor: Bayesian learning of decay rates.
//!
//! Maintains a Gamma posterior over the decay rate λ and supports
//! [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling)
//! for exploration/exploitation trade-offs.
//!
//! # Model
//!
//! The prior is `λ ~ Gamma(α, β)` with mean `α/β`.
//! Each call to [`DecayPredictor::observe`] updates the posterior using
//! an incremental mean estimator so that `predict()` converges to the
//! observed average decay rate.
//!
//! # Example
//!
//! ```
//! use synadb::davo::DecayPredictor;
//!
//! let mut p = DecayPredictor::new();
//! for _ in 0..100 {
//!     p.observe(0.05);
//! }
//! assert!((p.predict() - 0.05).abs() < 0.01);
//! ```

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Bayesian decay-rate predictor using a Gamma conjugate prior.
///
/// The predictor learns the optimal λ from observed staleness outcomes.
/// Use [`predict`](Self::predict) for a point estimate or
/// [`sample`](Self::sample) for Thompson Sampling.
#[derive(Debug, Clone)]
pub struct DecayPredictor {
    /// Gamma prior shape (α₀).
    alpha_prior: f32,

    /// Gamma prior rate (β₀).
    beta_prior: f32,

    /// Posterior shape (α).
    alpha: f32,

    /// Posterior rate (β).
    beta: f32,

    /// Global multiplier applied to predictions (for external adjustments).
    pub global_multiplier: f32,

    /// Number of observations incorporated so far.
    observation_count: u64,

    /// PRNG for Thompson Sampling.
    rng: SmallRng,
}

impl DecayPredictor {
    /// Create a new predictor with a weak default prior.
    ///
    /// Default prior: `α=1, β=100` → E\[λ\] = 0.01 with high variance.
    pub fn new() -> Self {
        Self::with_prior(1.0, 100.0)
    }

    /// Create a predictor with a custom Gamma prior `(α, β)`.
    pub fn with_prior(alpha: f32, beta: f32) -> Self {
        Self {
            alpha_prior: alpha,
            beta_prior: beta,
            alpha,
            beta,
            global_multiplier: 1.0,
            observation_count: 0,
            rng: SmallRng::from_entropy(),
        }
    }

    /// Point estimate of the decay rate: posterior mean `α/β × global_multiplier`.
    pub fn predict(&self) -> f32 {
        (self.alpha / self.beta) * self.global_multiplier
    }

    /// Sample a decay rate from the posterior (Thompson Sampling).
    ///
    /// Uses a Normal approximation `N(mean, variance)` of the Gamma
    /// posterior, clamped to a minimum of 0.0001 to guarantee a positive
    /// decay rate.
    pub fn sample(&mut self) -> f32 {
        let mean = self.predict();
        let std_dev = self.uncertainty().sqrt();

        // Sample from N(mean, std_dev²) and clamp to positive
        let noise: f32 = self.rng.gen_range(-1.0..1.0);
        (mean + std_dev * noise).max(0.0001)
    }

    /// Update the posterior from a single observed decay rate.
    ///
    /// Uses an incremental mean estimator: after *n* observations the
    /// posterior mean converges to the arithmetic mean of all observed
    /// values.
    pub fn observe(&mut self, actual_decay: f32) {
        self.observation_count += 1;

        if actual_decay > 0.0 {
            let n = self.observation_count as f32;
            let old_mean = self.alpha / self.beta;
            let new_mean = old_mean + (actual_decay - old_mean) / n;

            // α grows with sample size; β is set so that α/β = new_mean.
            self.alpha = self.alpha_prior + n;
            self.beta = self.alpha / new_mean.max(0.0001);
        }
    }

    /// Partially reset the posterior toward the prior.
    ///
    /// `blend_factor` in \[0, 1\]: 1.0 keeps the current posterior,
    /// 0.0 fully resets to the prior.
    pub fn reset(&mut self, blend_factor: f32) {
        self.alpha = self.alpha * blend_factor + self.alpha_prior * (1.0 - blend_factor);
        self.beta = self.beta * blend_factor + self.beta_prior * (1.0 - blend_factor);
    }

    /// Posterior variance `α / β²`.
    ///
    /// Decreases as more observations are incorporated.
    pub fn uncertainty(&self) -> f32 {
        self.alpha / (self.beta * self.beta)
    }

    /// Number of observations incorporated so far.
    pub fn observations(&self) -> u64 {
        self.observation_count
    }
}

impl Default for DecayPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_convergence() {
        let mut predictor = DecayPredictor::new();

        // Observe true decay rate of 0.01
        for _ in 0..100 {
            predictor.observe(0.01);
        }

        // Prediction should be close to 0.01
        let prediction = predictor.predict();
        assert!((prediction - 0.01).abs() < 0.005);
    }

    #[test]
    fn test_uncertainty_decreases() {
        let mut predictor = DecayPredictor::new();
        let initial_uncertainty = predictor.uncertainty();

        for _ in 0..10 {
            predictor.observe(0.01);
        }

        assert!(predictor.uncertainty() < initial_uncertainty);
    }

    #[test]
    fn test_thompson_sampling() {
        let mut predictor = DecayPredictor::new();

        // Sample should return a positive value
        let sample = predictor.sample();
        assert!(sample > 0.0);
    }
}

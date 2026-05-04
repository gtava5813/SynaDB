//! DAVO: Decay-Aware Value Optimization (Experimental).
//!
//! A self-improving database layer that treats data as decaying assets.
//! Every value can carry a decay rate λ so that freshness degrades over
//! time according to `e^(-λ × age_seconds)`.
//!
//! # Core Concepts
//!
//! | Concept | Type | Purpose |
//! |---------|------|---------|
//! | **DAVOAtom** | [`DAVOAtom`] | Value with decay metadata |
//! | **Forward Decay** | [`FreshnessIndexV2`] | O(k + log N) staleness queries via deadline index |
//! | **Bayesian Learning** | [`DecayPredictor`] | Learn optimal λ from observations |
//! | **Asymmetric Loss** | [`OutcomeTracker`] | Track TP/FP/TN/FN with weighted loss |
//! | **Lazy Evaluation** | [`Thunk`] / [`ThunkRegistry`] | Defer computation with probation GC |
//!
//! # Quick Start
//!
//! ```
//! use synadb::davo::{FreshnessIndexV2, DecayPredictor};
//!
//! // Track freshness of keys
//! let mut index = FreshnessIndexV2::new();
//! index.insert("sensor/temp", 0.001);   // slow decay
//! index.insert("cache/user_1", 10.0);   // fast decay
//!
//! // Query freshness
//! let f = index.get_freshness("sensor/temp").unwrap();
//! assert!(f > 0.99); // just inserted
//!
//! // Learn decay rates from observations
//! let mut predictor = DecayPredictor::new();
//! for _ in 0..50 {
//!     predictor.observe(0.05);
//! }
//! assert!((predictor.predict() - 0.05).abs() < 0.02);
//! ```
//!
//! # Freshness Index Versions
//!
//! - **FreshnessIndex** (V1) — HashMap-based, O(N) staleness scans. **Deprecated.**
//! - **FreshnessIndexV2** (V2) — BTreeMap deadline index, O(k + log N) staleness scans.
//!
//! Always use [`FreshnessIndexV2`] for new code.
//!
//! # Status
//!
//! **Experimental** — API may change between minor versions.

pub mod atom;
pub mod freshness;
pub mod freshness_v2;
pub mod outcomes;
pub mod predictor;
pub mod thunk;

// Re-exports
pub use atom::{DAVOAtom, DecayObservation, DecayedValue};
#[allow(deprecated)]
pub use freshness::FreshnessIndex;
pub use freshness_v2::FreshnessIndexV2;
pub use outcomes::{ClassifiedOutcomes, Observation, OutcomeTracker};
pub use predictor::DecayPredictor;
pub use thunk::{Thunk, ThunkRegistry};

//! DAVOAtom: Values with decay metadata.
//!
//! Extends SynaDB's [`Atom`] type with temporal decay information.
//! Every value can carry a decay rate λ so that freshness degrades
//! over time according to `e^(-λ × age_seconds)`.
//!
//! # Variants
//!
//! | Variant | λ | Freshness behaviour |
//! |---------|---|---------------------|
//! | [`DAVOAtom::Static`] | 0 | Always 1.0 |
//! | [`DAVOAtom::Decaying`] | fixed | `e^(-λ × age)` |
//! | [`DAVOAtom::SelfImproving`] | learned | Uses external [`super::DecayPredictor`] |
//! | [`DAVOAtom::Thunk`] | n/a | 1.0 during probation, 0.0 after |

use crate::types::Atom;
use serde::{Deserialize, Serialize};

/// Default decay rate used when a [`DAVOAtom::SelfImproving`] variant
/// cannot resolve its predictor.
pub const DEFAULT_SELF_IMPROVING_DECAY: f32 = 0.001;

/// Helper to get current time in microseconds.
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default()
}

/// DAVO-aware Atom: a value wrapped with decay metadata.
///
/// This is the fundamental data type of the DAVO module. It extends
/// SynaDB's [`Atom`] with temporal decay semantics so that every stored
/// value can express how quickly it becomes stale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DAVOAtom {
    /// Static value (λ = 0, never decays). Freshness is always 1.0.
    Static(Atom),

    /// Decaying value with a fixed decay rate λ.
    ///
    /// Freshness at query time *t* is `e^(-λ × (t − stored_at) / 1_000_000)`.
    Decaying {
        /// The wrapped value.
        value: Atom,
        /// Timestamp (microseconds since epoch) when the value was stored.
        stored_at: u64,
        /// Decay rate λ (per second). Must be ≥ 0.
        decay_rate: f32,
        /// Optional context embedding used by higher-level predictors.
        decay_context: Vec<f32>,
    },

    /// Self-improving value whose decay rate is learned by a
    /// [`super::DecayPredictor`] identified by `predictor_id`.
    ///
    /// Call [`DAVOAtom::freshness_at_with_predictor`] to supply the
    /// predictor map; the plain [`DAVOAtom::freshness_at`] method falls
    /// back to [`DEFAULT_SELF_IMPROVING_DECAY`].
    SelfImproving {
        /// The wrapped value.
        value: Atom,
        /// Timestamp (microseconds since epoch) when the value was stored.
        stored_at: u64,
        /// Key into an external predictor registry.
        predictor_id: String,
        /// Historical observations for self-improvement.
        observations: Vec<DecayObservation>,
    },

    /// Lazy value (thunk) — not yet evaluated.
    ///
    /// Returns freshness 1.0 while within the probation window and 0.0
    /// after the deadline expires.
    Thunk {
        /// Unique thunk identifier.
        thunk_id: u64,
        /// Timestamp (microseconds since epoch) when the thunk was created.
        created_at: u64,
        /// Deadline (microseconds since epoch) after which the thunk expires.
        probation_deadline: u64,
    },
}

/// A single staleness observation used by the self-improvement loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayObservation {
    /// When the query was made (microseconds since epoch).
    pub query_time: u64,
    /// Freshness predicted at `query_time`.
    pub predicted_freshness: f32,
    /// Measured error between predicted and actual value.
    pub actual_error: f32,
}

/// A value together with its computed freshness metadata.
#[derive(Debug, Clone)]
pub struct DecayedValue {
    /// The underlying value.
    pub value: Atom,
    /// Freshness score in \[0.0, 1.0\].
    pub freshness: f32,
    /// Age of the value in microseconds.
    pub age_micros: u64,
    /// Decay rate λ that was used for the freshness computation.
    pub decay_rate: f32,
}

impl DAVOAtom {
    /// Create a static (non-decaying) atom. Freshness is always 1.0.
    pub fn static_value(value: Atom) -> Self {
        DAVOAtom::Static(value)
    }

    /// Create a decaying atom with the specified decay rate λ.
    ///
    /// The `stored_at` timestamp is set to the current wall-clock time.
    pub fn decaying(value: Atom, decay_rate: f32) -> Self {
        DAVOAtom::Decaying {
            value,
            stored_at: now_micros(),
            decay_rate,
            decay_context: Vec::new(),
        }
    }

    /// Calculate freshness at the given `query_time` (microseconds since epoch).
    ///
    /// For [`DAVOAtom::SelfImproving`] this falls back to
    /// [`DEFAULT_SELF_IMPROVING_DECAY`]. Use
    /// [`freshness_at_with_predictor`](Self::freshness_at_with_predictor)
    /// to supply an actual predictor map.
    pub fn freshness_at(&self, query_time: u64) -> f32 {
        match self {
            DAVOAtom::Static(_) => 1.0,
            DAVOAtom::Decaying {
                stored_at,
                decay_rate,
                ..
            } => {
                let age = query_time.saturating_sub(*stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                (-decay_rate * age_secs).exp()
            }
            DAVOAtom::SelfImproving { stored_at, .. } => {
                let age = query_time.saturating_sub(*stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                (-DEFAULT_SELF_IMPROVING_DECAY * age_secs).exp()
            }
            DAVOAtom::Thunk {
                probation_deadline, ..
            } => {
                if query_time > *probation_deadline {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    /// Calculate freshness using an external predictor map.
    ///
    /// For [`DAVOAtom::SelfImproving`] the `predictor_id` is looked up in
    /// `predictors`. If the predictor is found its
    /// [`predict()`](super::DecayPredictor::predict) value is used as λ;
    /// otherwise [`DEFAULT_SELF_IMPROVING_DECAY`] is used.
    ///
    /// All other variants delegate to [`freshness_at`](Self::freshness_at).
    pub fn freshness_at_with_predictor(
        &self,
        query_time: u64,
        predictors: &std::collections::HashMap<String, super::DecayPredictor>,
    ) -> f32 {
        match self {
            DAVOAtom::SelfImproving {
                stored_at,
                predictor_id,
                ..
            } => {
                let decay_rate = predictors
                    .get(predictor_id)
                    .map(|p| p.predict())
                    .unwrap_or(DEFAULT_SELF_IMPROVING_DECAY);
                let age = query_time.saturating_sub(*stored_at);
                let age_secs = age as f32 / 1_000_000.0;
                (-decay_rate * age_secs).exp()
            }
            _ => self.freshness_at(query_time),
        }
    }

    /// Get the inner value, or `None` for unevaluated [`DAVOAtom::Thunk`]s.
    pub fn value(&self) -> Option<&Atom> {
        match self {
            DAVOAtom::Static(v) => Some(v),
            DAVOAtom::Decaying { value, .. } => Some(value),
            DAVOAtom::SelfImproving { value, .. } => Some(value),
            DAVOAtom::Thunk { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_freshness() {
        let atom = DAVOAtom::static_value(Atom::Float(3.14));
        assert_eq!(atom.freshness_at(0), 1.0);
        assert_eq!(atom.freshness_at(u64::MAX), 1.0);
    }

    #[test]
    fn test_decaying_freshness() {
        let atom = DAVOAtom::Decaying {
            value: Atom::Float(3.14),
            stored_at: 0,
            decay_rate: 1.0, // 1/second
            decay_context: vec![],
        };

        // At t=0, freshness = 1.0
        assert!((atom.freshness_at(0) - 1.0).abs() < 0.001);

        // At t=1 second (1_000_000 micros), freshness ≈ 0.368
        assert!((atom.freshness_at(1_000_000) - 0.368).abs() < 0.01);
    }
}

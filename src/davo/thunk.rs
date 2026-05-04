//! Thunk: Lazy evaluation for potential data.
//!
//! Implements the "probationary segment" concept from the DAVO architecture.
//! A thunk is a promise of data that is only materialised when explicitly
//! evaluated. If the probation deadline passes without evaluation the thunk
//! can be garbage-collected, saving the cost of the computation entirely.

use crate::types::Atom;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global monotonic thunk-ID counter.
static THUNK_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A thunk represents potential data that has not been evaluated yet.
///
/// During the probation window the thunk is considered "fresh" (freshness 1.0).
/// After the deadline it is considered expired (freshness 0.0) and eligible
/// for garbage collection by [`ThunkRegistry::gc`].
#[derive(Debug, Clone)]
pub struct Thunk {
    /// Unique identifier (monotonically increasing).
    pub id: u64,

    /// When the thunk was created (microseconds since epoch).
    pub created_at: u64,

    /// Deadline after which the thunk expires (microseconds since epoch).
    pub probation_deadline: u64,

    /// Identifier of the generator function that can produce the value.
    pub generator_id: String,

    /// Opaque context bytes passed to the generator on evaluation.
    pub context: Vec<u8>,

    /// Memoised result, set by [`evaluate`](Self::evaluate).
    evaluated: Option<Atom>,
}

impl Thunk {
    /// Create a new thunk with probation period
    pub fn new(generator_id: &str, context: Vec<u8>, probation_micros: u64) -> Self {
        let now = now_micros();
        Self {
            id: THUNK_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            created_at: now,
            probation_deadline: now + probation_micros,
            generator_id: generator_id.to_string(),
            context,
            evaluated: None,
        }
    }

    /// Check if thunk is still in probation
    pub fn in_probation(&self) -> bool {
        now_micros() < self.probation_deadline
    }

    /// Check if thunk has been evaluated
    pub fn is_evaluated(&self) -> bool {
        self.evaluated.is_some()
    }

    /// Evaluate the thunk (memoize result)
    pub fn evaluate(&mut self, value: Atom) {
        self.evaluated = Some(value);
    }

    /// Get evaluated value
    pub fn value(&self) -> Option<&Atom> {
        self.evaluated.as_ref()
    }

    /// Time remaining in probation (0 if expired)
    pub fn time_remaining(&self) -> u64 {
        self.probation_deadline.saturating_sub(now_micros())
    }
}

/// Registry for managing active thunks.
///
/// Provides registration, evaluation, and garbage collection of thunks.
/// Expired unevaluated thunks are removed by [`gc`](Self::gc).
#[derive(Debug)]
pub struct ThunkRegistry {
    /// Active thunks keyed by ID.
    thunks: HashMap<u64, Thunk>,

    /// Default probation period for new thunks (microseconds).
    default_probation: u64,
}

impl ThunkRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self::with_probation(5_000_000) // 5 seconds default
    }

    /// Create with custom probation period
    pub fn with_probation(probation_micros: u64) -> Self {
        Self {
            thunks: HashMap::new(),
            default_probation: probation_micros,
        }
    }

    /// Register a new thunk
    pub fn register(&mut self, generator_id: &str, context: Vec<u8>) -> u64 {
        let thunk = Thunk::new(generator_id, context, self.default_probation);
        let id = thunk.id;
        self.thunks.insert(id, thunk);
        id
    }

    /// Get a thunk by ID
    pub fn get(&self, id: u64) -> Option<&Thunk> {
        self.thunks.get(&id)
    }

    /// Get mutable thunk by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Thunk> {
        self.thunks.get_mut(&id)
    }

    /// Evaluate a thunk
    pub fn evaluate(&mut self, id: u64, value: Atom) -> bool {
        if let Some(thunk) = self.thunks.get_mut(&id) {
            thunk.evaluate(value);
            true
        } else {
            false
        }
    }

    /// Garbage collect expired thunks
    pub fn gc(&mut self) -> usize {
        let before = self.thunks.len();

        self.thunks.retain(|_, thunk| {
            // Keep if: evaluated OR still in probation
            thunk.is_evaluated() || thunk.in_probation()
        });

        before - self.thunks.len()
    }

    /// Count active thunks
    pub fn len(&self) -> usize {
        self.thunks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.thunks.is_empty()
    }

    /// Count unevaluated thunks
    pub fn unevaluated_count(&self) -> usize {
        self.thunks.values().filter(|t| !t.is_evaluated()).count()
    }

    /// Count expired (garbage-collectable) thunks
    pub fn expired_count(&self) -> usize {
        self.thunks
            .values()
            .filter(|t| !t.is_evaluated() && !t.in_probation())
            .count()
    }
}

impl Default for ThunkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to get current time in microseconds
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thunk_probation() {
        let thunk = Thunk::new("test_gen", vec![], 1_000_000); // 1 second

        assert!(thunk.in_probation());
        assert!(!thunk.is_evaluated());
    }

    #[test]
    fn test_thunk_evaluation() {
        let mut thunk = Thunk::new("test_gen", vec![], 1_000_000);

        thunk.evaluate(Atom::Float(3.14));

        assert!(thunk.is_evaluated());
        assert_eq!(thunk.value(), Some(&Atom::Float(3.14)));
    }

    #[test]
    fn test_registry_gc() {
        let mut registry = ThunkRegistry::with_probation(0); // Immediate expiry

        registry.register("gen1", vec![]);
        registry.register("gen2", vec![]);

        // Both should be expired immediately
        std::thread::sleep(std::time::Duration::from_millis(1));

        let collected = registry.gc();
        assert_eq!(collected, 2);
        assert!(registry.is_empty());
    }
}

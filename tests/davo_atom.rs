//! Property-based tests for DAVOAtom.
//!
//! - Property 19: Freshness Monotonicity
//! - Property 20: Static Invariance

#![cfg(feature = "davo")]

use proptest::prelude::*;
use synadb::davo::DAVOAtom;
use synadb::types::Atom;

// ── Strategies ───────────────────────────────────────────────────────

fn arb_decay_rate() -> impl Strategy<Value = f32> {
    0.0001f32..100.0f32
}

fn arb_timestamp() -> impl Strategy<Value = u64> {
    0u64..1_000_000_000_000u64 // up to ~11.5 days in micros
}

// ── Property 19: Freshness Monotonicity ──────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 19: DAVOAtom Freshness Monotonicity**
    ///
    /// For any Decaying atom, freshness_at(t1) >= freshness_at(t2) when t1 < t2.
    /// Freshness is monotonically non-increasing over time.
    #[test]
    fn prop_freshness_monotonicity(
        decay_rate in arb_decay_rate(),
        stored_at in arb_timestamp(),
        delta1 in 0u64..10_000_000u64,
        delta2 in 0u64..10_000_000u64,
    ) {
        let atom = DAVOAtom::Decaying {
            value: Atom::Float(1.0),
            stored_at,
            decay_rate,
            decay_context: vec![],
        };

        let t1 = stored_at.saturating_add(delta1.min(delta2));
        let t2 = stored_at.saturating_add(delta1.max(delta2));

        let f1 = atom.freshness_at(t1);
        let f2 = atom.freshness_at(t2);

        prop_assert!(
            f1 >= f2 - 1e-6,
            "Freshness should be monotonically non-increasing: f({})={} >= f({})={}",
            t1, f1, t2, f2
        );
    }
}

// ── Property 20: Static Invariance ───────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 20: DAVOAtom Static Invariance**
    ///
    /// Static atoms always return freshness 1.0 regardless of query time.
    #[test]
    fn prop_static_invariance(
        value in any::<f64>(),
        query_time in any::<u64>(),
    ) {
        let atom = DAVOAtom::static_value(Atom::Float(value));
        let freshness = atom.freshness_at(query_time);
        prop_assert_eq!(freshness, 1.0, "Static atom freshness must always be 1.0");
    }
}

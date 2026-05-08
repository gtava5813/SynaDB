//! Property-based tests for FreshnessIndexV2.
//!
//! - Property 21: Deadline Correctness
//! - Property 23: Eviction Completeness

#![cfg(feature = "davo")]

use proptest::prelude::*;
use synadb::davo::FreshnessIndexV2;

// ── Property 21: Deadline Correctness ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// **Property 21: FreshnessIndexV2 Deadline Correctness**
    ///
    /// Tests the mathematical equivalence: `is_stale()` (deadline comparison)
    /// should agree with `freshness < threshold`, modulo a small epsilon at
    /// the exact boundary. We test immediately after insert where the entry
    /// must not yet be stale by construction.
    ///
    /// Decay rate is bounded below to ensure time_to_stale > 1 second
    /// even for threshold close to 1.0, making the test CI-robust.
    #[test]
    fn prop_deadline_correctness(
        threshold in 0.1f32..0.9f32,
        decay_rate in 0.001f32..0.1f32,
    ) {
        let mut index = FreshnessIndexV2::with_threshold(threshold);
        index.insert("test_key", decay_rate);

        // Worst case time-to-stale: threshold=0.9, decay_rate=0.1
        // -ln(0.9) / 0.1 = 1.05 seconds. CI can't miss that window.
        let is_stale = index.is_stale("test_key").unwrap();
        let freshness = index.get_freshness("test_key").unwrap();

        prop_assert!(!is_stale, "just-inserted key should not be stale");
        prop_assert!(
            freshness >= threshold,
            "freshness {} must be >= threshold {}",
            freshness,
            threshold
        );
    }
}

// ── Property 23: Eviction Completeness ───────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property 23: FreshnessIndexV2 Eviction Completeness**
    ///
    /// After evict_stale(), count_stale() must return 0.
    #[test]
    fn prop_eviction_completeness(
        num_fast in 1usize..20,
        num_slow in 0usize..10,
    ) {
        let mut index = FreshnessIndexV2::new();

        // Insert fast-decaying keys (will be stale almost immediately)
        for i in 0..num_fast {
            index.insert(&format!("fast_{}", i), 100_000.0);
        }

        // Insert slow/static keys
        for i in 0..num_slow {
            index.insert(&format!("slow_{}", i), 0.0);
        }

        // Wait for fast keys to become stale
        std::thread::sleep(std::time::Duration::from_millis(2));

        // Evict
        let _evicted = index.evict_stale();

        // After eviction, no stale keys should remain
        let stale_count = index.count_stale();
        prop_assert_eq!(stale_count, 0, "After evict_stale(), count_stale() must be 0");

        // Only slow keys should remain
        prop_assert_eq!(index.len(), num_slow, "Only static keys should remain");
    }
}

// ── Property 25: Persistence Round-Trip ──────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// **Property 25: FreshnessIndexV2 Persistence Round-Trip**
    ///
    /// Save → Load produces an index with identical len(), threshold,
    /// and freshness values for all keys.
    #[test]
    fn prop_persistence_roundtrip(
        entries in prop::collection::vec(
            ("[a-z]{1,8}", 0.0001f32..10.0f32),
            1..20
        ),
        threshold in 0.1f32..0.9f32,
    ) {
        let mut original = FreshnessIndexV2::with_threshold(threshold);
        for (key, rate) in &entries {
            original.insert(key, *rate);
        }

        let tmpdir = tempfile::tempdir().unwrap();
        let file = tmpdir.path().join("freshness.bin");

        original.save(&file).unwrap();
        let loaded = FreshnessIndexV2::load(&file).unwrap();

        prop_assert_eq!(loaded.len(), original.len());
        prop_assert_eq!(loaded.threshold(), original.threshold());

        // Every key in original should exist in loaded with same freshness
        // (within float tolerance — a few micros of clock drift between calls)
        for (key, _) in &entries {
            let orig_fresh = original.get_freshness(key);
            let loaded_fresh = loaded.get_freshness(key);
            prop_assert_eq!(orig_fresh.is_some(), loaded_fresh.is_some());
            if let (Some(o), Some(l)) = (orig_fresh, loaded_fresh) {
                prop_assert!((o - l).abs() < 0.01, "freshness diverged: {} vs {}", o, l);
            }
        }
    }
}

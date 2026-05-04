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
    /// An entry is stale if and only if current_time >= its precomputed deadline.
    /// We verify that `is_stale()` and `get_freshness() < threshold` agree.
    #[test]
    fn prop_deadline_correctness(
        threshold in 0.1f32..0.9f32,
        decay_rate in 0.001f32..100.0f32,
    ) {
        let mut index = FreshnessIndexV2::with_threshold(threshold);
        index.insert("test_key", decay_rate);

        // Immediately after insert, should NOT be stale
        let is_stale = index.is_stale("test_key").unwrap();
        let freshness = index.get_freshness("test_key").unwrap();

        // Freshness should be ~1.0 (just inserted)
        prop_assert!(freshness > 0.99, "Just-inserted freshness should be ~1.0, got {}", freshness);

        // is_stale should be false (freshness > threshold)
        prop_assert!(!is_stale, "Just-inserted key should not be stale");
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

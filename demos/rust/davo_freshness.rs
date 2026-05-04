//! DAVO Freshness Demo
//!
//! Demonstrates decay-aware value optimization:
//! - Tracking sensor data with different decay rates
//! - Querying freshness of keys
//! - Evicting stale data automatically
//! - Learning optimal decay rates with Bayesian prediction
//! - Thompson Sampling for exploration
//!
//! Run with: cargo run --example davo_freshness --features davo

use std::collections::HashMap;
use synadb::davo::{DAVOAtom, DecayPredictor, FreshnessIndexV2, OutcomeTracker};
use synadb::types::Atom;

fn main() {
    println!("╔════════════════════════════════════════════╗");
    println!("║   DAVO: Decay-Aware Value Optimization     ║");
    println!("║   Freshness Demo                           ║");
    println!("╚════════════════════════════════════════════╝\n");

    demo_davo_atoms();
    demo_freshness_index();
    demo_decay_predictor();
    demo_outcome_tracking();
    demo_self_improving();

    println!("\n✅ All demos completed successfully!");
}

/// Demo 1: DAVOAtom — values with decay metadata
fn demo_davo_atoms() {
    println!("━━━ Demo 1: DAVOAtom — Values with Decay ━━━\n");

    // Static data never decays (config, constants)
    let config = DAVOAtom::static_value(Atom::Text("production".to_string()));
    println!("  Static config freshness:  {:.4}  (always 1.0)", config.freshness_at(0));
    println!(
        "  Static config freshness:  {:.4}  (even at t=MAX)",
        config.freshness_at(u64::MAX)
    );

    // Sensor reading decays quickly (λ = 1.0 → half-life ≈ 0.7s)
    let sensor = DAVOAtom::Decaying {
        value: Atom::Float(23.5),
        stored_at: 0,
        decay_rate: 1.0,
        decay_context: vec![],
    };

    println!("\n  Sensor reading (λ=1.0/s, half-life ≈ 0.7s):");
    println!("    t=0.0s  freshness = {:.4}", sensor.freshness_at(0));
    println!(
        "    t=0.5s  freshness = {:.4}",
        sensor.freshness_at(500_000)
    );
    println!(
        "    t=1.0s  freshness = {:.4}",
        sensor.freshness_at(1_000_000)
    );
    println!(
        "    t=2.0s  freshness = {:.4}",
        sensor.freshness_at(2_000_000)
    );
    println!(
        "    t=5.0s  freshness = {:.4}",
        sensor.freshness_at(5_000_000)
    );

    // ML embedding decays slowly (λ = 0.0001 → half-life ≈ 1.9 hours)
    let embedding = DAVOAtom::Decaying {
        value: Atom::Float(0.95),
        stored_at: 0,
        decay_rate: 0.0001,
        decay_context: vec![],
    };

    println!("\n  ML embedding (λ=0.0001/s, half-life ≈ 1.9 hours):");
    println!("    t=0s     freshness = {:.4}", embedding.freshness_at(0));
    println!(
        "    t=1min   freshness = {:.4}",
        embedding.freshness_at(60_000_000)
    );
    println!(
        "    t=1hr    freshness = {:.4}",
        embedding.freshness_at(3_600_000_000)
    );
    println!(
        "    t=24hr   freshness = {:.4}",
        embedding.freshness_at(86_400_000_000)
    );
    println!();
}

/// Demo 2: FreshnessIndexV2 — scalable staleness queries
fn demo_freshness_index() {
    println!("━━━ Demo 2: FreshnessIndexV2 — Staleness Queries ━━━\n");

    let mut index = FreshnessIndexV2::with_threshold(0.5);

    // Simulate a mixed workload
    let keys = vec![
        ("sensor/lidar/front", 10.0),    // Very fast decay
        ("sensor/lidar/rear", 10.0),     // Very fast decay
        ("sensor/imu/accel_x", 5.0),     // Fast decay
        ("sensor/imu/accel_y", 5.0),     // Fast decay
        ("sensor/temp", 0.01),           // Slow decay
        ("model/weights/v3", 0.0001),    // Very slow decay
        ("config/robot_id", 0.0),        // Static (never decays)
    ];

    println!("  Inserting {} keys with varying decay rates:", keys.len());
    for (key, rate) in &keys {
        index.insert(key, *rate);
        let half_life = if *rate > 0.0 {
            format!("{:.2}s", 0.693 / rate)
        } else {
            "∞ (static)".to_string()
        };
        println!("    {:<30} λ={:<8.4}  half-life={}", key, rate, half_life);
    }

    println!("\n  Freshness immediately after insert:");
    for (key, _) in &keys {
        let f = index.get_freshness(key).unwrap_or(0.0);
        let stale = index.is_stale(key).unwrap_or(false);
        println!("    {:<30} freshness={:.4}  stale={}", key, f, stale);
    }

    // Wait for fast-decaying keys to become stale
    println!("\n  Waiting 100ms for fast keys to decay...");
    std::thread::sleep(std::time::Duration::from_millis(100));

    let stale_keys = index.query_stale();
    println!(
        "  Stale keys (freshness < 0.5): {} of {}",
        stale_keys.len(),
        index.len()
    );
    for key in &stale_keys {
        println!("    ✗ {}", key);
    }

    let fresh_keys = index.query_fresh();
    println!(
        "  Fresh keys (freshness ≥ 0.5): {} of {}",
        fresh_keys.len(),
        index.len()
    );
    for key in &fresh_keys {
        println!("    ✓ {}", key);
    }

    // Evict stale entries
    let evicted = index.evict_stale();
    println!(
        "\n  Evicted {} stale entries. Remaining: {}",
        evicted.len(),
        index.len()
    );
    println!();
}

/// Demo 3: DecayPredictor — Bayesian learning of decay rates
fn demo_decay_predictor() {
    println!("━━━ Demo 3: DecayPredictor — Bayesian Learning ━━━\n");

    let mut predictor = DecayPredictor::new();
    println!(
        "  Initial prediction: {:.6}  (prior mean, high uncertainty)",
        predictor.predict()
    );
    println!("  Initial uncertainty: {:.6}\n", predictor.uncertainty());

    // Simulate observing actual decay rates from a sensor domain
    let true_lambda = 0.05; // True decay rate: 0.05/s (half-life ≈ 14s)
    println!(
        "  Feeding 100 observations of true λ = {} ...",
        true_lambda
    );

    let checkpoints = [10, 25, 50, 100];
    let mut obs_count = 0;
    for &checkpoint in &checkpoints {
        while obs_count < checkpoint {
            predictor.observe(true_lambda);
            obs_count += 1;
        }
        println!(
            "    After {:>3} obs: prediction={:.6}  uncertainty={:.8}",
            checkpoint,
            predictor.predict(),
            predictor.uncertainty()
        );
    }

    println!(
        "\n  Final prediction: {:.6}  (true λ = {})",
        predictor.predict(),
        true_lambda
    );
    println!(
        "  Error: {:.4}%",
        ((predictor.predict() - true_lambda) / true_lambda * 100.0).abs()
    );

    // Thompson Sampling
    println!("\n  Thompson Sampling (5 samples from posterior):");
    for i in 1..=5 {
        let s = predictor.sample();
        println!("    Sample {}: {:.6}", i, s);
    }
    println!();
}

/// Demo 4: OutcomeTracker — asymmetric loss tracking
fn demo_outcome_tracking() {
    println!("━━━ Demo 4: OutcomeTracker — Asymmetric Loss ━━━\n");

    let mut tracker = OutcomeTracker::new();

    // Simulate predictions vs reality
    let scenarios = vec![
        // (predicted_freshness, actual_error, acceptable_error, description)
        (0.9, 0.02, 0.1, "TP: predicted fresh, was fresh"),
        (0.8, 0.05, 0.1, "TN: predicted fresh, was fresh"),
        (0.3, 0.50, 0.1, "TP: predicted stale, was stale"),
        (0.2, 0.80, 0.1, "TP: predicted stale, was stale"),
        (0.85, 0.60, 0.1, "FN: predicted fresh, WAS STALE ⚠️"),
        (0.15, 0.03, 0.1, "FP: predicted stale, was fresh"),
    ];

    println!("  Recording {} observations:", scenarios.len());
    for (i, (pred, err, acc, desc)) in scenarios.iter().enumerate() {
        tracker.record(&format!("key_{}", i), *pred, *err, *acc);
        println!("    {} — pred={:.1}, err={:.2}, acc={:.1}", desc, pred, err, acc);
    }

    let outcomes = tracker.cumulative();
    println!("\n  Classification results:");
    println!("    True Positives  (predicted stale, was stale):  {}", outcomes.tp);
    println!("    True Negatives  (predicted fresh, was fresh):  {}", outcomes.tn);
    println!("    False Positives (predicted stale, was fresh):  {}", outcomes.fp);
    println!(
        "    False Negatives (predicted fresh, was stale):  {} ⚠️",
        outcomes.fn_
    );

    println!("\n  Safety metrics:");
    println!(
        "    False Negative Rate: {:.1}%  (serving stale data)",
        tracker.false_negative_rate() * 100.0
    );
    println!(
        "    False Positive Rate: {:.1}%  (unnecessary re-fetches)",
        tracker.false_positive_rate() * 100.0
    );

    // Asymmetric loss: FN is 10x worse than FP
    let loss = tracker.compute_loss(10.0, 1.0);
    println!(
        "    Asymmetric Loss (FN=10×, FP=1×): {:.1}",
        loss
    );
    println!();
}

/// Demo 5: SelfImproving atoms with predictor lookup
fn demo_self_improving() {
    println!("━━━ Demo 5: SelfImproving — Predictor Lookup ━━━\n");

    // Create a predictor registry
    let mut predictors: HashMap<String, DecayPredictor> = HashMap::new();

    // Train a predictor for the "sensor" domain
    let mut sensor_pred = DecayPredictor::new();
    for _ in 0..100 {
        sensor_pred.observe(0.5); // Sensors decay at 0.5/s
    }
    println!(
        "  Trained 'sensor' predictor: λ = {:.4}",
        sensor_pred.predict()
    );
    predictors.insert("sensor".to_string(), sensor_pred);

    // Train a predictor for the "embedding" domain
    let mut embed_pred = DecayPredictor::new();
    for _ in 0..100 {
        embed_pred.observe(0.001); // Embeddings decay slowly
    }
    println!(
        "  Trained 'embedding' predictor: λ = {:.6}",
        embed_pred.predict()
    );
    predictors.insert("embedding".to_string(), embed_pred);

    // Create SelfImproving atoms
    let sensor_atom = DAVOAtom::SelfImproving {
        value: Atom::Float(23.5),
        stored_at: 0,
        predictor_id: "sensor".to_string(),
        observations: vec![],
    };

    let embed_atom = DAVOAtom::SelfImproving {
        value: Atom::Float(0.95),
        stored_at: 0,
        predictor_id: "embedding".to_string(),
        observations: vec![],
    };

    let unknown_atom = DAVOAtom::SelfImproving {
        value: Atom::Float(1.0),
        stored_at: 0,
        predictor_id: "unknown_domain".to_string(),
        observations: vec![],
    };

    // Query freshness at t=1 second using predictor lookup
    let t = 1_000_000; // 1 second in microseconds
    println!("\n  Freshness at t=1s (with predictor lookup):");
    println!(
        "    sensor atom:    {:.4}  (λ learned = {:.4})",
        sensor_atom.freshness_at_with_predictor(t, &predictors),
        predictors.get("sensor").map(|p| p.predict()).unwrap_or(0.0)
    );
    println!(
        "    embedding atom: {:.4}  (λ learned = {:.6})",
        embed_atom.freshness_at_with_predictor(t, &predictors),
        predictors.get("embedding").map(|p| p.predict()).unwrap_or(0.0)
    );
    println!(
        "    unknown atom:   {:.4}  (fallback λ = 0.001)",
        unknown_atom.freshness_at_with_predictor(t, &predictors)
    );

    // Compare with default fallback
    println!("\n  Freshness at t=1s (without predictor — default λ=0.001):");
    println!("    sensor atom:    {:.4}", sensor_atom.freshness_at(t));
    println!("    embedding atom: {:.4}", embed_atom.freshness_at(t));
    println!(
        "    → The predictor-aware version uses learned λ, not the default!"
    );
    println!();
}

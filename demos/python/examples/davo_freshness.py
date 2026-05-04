#!/usr/bin/env python3
"""
DAVO Freshness Demo (Python)

Demonstrates decay-aware value optimization from Python:
1. Tracking IoT sensor data with different decay rates
2. Querying freshness and detecting stale data
3. Automatic eviction of stale entries
4. Learning optimal decay rates with Bayesian prediction
5. Thompson Sampling for exploration

Usage:
    python davo_freshness.py

Requirements:
    - SynaDB shared library built with: cargo build --release --features davo
"""

import sys
import os
import time
import math

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synadb.davo import FreshnessIndex, DecayPredictor


def demo_freshness_index():
    """Demo 1: Track sensor data freshness and evict stale entries."""
    print("━━━ Demo 1: FreshnessIndex — Sensor Data Freshness ━━━\n")

    with FreshnessIndex("sensor_demo", threshold=0.5) as idx:
        # Simulate a robot with multiple sensors
        sensors = {
            "lidar/front":     100.0,    # Very fast decay (half-life ~7ms)
            "lidar/rear":      100.0,
            "imu/accel_x":     10.0,     # Fast decay (half-life ~70ms)
            "imu/accel_y":     10.0,
            "temperature":     0.01,     # Slow decay (half-life ~69s)
            "model/weights":   0.0001,   # Very slow (half-life ~1.9hr)
            "config/robot_id": 0.0,      # Static (never decays)
        }

        print(f"  Inserting {len(sensors)} sensor keys:\n")
        for key, rate in sensors.items():
            idx.insert(key, decay_rate=rate)
            if rate > 0:
                half_life = math.log(2) / rate
                print(f"    {key:<25} λ={rate:<10.4f} half-life={half_life:.3f}s")
            else:
                print(f"    {key:<25} λ={rate:<10.4f} half-life=∞ (static)")

        # Check freshness immediately
        print(f"\n  Freshness immediately after insert:")
        for key in sensors:
            f = idx.get_freshness(key)
            print(f"    {key:<25} freshness={f:.4f}")

        print(f"\n  Total tracked keys: {len(idx)}")

        # Wait for fast sensors to become stale
        print(f"\n  Waiting 100ms for fast sensors to decay...")
        time.sleep(0.1)

        # Query stale keys
        stale = idx.query_stale()
        print(f"\n  Stale keys ({len(stale)} of {len(idx)}):")
        for key in stale:
            f = idx.get_freshness(key)
            print(f"    ✗ {key:<25} freshness={f:.6f}" if f else f"    ✗ {key}")

        # Evict stale entries
        evicted = idx.evict_stale()
        print(f"\n  Evicted {evicted} stale entries. Remaining: {len(idx)}")

        # Show what survived
        print(f"\n  Surviving keys:")
        for key in sensors:
            f = idx.get_freshness(key)
            if f is not None:
                print(f"    ✓ {key:<25} freshness={f:.4f}")

    print()


def demo_decay_predictor():
    """Demo 2: Learn optimal decay rates from observations."""
    print("━━━ Demo 2: DecayPredictor — Bayesian Learning ━━━\n")

    with DecayPredictor("learning_demo") as pred:
        print(f"  Initial prediction: {pred.predict():.6f}  (prior mean)")
        print(f"  Initial uncertainty: {pred.uncertainty():.8f}\n")

        # Simulate observing actual decay rates from a sensor domain
        true_lambda = 0.05  # True decay rate: half-life ≈ 14s
        print(f"  Feeding observations of true λ = {true_lambda}...\n")

        checkpoints = [5, 10, 25, 50, 100]
        obs_count = 0
        for checkpoint in checkpoints:
            while obs_count < checkpoint:
                pred.observe(true_lambda)
                obs_count += 1
            prediction = pred.predict()
            uncertainty = pred.uncertainty()
            error_pct = abs(prediction - true_lambda) / true_lambda * 100
            print(
                f"    After {checkpoint:>3} obs: "
                f"prediction={prediction:.6f}  "
                f"uncertainty={uncertainty:.8f}  "
                f"error={error_pct:.2f}%"
            )

        print(f"\n  Final prediction: {pred.predict():.6f}  (true λ = {true_lambda})")

        # Thompson Sampling
        print(f"\n  Thompson Sampling (5 samples from posterior):")
        for i in range(1, 6):
            s = pred.sample()
            print(f"    Sample {i}: {s:.6f}")

    print()


def demo_multi_domain():
    """Demo 3: Multiple predictors for different data domains."""
    print("━━━ Demo 3: Multi-Domain Decay Learning ━━━\n")

    domains = {
        "iot_sensors":   (0.5,   "IoT sensor readings — fast decay"),
        "user_profiles": (0.001, "User profile data — slow decay"),
        "ml_embeddings": (0.01,  "ML embeddings — moderate decay"),
        "config_data":   (0.0,   "Configuration — no decay"),
    }

    predictors = {}
    for domain, (true_lambda, description) in domains.items():
        pred = DecayPredictor(f"domain_{domain}")
        # Train with 100 observations
        if true_lambda > 0:
            for _ in range(100):
                pred.observe(true_lambda)
        predictors[domain] = pred

    print("  Trained predictors for 4 domains:\n")
    print(f"    {'Domain':<20} {'True λ':<12} {'Learned λ':<12} {'Description'}")
    print(f"    {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 30}")

    for domain, (true_lambda, description) in domains.items():
        pred = predictors[domain]
        learned = pred.predict()
        print(f"    {domain:<20} {true_lambda:<12.4f} {learned:<12.6f} {description}")

    # Use the learned rates in a freshness index
    print(f"\n  Using learned rates in FreshnessIndex:\n")
    with FreshnessIndex("multi_domain", threshold=0.5) as idx:
        for domain, (_, _) in domains.items():
            pred = predictors[domain]
            learned_rate = pred.predict()
            key = f"{domain}/latest"
            idx.insert(key, decay_rate=learned_rate)

        # Wait and check
        time.sleep(0.05)

        for domain in domains:
            key = f"{domain}/latest"
            f = idx.get_freshness(key)
            stale = "STALE" if f is not None and f < 0.5 else "fresh"
            print(f"    {key:<30} freshness={f:.4f}  [{stale}]" if f else "")

    # Clean up predictors
    for pred in predictors.values():
        pred.close()

    print()


def demo_cache_eviction():
    """Demo 4: Simulate a freshness-aware cache with automatic eviction."""
    print("━━━ Demo 4: Freshness-Aware Cache Eviction ━━━\n")

    with FreshnessIndex("cache_demo", threshold=0.3) as cache:
        # Simulate cache entries with varying lifetimes
        print("  Simulating cache with 20 entries...\n")

        for i in range(20):
            # Mix of fast and slow decay
            if i < 5:
                rate = 500.0  # Very fast: stale in ~1ms
                label = "ephemeral"
            elif i < 10:
                rate = 50.0   # Fast: stale in ~25ms
                label = "short-lived"
            elif i < 15:
                rate = 1.0    # Moderate: stale in ~1s
                label = "normal"
            else:
                rate = 0.0    # Static: never stale
                label = "permanent"
            cache.insert(f"cache/{label}/{i}", decay_rate=rate)

        print(f"  Inserted {len(cache)} entries")

        # Run eviction cycles
        for cycle in range(1, 4):
            time.sleep(0.03)  # 30ms between cycles
            stale_count = len(cache.query_stale())
            evicted = cache.evict_stale()
            print(
                f"  Cycle {cycle} (t={cycle * 30}ms): "
                f"stale={stale_count}, evicted={evicted}, remaining={len(cache)}"
            )

        print(f"\n  Final cache size: {len(cache)} entries (only long-lived and permanent)")

    print()


def main():
    print("╔════════════════════════════════════════════════╗")
    print("║   DAVO: Decay-Aware Value Optimization         ║")
    print("║   Python Freshness Demo                        ║")
    print("╚════════════════════════════════════════════════╝\n")

    demo_freshness_index()
    demo_decay_predictor()
    demo_multi_domain()
    demo_cache_eviction()

    print("✅ All demos completed successfully!")


if __name__ == "__main__":
    main()

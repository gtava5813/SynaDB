//! Compression Demo
//!
//! This demo shows syna's compression capabilities:
//! - Writing data with compression disabled
//! - Writing data with LZ4 compression only
//! - Writing data with delta compression only
//! - Writing data with both LZ4 and delta compression
//! - Comparing file sizes and compression ratios
//! - Verifying data integrity after compression
//!
//! Run with: cargo run --example compression

use synadb::{Atom, DbConfig, synaDB, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== syna Compression Demo ===\n");

    // Test data: time-series with gradual changes (ideal for delta compression)
    let num_entries = 1000;
    
    // Generate test data: temperature readings with small variations
    let test_data: Vec<f64> = (0..num_entries)
        .map(|i| 20.0 + 0.01 * i as f64 + 0.001 * (i as f64).sin())
        .collect();

    // 1. No compression
    println!("1. Writing {} entries with NO compression...", num_entries);
    let size_none = write_with_config("demo_compress_none.db", &test_data, DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    })?;
    println!("   ✓ File size: {} bytes\n", size_none);

    // 2. LZ4 compression only
    println!("2. Writing {} entries with LZ4 compression only...", num_entries);
    let size_lz4 = write_with_config("demo_compress_lz4.db", &test_data, DbConfig {
        enable_compression: true,
        enable_delta: false,
        sync_on_write: false,
    })?;
    println!("   ✓ File size: {} bytes", size_lz4);
    println!("   ✓ Ratio vs none: {:.2}x\n", size_none as f64 / size_lz4 as f64);

    // 3. Delta compression only
    println!("3. Writing {} entries with DELTA compression only...", num_entries);
    let size_delta = write_with_config("demo_compress_delta.db", &test_data, DbConfig {
        enable_compression: false,
        enable_delta: true,
        sync_on_write: false,
    })?;
    println!("   ✓ File size: {} bytes", size_delta);
    println!("   ✓ Ratio vs none: {:.2}x\n", size_none as f64 / size_delta as f64);

    // 4. Both LZ4 and delta compression
    println!("4. Writing {} entries with BOTH LZ4 and delta compression...", num_entries);
    let size_both = write_with_config("demo_compress_both.db", &test_data, DbConfig {
        enable_compression: true,
        enable_delta: true,
        sync_on_write: false,
    })?;
    println!("   ✓ File size: {} bytes", size_both);
    println!("   ✓ Ratio vs none: {:.2}x\n", size_none as f64 / size_both as f64);

    // 5. Summary comparison
    println!("5. Compression Summary:");
    println!("   ┌─────────────────────┬──────────────┬─────────────┐");
    println!("   │ Configuration       │ Size (bytes) │ Ratio       │");
    println!("   ├─────────────────────┼──────────────┼─────────────┤");
    println!("   │ No compression      │ {:>12} │ 1.00x       │", size_none);
    println!("   │ LZ4 only            │ {:>12} │ {:.2}x       │", size_lz4, size_none as f64 / size_lz4 as f64);
    println!("   │ Delta only          │ {:>12} │ {:.2}x       │", size_delta, size_none as f64 / size_delta as f64);
    println!("   │ LZ4 + Delta         │ {:>12} │ {:.2}x       │", size_both, size_none as f64 / size_both as f64);
    println!("   └─────────────────────┴──────────────┴─────────────┘\n");

    // 6. Verify data integrity
    println!("6. Verifying data integrity after compression...");
    verify_data_integrity("demo_compress_none.db", &test_data)?;
    verify_data_integrity("demo_compress_lz4.db", &test_data)?;
    verify_data_integrity("demo_compress_delta.db", &test_data)?;
    verify_data_integrity("demo_compress_both.db", &test_data)?;
    println!("   ✓ All data verified successfully!\n");

    // 7. Test with different data patterns
    println!("7. Testing compression with different data patterns...\n");
    
    // Random data (poor for delta compression)
    let random_data: Vec<f64> = (0..100)
        .map(|i| rand_float(i as u64) * 1000.0)
        .collect();
    
    println!("   a) Random data (100 entries):");
    let rand_none = write_with_config("demo_rand_none.db", &random_data, DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    })?;
    let rand_delta = write_with_config("demo_rand_delta.db", &random_data, DbConfig {
        enable_compression: false,
        enable_delta: true,
        sync_on_write: false,
    })?;
    println!("      No compression: {} bytes", rand_none);
    println!("      Delta only: {} bytes (ratio: {:.2}x)", rand_delta, rand_none as f64 / rand_delta as f64);
    println!("      Note: Delta compression may not help with random data\n");

    // Constant data (excellent for delta compression)
    let constant_data: Vec<f64> = vec![42.0; 100];
    
    println!("   b) Constant data (100 identical values):");
    let const_none = write_with_config("demo_const_none.db", &constant_data, DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    })?;
    let const_delta = write_with_config("demo_const_delta.db", &constant_data, DbConfig {
        enable_compression: false,
        enable_delta: true,
        sync_on_write: false,
    })?;
    println!("      No compression: {} bytes", const_none);
    println!("      Delta only: {} bytes (ratio: {:.2}x)", const_delta, const_none as f64 / const_delta as f64);
    println!("      Note: Delta compression excels with constant/slowly-changing data\n");

    // Cleanup
    cleanup_demo_files()?;
    println!("=== Demo Complete ===");

    Ok(())
}

/// Writes test data to a database with the given configuration
fn write_with_config(path: &str, data: &[f64], config: DbConfig) -> Result<u64> {
    // Clean up existing file
    if Path::new(path).exists() {
        std::fs::remove_file(path)?;
    }

    let mut db = synaDB::with_config(path, config)?;

    for value in data {
        db.append("sensor/temp", Atom::Float(*value))?;
    }

    db.close()?;

    // Return file size
    Ok(std::fs::metadata(path)?.len())
}

/// Verifies that data can be read back correctly
fn verify_data_integrity(path: &str, expected: &[f64]) -> Result<()> {
    let mut db = synaDB::new(path)?;
    let history = db.get_history_floats("sensor/temp")?;

    assert_eq!(history.len(), expected.len(), "Length mismatch for {}", path);

    for (i, (actual, expected)) in history.iter().zip(expected.iter()).enumerate() {
        // Use approximate comparison for floating point
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-10,
            "Value mismatch at index {} in {}: expected {}, got {} (diff: {})",
            i, path, expected, actual, diff
        );
    }

    db.close()?;
    println!("   ✓ {} - {} values verified", path, expected.len());
    Ok(())
}

/// Simple pseudo-random float generator
fn rand_float(seed: u64) -> f64 {
    let s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (s >> 33) as f64 / (1u64 << 31) as f64
}

/// Cleans up demo files
fn cleanup_demo_files() -> Result<()> {
    let files = [
        "demo_compress_none.db",
        "demo_compress_lz4.db",
        "demo_compress_delta.db",
        "demo_compress_both.db",
        "demo_rand_none.db",
        "demo_rand_delta.db",
        "demo_const_none.db",
        "demo_const_delta.db",
    ];

    for file in &files {
        if Path::new(file).exists() {
            std::fs::remove_file(file)?;
        }
    }

    Ok(())
}



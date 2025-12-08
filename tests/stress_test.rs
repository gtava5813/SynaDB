//! Stress test for high-throughput writes.
//!
//! This test writes 100,000 entries as fast as possible, measures throughput,
//! and verifies all entries are readable after.
//!
//! Requirements: 2.1
//!
//! Run with: cargo test --release --test stress_test -- --nocapture

use std::time::Instant;
use tempfile::tempdir;

use synadb::{Atom, DbConfig, SynaDB};

/// Number of entries to write in the stress test.
const NUM_ENTRIES: usize = 100_000;

/// Test high-throughput sequential writes to different keys.
#[test]
fn stress_test_sequential_writes() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("stress_sequential.db");

    // Use config with sync_on_write disabled for maximum throughput
    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false, // Disable sync for speed
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    println!("\n=== Stress Test: Sequential Writes ===");
    println!("Writing {} entries...", NUM_ENTRIES);

    // Write entries
    let start = Instant::now();
    for i in 0..NUM_ENTRIES {
        let key = format!("key_{:06}", i);
        let value = Atom::Float(i as f64 * 1.5);
        db.append(&key, value).expect("Failed to append");
    }
    let write_duration = start.elapsed();

    // Calculate throughput
    let writes_per_sec = NUM_ENTRIES as f64 / write_duration.as_secs_f64();
    println!(
        "Write time: {:.2}s ({:.0} entries/sec)",
        write_duration.as_secs_f64(),
        writes_per_sec
    );

    // Verify all entries are readable
    println!("Verifying {} entries...", NUM_ENTRIES);
    let start = Instant::now();
    let mut verified = 0;
    for i in 0..NUM_ENTRIES {
        let key = format!("key_{:06}", i);
        let expected = i as f64 * 1.5;
        match db.get(&key) {
            Ok(Some(Atom::Float(f))) => {
                assert!(
                    (f - expected).abs() < 1e-10,
                    "Value mismatch for {}: got {}, expected {}",
                    key,
                    f,
                    expected
                );
                verified += 1;
            }
            Ok(other) => panic!("Unexpected value for {}: {:?}", key, other),
            Err(e) => panic!("Error reading {}: {:?}", key, e),
        }
    }
    let read_duration = start.elapsed();

    let reads_per_sec = NUM_ENTRIES as f64 / read_duration.as_secs_f64();
    println!(
        "Read time: {:.2}s ({:.0} entries/sec)",
        read_duration.as_secs_f64(),
        reads_per_sec
    );
    println!("Verified: {}/{} entries", verified, NUM_ENTRIES);

    assert_eq!(verified, NUM_ENTRIES, "Not all entries were verified");

    // Report file size
    let file_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Database size: {:.2} MB ({} bytes per entry avg)",
        file_size as f64 / 1_000_000.0,
        file_size / NUM_ENTRIES as u64
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

/// Test high-throughput writes to the same key (time-series pattern).
#[test]
fn stress_test_timeseries_writes() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("stress_timeseries.db");

    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    println!("\n=== Stress Test: Time-Series Writes (same key) ===");
    println!("Writing {} entries to single key...", NUM_ENTRIES);

    let key = "sensor_data";

    // Write entries
    let start = Instant::now();
    for i in 0..NUM_ENTRIES {
        let value = Atom::Float(i as f64 * 0.001);
        db.append(key, value).expect("Failed to append");
    }
    let write_duration = start.elapsed();

    let writes_per_sec = NUM_ENTRIES as f64 / write_duration.as_secs_f64();
    println!(
        "Write time: {:.2}s ({:.0} entries/sec)",
        write_duration.as_secs_f64(),
        writes_per_sec
    );

    // Verify latest value
    let expected_latest = (NUM_ENTRIES - 1) as f64 * 0.001;
    match db.get(key) {
        Ok(Some(Atom::Float(f))) => {
            assert!(
                (f - expected_latest).abs() < 1e-10,
                "Latest value mismatch: got {}, expected {}",
                f,
                expected_latest
            );
            println!("Latest value verified: {}", f);
        }
        Ok(other) => panic!("Unexpected value: {:?}", other),
        Err(e) => panic!("Error reading: {:?}", e),
    }

    // Verify history length
    println!("Extracting history tensor...");
    let start = Instant::now();
    let history = db.get_history_floats(key).expect("Failed to get history");
    let history_duration = start.elapsed();

    println!(
        "History extraction: {:.2}s ({} entries)",
        history_duration.as_secs_f64(),
        history.len()
    );

    assert_eq!(
        history.len(),
        NUM_ENTRIES,
        "History length mismatch: got {}, expected {}",
        history.len(),
        NUM_ENTRIES
    );

    // Verify first and last values in history
    assert!(history[0].abs() < 1e-10, "First history value should be ~0");
    assert!(
        (history[NUM_ENTRIES - 1] - expected_latest).abs() < 1e-10,
        "Last history value mismatch"
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

/// Test writes with compression enabled.
#[test]
fn stress_test_compressed_writes() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("stress_compressed.db");

    let config = DbConfig {
        enable_compression: true,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    // Use fewer entries for compressed test (compression has overhead)
    let num_entries = 10_000;

    println!("\n=== Stress Test: Compressed Writes ===");
    println!("Writing {} entries with LZ4 compression...", num_entries);

    // Write large text entries (compression benefits larger data)
    let start = Instant::now();
    for i in 0..num_entries {
        let key = format!("doc_{:05}", i);
        // Create a larger text value that compresses well
        let value = Atom::Text(format!(
            "Document {} contains repeated text. Document {} contains repeated text. \
             Document {} contains repeated text. Document {} contains repeated text.",
            i, i, i, i
        ));
        db.append(&key, value).expect("Failed to append");
    }
    let write_duration = start.elapsed();

    let writes_per_sec = num_entries as f64 / write_duration.as_secs_f64();
    println!(
        "Write time: {:.2}s ({:.0} entries/sec)",
        write_duration.as_secs_f64(),
        writes_per_sec
    );

    // Verify sample entries
    println!("Verifying sample entries...");
    for i in [0, num_entries / 2, num_entries - 1] {
        let key = format!("doc_{:05}", i);
        match db.get(&key) {
            Ok(Some(Atom::Text(t))) => {
                assert!(
                    t.contains(&format!("Document {}", i)),
                    "Content mismatch for {}",
                    key
                );
            }
            Ok(other) => panic!("Unexpected value for {}: {:?}", key, other),
            Err(e) => panic!("Error reading {}: {:?}", key, e),
        }
    }
    println!("Sample entries verified");

    let file_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Database size: {:.2} MB ({} bytes per entry avg)",
        file_size as f64 / 1_000_000.0,
        file_size / num_entries as u64
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

/// Test mixed workload with various data types.
#[test]
fn stress_test_mixed_workload() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("stress_mixed.db");

    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    let num_entries = 25_000; // 25k of each type = 100k total

    println!("\n=== Stress Test: Mixed Workload ===");
    println!(
        "Writing {} entries of each type (Float, Int, Text, Bytes)...",
        num_entries
    );

    let start = Instant::now();

    // Write floats
    for i in 0..num_entries {
        let key = format!("float_{:05}", i);
        db.append(&key, Atom::Float(i as f64 * 3.14159))
            .expect("Failed to append float");
    }

    // Write ints
    for i in 0..num_entries {
        let key = format!("int_{:05}", i);
        db.append(&key, Atom::Int(i as i64 * 1000))
            .expect("Failed to append int");
    }

    // Write text
    for i in 0..num_entries {
        let key = format!("text_{:05}", i);
        db.append(&key, Atom::Text(format!("Value number {}", i)))
            .expect("Failed to append text");
    }

    // Write bytes
    for i in 0..num_entries {
        let key = format!("bytes_{:05}", i);
        let bytes: Vec<u8> = (0..32).map(|j| ((i + j) % 256) as u8).collect();
        db.append(&key, Atom::Bytes(bytes))
            .expect("Failed to append bytes");
    }

    let write_duration = start.elapsed();
    let total_entries = num_entries * 4;
    let writes_per_sec = total_entries as f64 / write_duration.as_secs_f64();

    println!(
        "Write time: {:.2}s ({:.0} entries/sec for {} total entries)",
        write_duration.as_secs_f64(),
        writes_per_sec,
        total_entries
    );

    // Verify samples from each type
    println!("Verifying samples...");

    // Check float
    match db.get("float_00100") {
        Ok(Some(Atom::Float(f))) => {
            assert!((f - 100.0 * 3.14159).abs() < 1e-10);
        }
        other => panic!("Float verification failed: {:?}", other),
    }

    // Check int
    match db.get("int_00100") {
        Ok(Some(Atom::Int(i))) => {
            assert_eq!(i, 100_000);
        }
        other => panic!("Int verification failed: {:?}", other),
    }

    // Check text
    match db.get("text_00100") {
        Ok(Some(Atom::Text(t))) => {
            assert_eq!(t, "Value number 100");
        }
        other => panic!("Text verification failed: {:?}", other),
    }

    // Check bytes
    match db.get("bytes_00100") {
        Ok(Some(Atom::Bytes(b))) => {
            assert_eq!(b.len(), 32);
            assert_eq!(b[0], 100);
        }
        other => panic!("Bytes verification failed: {:?}", other),
    }

    println!("All samples verified");

    let file_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Database size: {:.2} MB ({} bytes per entry avg)",
        file_size as f64 / 1_000_000.0,
        file_size / total_entries as u64
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

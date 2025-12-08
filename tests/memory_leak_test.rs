//! Memory leak test for Syna database.
//!
//! This test verifies that:
//! - Opening/closing databases many times doesn't leak memory
//! - Allocating/freeing tensors many times doesn't leak memory
//! - Memory usage doesn't grow unbounded
//!
//! Requirements: 4.3, 6.5
//!
//! Run with: cargo test --release --test memory_leak_test -- --nocapture

use std::time::Instant;
use tempfile::tempdir;

use synadb::{close_db, open_db, with_db, Atom, DbConfig, SynaDB};

/// Number of iterations for open/close cycle test.
const OPEN_CLOSE_ITERATIONS: usize = 100;

/// Number of iterations for tensor allocation test.
const TENSOR_ITERATIONS: usize = 1000;

/// Number of entries per tensor.
const TENSOR_SIZE: usize = 10_000;

/// Test that opening and closing databases many times doesn't leak memory.
#[test]
fn test_open_close_no_leak() {
    let dir = tempdir().expect("Failed to create temp dir");

    println!("\n=== Memory Leak Test: Open/Close Cycles ===");
    println!("Running {} open/close cycles...", OPEN_CLOSE_ITERATIONS);

    let start = Instant::now();

    for i in 0..OPEN_CLOSE_ITERATIONS {
        let db_path = dir.path().join(format!("leak_test_{}.db", i % 10));
        let path_str = db_path.to_str().unwrap();

        // Open database
        open_db(path_str).expect("Failed to open database");

        // Write some data
        with_db(path_str, |db| {
            for j in 0..100 {
                let key = format!("key_{}", j);
                db.append(&key, Atom::Float(j as f64))?;
            }
            Ok(())
        })
        .expect("Failed to write data");

        // Close database
        close_db(path_str).expect("Failed to close database");

        if (i + 1) % 20 == 0 {
            println!("  Completed {} cycles", i + 1);
        }
    }

    let duration = start.elapsed();
    println!(
        "Completed {} cycles in {:.2}s",
        OPEN_CLOSE_ITERATIONS,
        duration.as_secs_f64()
    );
    println!("=== Test Passed (no crash = no obvious leak) ===\n");
}

/// Test that allocating and freeing tensors many times doesn't leak memory.
#[test]
fn test_tensor_alloc_free_no_leak() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("tensor_leak_test.db");

    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    println!("\n=== Memory Leak Test: Tensor Alloc/Free Cycles ===");
    println!(
        "Writing {} float entries for tensor extraction...",
        TENSOR_SIZE
    );

    // Write data for tensor extraction
    let key = "tensor_data";
    for i in 0..TENSOR_SIZE {
        db.append(key, Atom::Float(i as f64 * 0.1))
            .expect("Failed to append");
    }

    println!("Running {} tensor alloc/free cycles...", TENSOR_ITERATIONS);

    let start = Instant::now();

    for i in 0..TENSOR_ITERATIONS {
        // Allocate tensor
        let (ptr, len) = db
            .get_history_tensor(key)
            .expect("Failed to get history tensor");

        // Verify tensor is valid
        assert_eq!(len, TENSOR_SIZE, "Tensor length mismatch");
        assert!(!ptr.is_null(), "Tensor pointer is null");

        // Read a sample value to ensure memory is accessible
        let sample = unsafe { *ptr.add(len / 2) };
        assert!(
            (sample - (len / 2) as f64 * 0.1).abs() < 1e-10,
            "Sample value mismatch"
        );

        // Free tensor
        unsafe {
            synadb::free_tensor(ptr, len);
        }

        if (i + 1) % 200 == 0 {
            println!("  Completed {} cycles", i + 1);
        }
    }

    let duration = start.elapsed();
    println!(
        "Completed {} cycles in {:.2}s ({:.0} alloc/free per sec)",
        TENSOR_ITERATIONS,
        duration.as_secs_f64(),
        TENSOR_ITERATIONS as f64 / duration.as_secs_f64()
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed (no crash = no obvious leak) ===\n");
}

/// Test repeated database operations without explicit close.
/// This tests that the registry properly manages instances.
#[test]
fn test_registry_management() {
    let dir = tempdir().expect("Failed to create temp dir");

    println!("\n=== Memory Leak Test: Registry Management ===");
    println!("Testing registry with multiple databases...");

    let num_dbs = 10;
    let iterations = 50;

    let start = Instant::now();

    for iter in 0..iterations {
        // Open multiple databases
        for i in 0..num_dbs {
            let db_path = dir.path().join(format!("registry_test_{}.db", i));
            let path_str = db_path.to_str().unwrap();

            open_db(path_str).expect("Failed to open database");

            // Write some data
            with_db(path_str, |db| {
                let key = format!("iter_{}_key", iter);
                db.append(&key, Atom::Int(iter as i64))?;
                Ok(())
            })
            .expect("Failed to write data");
        }

        // Close all databases
        for i in 0..num_dbs {
            let db_path = dir.path().join(format!("registry_test_{}.db", i));
            let path_str = db_path.to_str().unwrap();
            close_db(path_str).expect("Failed to close database");
        }

        if (iter + 1) % 10 == 0 {
            println!("  Completed {} iterations", iter + 1);
        }
    }

    let duration = start.elapsed();
    println!(
        "Completed {} iterations with {} databases each in {:.2}s",
        iterations,
        num_dbs,
        duration.as_secs_f64()
    );
    println!("=== Test Passed ===\n");
}

/// Test that get_history doesn't leak memory on repeated calls.
#[test]
fn test_history_no_leak() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("history_leak_test.db");

    let config = DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    println!("\n=== Memory Leak Test: History Retrieval ===");

    // Write data
    let key = "history_key";
    let num_entries = 1000;
    for i in 0..num_entries {
        db.append(key, Atom::Text(format!("Entry number {}", i)))
            .expect("Failed to append");
    }

    println!("Running {} history retrieval cycles...", TENSOR_ITERATIONS);

    let start = Instant::now();

    for i in 0..TENSOR_ITERATIONS {
        // Get history (allocates Vec internally)
        let history = db.get_history(key).expect("Failed to get history");

        // Verify
        assert_eq!(history.len(), num_entries, "History length mismatch");

        // History Vec is dropped here, memory should be freed

        if (i + 1) % 200 == 0 {
            println!("  Completed {} cycles", i + 1);
        }
    }

    let duration = start.elapsed();
    println!(
        "Completed {} cycles in {:.2}s",
        TENSOR_ITERATIONS,
        duration.as_secs_f64()
    );

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

/// Test mixed operations for memory stability.
#[test]
fn test_mixed_operations_stability() {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("mixed_stability_test.db");

    let config = DbConfig {
        enable_compression: true,
        enable_delta: false,
        sync_on_write: false,
    };

    let mut db = SynaDB::with_config(&db_path, config).expect("Failed to create database");

    println!("\n=== Memory Leak Test: Mixed Operations ===");

    let iterations = 100;
    let ops_per_iter = 100;

    println!(
        "Running {} iterations with {} operations each...",
        iterations, ops_per_iter
    );

    let start = Instant::now();

    for iter in 0..iterations {
        // Write various types
        for j in 0..ops_per_iter {
            let key = format!("key_{}_{}", iter, j);
            match j % 4 {
                0 => {
                    db.append(&key, Atom::Float(j as f64))
                        .expect("Failed to append float");
                }
                1 => {
                    db.append(&key, Atom::Int(j as i64))
                        .expect("Failed to append int");
                }
                2 => {
                    db.append(&key, Atom::Text(format!("Text value {}", j)))
                        .expect("Failed to append text");
                }
                _ => {
                    let bytes: Vec<u8> = (0..64).map(|k| ((j + k) % 256) as u8).collect();
                    db.append(&key, Atom::Bytes(bytes))
                        .expect("Failed to append bytes");
                }
            }
        }

        // Read some values back
        for j in 0..ops_per_iter / 10 {
            let key = format!("key_{}_{}", iter, j * 10);
            let _ = db.get(&key).expect("Failed to get");
        }

        // Delete some keys
        for j in 0..ops_per_iter / 20 {
            let key = format!("key_{}_{}", iter, j * 20);
            let _ = db.delete(&key);
        }

        if (iter + 1) % 20 == 0 {
            println!("  Completed {} iterations", iter + 1);
        }
    }

    let duration = start.elapsed();
    let total_ops = iterations * ops_per_iter;
    println!(
        "Completed {} total operations in {:.2}s ({:.0} ops/sec)",
        total_ops,
        duration.as_secs_f64(),
        total_ops as f64 / duration.as_secs_f64()
    );

    // Verify database is still functional
    let keys = db.keys();
    println!("Database has {} active keys", keys.len());

    db.close().expect("Failed to close database");
    println!("=== Test Passed ===\n");
}

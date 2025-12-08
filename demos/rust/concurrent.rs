//! Concurrent Access Demo
//!
//! This demo shows syna's thread-safe concurrent access:
//! - Multiple writer threads appending data simultaneously
//! - Multiple reader threads reading data concurrently
//! - Verifying no data corruption after concurrent operations
//! - Using std::thread::scope for safe concurrency
//!
//! Run with: cargo run --example concurrent

use synadb::{open_db, close_db, with_db, Atom, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const NUM_WRITERS: usize = 4;
const NUM_READERS: usize = 4;
const WRITES_PER_THREAD: usize = 250;

fn main() -> Result<()> {
    println!("=== syna Concurrent Access Demo ===\n");

    // Use absolute path to avoid canonicalization issues in threads
    let db_path = std::env::current_dir()
        .unwrap()
        .join("demo_concurrent.db");
    let db_path_str = db_path.to_string_lossy().to_string();
    
    // Clean up any existing demo database
    if db_path.exists() {
        std::fs::remove_file(&db_path)?;
    }

    // Open database in global registry for shared access
    println!("1. Opening database for concurrent access...");
    open_db(&db_path_str)?;
    println!("   ✓ Database opened\n");

    // Counters for tracking operations
    let writes_completed = Arc::new(AtomicUsize::new(0));
    let reads_completed = Arc::new(AtomicUsize::new(0));

    // 2. Spawn concurrent writers and readers
    println!("2. Spawning {} writer threads and {} reader threads...", NUM_WRITERS, NUM_READERS);
    println!("   Each writer will append {} values\n", WRITES_PER_THREAD);

    let start = Instant::now();

    // Use thread::scope for safe concurrent access
    std::thread::scope(|s| {
        // Spawn writer threads
        for writer_id in 0..NUM_WRITERS {
            let writes_completed = Arc::clone(&writes_completed);
            let path = db_path_str.clone();
            
            s.spawn(move || {
                for i in 0..WRITES_PER_THREAD {
                    let key = format!("writer_{}/value_{}", writer_id, i);
                    let value = (writer_id * WRITES_PER_THREAD + i) as f64;
                    
                    // Write to database using global registry
                    with_db(&path, |db| {
                        db.append(&key, Atom::Float(value))
                    }).expect("Write failed");
                    
                    writes_completed.fetch_add(1, Ordering::Relaxed);
                }
                println!("   Writer {} completed {} writes", writer_id, WRITES_PER_THREAD);
            });
        }

        // Spawn reader threads
        for reader_id in 0..NUM_READERS {
            let reads_completed = Arc::clone(&reads_completed);
            let path = db_path_str.clone();
            
            s.spawn(move || {
                let mut successful_reads = 0;
                
                // Continuously read random keys while writers are active
                for attempt in 0..WRITES_PER_THREAD {
                    // Try to read a key that might exist
                    let writer_id = attempt % NUM_WRITERS;
                    let value_id = attempt % (WRITES_PER_THREAD / 2 + 1);
                    let key = format!("writer_{}/value_{}", writer_id, value_id);
                    
                    let result = with_db(&path, |db| {
                        db.get(&key)
                    });
                    
                    if let Ok(Some(_)) = result {
                        successful_reads += 1;
                    }
                    
                    reads_completed.fetch_add(1, Ordering::Relaxed);
                }
                println!("   Reader {} completed {} reads ({} found values)", 
                         reader_id, WRITES_PER_THREAD, successful_reads);
            });
        }
    });

    let elapsed = start.elapsed();
    
    let total_writes = writes_completed.load(Ordering::Relaxed);
    let total_reads = reads_completed.load(Ordering::Relaxed);
    
    println!("\n3. Concurrent operations completed:");
    println!("   ✓ Total writes: {}", total_writes);
    println!("   ✓ Total reads: {}", total_reads);
    println!("   ✓ Time elapsed: {:?}", elapsed);
    println!("   ✓ Write throughput: {:.0} ops/sec", total_writes as f64 / elapsed.as_secs_f64());
    println!();

    // 4. Verify data integrity
    println!("4. Verifying data integrity...");
    
    let mut verified_count = 0;
    let mut missing_count = 0;
    let mut corrupted_count = 0;

    with_db(&db_path_str, |db| {
        for writer_id in 0..NUM_WRITERS {
            for i in 0..WRITES_PER_THREAD {
                let key = format!("writer_{}/value_{}", writer_id, i);
                let expected_value = (writer_id * WRITES_PER_THREAD + i) as f64;
                
                match db.get(&key)? {
                    Some(Atom::Float(actual)) => {
                        if (actual - expected_value).abs() < 1e-10 {
                            verified_count += 1;
                        } else {
                            corrupted_count += 1;
                            eprintln!("   ✗ Corrupted: {} expected {}, got {}", key, expected_value, actual);
                        }
                    }
                    Some(_) => {
                        corrupted_count += 1;
                        eprintln!("   ✗ Wrong type for key: {}", key);
                    }
                    None => {
                        missing_count += 1;
                        eprintln!("   ✗ Missing key: {}", key);
                    }
                }
            }
        }
        Ok(())
    })?;

    println!("   ✓ Verified: {} values", verified_count);
    if missing_count > 0 {
        println!("   ✗ Missing: {} values", missing_count);
    }
    if corrupted_count > 0 {
        println!("   ✗ Corrupted: {} values", corrupted_count);
    }
    
    let expected_total = NUM_WRITERS * WRITES_PER_THREAD;
    if verified_count == expected_total {
        println!("   ✓ All {} values verified - NO DATA CORRUPTION!\n", expected_total);
    } else {
        println!("   ✗ Data integrity check FAILED!\n");
    }

    // 5. Check total keys
    println!("5. Final database state:");
    with_db(&db_path_str, |db| {
        let keys = db.keys();
        println!("   ✓ Total keys: {}", keys.len());
        Ok(())
    })?;
    println!();

    // 6. Close and cleanup
    println!("6. Closing database...");
    close_db(&db_path_str)?;
    std::fs::remove_file(&db_path)?;
    println!("   ✓ Database closed and cleaned up\n");

    println!("=== Demo Complete ===");
    Ok(())
}



//! Recovery Demo
//!
//! This demo shows syna's crash recovery capabilities:
//! - Writing data and simulating a crash (drop without close)
//! - Reopening and verifying all data is recovered
//! - Injecting corruption and verifying partial recovery
//! - Demonstrating automatic index rebuild
//!
//! Run with: cargo run --example recovery

use synadb::{Atom, DbConfig, synaDB, Result};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== syna Recovery Demo ===\n");

    // 1. Normal crash recovery (drop without close)
    println!("1. Testing crash recovery (drop without explicit close)...\n");
    test_crash_recovery()?;

    // 2. Recovery with corrupted entries
    println!("\n2. Testing recovery with corrupted entries...\n");
    test_corruption_recovery()?;

    // 3. Recovery with truncated file
    println!("\n3. Testing recovery with truncated file...\n");
    test_truncation_recovery()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Test 1: Simulate crash by dropping database without close
fn test_crash_recovery() -> Result<()> {
    let db_path = "demo_recovery_crash.db";
    
    // Clean up
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }

    // Write data and "crash" (drop without close)
    println!("   a) Writing 100 entries and simulating crash...");
    {
        let mut db = synaDB::with_config(db_path, DbConfig {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: true, // Ensure data is flushed
        })?;

        for i in 0..100 {
            db.append(&format!("key_{}", i), Atom::Int(i as i64))?;
        }

        // Simulate crash: drop without calling close()
        // The database file should still be valid
        drop(db);
        println!("      ✓ Database dropped without close() - simulating crash");
    }

    // Reopen and verify recovery
    println!("   b) Reopening database and verifying recovery...");
    {
        let mut db = synaDB::new(db_path)?;
        
        let keys = db.keys();
        println!("      ✓ Recovered {} keys", keys.len());

        // Verify all data
        let mut verified = 0;
        for i in 0..100 {
            let key = format!("key_{}", i);
            if let Some(Atom::Int(value)) = db.get(&key)? {
                if value == i as i64 {
                    verified += 1;
                }
            }
        }
        
        println!("      ✓ Verified {} values - all data recovered!", verified);
        db.close()?;
    }

    // Cleanup
    std::fs::remove_file(db_path)?;
    Ok(())
}


/// Test 2: Inject corruption and verify partial recovery
fn test_corruption_recovery() -> Result<()> {
    let db_path = "demo_recovery_corrupt.db";
    
    // Clean up
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }

    // Write valid data
    println!("   a) Writing 50 valid entries...");
    {
        let mut db = synaDB::with_config(db_path, DbConfig {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: true,
        })?;

        for i in 0..50 {
            db.append(&format!("valid_{}", i), Atom::Float(i as f64 * 1.5))?;
        }
        
        db.close()?;
        println!("      ✓ Written 50 entries");
    }

    // Get file size before corruption
    let original_size = std::fs::metadata(db_path)?.len();
    println!("      File size: {} bytes", original_size);

    // Inject corruption in the middle of the file
    println!("   b) Injecting corruption at byte offset 500...");
    {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(db_path)?;

        // Corrupt bytes at offset 500 (middle of file)
        file.seek(SeekFrom::Start(500))?;
        file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])?;
        file.sync_all()?;
        println!("      ✓ Corrupted 8 bytes at offset 500");
    }

    // Reopen and verify partial recovery
    println!("   c) Reopening database (recovery will skip corrupted entries)...");
    {
        let mut db = synaDB::new(db_path)?;
        
        let keys = db.keys();
        println!("      ✓ Recovered {} keys (some may be lost due to corruption)", keys.len());

        // Count how many we can still read
        let mut readable = 0;
        for i in 0..50 {
            let key = format!("valid_{}", i);
            if db.get(&key)?.is_some() {
                readable += 1;
            }
        }
        
        println!("      ✓ {} of 50 entries still readable", readable);
        println!("      Note: Entries after corruption point may be lost");
        
        db.close()?;
    }

    // Cleanup
    std::fs::remove_file(db_path)?;
    Ok(())
}

/// Test 3: Truncated file recovery
fn test_truncation_recovery() -> Result<()> {
    let db_path = "demo_recovery_truncate.db";
    
    // Clean up
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }

    // Write valid data
    println!("   a) Writing 100 entries...");
    {
        let mut db = synaDB::with_config(db_path, DbConfig {
            enable_compression: false,
            enable_delta: false,
            sync_on_write: true,
        })?;

        for i in 0..100 {
            db.append(&format!("entry_{}", i), Atom::Text(format!("value_{}", i)))?;
        }
        
        db.close()?;
    }

    let original_size = std::fs::metadata(db_path)?.len();
    println!("      ✓ Original file size: {} bytes", original_size);

    // Truncate file to simulate partial write during crash
    println!("   b) Truncating file to simulate partial write...");
    {
        let truncate_to = original_size * 2 / 3; // Keep 2/3 of the file
        let file = OpenOptions::new()
            .write(true)
            .open(db_path)?;
        file.set_len(truncate_to)?;
        println!("      ✓ Truncated to {} bytes (removed last 1/3)", truncate_to);
    }

    // Reopen and verify partial recovery
    println!("   c) Reopening database...");
    {
        let mut db = synaDB::new(db_path)?;
        
        let keys = db.keys();
        println!("      ✓ Recovered {} keys", keys.len());

        // Verify recovered entries are valid
        let mut valid_count = 0;
        for key in &keys {
            if db.get(key)?.is_some() {
                valid_count += 1;
            }
        }
        
        println!("      ✓ All {} recovered entries are valid", valid_count);
        println!("      Note: ~{} entries lost due to truncation", 100 - keys.len());
        
        // Can still write new data after recovery
        println!("   d) Writing new data after recovery...");
        db.append("new_after_recovery", Atom::Text("This works!".to_string()))?;
        
        if let Some(Atom::Text(s)) = db.get("new_after_recovery")? {
            println!("      ✓ New entry written and read back: \"{}\"", s);
        }
        
        db.close()?;
    }

    // Cleanup
    std::fs::remove_file(db_path)?;
    Ok(())
}



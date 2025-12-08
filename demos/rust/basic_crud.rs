//! Basic CRUD Operations Demo
//!
//! This demo shows fundamental database operations:
//! - Creating/opening a database
//! - Writing values of different types
//! - Reading values back
//! - Deleting keys
//! - Listing keys
//!
//! Run with: cargo run --example basic_crud

use synadb::{Atom, synaDB, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== syna Basic CRUD Demo ===\n");
    
    // Clean up any existing demo database
    let db_path = "demo_basic.db";
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }
    
    // 1. Create/Open Database
    println!("1. Creating database at '{}'...", db_path);
    let mut db = synaDB::new(db_path)?;
    println!("   ✓ Database created successfully\n");
    
    // 2. Write Different Data Types
    println!("2. Writing values of different types...");
    
    // Float (f64)
    let offset = db.append("sensor/temperature", Atom::Float(23.5))?;
    println!("   ✓ Float: sensor/temperature = 23.5 (offset: {})", offset);
    
    // Integer (i64)
    let offset = db.append("counter/visits", Atom::Int(42))?;
    println!("   ✓ Int: counter/visits = 42 (offset: {})", offset);
    
    // Text (String)
    let offset = db.append("config/name", Atom::Text("syna Demo".to_string()))?;
    println!("   ✓ Text: config/name = \"syna Demo\" (offset: {})", offset);
    
    // Bytes (Vec<u8>)
    let data = vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello" in ASCII
    let offset = db.append("binary/greeting", Atom::Bytes(data))?;
    println!("   ✓ Bytes: binary/greeting = [0x48, 0x65, 0x6c, 0x6c, 0x6f] (offset: {})", offset);
    
    // Null
    let offset = db.append("optional/value", Atom::Null)?;
    println!("   ✓ Null: optional/value = Null (offset: {})\n", offset);
    
    // 3. Read Values Back
    println!("3. Reading values back...");
    
    if let Some(Atom::Float(temp)) = db.get("sensor/temperature")? {
        println!("   ✓ sensor/temperature = {}", temp);
    }
    
    if let Some(Atom::Int(count)) = db.get("counter/visits")? {
        println!("   ✓ counter/visits = {}", count);
    }
    
    if let Some(Atom::Text(name)) = db.get("config/name")? {
        println!("   ✓ config/name = \"{}\"", name);
    }
    
    if let Some(Atom::Bytes(bytes)) = db.get("binary/greeting")? {
        let text = String::from_utf8_lossy(&bytes);
        println!("   ✓ binary/greeting = {:?} (\"{}\")", bytes, text);
    }
    
    // Reading non-existent key
    match db.get("nonexistent/key")? {
        Some(_) => println!("   ✗ Unexpected value for nonexistent key"),
        None => println!("   ✓ nonexistent/key = None (as expected)"),
    }
    println!();
    
    // 4. Update (Append new value to same key)
    println!("4. Updating values (append new value)...");
    db.append("sensor/temperature", Atom::Float(24.1))?;
    db.append("sensor/temperature", Atom::Float(24.8))?;
    
    if let Some(Atom::Float(temp)) = db.get("sensor/temperature")? {
        println!("   ✓ sensor/temperature updated to {} (latest value)", temp);
    }
    println!();
    
    // 5. List All Keys
    println!("5. Listing all keys...");
    let keys = db.keys();
    for key in &keys {
        println!("   - {}", key);
    }
    println!("   Total: {} keys\n", keys.len());
    
    // 6. Check Key Existence
    println!("6. Checking key existence...");
    println!("   ✓ exists(\"sensor/temperature\") = {}", db.exists("sensor/temperature"));
    println!("   ✓ exists(\"nonexistent\") = {}\n", db.exists("nonexistent"));
    
    // 7. Delete a Key
    println!("7. Deleting a key...");
    db.delete("counter/visits")?;
    println!("   ✓ Deleted counter/visits");
    println!("   ✓ exists(\"counter/visits\") = {}", db.exists("counter/visits"));
    println!("   ✓ get(\"counter/visits\") = {:?}\n", db.get("counter/visits")?);
    
    // 8. List Keys After Delete
    println!("8. Keys after deletion...");
    let keys = db.keys();
    for key in &keys {
        println!("   - {}", key);
    }
    println!("   Total: {} keys\n", keys.len());
    
    // 9. Close Database
    println!("9. Closing database...");
    db.close()?;
    println!("   ✓ Database closed\n");
    
    // 10. Reopen and Verify
    println!("10. Reopening database to verify persistence...");
    let mut db = synaDB::new(db_path)?;
    
    if let Some(Atom::Float(temp)) = db.get("sensor/temperature")? {
        println!("   ✓ sensor/temperature = {} (persisted!)", temp);
    }
    
    let keys = db.keys();
    println!("   ✓ {} keys recovered\n", keys.len());
    
    // Cleanup
    db.close()?;
    std::fs::remove_file(db_path)?;
    println!("=== Demo Complete ===");
    
    Ok(())
}



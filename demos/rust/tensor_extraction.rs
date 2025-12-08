//! Tensor Extraction Demo
//!
//! This demo shows syna's tensor extraction capabilities for ML:
//! - Storing float sequences for multiple keys
//! - Extracting history as Vec<f64> for Rust ML
//! - Extracting as raw pointers (simulating FFI for Python/C++)
//! - Demonstrating memory management with free_tensor
//!
//! Run with: cargo run --example tensor_extraction

use synadb::{free_tensor, Atom, synaDB, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== syna Tensor Extraction Demo ===\n");

    let db_path = "demo_tensor.db";
    
    // Clean up
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }

    // Use fast config (no sync on write) for demo
    let mut db = synaDB::with_config(db_path, synadb::DbConfig {
        enable_compression: false,
        enable_delta: false,
        sync_on_write: false,
    })?;

    // 1. Store float sequences for multiple sensors
    println!("1. Storing float sequences for multiple sensors...\n");
    
    let sensors = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"];
    let num_samples = 500;

    for (sensor_idx, sensor) in sensors.iter().enumerate() {
        let key = format!("imu/{}", sensor);
        
        for i in 0..num_samples {
            // Generate realistic IMU-like data
            let t = i as f64 * 0.01; // 100 Hz sampling
            let value = match sensor_idx {
                0 => 0.1 * (t * 2.0).sin(),           // accel_x: oscillating
                1 => 0.05 * (t * 3.0).cos(),          // accel_y: oscillating
                2 => 9.81 + 0.02 * (t * 0.5).sin(),   // accel_z: gravity + noise
                3 => 0.01 * (t * 5.0).sin(),          // gyro_x: small rotation
                4 => 0.02 * (t * 4.0).cos(),          // gyro_y: small rotation
                5 => 0.005 * t,                        // gyro_z: slow drift
                _ => 0.0,
            };
            
            db.append(&key, Atom::Float(value))?;
        }
        
        println!("   ✓ {} - {} samples stored", sensor, num_samples);
    }
    println!();

    // 2. Extract as Vec<f64> (Rust-native)
    println!("2. Extracting tensors as Vec<f64> (Rust-native)...\n");
    
    for sensor in &sensors {
        let key = format!("imu/{}", sensor);
        let tensor = db.get_history_floats(&key)?;
        
        // Calculate statistics
        let mean = tensor.iter().sum::<f64>() / tensor.len() as f64;
        let variance = tensor.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / tensor.len() as f64;
        let std_dev = variance.sqrt();
        let min = tensor.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = tensor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        println!("   {} [{}]:", sensor, tensor.len());
        println!("      Mean: {:>10.6}  Std: {:>10.6}", mean, std_dev);
        println!("      Min:  {:>10.6}  Max: {:>10.6}", min, max);
    }
    println!();

    // 3. Extract as raw pointer (FFI simulation)
    println!("3. Extracting as raw pointer (simulating FFI for Python/C++)...\n");
    
    let key = "imu/accel_z";
    let (ptr, len) = db.get_history_tensor(key)?;
    
    println!("   Key: {}", key);
    println!("   Pointer: {:?}", ptr);
    println!("   Length: {} elements ({} bytes)", len, len * std::mem::size_of::<f64>());
    
    // Access data through raw pointer (unsafe, but this is what FFI does)
    unsafe {
        println!("\n   First 5 values (via raw pointer):");
        for i in 0..5.min(len) {
            let value = *ptr.add(i);
            println!("      [{}] = {:.6}", i, value);
        }
        
        println!("   ...");
        
        println!("   Last 5 values (via raw pointer):");
        for i in (len - 5).max(0)..len {
            let value = *ptr.add(i);
            println!("      [{}] = {:.6}", i, value);
        }
    }
    println!();

    // 4. Demonstrate memory management
    println!("4. Demonstrating memory management with free_tensor...\n");
    
    println!("   Before free: pointer is valid at {:?}", ptr);
    
    // IMPORTANT: Must free the tensor to avoid memory leak
    unsafe {
        free_tensor(ptr, len);
    }
    
    println!("   After free: memory has been deallocated");
    println!("   ✓ No memory leak - tensor properly freed\n");

    // 5. Multi-tensor extraction for ML batch processing
    println!("5. Multi-tensor extraction for ML batch processing...\n");
    
    // Extract all accelerometer data as a "batch"
    let accel_keys = ["imu/accel_x", "imu/accel_y", "imu/accel_z"];
    let mut batch: Vec<Vec<f64>> = Vec::new();
    
    for key in &accel_keys {
        let tensor = db.get_history_floats(key)?;
        batch.push(tensor);
    }
    
    println!("   Extracted batch shape: [{} channels, {} samples]", batch.len(), batch[0].len());
    
    // Simulate computing magnitude: sqrt(x^2 + y^2 + z^2)
    let magnitude: Vec<f64> = (0..batch[0].len())
        .map(|i| {
            let x = batch[0][i];
            let y = batch[1][i];
            let z = batch[2][i];
            (x * x + y * y + z * z).sqrt()
        })
        .collect();
    
    let mag_mean = magnitude.iter().sum::<f64>() / magnitude.len() as f64;
    let mag_min = magnitude.iter().cloned().fold(f64::INFINITY, f64::min);
    let mag_max = magnitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    println!("   Computed acceleration magnitude:");
    println!("      Mean: {:.6} m/s²", mag_mean);
    println!("      Min:  {:.6} m/s²", mag_min);
    println!("      Max:  {:.6} m/s²", mag_max);
    println!("   (Expected ~9.81 m/s² for gravity)\n");

    // 6. Mixed type handling
    println!("6. Demonstrating mixed type filtering...\n");
    
    // Store mixed types under same key
    db.append("mixed/data", Atom::Float(1.0))?;
    db.append("mixed/data", Atom::Int(42))?;  // This will be filtered out
    db.append("mixed/data", Atom::Float(2.0))?;
    db.append("mixed/data", Atom::Text("hello".to_string()))?;  // Filtered out
    db.append("mixed/data", Atom::Float(3.0))?;
    
    let all_history = db.get_history("mixed/data")?;
    let float_tensor = db.get_history_floats("mixed/data")?;
    
    println!("   Stored 5 values (3 Float, 1 Int, 1 Text)");
    println!("   get_history() returns: {} atoms", all_history.len());
    println!("   get_history_floats() returns: {} floats (non-floats filtered)", float_tensor.len());
    println!("   Float values: {:?}\n", float_tensor);

    // Cleanup
    db.close()?;
    std::fs::remove_file(db_path)?;
    
    println!("=== Demo Complete ===");
    Ok(())
}



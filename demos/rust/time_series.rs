//! Time-Series Data Demo
//!
//! This demo shows how to use syna for time-series data:
//! - Appending sequential sensor readings
//! - Extracting full history
//! - Getting float tensors for ML
//!
//! Run with: cargo run --example time_series

use synadb::{Atom, synaDB, Result};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== syna Time-Series Demo ===\n");
    
    let db_path = "demo_timeseries.db";
    if Path::new(db_path).exists() {
        std::fs::remove_file(db_path)?;
    }
    
    let mut db = synaDB::new(db_path)?;
    
    // 1. Simulate Sensor Data
    println!("1. Simulating sensor data (1000 readings)...");
    
    let sensors = ["temperature", "humidity", "pressure"];
    
    for i in 0..1000 {
        // Simulate realistic sensor values with some noise
        let time = i as f64 * 0.1; // 0.1 second intervals
        
        // Temperature: 20-25°C with daily cycle
        let temp = 22.5 + 2.5 * (time * 0.01).sin() + (rand_float() - 0.5) * 0.5;
        db.append("sensor/temperature", Atom::Float(temp))?;
        
        // Humidity: 40-60% with inverse correlation to temperature
        let humidity = 50.0 - 10.0 * (time * 0.01).sin() + (rand_float() - 0.5) * 2.0;
        db.append("sensor/humidity", Atom::Float(humidity))?;
        
        // Pressure: 1013 hPa with slow drift
        let pressure = 1013.0 + 5.0 * (time * 0.001).sin() + (rand_float() - 0.5) * 0.5;
        db.append("sensor/pressure", Atom::Float(pressure))?;
    }
    
    println!("   ✓ Written 3000 sensor readings (1000 per sensor)\n");
    
    // 2. Get Latest Values
    println!("2. Latest sensor values:");
    for sensor in &sensors {
        let key = format!("sensor/{}", sensor);
        if let Some(Atom::Float(value)) = db.get(&key)? {
            println!("   {} = {:.2}", sensor, value);
        }
    }
    println!();
    
    // 3. Extract Full History
    println!("3. Extracting history for temperature sensor...");
    let history = db.get_history("sensor/temperature")?;
    println!("   ✓ Retrieved {} values", history.len());
    
    // Show first and last few values
    println!("   First 5 values:");
    for (i, atom) in history.iter().take(5).enumerate() {
        if let Atom::Float(v) = atom {
            println!("      [{}] {:.2}°C", i, v);
        }
    }
    println!("   ...");
    println!("   Last 5 values:");
    for (i, atom) in history.iter().rev().take(5).rev().enumerate() {
        if let Atom::Float(v) = atom {
            println!("      [{}] {:.2}°C", history.len() - 5 + i, v);
        }
    }
    println!();
    
    // 4. Extract as Float Tensor (for ML)
    println!("4. Extracting as float tensor (for ML)...");
    let tensor = db.get_history_floats("sensor/temperature")?;
    println!("   ✓ Tensor shape: [{}]", tensor.len());
    
    // Calculate statistics
    let sum: f64 = tensor.iter().sum();
    let mean = sum / tensor.len() as f64;
    let min = tensor.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = tensor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    println!("   Statistics:");
    println!("      Mean: {:.2}°C", mean);
    println!("      Min:  {:.2}°C", min);
    println!("      Max:  {:.2}°C", max);
    println!();
    
    // 5. Multi-Sensor Analysis
    println!("5. Multi-sensor correlation analysis...");
    let temp_data = db.get_history_floats("sensor/temperature")?;
    let humidity_data = db.get_history_floats("sensor/humidity")?;
    
    // Simple correlation calculation
    let n = temp_data.len().min(humidity_data.len());
    let temp_mean: f64 = temp_data.iter().take(n).sum::<f64>() / n as f64;
    let hum_mean: f64 = humidity_data.iter().take(n).sum::<f64>() / n as f64;
    
    let mut cov = 0.0;
    let mut temp_var = 0.0;
    let mut hum_var = 0.0;
    
    for i in 0..n {
        let temp_diff = temp_data[i] - temp_mean;
        let hum_diff = humidity_data[i] - hum_mean;
        cov += temp_diff * hum_diff;
        temp_var += temp_diff * temp_diff;
        hum_var += hum_diff * hum_diff;
    }
    
    let correlation = cov / (temp_var.sqrt() * hum_var.sqrt());
    println!("   Temperature-Humidity correlation: {:.3}", correlation);
    println!("   (Negative correlation expected - as temp rises, humidity falls)\n");
    
    // 6. Hierarchical Key Organization
    println!("6. Demonstrating hierarchical key organization...");
    
    // Add some metadata
    db.append("sensor/temperature/unit", Atom::Text("celsius".to_string()))?;
    db.append("sensor/temperature/location", Atom::Text("room-1".to_string()))?;
    db.append("sensor/humidity/unit", Atom::Text("percent".to_string()))?;
    
    let all_keys = db.keys();
    println!("   All keys in database:");
    for key in &all_keys {
        println!("      {}", key);
    }
    println!();
    
    // Cleanup
    db.close()?;
    std::fs::remove_file(db_path)?;
    println!("=== Demo Complete ===");
    
    Ok(())
}

/// Simple pseudo-random float generator (deterministic for demo)
fn rand_float() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(12345);
    }
    
    SEED.with(|seed| {
        let mut s = seed.get();
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed.set(s);
        (s >> 33) as f64 / (1u64 << 31) as f64
    })
}



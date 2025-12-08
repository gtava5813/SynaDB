# Syna Rust Demos

This directory contains Rust examples demonstrating Syna's core functionality.

## Prerequisites

Build the Syna library from the repository root:

```bash
cargo build --release
```

## Running Demos

### Run All Demos

```bash
cd demos/rust
cargo run
```

### Run Specific Demo

```bash
# Using the demo runner
cargo run -- basic_crud
cargo run -- time_series
cargo run -- compression

# Or as cargo examples
cargo run --example basic_crud
cargo run --example time_series
cargo run --example compression
```

### List Available Demos

```bash
cargo run -- --list
```

## Demo Descriptions

| Demo | File | Description |
|------|------|-------------|
| `basic_crud` | `basic_crud.rs` | Create, read, update, delete operations |
| `time_series` | `time_series.rs` | Append sequential data, extract history |
| `compression` | `compression.rs` | LZ4 and delta compression comparison |
| `concurrent` | `concurrent.rs` | Multi-threaded read/write access |
| `recovery` | `recovery.rs` | Crash recovery simulation |
| `tensor_extraction` | `tensor_extraction.rs` | ML-ready float array extraction |

## Demo Details

### 1. Basic CRUD (`basic_crud.rs`)

Demonstrates fundamental database operations:
- Opening and closing databases
- Writing all Atom types (Float, Int, Text, Bytes)
- Reading values back
- Deleting keys
- Listing keys
- Error handling

```rust
use synadb::{synadb, Atom};

let mut db = synadb::new("my.db")?;
db.append("temperature", Atom::Float(23.5))?;
let value = db.get("temperature")?;
db.close()?;
```

### 2. Time-Series (`time_series.rs`)

Shows time-series data patterns:
- Appending sequential sensor readings
- Extracting full history
- Using `get_history_floats()` for ML

```rust
// Append 1000 temperature readings
for i in 0..1000 {
    db.append("sensor/temp", Atom::Float(20.0 + (i as f64 * 0.01)))?;
}

// Extract as float array for ML
let history = db.get_history_floats("sensor/temp")?;
```

### 3. Compression (`compression.rs`)

Compares compression strategies:
- No compression (baseline)
- LZ4 only (for large values)
- Delta only (for float sequences)
- Both LZ4 and delta

Reports file sizes and compression ratios.

### 4. Concurrent Access (`concurrent.rs`)

Demonstrates thread-safe operations:
- 4 writer threads appending data
- 4 reader threads reading data
- Uses `std::thread::scope` for safe concurrency
- Verifies no data corruption

```rust
std::thread::scope(|s| {
    // Spawn writers
    for i in 0..4 {
        s.spawn(|| {
            for j in 0..100 {
                db.append(&format!("thread{}/key{}", i, j), Atom::Int(j))?;
            }
        });
    }
});
```

### 5. Recovery (`recovery.rs`)

Shows crash recovery capabilities:
- Write data, simulate crash (drop without close)
- Reopen and verify all data recovered
- Inject corruption and verify partial recovery
- Demonstrates automatic index rebuild

### 6. Tensor Extraction (`tensor_extraction.rs`)

ML-focused tensor operations:
- Store float sequences for multiple keys
- Extract as raw pointers (simulating FFI)
- Demonstrate memory management with `free_tensor`
- Show zero-copy patterns

## Configuration Options

```rust
use synadb::{synadb, DbConfig};

let config = DbConfig {
    enable_compression: true,  // LZ4 for large values
    enable_delta: true,        // Delta encoding for floats
    sync_on_write: false,      // Disable fsync for speed
};

let db = synadb::with_config("my.db", config)?;
```

## Performance Tips

1. **Disable sync_on_write** for bulk loading:
   ```rust
   let config = DbConfig { sync_on_write: false, ..Default::default() };
   ```

2. **Enable delta compression** for time-series:
   ```rust
   let config = DbConfig { enable_delta: true, ..Default::default() };
   ```

3. **Use `get_history_floats()`** for ML tensors:
   ```rust
   let tensor: Vec<f64> = db.get_history_floats("sensor/temp")?;
   ```

## Error Handling

All operations return `Result<T, SynaError>`:

```rust
match db.get("key") {
    Ok(Some(value)) => println!("Found: {:?}", value),
    Ok(None) => println!("Key not found"),
    Err(e) => eprintln!("Error: {}", e),
}
```

## File Structure

```
demos/rust/
├── Cargo.toml              # Package manifest
├── main.rs                 # Demo runner
├── basic_crud.rs           # Basic operations
├── time_series.rs          # Time-series patterns
├── compression.rs          # Compression comparison
├── concurrent.rs           # Multi-threaded access
├── recovery.rs             # Crash recovery
└── tensor_extraction.rs    # ML tensor extraction
```


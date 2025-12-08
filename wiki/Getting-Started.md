# Getting Started with SynaDB

## Installation

### Python

```bash
pip install synadb
```

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
synadb = "0.5.1"
```

### Building from Source

```bash
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB
cargo build --release
```

## Your First Database

### Python

```python
from synadb import SynaDB

# Create or open a database
with SynaDB("my_data.db") as db:
    # Store different data types
    db.put_float("temperature", 23.5)
    db.put_int("count", 42)
    db.put_text("name", "sensor-1")
    db.put_bytes("raw", b"\x01\x02\x03")
    
    # Read values back
    temp = db.get_float("temperature")
    print(f"Temperature: {temp}")
    
    # Build history by appending more values
    db.put_float("temperature", 24.1)
    db.put_float("temperature", 24.8)
    
    # Get history as numpy array (perfect for ML!)
    history = db.get_history_tensor("temperature")
    print(f"History: {history}")  # [23.5, 24.1, 24.8]
    
    # List all keys
    print(f"Keys: {db.keys()}")
    
    # Delete a key
    db.delete("count")
```

### Rust

```rust
use synadb::{SynaDB, Atom, Result};

fn main() -> Result<()> {
    // Create or open a database
    let mut db = SynaDB::new("my_data.db")?;
    
    // Store different data types
    db.append("temperature", Atom::Float(23.5))?;
    db.append("count", Atom::Int(42))?;
    db.append("name", Atom::Text("sensor-1".to_string()))?;
    
    // Read values back
    if let Some(temp) = db.get("temperature")? {
        println!("Temperature: {:?}", temp);
    }
    
    // Build history
    db.append("temperature", Atom::Float(24.1))?;
    db.append("temperature", Atom::Float(24.8))?;
    
    // Get history as vector
    let history = db.get_history_floats("temperature")?;
    println!("History: {:?}", history);  // [23.5, 24.1, 24.8]
    
    Ok(())
}
```


## Key Concepts

### 1. Append-Only Storage

SynaDB never overwrites data. Every write appends to the end of the file:

```python
db.put_float("temp", 20.0)  # Entry 0
db.put_float("temp", 21.0)  # Entry 1 (doesn't overwrite!)
db.put_float("temp", 22.0)  # Entry 2

# get() returns the latest value
db.get_float("temp")  # Returns 22.0

# get_history_tensor() returns all values
db.get_history_tensor("temp")  # Returns [20.0, 21.0, 22.0]
```

### 2. Schema-Free

No need to define schemas. Store any type with any key:

```python
db.put_float("sensor/temp", 23.5)
db.put_int("sensor/count", 100)
db.put_text("sensor/name", "living-room")
db.put_bytes("sensor/raw", b"\x01\x02\x03")
```

### 3. History as Tensors

Perfect for ML - extract time-series as numpy arrays:

```python
import numpy as np

# Simulate sensor readings
for i in range(1000):
    db.put_float("sensor/reading", np.sin(i * 0.1))

# Get all readings as a tensor
X = db.get_history_tensor("sensor/reading")
print(X.shape)  # (1000,)

# Use directly with PyTorch/TensorFlow
import torch
tensor = torch.from_numpy(X)
```

### 4. Compression

SynaDB automatically compresses data:

- **Delta compression**: For sequential float values (time-series)
- **LZ4 compression**: For large text/bytes values

```python
# Configure compression
from synadb import SynaDB, DbConfig

config = DbConfig(
    enable_compression=True,  # LZ4 for large values
    enable_delta=True,        # Delta for floats
    sync_on_write=False       # Faster, less safe
)

db = SynaDB("data.db", config=config)
```

## Common Patterns

### Time-Series Data

```python
import time

# Log sensor readings
while True:
    reading = get_sensor_reading()
    db.put_float(f"sensor/{sensor_id}", reading)
    time.sleep(1)

# Later: extract for analysis
data = db.get_history_tensor(f"sensor/{sensor_id}")
```

### Key-Value Cache

```python
# Simple caching
db.put_text("cache/user:123", json.dumps(user_data))
cached = json.loads(db.get_text("cache/user:123"))
```

### ML Feature Storage

```python
# Store features
for user_id, features in compute_features():
    db.put_bytes(f"features/{user_id}", features.tobytes())

# Load for training
X = []
for key in db.keys():
    if key.startswith("features/"):
        data = db.get_bytes(key)
        X.append(np.frombuffer(data, dtype=np.float32))
X = np.stack(X)
```

## Next Steps

- [Python Guide](Python-Guide) - Detailed Python API
- [Rust Guide](Rust-Guide) - Detailed Rust API
- [Architecture](Architecture) - How SynaDB works
- [Roadmap](Roadmap) - Upcoming features

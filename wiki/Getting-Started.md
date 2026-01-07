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
synadb = "1.0.6"
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

## Performance Tips

### 1. Batch Writes for High Throughput

Instead of writing one value at a time, batch your writes:

```python
# Slower: Individual writes
for i in range(10000):
    db.put_float(f"sensor/{i}", values[i])

# Faster: Use TensorEngine for batch operations
from synadb import TensorEngine
engine = TensorEngine("data.db")
engine.put_tensor("sensor/", values)  # Single operation
```

### 2. Disable Sync for Speed

By default, SynaDB syncs to disk after each write for durability. For high-throughput scenarios where you can tolerate some data loss on crash:

```python
from synadb import SynaDB, DbConfig

config = DbConfig(
    sync_on_write=False  # 10x faster, but less safe
)
db = SynaDB("data.db", config=config)
```

### 3. Choose the Right Vector Store

SynaDB offers multiple vector storage options:

| Store | Best For | Write Speed | Build Time |
|-------|----------|-------------|------------|
| `VectorStore` | General use | 67K/sec | O(N²) HNSW |
| `MmapVectorStore` | High throughput | 490K/sec | O(N²) HNSW |
| `GravityWellIndex` | Fast build | 90K/sec | O(N) |
| `CascadeIndex` | Tunable recall (Experimental) | 80K/sec | O(N) |

```python
from synadb import VectorStore, MmapVectorStore, GravityWellIndex, CascadeIndex

# General use
store = VectorStore("vectors.db", dimensions=768)

# High-throughput ingestion
store = MmapVectorStore("vectors.mmap", dimensions=768, initial_capacity=100000)

# Fast index build (faster than HNSW)
gwi = GravityWellIndex("vectors.gwi", dimensions=768)
gwi.initialize(sample_vectors)  # Required for GWI

# Tunable recall/latency (Experimental)
cascade = CascadeIndex("vectors.cascade", dimensions=768, preset="large")
```

### 4. Tune HNSW Parameters

For better recall vs. speed tradeoff:

```rust
// Rust API for custom HNSW config
let config = HnswConfig::with_m(32)  // More connections = better recall
    .ef_construction(200)             // Higher = better index quality
    .ef_search(100);                  // Higher = better search recall
```

| Use Case | m | ef_construction | ef_search |
|----------|---|-----------------|-----------|
| Speed priority | 8 | 100 | 50 |
| Balanced | 16 | 200 | 100 |
| Recall priority | 32 | 400 | 200 |

### 5. Use Pattern Matching for Bulk Reads

Instead of reading keys one by one:

```python
# Slower: Individual reads
data = []
for key in db.keys():
    if key.startswith("sensor/"):
        data.append(db.get_float(key))

# Faster: Use TensorEngine with pattern matching
engine = TensorEngine("data.db")
data, shape = engine.get_tensor("sensor/*")
```

### 7. Compact Periodically

After many deletes, compact to reclaim space:

```python
# Compaction rewrites the file with only latest values
db.compact()
```

### 8. Use Multiple Databases for Parallelism

For read-heavy workloads, split data across multiple databases:

```python
# Instead of one large database
db = SynaDB("all_data.db")

# Use multiple databases for parallel access
db_sensors = SynaDB("sensors.db")
db_models = SynaDB("models.db")
db_experiments = SynaDB("experiments.db")
```

### Performance Expectations

| Operation | Expected Throughput | Latency (p50) |
|-----------|---------------------|---------------|
| Write (64B) | 139K ops/sec | 5.6 μs |
| Write (1KB) | 98K ops/sec | 6.8 μs |
| Read (single) | 135K ops/sec | 6.2 μs |
| Vector search (10K) | ~1 ms | - |
| Vector search (1M, HNSW) | ~5-10 ms | - |

### FAISS vs HNSW Comparison

SynaDB includes benchmarks comparing its native HNSW index against FAISS:

```bash
cd benchmarks

# Quick comparison
cargo run --release -- faiss --quick

# Full comparison (100K and 1M vectors)
cargo run --release -- faiss --full
```

| Index | Insert | Search p50 | Recall@10 |
|-------|--------|------------|-----------|
| HNSW | 50K v/s | 0.5ms | 95% |
| FAISS-Flat | 100K v/s | 10ms | 100% |
| FAISS-IVF | 80K v/s | 1ms | 92% |

See [Architecture](Architecture#performance-characteristics) for detailed benchmarks.

## Next Steps

- [Python Guide](Python-Guide) - Detailed Python API
- [Rust Guide](Rust-Guide) - Detailed Rust API
- [Architecture](Architecture) - How SynaDB works
- [Roadmap](Roadmap) - Upcoming features

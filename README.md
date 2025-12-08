# SynaDB

[![CI](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml/badge.svg)](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synadb.svg)](https://pypi.org/project/synadb/)
[![Crates.io](https://img.shields.io/crates/v/synadb.svg)](https://crates.io/crates/synadb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An AI-native embedded database.

An embedded, log-structured, columnar-mapped database engine written in Rust. Syna combines the embedded simplicity of SQLite, the columnar analytical speed of DuckDB, and the schema flexibility of MongoDB.

## Features

- **Append-only log structure** - Fast sequential writes, immutable history
- **Schema-free** - Store heterogeneous data types without migrations
- **AI/ML optimized** - Extract time-series data as contiguous tensors for PyTorch/TensorFlow
- **Vector Store** - Native embedding storage with similarity search for RAG applications
- **C-ABI interface** - Use from Python, Node.js, C++, or any FFI-capable language
- **Delta & LZ4 compression** - Minimize storage for time-series data
- **Crash recovery** - Automatic index rebuild on open
- **Thread-safe** - Concurrent read/write access with mutex-protected writes

## Installation

### Python (Recommended)

```bash
pip install synadb
```

With optional ML dependencies:
```bash
pip install synadb[ml]      # PyTorch, transformers
pip install synadb[pandas]  # Pandas integration
pip install synadb[all]     # Everything
```

### Rust

```toml
[dependencies]
synadb = "0.5.0"
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB

# Build release version
cargo build --release

# Run tests
cargo test
```

The compiled library will be at:
- Linux: `target/release/libsynadb.so`
- macOS: `target/release/libsynadb.dylib`
- Windows: `target/release/synadb.dll`

## Quick Start

### Rust Usage

```rust
use synadb::{synadb, Atom, Result};

fn main() -> Result<()> {
    // Open or create a database
    let mut db = synadb::new("my_data.db")?;
    
    // Write different data types
    db.append("temperature", Atom::Float(23.5))?;
    db.append("count", Atom::Int(42))?;
    db.append("name", Atom::Text("sensor-1".to_string()))?;
    db.append("raw_data", Atom::Bytes(vec![0x01, 0x02, 0x03]))?;
    
    // Read values back
    if let Some(temp) = db.get("temperature")? {
        println!("Temperature: {:?}", temp);
    }
    
    // Append more values to build history
    db.append("temperature", Atom::Float(24.1))?;
    db.append("temperature", Atom::Float(24.8))?;
    
    // Extract history as tensor for ML
    let history = db.get_history_floats("temperature")?;
    println!("Temperature history: {:?}", history); // [23.5, 24.1, 24.8]
    
    // Delete a key
    db.delete("count")?;
    assert!(db.get("count")?.is_none());
    
    // List all keys
    let keys = db.keys();
    println!("Keys: {:?}", keys);
    
    // Compact to reclaim space
    db.compact()?;
    
    // Close (optional - happens on drop)
    db.close()?;
    
    Ok(())
}
```


### Python Usage (ctypes)

```python
import ctypes
from ctypes import c_char_p, c_double, c_int64, c_int32, c_size_t, POINTER, byref

# Load the library
lib = ctypes.CDLL("./target/release/libsynadb.so")  # or .dylib/.dll

# Define function signatures
lib.syna_open.argtypes = [c_char_p]
lib.syna_open.restype = c_int32

lib.syna_close.argtypes = [c_char_p]
lib.syna_close.restype = c_int32

lib.syna_put_float.argtypes = [c_char_p, c_char_p, c_double]
lib.syna_put_float.restype = c_int64

lib.syna_get_float.argtypes = [c_char_p, c_char_p, POINTER(c_double)]
lib.syna_get_float.restype = c_int32

lib.syna_get_history_tensor.argtypes = [c_char_p, c_char_p, POINTER(c_size_t)]
lib.syna_get_history_tensor.restype = POINTER(c_double)

lib.syna_free_tensor.argtypes = [POINTER(c_double), c_size_t]
lib.syna_free_tensor.restype = None

lib.syna_delete.argtypes = [c_char_p, c_char_p]
lib.syna_delete.restype = c_int32

# Usage
db_path = b"my_data.db"

# Open database
result = lib.syna_open(db_path)
assert result == 1, f"Failed to open database: {result}"

# Write float values
lib.syna_put_float(db_path, b"temperature", 23.5)
lib.syna_put_float(db_path, b"temperature", 24.1)
lib.syna_put_float(db_path, b"temperature", 24.8)

# Read latest value
value = c_double()
result = lib.syna_get_float(db_path, b"temperature", byref(value))
if result == 1:
    print(f"Temperature: {value.value}")

# Get history as numpy-compatible array
length = c_size_t()
ptr = lib.syna_get_history_tensor(db_path, b"temperature", byref(length))
if ptr:
    # Convert to Python list (or use numpy.ctypeslib for zero-copy)
    history = [ptr[i] for i in range(length.value)]
    print(f"History: {history}")
    
    # Free the tensor memory
    lib.syna_free_tensor(ptr, length)

# Close database
lib.syna_close(db_path)
```

### C/C++ Usage

```c
#include "synadb.h"
#include <stdio.h>

int main() {
    const char* db_path = "my_data.db";
    
    // Open database
    int result = syna_open(db_path);
    if (result != 1) {
        fprintf(stderr, "Failed to open database: %d\n", result);
        return 1;
    }
    
    // Write values
    syna_put_float(db_path, "temperature", 23.5);
    syna_put_float(db_path, "temperature", 24.1);
    syna_put_int(db_path, "count", 42);
    syna_put_text(db_path, "name", "sensor-1");
    
    // Read float value
    double temp;
    if (syna_get_float(db_path, "temperature", &temp) == 1) {
        printf("Temperature: %f\n", temp);
    }
    
    // Get history tensor for ML
    size_t len;
    double* tensor = syna_get_history_tensor(db_path, "temperature", &len);
    if (tensor) {
        printf("History (%zu values):", len);
        for (size_t i = 0; i < len; i++) {
            printf(" %f", tensor[i]);
        }
        printf("\n");
        
        // Free tensor memory
        syna_free_tensor(tensor, len);
    }
    
    // Delete a key
    syna_delete(db_path, "count");
    
    // Compact database
    syna_compact(db_path);
    
    // Close database
    syna_close(db_path);
    
    return 0;
}
```

Compile with:
```bash
gcc -o myapp myapp.c -L./target/release -lsynadb -Wl,-rpath,./target/release
```

## Vector Store

Store and search embeddings for RAG applications:

```python
from synadb import VectorStore
import numpy as np

# Create store with 768 dimensions (BERT-sized)
store = VectorStore("vectors.db", dimensions=768)

# Insert embeddings
embedding1 = np.random.randn(768).astype(np.float32)
embedding2 = np.random.randn(768).astype(np.float32)
store.insert("doc1", embedding1)
store.insert("doc2", embedding2)

# Search for similar vectors
query_embedding = np.random.randn(768).astype(np.float32)
results = store.search(query_embedding, k=5)
for r in results:
    print(f"{r.key}: {r.score:.4f}")
```

### Distance Metrics

The VectorStore supports three distance metrics:

| Metric | Description | Use Case |
|--------|-------------|----------|
| `cosine` (default) | Cosine distance (1 - cosine_similarity) | Text embeddings, normalized vectors |
| `euclidean` | Euclidean (L2) distance | Image embeddings, spatial data |
| `dot_product` | Negative dot product | Maximum inner product search |

```python
# Use euclidean distance
store = VectorStore("vectors.db", dimensions=768, metric="euclidean")

# Use dot product
store = VectorStore("vectors.db", dimensions=768, metric="dot_product")
```

### Supported Dimensions

Vector dimensions from 64 to 4096 are supported, covering all common embedding models:

| Model | Dimensions |
|-------|------------|
| MiniLM | 384 |
| BERT base | 768 |
| BERT large | 1024 |
| OpenAI ada-002 | 1536 |
| OpenAI text-embedding-3-large | 3072 |

### HNSW Index

For large-scale vector search (>10,000 vectors), SynaDB uses HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search with O(log N) complexity.

```python
from synadb import VectorStore
import numpy as np

# HNSW is automatically enabled when vector count exceeds threshold
store = VectorStore("vectors.db", dimensions=768)

# Insert many vectors - HNSW index builds automatically
for i in range(100000):
    embedding = np.random.randn(768).astype(np.float32)
    store.insert(f"doc{i}", embedding)

# Search is now O(log N) instead of O(N)
results = store.search(query_embedding, k=10)  # <10ms for 1M vectors
```

HNSW Configuration (Rust API):

```rust
use synadb::hnsw::{HnswIndex, HnswConfig};
use synadb::distance::DistanceMetric;

// Custom HNSW configuration
let config = HnswConfig::with_m(32)  // More connections = better recall
    .ef_construction(200)             // Higher = better index quality
    .ef_search(100);                  // Higher = better search recall

let mut index = HnswIndex::new(768, DistanceMetric::Cosine, config);
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per node (8-64 typical) |
| `m_max` | 32 | Max connections at higher layers (2×M) |
| `ef_construction` | 200 | Build quality (100-500 typical) |
| `ef_search` | 100 | Search quality (50-500 typical) |

## Tensor Engine

The TensorEngine provides efficient batch operations for ML data loading:

```python
from synadb import TensorEngine
import numpy as np

# Create tensor engine
engine = TensorEngine("training_data.db")

# Store training data
X_train = np.random.randn(10000, 784).astype(np.float32)
engine.put_tensor("train/", X_train)

# Load as tensor (pattern matching)
X = engine.get_tensor("train/*", dtype=np.float32)

# Load with specific shape
X = engine.get_tensor("train/*", shape=(10000, 784), dtype=np.float32)

# Stream in batches for training
for batch in engine.stream("train/*", batch_size=32):
    model.train_step(batch)
```

### PyTorch Integration

```python
# Load directly as PyTorch tensor
X = engine.get_tensor_torch("train/*", device="cuda")

# Or use with DataLoader
from torch.utils.data import TensorDataset, DataLoader

X = engine.get_tensor_torch("train/*")
y = engine.get_tensor_torch("labels/*")
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### TensorFlow Integration

```python
# Load directly as TensorFlow tensor
X = engine.get_tensor_tf("train/*")

# Use with tf.data
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(X).batch(32)
```

### Rust API

```rust
use synadb::{SynaDB, Atom};
use synadb::tensor::{TensorEngine, DType};

// Create database and populate with data
let mut db = SynaDB::new("data.db")?;
for i in 0..100 {
    db.append(&format!("sensor/{:04}", i), Atom::Float(i as f64 * 0.1))?;
}

// Create tensor engine
let mut engine = TensorEngine::new(db);

// Load all sensor data as a tensor
let (data, shape) = engine.get_tensor("sensor/*", DType::Float64)?;
assert_eq!(shape[0], 100);

// Store tensor with auto-generated keys
let values: Vec<u8> = vec![1.0f64, 2.0, 3.0, 4.0]
    .iter()
    .flat_map(|f| f.to_le_bytes())
    .collect();
let count = engine.put_tensor("values/", &values, &[4], DType::Float64)?;
```

### Supported Data Types

| DType | Size | Description |
|-------|------|-------------|
| `Float32` | 4 bytes | 32-bit floating point |
| `Float64` | 8 bytes | 64-bit floating point |
| `Int32` | 4 bytes | 32-bit signed integer |
| `Int64` | 8 bytes | 64-bit signed integer |

## Model Registry

Store and version ML models with automatic checksum verification:

### Python Usage

```python
from synadb import ModelRegistry

# Create a model registry
registry = ModelRegistry("models.db")

# Save a model with metadata
model_data = open("model.pt", "rb").read()
metadata = {"accuracy": "0.95", "framework": "pytorch"}
version = registry.save_model("classifier", model_data, metadata)
print(f"Saved version {version.version} with checksum {version.checksum}")

# Load the latest version (with automatic checksum verification)
data, info = registry.load_model("classifier")
print(f"Loaded {info.size_bytes} bytes, stage: {info.stage}")

# Load a specific version
data, info = registry.load_model("classifier", version=1)

# List all versions
versions = registry.list_versions("classifier")
for v in versions:
    print(f"v{v.version}: {v.stage} ({v.size_bytes} bytes)")

# Promote to production
registry.set_stage("classifier", version.version, "Production")

# Get the production model
prod = registry.get_production("classifier")
if prod:
    print(f"Production version: {prod.version}")
```

### Rust Usage

```rust
use synadb::model_registry::{ModelRegistry, ModelStage};
use std::collections::HashMap;

// Create a model registry
let mut registry = ModelRegistry::new("models.db")?;

// Save a model with metadata
let model_data = vec![0u8; 1024]; // Your model bytes
let mut metadata = HashMap::new();
metadata.insert("accuracy".to_string(), "0.95".to_string());
metadata.insert("framework".to_string(), "pytorch".to_string());

let version = registry.save_model("classifier", &model_data, metadata)?;
println!("Saved version {} with checksum {}", version.version, version.checksum);

// Load the latest version (with automatic checksum verification)
let (data, info) = registry.load_model("classifier", None)?;
println!("Loaded {} bytes", data.len());

// Load a specific version
let (data, info) = registry.load_model("classifier", Some(1))?;

// List all versions
let versions = registry.list_versions("classifier")?;
for v in versions {
    println!("v{}: {} ({} bytes)", v.version, v.stage, v.size_bytes);
}

// Promote to production
registry.set_stage("classifier", version.version, ModelStage::Production)?;

// Get the production model
if let Some(prod) = registry.get_production("classifier")? {
    println!("Production version: {}", prod.version);
}
```

### Model Stages

Models progress through deployment stages:

| Stage | Description |
|-------|-------------|
| `Development` | Initial stage for new models (default) |
| `Staging` | Models being tested before production |
| `Production` | Models actively serving predictions |
| `Archived` | Retired models kept for reference |

### Checksum Verification

Every model is stored with a SHA-256 checksum. When loading, the checksum is automatically verified to detect corruption:

```python
# If the model data is corrupted, load_model raises an error
try:
    data, info = registry.load_model("classifier")
except SynaError as e:
    print(f"Checksum mismatch: {e}")
```

## Experiment Tracking

Track ML experiments with parameters, metrics, and artifacts:

### Python Usage

```python
from synadb import Experiment

# Create an experiment
exp = Experiment("mnist_classifier", "experiments.db")

# Start a run with tags
with exp.start_run(tags=["baseline", "v1"]) as run:
    # Log hyperparameters
    run.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam"
    })
    
    # Log metrics during training
    for epoch in range(100):
        loss = 1.0 / (epoch + 1)
        accuracy = 0.5 + 0.005 * epoch
        run.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)
    
    # Log artifacts
    run.log_artifact("model.pt", model.state_dict())
    run.log_artifact("config.json", json.dumps(config).encode())

# Query runs
completed_runs = exp.query(filter={"status": "completed"})
best_runs = exp.query(sort_by="accuracy", ascending=False)

# Get metrics as numpy array for plotting
loss_history = exp.get_metric_tensor(run.id, "loss")
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.title("Training Loss")
plt.show()

# Compare runs
comparison = exp.compare_runs([run1.id, run2.id])
print(comparison)
```

### Rust Usage

```rust
use synadb::experiment::{ExperimentTracker, RunStatus};

// Create an experiment tracker
let mut tracker = ExperimentTracker::new("experiments.db")?;

// Start a run with tags
let run_id = tracker.start_run("mnist_classifier", vec!["baseline".to_string()])?;

// Log hyperparameters
tracker.log_param(&run_id, "learning_rate", "0.001")?;
tracker.log_param(&run_id, "batch_size", "32")?;
tracker.log_param(&run_id, "epochs", "100")?;

// Log metrics during training
for epoch in 0..100 {
    let loss = 1.0 / (epoch + 1) as f64;
    let accuracy = 0.5 + 0.005 * epoch as f64;
    tracker.log_metric(&run_id, "loss", loss, Some(epoch as u64))?;
    tracker.log_metric(&run_id, "accuracy", accuracy, Some(epoch as u64))?;
}

// Log artifacts
let model_data = vec![0u8; 1024]; // Your model bytes
tracker.log_artifact(&run_id, "model.pt", &model_data)?;

// End the run
tracker.end_run(&run_id, RunStatus::Completed)?;

// Query runs
let runs = tracker.list_runs("mnist_classifier")?;
for run in runs {
    println!("Run {}: status={}", run.id, run.status);
}

// Get metrics
let loss_values = tracker.get_metric(&run_id, "loss")?;
for (step, value) in loss_values {
    println!("Step {}: loss = {:.4}", step, value);
}
```

### Run Status

Runs progress through states:

| Status | Description |
|--------|-------------|
| `Running` | Run is currently in progress |
| `Completed` | Run finished successfully |
| `Failed` | Run encountered an error |
| `Killed` | Run was manually terminated |

### Context Manager Support

The Python API supports context managers for automatic run completion:

```python
# Automatic completion on success
with exp.start_run() as run:
    run.log_param("lr", 0.001)
    # ... training code ...
# Run automatically marked as "completed"

# Automatic failure on exception
with exp.start_run() as run:
    run.log_param("lr", 0.001)
    raise ValueError("Training failed!")
# Run automatically marked as "failed"
```

### Querying and Filtering

```python
# Filter by status
completed = exp.query(filter={"status": "completed"})

# Filter by tags
baseline_runs = exp.query(filter={"tags": ["baseline"]})

# Filter by parameter value
lr_runs = exp.query(filter={"learning_rate": "0.001"})

# Sort by metric (descending for best first)
best_runs = exp.query(sort_by="accuracy", ascending=False)

# Combine filters
best_baseline = exp.query(
    filter={"status": "completed", "tags": ["baseline"]},
    sort_by="accuracy",
    ascending=False
)
```

## Data Types

Syna supports six atomic data types:

| Type | Rust | C/FFI | Description |
|------|------|-------|-------------|
| Null | `Atom::Null` | N/A | Absence of value |
| Float | `Atom::Float(f64)` | `SYNA_put_float` | 64-bit floating point |
| Int | `Atom::Int(i64)` | `SYNA_put_int` | 64-bit signed integer |
| Text | `Atom::Text(String)` | `SYNA_put_text` | UTF-8 string |
| Bytes | `Atom::Bytes(Vec<u8>)` | `SYNA_put_bytes` | Raw byte array |
| Vector | `Atom::Vector(Vec<f32>, u16)` | `SYNA_put_vector` | Embedding vector (64-4096 dims) |

## Configuration

```rust
use synadb::{synadb, DbConfig};

let config = DbConfig {
    enable_compression: true,   // LZ4 compression for large values
    enable_delta: true,         // Delta encoding for float sequences
    sync_on_write: true,        // fsync after each write (safer but slower)
};

let db = synadb::with_config("my_data.db", config)?;
```

## Error Codes (FFI)

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `ERR_SUCCESS` | Operation successful |
| 0 | `ERR_GENERIC` | Generic error |
| -1 | `ERR_DB_NOT_FOUND` | Database not in registry |
| -2 | `ERR_INVALID_PATH` | Invalid path or UTF-8 |
| -3 | `ERR_IO` | I/O error |
| -4 | `ERR_SERIALIZATION` | Serialization error |
| -5 | `ERR_KEY_NOT_FOUND` | Key not found |
| -6 | `ERR_TYPE_MISMATCH` | Type mismatch on read |
| -100 | `ERR_INTERNAL_PANIC` | Internal panic |

## Architecture

Syna uses an append-only log structure inspired by the "physics of time" principle:

```
┌─────────────────────────────────────────────────────────────┐
│ Entry 0                                                     │
├──────────────┬──────────────────┬───────────────────────────┤
│ LogHeader    │ Key (UTF-8)      │ Value (bincode)           │
│ (15 bytes)   │ (key_len bytes)  │ (val_len bytes)           │
├──────────────┴──────────────────┴───────────────────────────┤
│ Entry 1 ...                                                 │
└─────────────────────────────────────────────────────────────┘
```

- **Writes**: Always append to end of file (sequential I/O)
- **Reads**: Use in-memory index for O(1) key lookup
- **Recovery**: Scan file on open to rebuild index
- **Compaction**: Rewrite file with only latest values

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.



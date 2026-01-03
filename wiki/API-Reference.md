# API Reference

## Python API

### SynaDB Class

```python
from synadb import SynaDB

# Constructor
db = SynaDB(path: str, config: DbConfig = None)

# Context manager (recommended)
with SynaDB("data.db") as db:
    ...
```

### Write Operations

```python
# Store float value
db.put_float(key: str, value: float) -> int  # Returns offset

# Store integer value
db.put_int(key: str, value: int) -> int

# Store text value
db.put_text(key: str, value: str) -> int

# Store bytes value
db.put_bytes(key: str, value: bytes) -> int
```

### Read Operations

```python
# Get latest float value
db.get_float(key: str) -> Optional[float]

# Get latest integer value
db.get_int(key: str) -> Optional[int]

# Get latest text value
db.get_text(key: str) -> Optional[str]

# Get latest bytes value
db.get_bytes(key: str) -> Optional[bytes]

# Get history as numpy array
db.get_history_tensor(key: str) -> np.ndarray
```

### Key Operations

```python
# List all keys
db.keys() -> List[str]

# Check if key exists
db.exists(key: str) -> bool

# Delete a key (tombstone)
db.delete(key: str) -> None
```

### Maintenance

```python
# Compact database (remove old values)
db.compact() -> None

# Close database
db.close() -> None
```

### Configuration

```python
from synadb import DbConfig

config = DbConfig(
    enable_compression: bool = True,   # LZ4 for large values
    enable_delta: bool = True,         # Delta encoding for floats
    sync_on_write: bool = False        # fsync after each write
)

db = SynaDB("data.db", config=config)
```

---

## Rust API

### SynaDB Struct

```rust
use synadb::{SynaDB, Atom, DbConfig, Result};

// Create with default config
let mut db = SynaDB::new("data.db")?;

// Create with custom config
let config = DbConfig {
    enable_compression: true,
    enable_delta: true,
    sync_on_write: false,
};
let mut db = SynaDB::with_config("data.db", config)?;
```


### Write Operations

```rust
// Append any Atom type
db.append(key: &str, value: Atom) -> Result<u64>  // Returns offset

// Examples
db.append("temp", Atom::Float(23.5))?;
db.append("count", Atom::Int(42))?;
db.append("name", Atom::Text("sensor".to_string()))?;
db.append("raw", Atom::Bytes(vec![1, 2, 3]))?;
```

### Read Operations

```rust
// Get latest value
db.get(key: &str) -> Result<Option<Atom>>

// Get history as Vec<Atom>
db.get_history(key: &str) -> Result<Vec<Atom>>

// Get history as Vec<f64> (floats only)
db.get_history_floats(key: &str) -> Result<Vec<f64>>
```

### Key Operations

```rust
// List all keys
db.keys() -> Vec<String>

// Check if key exists
db.exists(key: &str) -> bool

// Delete a key
db.delete(key: &str) -> Result<()>
```

### Maintenance

```rust
// Compact database
db.compact() -> Result<()>

// Close database
db.close() -> Result<()>
```

### Atom Enum

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Null,
    Float(f64),
    Int(i64),
    Text(String),
    Bytes(Vec<u8>),
    Vector(Vec<f32>, u16),  // v0.2.0+
}

// Helper methods
atom.is_null() -> bool
atom.is_float() -> bool
atom.as_float() -> Option<f64>
atom.as_int() -> Option<i64>
atom.as_text() -> Option<&str>
atom.as_bytes() -> Option<&[u8]>
```

---

## C/FFI API

### Lifecycle

```c
// Open database (adds to global registry)
int32_t syna_open(const char* path);

// Close database (removes from registry)
int32_t syna_close(const char* path);
```

### Write Operations

```c
// Store float
int64_t syna_put_float(const char* path, const char* key, double value);

// Store integer
int64_t syna_put_int(const char* path, const char* key, int64_t value);

// Store text
int64_t syna_put_text(const char* path, const char* key, const char* value);

// Store bytes
int64_t syna_put_bytes(const char* path, const char* key, 
                       const uint8_t* data, size_t len);
```

### Read Operations

```c
// Get float (returns 1 on success, negative on error)
int32_t syna_get_float(const char* path, const char* key, double* out);

// Get integer
int32_t syna_get_int(const char* path, const char* key, int64_t* out);

// Get history as tensor (caller must free with syna_free_tensor)
double* syna_get_history_tensor(const char* path, const char* key, 
                                 size_t* out_len);

// Free tensor memory
void syna_free_tensor(double* ptr, size_t len);
```

### Key Operations

```c
// Delete key
int32_t syna_delete(const char* path, const char* key);

// Check if key exists
int32_t syna_exists(const char* path, const char* key);

// List keys (caller must free with syna_free_keys)
char** syna_keys(const char* path, size_t* out_len);
void syna_free_keys(char** keys, size_t len);
```

### Maintenance

```c
// Compact database
int32_t syna_compact(const char* path);
```

### Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `SYNA_SUCCESS` | Operation successful |
| 0 | `SYNA_ERR_GENERIC` | Generic error |
| -1 | `SYNA_ERR_DB_NOT_FOUND` | Database not in registry |
| -2 | `SYNA_ERR_INVALID_PATH` | Invalid path or UTF-8 |
| -3 | `SYNA_ERR_IO` | I/O error |
| -4 | `SYNA_ERR_SERIALIZATION` | Serialization error |
| -5 | `SYNA_ERR_KEY_NOT_FOUND` | Key not found |
| -6 | `SYNA_ERR_TYPE_MISMATCH` | Type mismatch on read |
| -100 | `SYNA_ERR_INTERNAL_PANIC` | Internal panic |


---

## VectorStore (Python)

```python
from synadb import VectorStore

# Create store
store = VectorStore(path: str, dimensions: int, metric: str = "cosine")

# Insert vector
store.insert(key: str, vector: np.ndarray) -> None

# Search for similar vectors
store.search(query: np.ndarray, k: int = 10) -> List[SearchResult]

# Get vector by key
store.get(key: str) -> Optional[np.ndarray]

# Delete vector
store.delete(key: str) -> None

# Build HNSW index (for large datasets)
store.build_index() -> None

# Properties
len(store) -> int
store.dimensions -> int
```

### SearchResult

```python
@dataclass
class SearchResult:
    key: str      # Vector key
    score: float  # Distance score (lower = more similar)
```

### Distance Metrics

| Metric | Description |
|--------|-------------|
| `cosine` | Cosine distance (1 - cosine_similarity) |
| `euclidean` | Euclidean (L2) distance |
| `dot_product` | Negative dot product |

---

## TensorEngine (Python)

```python
from synadb import TensorEngine

# Create engine
engine = TensorEngine(path: str)

# Store tensor with chunked storage
engine.put_tensor_chunked(name: str, data: np.ndarray) -> int

# Load tensor
data, shape = engine.get_tensor_chunked(name: str) -> Tuple[np.ndarray, Tuple]

# Pattern-based loading
data, shape = engine.get_tensor(pattern: str, dtype) -> Tuple[np.ndarray, Tuple]
```

---

## ModelRegistry (Python)

```python
from synadb import ModelRegistry

# Create registry
registry = ModelRegistry(path: str)

# Save model with metadata
version = registry.save_model(
    name: str, 
    data: bytes, 
    metadata: Dict[str, str] = {}
) -> ModelVersion

# Load model (with checksum verification)
data, info = registry.load_model(name: str, version: int = None) -> Tuple[bytes, ModelVersion]

# List versions
versions = registry.list_versions(name: str) -> List[ModelVersion]

# Set deployment stage
registry.set_stage(name: str, version: int, stage: str) -> None

# Get production model
prod = registry.get_production(name: str) -> Optional[ModelVersion]
```

### ModelVersion

```python
@dataclass
class ModelVersion:
    version: int
    checksum: str      # SHA-256 hash
    size_bytes: int
    stage: str         # Development, Staging, Production, Archived
    metadata: Dict[str, str]
    created_at: int    # Unix timestamp
```

---

## ExperimentTracker (Python)

```python
from synadb import ExperimentTracker

# Create tracker
tracker = ExperimentTracker(path: str)

# Start a run
run_id = tracker.start_run(experiment: str, tags: List[str] = []) -> str

# Log parameters
tracker.log_param(run_id: str, key: str, value: str) -> None

# Log metrics
tracker.log_metric(run_id: str, key: str, value: float, step: int = None) -> None

# Log artifacts
tracker.log_artifact(run_id: str, name: str, data: bytes) -> None

# End run
tracker.end_run(run_id: str, status: str) -> None

# Query runs
run = tracker.get_run(run_id: str) -> Run
runs = tracker.list_runs(experiment: str) -> List[Run]
metrics = tracker.get_metric(run_id: str, metric_name: str) -> List[Tuple[int, float]]
```

### Run Status

| Status | Description |
|--------|-------------|
| `Running` | Run in progress |
| `Completed` | Run finished successfully |
| `Failed` | Run encountered error |
| `Killed` | Run manually terminated |


---

## MmapVectorStore (Python)

Ultra-high-throughput vector storage using memory-mapped files.

```python
from synadb import MmapVectorStore

# Create store with pre-allocated capacity
store = MmapVectorStore(
    path: str,
    dimensions: int,
    initial_capacity: int = 100000,
    metric: str = "cosine"
)

# Insert single vector
store.insert(key: str, vector: np.ndarray) -> None

# Insert batch (490K vectors/sec for 768 dims)
store.insert_batch(keys: List[str], vectors: np.ndarray) -> None

# Build HNSW index
store.build_index() -> None

# Search for similar vectors
store.search(query: np.ndarray, k: int = 10) -> List[SearchResult]

# Get vector by key
store.get(key: str) -> Optional[np.ndarray]

# Checkpoint to disk
store.checkpoint() -> None

# Close store
store.close() -> None

# Properties
len(store) -> int
store.dimensions -> int
```

### Performance Comparison

| Aspect | VectorStore | MmapVectorStore |
|--------|-------------|-----------------|
| Write speed | ~67K/sec | ~490K/sec |
| Durability | Per-write | Checkpoint |
| Capacity | Dynamic | Pre-allocated |

### Usage Example

```python
from synadb import MmapVectorStore
import numpy as np

# Create store for 100K vectors
store = MmapVectorStore("vectors.mmap", dimensions=768, initial_capacity=100000)

# Batch insert (fastest)
keys = [f"doc_{i}" for i in range(10000)]
vectors = np.random.randn(10000, 768).astype(np.float32)
store.insert_batch(keys, vectors)  # 490K vectors/sec

# Build HNSW index for fast search
store.build_index()

# Search
query = np.random.randn(768).astype(np.float32)
results = store.search(query, k=10)

# Checkpoint to persist
store.checkpoint()
store.close()
```

---

## GravityWellIndex (Python)

Novel append-only vector index with O(N) build time.

```python
from synadb import GravityWellIndex

# Create index
gwi = GravityWellIndex(
    path: str,
    dimensions: int,
    num_attractors: int = 256,
    metric: str = "cosine"
)

# Initialize attractors from sample vectors (required before insert)
gwi.initialize(sample_vectors: np.ndarray) -> None

# Insert single vector
gwi.insert(key: str, vector: np.ndarray) -> None

# Insert batch
gwi.insert_batch(keys: List[str], vectors: np.ndarray) -> None

# Search with tunable recall
gwi.search(
    query: np.ndarray,
    k: int = 10,
    nprobe: int = 50  # Higher = better recall, slower
) -> List[SearchResult]

# Get vector by key
gwi.get(key: str) -> Optional[np.ndarray]

# Close index
gwi.close() -> None

# Properties
len(gwi) -> int
gwi.dimensions -> int
```

### Recall vs nprobe

| nprobe | Recall@10 | Latency |
|--------|-----------|---------|
| 3 | ~50% | 0.23ms |
| 10 | ~70% | 0.37ms |
| 30 | ~90% | 0.59ms |
| 50 | ~98% | 0.68ms |
| 100 | ~100% | 0.86ms |

### Build Time Comparison

| Dataset | GWI Build | HNSW Build | Speedup |
|---------|-----------|------------|---------|
| 10K × 384 | 1.0s | 8.8s | 8.6x |
| 10K × 768 | 2.1s | 18.4s | 8.9x |
| 50K × 384 | 1.5s | 272s | 186x |
| 50K × 768 | 3.0s | 504s | 169x |

### Usage Example

```python
from synadb import GravityWellIndex
import numpy as np

# Create index
gwi = GravityWellIndex("vectors.gwi", dimensions=768)

# Initialize with sample vectors (use ~1000 representative samples)
sample = np.random.randn(1000, 768).astype(np.float32)
gwi.initialize(sample)

# Insert vectors
keys = [f"doc_{i}" for i in range(50000)]
vectors = np.random.randn(50000, 768).astype(np.float32)
gwi.insert_batch(keys, vectors)

# Search with 98% recall
query = np.random.randn(768).astype(np.float32)
results = gwi.search(query, k=10, nprobe=50)

for r in results:
    print(f"{r.key}: {r.score:.4f}")

gwi.close()
```

### When to Use GWI vs HNSW

| Use Case | Recommended |
|----------|-------------|
| Index build time critical | GWI |
| Streaming/real-time data | GWI |
| Append-only storage required | GWI |
| Search latency critical | HNSW |
| Index built once, queried many times | HNSW |
| Highest recall required | HNSW |

---

## CascadeIndex (Python)

Three-stage hybrid vector index combining LSH, adaptive bucket trees, and sparse graphs.

```python
from synadb import CascadeIndex

# Create with preset configuration
index = CascadeIndex(
    path: str,
    dimensions: int,
    preset: str = "small"  # "small", "large", "high_recall", "fast_search"
)

# Or custom configuration
index = CascadeIndex(
    path: str,
    dimensions: int,
    num_hyperplanes: int = 12,    # LSH hyperplanes
    bucket_capacity: int = 64,     # Max vectors per bucket before split
    nprobe: int = 4,               # Buckets to probe during search
    graph_neighbors: int = 16,     # Neighbors in sparse graph
    metric: str = "cosine"
)

# Insert single vector
index.insert(key: str, vector: np.ndarray) -> None

# Insert batch
index.insert_batch(keys: List[str], vectors: np.ndarray) -> None

# Search for similar vectors
index.search(query: np.ndarray, k: int = 10) -> List[SearchResult]

# Get vector by key
index.get(key: str) -> Optional[np.ndarray]

# Save index to disk
index.save() -> None

# Close index
index.close() -> None

# Properties
len(index) -> int
index.dimensions -> int
```

### Configuration Presets

| Preset | num_hyperplanes | bucket_capacity | nprobe | graph_neighbors | Use Case |
|--------|-----------------|-----------------|--------|-----------------|----------|
| `small` | 12 | 64 | 4 | 16 | <100K vectors |
| `large` | 16 | 128 | 8 | 24 | 1M+ vectors |
| `high_recall` | 20 | 256 | 16 | 32 | Accuracy critical |
| `fast_search` | 8 | 32 | 2 | 8 | Latency critical |

### Architecture

The Cascade Index uses a three-stage approach:

1. **LSH Layer** - Hyperplane-based locality-sensitive hashing partitions vectors into buckets. Multi-probe queries nearby buckets for better recall.

2. **Bucket Tree** - Adaptive binary tree that splits buckets when they exceed capacity. Maintains balanced distribution as data grows.

3. **Sparse Graph** - Local neighbor connections for final ranking refinement. Improves recall without full graph traversal.

### Usage Example

```python
from synadb import CascadeIndex
import numpy as np

# Create index with large dataset preset
index = CascadeIndex("vectors.cascade", dimensions=768, preset="large")

# Insert vectors
keys = [f"doc_{i}" for i in range(100000)]
vectors = np.random.randn(100000, 768).astype(np.float32)
index.insert_batch(keys, vectors)

# Search
query = np.random.randn(768).astype(np.float32)
results = index.search(query, k=10)

for r in results:
    print(f"{r.key}: {r.score:.4f}")

# Save and close
index.save()
index.close()
```

### When to Use Cascade Index

| Use Case | Recommended Index |
|----------|-------------------|
| Balanced build/search performance | Cascade |
| Tunable recall/latency trade-off | Cascade |
| Medium datasets (100K-10M) | Cascade |
| Maximum throughput | MmapVectorStore |
| Fastest build time | GWI |
| Lowest search latency | HNSW |
| Billion-scale | FAISS |

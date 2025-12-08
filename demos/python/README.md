# SynaDB

[![CI](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml/badge.svg)](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synadb.svg)](https://pypi.org/project/synadb/)
[![Crates.io](https://img.shields.io/crates/v/synadb.svg)](https://crates.io/crates/synadb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AI-native embedded database for Python

SynaDB is an embedded database designed for AI/ML workloads. It combines the simplicity of SQLite with native support for vectors, tensors, model versioning, and experiment tracking.

## Installation

```bash
pip install synadb
```

With optional dependencies:
```bash
pip install synadb[ml]      # PyTorch, transformers
pip install synadb[pandas]  # Pandas integration
pip install synadb[all]     # Everything
```

## Features

| Feature | Description |
|---------|-------------|
| **Vector Store** | Embedding storage with similarity search (cosine, euclidean, dot product) |
| **HNSW Index** | O(log N) approximate nearest neighbor search for large-scale vectors |
| **Tensor Engine** | Batch tensor operations for ML data loading |
| **Model Registry** | Version and stage ML models with checksum verification |
| **Experiment Tracking** | Log parameters, metrics, and artifacts |
| **Core Database** | Schema-free key-value storage with history |

## Quick Start

### Basic Key-Value Storage

```python
from synadb import SynaDB

with SynaDB("my_data.db") as db:
    # Store different types
    db.put_float("temperature", 23.5)
    db.put_int("count", 42)
    db.put_text("name", "sensor-1")
    
    # Read values
    temp = db.get_float("temperature")  # 23.5
    
    # Build history
    db.put_float("temperature", 24.1)
    db.put_float("temperature", 24.8)
    
    # Extract as numpy array for ML
    history = db.get_history_tensor("temperature")  # [23.5, 24.1, 24.8]
```

### Vector Store (RAG Applications)

```python
from synadb import VectorStore
import numpy as np

# Create store with 768 dimensions (BERT-sized)
store = VectorStore("vectors.db", dimensions=768)

# Insert embeddings
embedding = np.random.randn(768).astype(np.float32)
store.insert("doc1", embedding)

# Search for similar vectors
query = np.random.randn(768).astype(np.float32)
results = store.search(query, k=5)
for r in results:
    print(f"{r.key}: {r.score:.4f}")
```

**Distance Metrics:**
- `cosine` (default) - Best for text embeddings
- `euclidean` - Best for image embeddings  
- `dot_product` - Maximum inner product search

**Supported Dimensions:** 64-4096 (covers MiniLM, BERT, OpenAI ada-002, etc.)

### Tensor Engine (ML Data Loading)

```python
from synadb import TensorEngine
import numpy as np

engine = TensorEngine("training.db")

# Store training data
X_train = np.random.randn(10000, 784).astype(np.float32)
engine.put_tensor_chunked("train/X", X_train)

# Load as tensor
X, shape = engine.get_tensor_chunked("train/X")
```

### Model Registry

```python
from synadb import ModelRegistry

registry = ModelRegistry("models.db")

# Save model with metadata
model_bytes = open("model.pt", "rb").read()
version = registry.save_model("classifier", model_bytes, {"accuracy": "0.95"})
print(f"Saved v{version.version}, checksum: {version.checksum}")

# Load with automatic checksum verification
data, info = registry.load_model("classifier")

# Promote to production
registry.set_stage("classifier", version.version, "Production")
```

### Experiment Tracking

```python
from synadb import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

# Start a run
run_id = tracker.start_run("mnist", tags=["baseline"])

# Log hyperparameters
tracker.log_param(run_id, "learning_rate", "0.001")
tracker.log_param(run_id, "batch_size", "32")

# Log metrics during training
for epoch in range(100):
    loss = 1.0 / (epoch + 1)
    tracker.log_metric(run_id, "loss", loss, step=epoch)

# Log artifacts
tracker.log_artifact(run_id, "model.pt", model_bytes)

# End run
tracker.end_run(run_id, "Completed")
```

## API Reference

### SynaDB (Core Database)

```python
from synadb import SynaDB

db = SynaDB("path.db")

# Write
db.put_float(key, value) -> int      # Returns offset
db.put_int(key, value) -> int
db.put_text(key, value) -> int
db.put_bytes(key, value) -> int

# Read
db.get_float(key) -> Optional[float]
db.get_int(key) -> Optional[int]
db.get_text(key) -> Optional[str]
db.get_bytes(key) -> Optional[bytes]

# History (for ML)
db.get_history_tensor(key) -> np.ndarray

# Operations
db.delete(key)
db.exists(key) -> bool
db.keys() -> List[str]
db.compact()
db.close()
```

### VectorStore

```python
from synadb import VectorStore

store = VectorStore(path, dimensions, metric="cosine")

store.insert(key, vector: np.ndarray)
store.search(query: np.ndarray, k=10) -> List[SearchResult]
store.get(key) -> Optional[np.ndarray]
store.delete(key)
store.build_index()  # Build HNSW for large datasets
len(store) -> int
```

### TensorEngine

```python
from synadb import TensorEngine

engine = TensorEngine(path)

# Pattern-based loading
data, shape = engine.get_tensor(pattern, dtype)
count = engine.put_tensor(prefix, data, shape, dtype)

# Chunked storage (large tensors)
chunks = engine.put_tensor_chunked(name, data, shape, dtype)
data, shape = engine.get_tensor_chunked(name)
```

### ModelRegistry

```python
from synadb import ModelRegistry

registry = ModelRegistry(path)

version = registry.save_model(name, data, metadata) -> ModelVersion
data, info = registry.load_model(name, version=None)
versions = registry.list_versions(name) -> List[ModelVersion]
registry.set_stage(name, version, stage)
prod = registry.get_production(name) -> Optional[ModelVersion]
```

### ExperimentTracker

```python
from synadb import ExperimentTracker

tracker = ExperimentTracker(path)

run_id = tracker.start_run(experiment, tags=[])
tracker.log_param(run_id, key, value)
tracker.log_metric(run_id, key, value, step=None)
tracker.log_artifact(run_id, name, data)
tracker.end_run(run_id, status)

run = tracker.get_run(run_id)
runs = tracker.list_runs(experiment)
metrics = tracker.get_metric(run_id, metric_name)
```

## Requirements

- Python 3.8+
- NumPy 1.21+
- The native library is bundled with the package

## Links

- [GitHub Repository](https://github.com/gtava5813/SynaDB)
- [Documentation](https://github.com/gtava5813/SynaDB/wiki)
- [Rust Crate](https://crates.io/crates/synadb)

## License

MIT License

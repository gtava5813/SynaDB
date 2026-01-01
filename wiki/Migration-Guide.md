# Migration Guide

This guide helps you upgrade SynaDB between major versions.

## Table of Contents

- [Upgrading from v0.5.x to v1.0.0](#upgrading-from-v05x-to-v100)
- [Upgrading from v0.2.x to v0.5.x](#upgrading-from-v02x-to-v05x)
- [Upgrading from v0.1.x to v0.2.x](#upgrading-from-v01x-to-v02x)

---

## Upgrading from v0.5.x to v1.0.0

### Overview

v1.0.0 is the production-ready release of SynaDB. This version focuses on stability, performance guarantees, and introduces a new licensing model.

### License Changes

**Important:** v1.0.0 introduces a BSL-style (Business Source License) model:

| Use Case | License |
|----------|---------|
| Personal use | ✅ Free |
| Companies under $10M ARR | ✅ Free |
| Companies under 1M MAUs | ✅ Free |
| Companies above thresholds | 💼 Commercial license required |
| Selling modified versions | ❌ Not permitted |

If you're using SynaDB in a commercial product and exceed the thresholds, please contact us for licensing options.

### New Features in v1.0.0

| Feature | Description |
|---------|-------------|
| LangChain Integration | VectorStore, ChatMessageHistory, DocumentLoader |
| LlamaIndex Integration | VectorStore, ChatStore |
| Haystack Integration | DocumentStore |
| PyTorch Integration | Dataset, DataLoader, DistributedSampler |
| TensorFlow Integration | tf.data.Dataset, tf.distribute support |
| Syna CLI | Command-line database management |
| Syna Studio | Web UI for database exploration |
| GPU Direct | CUDA tensor loading (optional feature) |
| FAISS Integration | Billion-scale vector search (optional feature) |
| Memory-Mapped I/O | Zero-copy tensor access |

### API Changes

v1.0.0 maintains backward compatibility with v0.5.x. No breaking API changes.

#### Python API

```python
# v0.5.x code continues to work in v1.0.0
from synadb import SynaDB, VectorStore, TensorEngine

with SynaDB("data.db") as db:
    db.put_float("key", 3.14)

# New in v1.0.0
from synadb.torch import SynaDataset, SynaDataLoader
from synadb.tensorflow import syna_dataset
from synadb.integrations.langchain import SynaVectorStore
```

#### Rust API

```rust
// v0.5.x code continues to work in v1.0.0
use synadb::{SynaDB, Atom};

let mut db = SynaDB::new("data.db")?;
db.append("key", Atom::Float(3.14))?;

// New in v1.0.0 (optional features)
#[cfg(feature = "gpu")]
use synadb::gpu::GpuContext;

#[cfg(feature = "faiss")]
use synadb::faiss_index::FaissIndex;
```

### Migration Steps

1. **Update your dependencies:**

   ```bash
   # Python
   pip install --upgrade synadb>=1.0.0
   
   # Rust
   cargo update -p synadb
   ```

2. **Review license compliance:**
   - Check if your organization exceeds the free tier thresholds
   - Contact us for commercial licensing if needed

3. **Test your application:**
   - Run your existing test suite
   - Verify all functionality works as expected

4. **No data migration required:**
   - Database files from v0.5.x are fully compatible with v1.0.0
   - HNSW index files (`.hnsw`) are compatible

### Deprecation Notices

No deprecations in v1.0.0. All v0.5.x APIs remain supported.

---

## Upgrading from v0.2.x to v0.5.x

### Overview

v0.5.0 is a major feature release that adds the AI Platform capabilities: HNSW indexing, Tensor Engine, Model Registry, and Experiment Tracking.

### New Features

| Feature | Module | Description |
|---------|--------|-------------|
| HNSW Index | `hnsw.rs` | O(log N) approximate nearest neighbor search |
| Tensor Engine | `tensor.rs` | Batch tensor operations with chunked storage |
| Model Registry | `model_registry.rs` | Version models with SHA-256 checksums |
| Experiment Tracking | `experiment.rs` | Log params, metrics, artifacts |

### API Additions

#### Python

```python
# New in v0.5.0
from synadb import TensorEngine, ModelRegistry, ExperimentTracker

# Tensor Engine
engine = TensorEngine("data.db")
engine.put_tensor_chunked("weights", large_array)
data, shape = engine.get_tensor_chunked("weights")

# Model Registry
registry = ModelRegistry("models.db")
version = registry.save_model("classifier", model_bytes, {"accuracy": "0.95"})
data, info = registry.load_model("classifier")

# Experiment Tracking
tracker = ExperimentTracker("experiments.db")
run_id = tracker.start_run("mnist", tags=["baseline"])
tracker.log_param(run_id, "lr", "0.001")
tracker.log_metric(run_id, "loss", 0.5, step=1)
tracker.end_run(run_id, "Completed")
```

#### Rust

```rust
// New in v0.5.0
use synadb::tensor::TensorEngine;
use synadb::model_registry::{ModelRegistry, ModelStage};
use synadb::experiment::{ExperimentTracker, RunStatus};

// Tensor Engine
let mut engine = TensorEngine::new(db);
engine.put_tensor_chunked("weights", &data, &shape, DType::Float32)?;

// Model Registry
let mut registry = ModelRegistry::new("models.db")?;
let version = registry.save_model("classifier", &bytes, metadata)?;

// Experiment Tracking
let mut tracker = ExperimentTracker::new("experiments.db")?;
let run_id = tracker.start_run("mnist", vec!["baseline".to_string()])?;
```

### VectorStore Enhancements

v0.5.0 adds automatic HNSW indexing to VectorStore:

```python
from synadb import VectorStore

store = VectorStore("vectors.db", dimensions=768)

# Insert many vectors
for i, embedding in enumerate(embeddings):
    store.insert(f"doc_{i}", embedding)

# HNSW index is automatically built when count > 10,000
# Or build manually:
store.build_index()

# Search is now O(log N) instead of O(N)
results = store.search(query, k=10)
```

### Migration Steps

1. **Update dependencies:**

   ```bash
   pip install --upgrade synadb>=0.5.0
   ```

2. **No code changes required:**
   - All v0.2.x APIs remain compatible
   - Existing databases work without modification

3. **Optional: Build HNSW indexes:**
   - For large vector stores (>10K vectors), call `store.build_index()`
   - This creates a `.hnsw` sidecar file for faster search

### Performance Notes

| Operation | v0.2.x | v0.5.0 |
|-----------|--------|--------|
| Vector search (10K) | ~10ms | ~1ms |
| Vector search (1M) | ~1000ms | ~5-10ms |
| Tensor load (1GB) | N/A | ~1s |

---

## Upgrading from v0.1.x to v0.2.x

### Overview

v0.2.0 introduced the Vector Store for embedding storage and similarity search.

### New Features

| Feature | Description |
|---------|-------------|
| `Atom::Vector` | New atom type for storing embeddings |
| `VectorStore` | High-level API for vector operations |
| Distance Metrics | Cosine, Euclidean, Dot Product |

### API Additions

#### Python

```python
# New in v0.2.0
from synadb import VectorStore
import numpy as np

store = VectorStore("vectors.db", dimensions=768, metric="cosine")
store.insert("doc1", embedding)
results = store.search(query, k=10)
```

#### Rust

```rust
// New in v0.2.0
use synadb::vector::{VectorStore, VectorConfig};
use synadb::distance::DistanceMetric;

let config = VectorConfig {
    dimensions: 768,
    metric: DistanceMetric::Cosine,
    ..Default::default()
};
let mut store = VectorStore::new("vectors.db", config)?;
store.insert("doc1", &embedding)?;
```

### Migration Steps

1. **Update dependencies:**

   ```bash
   pip install --upgrade synadb>=0.2.0
   ```

2. **No breaking changes:**
   - All v0.1.x code continues to work
   - New vector features are additive

---

## Database File Compatibility

### File Format Versions

| SynaDB Version | File Format | Compatible With |
|----------------|-------------|-----------------|
| v0.1.x | v1 | v0.1.x+ |
| v0.2.x | v1 | v0.1.x+ |
| v0.5.x | v1 | v0.1.x+ |
| v1.0.0 | v1 | v0.1.x+ |

All versions use the same underlying file format. Database files are forward and backward compatible.

### HNSW Index Files

HNSW index files (`.hnsw`) were introduced in v0.5.0:

- v0.5.x and v1.0.0 can read each other's `.hnsw` files
- If upgrading from v0.2.x, no `.hnsw` files exist (they'll be created on first `build_index()` call)

---

## Troubleshooting

### Common Issues

#### "Module not found" after upgrade

```bash
# Clear pip cache and reinstall
pip cache purge
pip uninstall synadb
pip install synadb
```

#### HNSW index not loading

```python
# Rebuild the index
store = VectorStore("vectors.db", dimensions=768)
store.build_index()  # Creates new .hnsw file
```

#### Performance degradation after upgrade

1. Check if HNSW index exists for large vector stores
2. Run `db.compact()` to optimize database file
3. Verify you're using the latest version

### Getting Help

- [GitHub Issues](https://github.com/gtava5813/SynaDB/issues)
- [API Reference](API-Reference)
- [Architecture](Architecture)

---

## Version History

| Version | Release Date | Highlights |
|---------|--------------|------------|
| v1.0.0 | Jan 2026 | Production release, integrations, tools, new license |
| v0.5.0 | Dec 2025 | HNSW, Tensor Engine, Model Registry, Experiments |
| v0.2.0 | Dec 8, 2025 | Vector Store |
| v0.1.0 | Dec 7, 2025 | Core Database |

See [Roadmap](Roadmap) for upcoming features.

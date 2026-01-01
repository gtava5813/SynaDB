# SynaDB Changelog

This document contains the complete release history for SynaDB.

---

## v1.0.3 - PyPI Native Library Fix

**Released:** January 2026  
**PyPI:** [synadb 1.0.3](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.3](https://crates.io/crates/synadb)

### Fixed

#### PyPI Native Library Bundling
- **Fixed:** `pip install synadb` now works on all platforms (Linux, macOS, Windows)
- **Issue:** Previous releases only bundled the Linux x86_64 native library
- **Solution:** Release workflow now copies all platform libraries into the PyPI package

#### Platform Support
| Platform | Library |
|----------|---------|
| Linux x86_64 | `libsynadb.so` |
| macOS x86_64 | `libsynadb-x86_64.dylib` |
| macOS ARM64 (Apple Silicon) | `libsynadb-arm64.dylib` |
| Windows x86_64 | `synadb.dll` |

#### Python Wrapper
- Enhanced `_find_library()` to detect platform AND architecture
- macOS ARM64 (Apple Silicon) now correctly loads ARM-specific library
- Library search now checks inside installed package directory first

---

## v1.0.2 - Bug Fixes

**Released:** January 2026

### Fixed
- Minor bug fixes and stability improvements

---

## v1.0.0 - Production Release

**Released:** January 2026  
**PyPI:** [synadb 1.0.0](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.0](https://crates.io/crates/synadb)

The first production-ready release of SynaDB with full AI/ML ecosystem integration.

### Highlights

- 🚀 **Production Ready** - Stable API, performance guarantees
- 🔗 **LLM Integrations** - LangChain, LlamaIndex, Haystack
- 🤖 **ML Integrations** - PyTorch Dataset/DataLoader, TensorFlow tf.data
- 🛠️ **Native Tools** - CLI and Studio Web UI
- ⚡ **Performance** - GPU Direct, FAISS, Memory-mapped I/O

### New Features

#### LLM Framework Integrations

##### LangChain

```python
from synadb.integrations.langchain import (
    SynaVectorStore,
    SynaChatMessageHistory,
    SynaDocumentLoader
)

# Vector store for RAG
vectorstore = SynaVectorStore.from_documents(documents, embedding, path="langchain.db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Chat history persistence
history = SynaChatMessageHistory(path="chat.db", session_id="user_123")
```

##### LlamaIndex

```python
from synadb.integrations.llamaindex import SynaVectorStore, SynaChatStore

vector_store = SynaVectorStore(path="index.db", dimensions=1536)
chat_store = SynaChatStore(path="chats.db")
```

##### Haystack

```python
from synadb.integrations.haystack import SynaDocumentStore

store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
```

#### ML Framework Integrations

##### PyTorch

```python
from synadb.torch import SynaDataset, SynaDataLoader, create_distributed_loader

dataset = SynaDataset(path="data.db", pattern="train/*")
loader = SynaDataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Distributed training support
loader, sampler = create_distributed_loader(dataset, batch_size=32)
```

##### TensorFlow

```python
from synadb.tensorflow import syna_dataset, create_distributed_dataset

dataset = syna_dataset(path="data.db", pattern="train/*", batch_size=32)

# Distributed training with tf.distribute
strategy = tf.distribute.MirroredStrategy()
dist_dataset = create_distributed_dataset(path="data.db", pattern="train/*", batch_size=32)
```

#### Native Tools

##### Syna CLI

Command-line interface for database inspection and management:

```bash
syna info mydata.db          # Database statistics
syna keys mydata.db          # List all keys
syna get mydata.db key       # Get value
syna export mydata.db out.json  # Export to JSON
```

##### Syna Studio Web UI

Web-based database explorer with:
- Keys Explorer with search and type filtering
- Model Registry dashboard
- 3D Embedding Clusters visualization (PCA)
- Statistics dashboard with customizable widgets
- Integrations scanner
- Custom Suite (compact, export, integrity check)
- Database switcher

```bash
# Launch with test data
cd demos/python/synadb
python run_ui.py --test

# Launch with HuggingFace embeddings
python run_ui.py --test --use-hf --samples 200

# Open existing database
python run_ui.py path/to/database.db
```

#### Performance Features

##### GPU Direct (Optional)

```python
from synadb.gpu import get_tensor_cuda, is_gpu_available

if is_gpu_available():
    tensor = get_tensor_cuda("data.db", "train/*", device=0)
```

##### FAISS Integration (Optional)

```rust
// Rust API with FAISS feature
let config = FaissConfig {
    index_type: "IVF1024,Flat".to_string(),
    train_size: 10000,
    nprobe: 10,
    use_gpu: false,
};
let mut index = FaissIndex::new(768, DistanceMetric::Cosine, config)?;
```

##### Memory-Mapped I/O

```rust
// Zero-copy tensor access
use synadb::mmap::{MmapReader, MmapTensorRef};

let reader = MmapReader::new("data.db")?;
let tensor_ref = reader.get_tensor_ref("weights")?;
```

### New Rust Modules

| Module | File | Description |
|--------|------|-------------|
| GPU Direct | `gpu.rs` | CUDA bindings for GPU memory access |
| FAISS Index | `faiss_index.rs` | FAISS wrapper for billion-scale search |
| Memory-Mapped I/O | `mmap.rs` | Zero-copy tensor access |

### New Python Modules

| Module | File | Description |
|--------|------|-------------|
| PyTorch | `torch.py` | Dataset, DataLoader, DistributedSampler |
| TensorFlow | `tensorflow.py` | tf.data.Dataset integration |
| Studio | `studio.py` | Flask-based web UI |
| GPU | `gpu.py` | Python GPU wrapper |
| MLflow | `integrations/mlflow.py` | MLflow backend |

---

## v0.5.0 - AI Platform Release

**Released:** December 2025  
**PyPI:** [synadb 0.5.0](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 0.5.0](https://crates.io/crates/synadb)

This is a major feature release that transforms SynaDB from a vector database into a complete AI/ML platform.

### Highlights

- 🚀 **HNSW Index** - O(log N) approximate nearest neighbor search
- 📊 **Tensor Engine** - Batch tensor operations with chunked storage
- 📦 **Model Registry** - Version models with SHA-256 checksum verification
- 🔬 **Experiment Tracking** - MLflow-style experiment logging
- 🔗 **LLM Integrations** - LangChain, LlamaIndex, and Haystack support

### New Features

#### HNSW Index

The Hierarchical Navigable Small World (HNSW) index provides fast approximate nearest neighbor search:

```python
from synadb import VectorStore

store = VectorStore("vectors.db", dimensions=768)

# Insert vectors (HNSW index builds automatically)
for doc_id, embedding in embeddings:
    store.insert(doc_id, embedding)

# Search is now O(log N) instead of O(N)
results = store.search(query_embedding, k=10)
```

**Features:**
- Multi-layer graph structure for efficient search
- Configurable parameters (M, ef_construction, ef_search)
- Automatic index building when vector count exceeds threshold (default: 1000)
- Save/load persistence to `.hnsw` sidecar files
- 95%+ recall on standard benchmarks

#### Tensor Engine

Batch tensor operations for ML training pipelines:

```python
from synadb import TensorEngine
import numpy as np

engine = TensorEngine("data.db")

# Store large tensors with automatic chunking
X_train = np.random.randn(10000, 768).astype(np.float32)
engine.put_tensor_chunked("train/X", X_train)

# Retrieve with shape preservation
X, shape = engine.get_tensor_chunked("train/X")

# Stream batches for training
for batch in engine.stream_batches("train/*", batch_size=32):
    model.train_step(batch)
```

**Features:**
- Pattern-based key matching (`sensor/*`, `train/*`)
- Chunked blob storage for large tensors (1MB chunks)
- Support for Float32, Float64, Int32, Int64 dtypes
- Memory-mapped access for zero-copy reads
- Direct I/O support for high throughput

#### Model Registry

Version and manage ML models with integrity verification:

```python
from synadb import ModelRegistry

registry = ModelRegistry("models.db")

# Save model with automatic versioning
version = registry.save_model(
    "classifier",
    model_bytes,
    metadata={"accuracy": "0.95", "framework": "pytorch"}
)

# Load with checksum verification
data, info = registry.load_model("classifier")
print(f"Version: {info.version}, Checksum: {info.checksum[:16]}...")

# Stage management
registry.set_stage("classifier", version.version, "Production")
prod_model = registry.get_production("classifier")
```

**Features:**
- Automatic version numbering
- SHA-256 checksum computation and verification
- Stage management (Development → Staging → Production → Archived)
- Metadata storage per version
- Corruption detection on load

#### Experiment Tracking

MLflow-style experiment logging built into the database:

```python
from synadb import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

# Start a run
run_id = tracker.start_run("mnist", tags=["baseline", "cnn"])

# Log parameters and metrics
tracker.log_param(run_id, "learning_rate", "0.001")
tracker.log_param(run_id, "batch_size", "32")

for epoch in range(100):
    loss = train_epoch()
    tracker.log_metric(run_id, "loss", loss, step=epoch)
    tracker.log_metric(run_id, "accuracy", accuracy, step=epoch)

# Log artifacts
tracker.log_artifact(run_id, "model.pt", model_bytes)

# End run
tracker.end_run(run_id, "Completed")

# Query runs
runs = tracker.query_runs("mnist", status="Completed", sort_by="accuracy")
```

**Features:**
- UUID-based run IDs
- Parameter and metric logging with step numbers
- Artifact storage (models, plots, configs)
- Run status management (Running, Completed, Failed, Killed)
- Query and filter runs by experiment, status, tags, parameters

#### LLM Framework Integrations

##### LangChain

```python
from langchain_openai import OpenAIEmbeddings
from synadb.integrations.langchain import SynaVectorStore, SynaChatMessageHistory

# Vector store for RAG
vectorstore = SynaVectorStore.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    path="langchain.db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Chat history persistence
history = SynaChatMessageHistory(path="chat.db", session_id="user_123")
history.add_user_message("Hello!")
history.add_ai_message("Hi there!")
```

##### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext
from synadb.integrations.llamaindex import SynaVectorStore

vector_store = SynaVectorStore(path="index.db", dimensions=1536)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
```

##### Haystack

```python
from synadb.integrations.haystack import SynaDocumentStore

store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
store.write_documents(documents)
results = store.filter_documents(filters={"category": "tech"})
```

**Note:** LangChain and LlamaIndex VectorStore integrations store document metadata in-memory only. Vectors are fully persisted. See [Known Limitations](#known-limitations) for details.

### Improvements

- VectorStore now supports optional HNSW indexing with automatic fallback to brute-force
- Improved error messages with more context
- Better memory management for large tensor operations
- FFI layer now uses canonicalized paths for consistent registry lookups

### Known Limitations

#### LangChain/LlamaIndex Metadata Persistence

When using `SynaVectorStore` with LangChain or LlamaIndex, document metadata (text content, custom metadata) is stored in-memory only. This means:

- ✅ Vectors persist across application restarts
- ✅ Similarity search works correctly
- ❌ Metadata is lost when the application restarts

**Workaround:** Store metadata in a separate database file:

```python
vectorstore = SynaVectorStore(path="vectors.db", ...)
metadata_db = SynaDB("metadata.db")  # Separate file for metadata
```

This limitation will be addressed in a future release by adding native metadata support to the Rust VectorStore.

---

## v0.2.0 - Vector Store

**Released:** December 8, 2025

This release adds vector embedding storage and similarity search capabilities.

### New Features

#### Vector Store

```python
from synadb import VectorStore
import numpy as np

store = VectorStore("vectors.db", dimensions=768, metric="cosine")

# Insert embeddings
store.insert("doc1", embedding1)
store.insert("doc2", embedding2)

# Search for similar vectors
results = store.search(query_embedding, k=5)
for r in results:
    print(f"{r.key}: {r.score:.4f}")
```

**Features:**
- `Atom::Vector(Vec<f32>, u16)` type for embedding storage
- Brute-force k-NN search (HNSW added in v0.5.0)
- Distance metrics: Cosine, Euclidean, DotProduct
- Dimension validation (64-4096)
- Python `VectorStore` class with NumPy integration

#### FFI Extensions

New C-ABI functions for vector operations:
- `SYNA_put_vector()` - Store vectors
- `SYNA_get_vector()` - Retrieve vectors
- `SYNA_free_vector()` - Free vector memory

### Property Tests

- Property 17: Vector Serialization Round-Trip
- Property 18: Similarity Search Correctness

---

## v0.1.0 - Core Database

**Released:** December 7, 2025

The initial release of SynaDB - an AI-native embedded database.

### Features

#### Core Database

```python
from synadb import SynaDB

with SynaDB("my.db") as db:
    # Write values
    db.put_float("sensor/temp", 72.5)
    db.put_int("counter", 42)
    db.put_text("config/name", "production")
    
    # Read values
    temp = db.get_float("sensor/temp")
    
    # Get history as tensor (for ML)
    tensor = db.get_history_tensor("sensor/temp")
```

**Features:**
- `Atom` enum: Null, Float, Int, Text, Bytes
- Append-only log storage with crash recovery
- In-memory index with O(1) key lookup
- Tombstone-based deletion
- Compaction to reclaim space

#### Compression

- LZ4 compression for values > 64 bytes
- Delta compression for consecutive floats
- Transparent decompression on read

#### FFI Layer

- C-ABI interface with `extern "C"` functions
- Global registry for instance management
- Panic safety with `catch_unwind`
- Integer error codes

#### Python Bindings

- `SynaDB` class with ctypes bindings
- Context manager support (`with` statement)
- NumPy integration for tensor extraction

### Property Tests (16 total)

| # | Property | Description |
|---|----------|-------------|
| 1 | Atom Serialization Round-Trip | Atoms serialize and deserialize correctly |
| 2 | LogHeader Serialization Round-Trip | Headers serialize and deserialize correctly |
| 3 | Write-Read Round-Trip | Written values can be read back |
| 4 | Index Rebuild on Reopen | Index rebuilds correctly after crash |
| 5 | Database Instance Isolation | Multiple databases don't interfere |
| 6 | Tensor Extraction Correctness | Float history extracts correctly |
| 7 | Tensor Filters Non-Float Types | Non-floats are filtered from tensors |
| 8 | Delta Compression Reduces Storage | Delta encoding saves space |
| 9 | LZ4 Compression Round-Trip | Compressed values decompress correctly |
| 10 | Concurrent Writes Preserve All Data | Parallel writes don't lose data |
| 11 | Corruption Recovery Skips Bad Entries | Bad entries are skipped on recovery |
| 12 | Schema-Free Key Acceptance | Any valid UTF-8 key is accepted |
| 13 | Delete Makes Key Unreadable | Deleted keys return None |
| 14 | Delete-Write Resurrection | Writing after delete resurrects key |
| 15 | Compaction Preserves Latest Values | Compaction keeps latest values |
| 16 | History Excludes Post-Tombstone | History stops at tombstone |

---

## Installation

### Python

```bash
pip install synadb
```

### Rust

```toml
[dependencies]
synadb = "1.0.3"
```

### Building from Source

```bash
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB
cargo build --release
```

---

## Links

- [GitHub Repository](https://github.com/gtava5813/SynaDB)
- [PyPI Package](https://pypi.org/project/synadb/)
- [Crates.io](https://crates.io/crates/synadb)
- [API Reference](API-Reference)
- [Getting Started Guide](Getting-Started)

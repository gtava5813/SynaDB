# SynaDB

[![CI](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml/badge.svg)](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synadb.svg)](https://pypi.org/project/synadb/)
[![License](https://img.shields.io/badge/License-SynaDB-blue.svg)](https://github.com/gtava5813/SynaDB/blob/main/LICENSE)

> AI-native embedded database for Python

SynaDB is an embedded database designed for AI/ML workloads. It combines the simplicity of SQLite with native support for vectors, tensors, model versioning, and experiment tracking.

## Architecture Philosophy

SynaDB uses a **modular architecture** where each component is a specialized class:

| Component | Purpose | Use Case |
|-----------|---------|----------|
| `SynaDB` | Core key-value store with history | Time-series, config, metadata |
| `VectorStore` | Embedding storage with HNSW search | RAG, semantic search |
| `MmapVectorStore` | High-throughput vector ingestion | Bulk embedding pipelines |
| `GravityWellIndex` | Fast-build vector index | Streaming/real-time data |
| `CascadeIndex` | Hybrid three-stage index | Balanced build/search (Experimental) |
| `SparseVectorStore` | Inverted index for sparse vectors | Lexical search (SPLADE, BM25) |
| `HybridVectorStore` | Hot/cold vector architecture | Ingestion + search combined |
| `TensorEngine` | Batch tensor operations | ML data loading |
| `ModelRegistry` | Model versioning with checksums | Model management |
| `Experiment` | Experiment tracking | MLOps workflows |

**Why modular?** Each component can be used independently or together, manages its own storage file, and is optimized for its specific workload.

**Typed API:** SynaDB uses typed methods (`put_float`, `put_int`, `put_text`) rather than a generic `set()` for type safety, performance, and FFI compatibility.

## Installation

```bash
pip install synadb
```

With optional dependencies:
```bash
pip install synadb[ml]        # PyTorch, TensorFlow, transformers
pip install synadb[langchain] # LangChain integration
pip install synadb[llama]     # LlamaIndex integration
pip install synadb[haystack]  # Haystack integration
pip install synadb[mlflow]    # MLflow integration
pip install synadb[pandas]    # Pandas integration
pip install synadb[all]       # Everything
```

## Features

| Feature | Description |
|---------|-------------|
| **Vector Store** | Embedding storage with similarity search (cosine, euclidean, dot product) |
| **MmapVectorStore** | Ultra-high-throughput vector storage (7x faster than VectorStore) |
| **HNSW Index** | O(log N) approximate nearest neighbor search for large-scale vectors |
| **Gravity Well Index** | O(N) build time index, faster build than HNSW |
| **Cascade Index** | Three-stage hybrid index (LSH + bucket tree + graph) with tunable recall (Experimental) |
| **Sparse Vector Store** | Inverted index for lexical embeddings (SPLADE, BM25, TF-IDF) |
| **HybridVectorStore** | Hot/cold architecture combining GWI (ingestion) + Cascade (search) |
| **Tensor Engine** | Batch tensor operations for ML data loading |
| **Model Registry** | Version and stage ML models with checksum verification |
| **Experiment Tracking** | Log parameters, metrics, and artifacts |
| **Core Database** | Schema-free key-value storage with history |
| **PyTorch Integration** | Native Dataset and DataLoader with distributed training support |
| **TensorFlow Integration** | tf.data.Dataset with tf.distribute strategy support |
| **LangChain Integration** | VectorStore, ChatMessageHistory, DocumentLoader |
| **LlamaIndex Integration** | VectorStore, ChatStore |
| **Haystack Integration** | DocumentStore |
| **MLflow Integration** | Tracking backend and artifact store |
| **GPU Operations** | CUDA tensor loading and prefetching |
| **Experience Collector** | RL experience storage with multi-machine sync |
| **Data Export/Import** | JSON, CSV, Parquet, Arrow, MessagePack formats |

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

**Supported Dimensions:** 64-8192 (covers MiniLM, BERT, OpenAI, DeepSeek-V3, etc.)

**High-Throughput Mode:**

```python
# Disable sync for 456x faster writes (use for bulk ingestion)
store = VectorStore("vectors.db", dimensions=768, sync_on_write=False)

# Context manager support (auto-saves index on exit)
with VectorStore("vectors.db", dimensions=768) as store:
    store.insert("doc1", embedding)
# Index automatically saved
```

### MmapVectorStore (Ultra-High-Throughput)

For maximum write throughput (7x faster than VectorStore):

```python
from synadb import MmapVectorStore
import numpy as np

# Pre-allocate capacity for best performance
store = MmapVectorStore("vectors.mmap", dimensions=768, initial_capacity=100_000)

# Batch insert for maximum throughput
keys = [f"doc_{i}" for i in range(10000)]
vectors = np.random.randn(10000, 768).astype(np.float32)
store.insert_batch(keys, vectors)  # 7x faster than VectorStore

# Build index for fast search
store.build_index()

# Search
results = store.search(query, k=10)  # 0.6ms
```

**Benchmark Results (10,000 vectors):**

| Model | Dims | Write/sec | Search | Storage |
|-------|------|-----------|--------|---------|
| MiniLM | 384 | 766,642 | 0.3ms | 18.8MB |
| BERT | 768 | 489,733 | 0.6ms | 34.9MB |
| OpenAI ada-002 | 1536 | 278,369 | 1.4ms | 67.2MB |

**Trade-offs vs VectorStore:**

| Aspect | VectorStore | MmapVectorStore |
|--------|-------------|-----------------|
| Write speed | ~67K/sec | ~490K/sec |
| Durability | Per-write | Checkpoint |
| Capacity | Dynamic | Pre-allocated |

### Gravity Well Index (Fastest Build Time)

For scenarios where index build time is critical (faster than HNSW):

```python
from synadb import GravityWellIndex
import numpy as np

# Initialize with sample vectors (required for attractor placement)
sample_vectors = np.random.randn(1000, 768).astype(np.float32)
gwi = GravityWellIndex("vectors.gwi", dimensions=768)
gwi.initialize(sample_vectors)

# Insert vectors (O(N) build time)
keys = [f"doc_{i}" for i in range(50000)]
vectors = np.random.randn(50000, 768).astype(np.float32)
gwi.insert_batch(keys, vectors)  # 46K vectors/sec

# Search with configurable recall
results = gwi.search(query, k=10, nprobe=50)  # 98% recall, 0.5ms
results = gwi.search(query, k=10, nprobe=100)  # 100% recall, 0.8ms
```

**GWI vs HNSW Build Time (50K vectors):**

| Model | GWI Build | HNSW Build | Speedup |
|-------|-----------|------------|---------|
| MiniLM (384d) | 1.5s | 272s | 186x |
| BERT (768d) | 3.0s | 504s | 168x |

**When to use GWI:**
- Index build time is critical
- Data is streaming/real-time
- Append-only storage required

**When to use HNSW:**
- Search latency is critical
- Index built once, queried many times
- Highest recall required

### Cascade Index (Experimental)

For balanced performance across build time, search speed, and recall:

```python
from synadb import CascadeIndex
import numpy as np

# Create with preset configuration
index = CascadeIndex("vectors.cascade", dimensions=768, preset="large")

# Or custom configuration
index = CascadeIndex("vectors.cascade", dimensions=768,
                     num_hyperplanes=16, bucket_capacity=128, nprobe=8)

# Insert vectors
keys = [f"doc_{i}" for i in range(50000)]
vectors = np.random.randn(50000, 768).astype(np.float32)
index.insert_batch(keys, vectors)

# Search
query = np.random.randn(768).astype(np.float32)
results = index.search(query, k=10)

# Save and close
index.save()
index.close()
```

**Configuration Presets:**

| Preset | Use Case | Build Speed | Search Speed | Recall |
|--------|----------|-------------|--------------|--------|
| `small` | <100K vectors | Fast | Fast | 95%+ |
| `large` | 1M+ vectors | Medium | Fast | 95%+ |
| `high_recall` | Accuracy critical | Slow | Medium | 99%+ |
| `fast_search` | Latency critical | Fast | Very Fast | 90%+ |

**Architecture:**
1. **LSH Layer** - Hyperplane-based locality-sensitive hashing with multi-probe
2. **Bucket Tree** - Adaptive splitting when buckets exceed threshold
3. **Sparse Graph** - Local neighbor refinement for final ranking

**When to use Cascade Index:**
- Need balanced build time and search speed
- Want tunable recall/latency trade-off
- Working with medium to large datasets (100K-10M vectors)

### HybridVectorStore (Hot/Cold Architecture)

Combines GWI (hot layer) with Cascade (cold layer) for optimal ingestion AND search:

```python
from synadb import HybridVectorStore
import numpy as np

# Create hybrid store with hot (GWI) and cold (Cascade) layers
store = HybridVectorStore(
    hot_path="vectors.gwi",
    cold_path="vectors.cascade",
    dimensions=768
)

# Initialize hot layer with sample vectors
sample = np.random.randn(1000, 768).astype(np.float32)
store.initialize_hot(sample)

# Ingest to hot layer (real-time, O(1) appends)
store.ingest("doc1", embedding)
store.ingest_batch(keys, vectors)  # High throughput

# Search both layers (results merged and deduplicated)
results = store.search(query, k=10)
for r in results:
    print(f"{r.key}: {r.score:.4f} (from {r.source})")  # source: "hot" or "cold"

# Promote hot data to cold layer (maintenance operation)
promoted = store.promote_to_cold()
print(f"Promoted {promoted} vectors to cold storage")

# Layer-specific operations
print(f"Hot: {store.hot_count()}, Cold: {store.cold_count()}")
store.flush_hot()  # Persist hot layer
store.save_cold()  # Persist cold layer
```

**Architecture:**

| Layer | Index | Role | Write | Read |
|-------|-------|------|-------|------|
| Hot | GWI | Real-time buffer | O(1) sync | Fallback |
| Cold | Cascade | Historical storage | Batch | Primary |

**When to use HybridVectorStore:**
- Need both high-throughput ingestion AND fast search
- Streaming data with periodic archival
- Want automatic hot→cold data lifecycle

### Sparse Vector Store (Lexical Search)

For lexical embeddings from sparse encoders like SPLADE, BM25, or TF-IDF:

```python
from synadb import SparseVectorStore

# Create store with vocabulary size (e.g., BERT vocab = 30522)
store = SparseVectorStore("lexical.svs", vocab_size=30522)

# Index sparse vectors from any encoder (SPLADE, BM25, TF-IDF)
store.index("doc1", indices=[101, 2054, 3000], values=[0.8, 0.5, 0.3])
store.index("doc2", indices=[101, 5678, 9012], values=[0.9, 0.4, 0.1])

# Search with sparse query
results = store.search(query_indices=[101, 2054], query_values=[0.7, 0.6], k=10)
for r in results:
    print(f"{r.key}: {r.score:.4f}")

# Get statistics
stats = store.stats()
print(f"Documents: {stats.num_vectors}, Avg NNZ: {stats.avg_nnz:.1f}")

# Persistence
store.save()
store.close()

# Reopen existing store
store = SparseVectorStore.open("lexical.svs")
```

**When to use SparseVectorStore:**
- Lexical/keyword search (BM25, TF-IDF)
- Learned sparse representations (SPLADE, SPLADE++)
- Hybrid search (combine with dense VectorStore)
- High-dimensional sparse data

**Architecture:**
- Inverted index maps vocabulary terms to document postings
- O(min(nnz)) search complexity (nnz = non-zero elements in query)
- Exact search (100% recall)

### Tensor Engine (ML Data Loading)

**Key Semantics:** When storing tensors, the first parameter is a **key prefix**, not a full key. Elements are stored with auto-generated keys. When loading, use glob patterns to retrieve all elements.

```python
from synadb import TensorEngine
import numpy as np

engine = TensorEngine("training.db")

# Store training data (prefix generates keys: train/X/0000, train/X/0001, ...)
X_train = np.random.randn(10000, 784).astype(np.float32)
engine.put_tensor("train/X/", X_train)  # Note: prefix ends with /

# Load as tensor (pattern matching with glob)
X = engine.get_tensor("train/X/*", dtype=np.float32, shape=(10000, 784))

# For large tensors, use chunked storage (more efficient)
engine.put_tensor_chunked("train/large/", large_tensor, chunk_size=10000)
X = engine.get_tensor_chunked("train/large/chunk_*")

# Stream in batches for training
for batch in engine.stream("train/X/*", batch_size=32):
    model.train_step(batch)
```

### Model Registry

```python
from synadb import ModelRegistry

registry = ModelRegistry("models.db")

# Save model with metadata (auto-versions, returns ModelVersion)
model_bytes = open("model.pt", "rb").read()
version = registry.save("classifier", model_bytes, {"accuracy": "0.95"})
print(f"Saved v{version.version}, checksum: {version.checksum}")

# Load with automatic checksum verification
model = registry.load("classifier")  # Latest version
model = registry.load("classifier", version=1)  # Specific version

# Promote to production
registry.promote("classifier", version.version, "production")

# Get production model directly
prod_model = registry.get_production_model("classifier")
```

### Experiment Tracking

```python
from synadb import Experiment

exp = Experiment("mnist", "experiments.db")

# Start a run with context manager
with exp.start_run(tags=["baseline"]) as run:
    # Log hyperparameters
    run.log_params({"learning_rate": 0.001, "batch_size": 32})
    
    # Log metrics during training
    for epoch in range(100):
        loss = 1.0 / (epoch + 1)
        run.log_metric("loss", loss, step=epoch)
    
    # Log artifacts
    run.log_artifact("model.pt", model_bytes)
    # Run automatically ends when context exits

# Query runs
runs = exp.list_runs()
best_run = exp.get_best_run(metric="loss", minimize=True)
```

---

## Framework Integrations

### LangChain Integration

```python
from synadb.integrations.langchain import (
    SynaVectorStore,
    SynaChatMessageHistory,
    SynaDocumentLoader
)
from langchain_openai import OpenAIEmbeddings

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

# Load documents from SynaDB
loader = SynaDocumentLoader(path="docs.db", pattern="documents/*")
docs = loader.load()
```

### LlamaIndex Integration

```python
from synadb.integrations.llamaindex import SynaVectorStore, SynaChatStore
from llama_index.core import VectorStoreIndex, StorageContext

# Vector store
vector_store = SynaVectorStore(path="index.db", dimensions=1536)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Chat store for conversation memory
chat_store = SynaChatStore(path="chats.db")
chat_store.set_messages("session_1", messages)
```

### Haystack Integration

```python
from synadb.integrations.haystack import SynaDocumentStore

document_store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
document_store.write_documents(documents)

# Query with embeddings
results = document_store.query_by_embedding(query_embedding, top_k=10)
```

### MLflow Integration

```python
from synadb.integrations.mlflow import SynaTrackingStore
import mlflow

# Use SynaDB as MLflow tracking backend
mlflow.set_tracking_uri("synadb:///experiments.db")

with mlflow.start_run():
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pt")
```

---

## Deep Learning Integrations

### PyTorch Integration

```python
from synadb.torch import SynaDataset, SynaDataLoader

# Create PyTorch Dataset backed by SynaDB
dataset = SynaDataset(
    path="mnist.db",
    pattern="train/*",
    transform=None  # Optional: add torchvision transforms
)

# Use with standard DataLoader
loader = SynaDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    prefetch_factor=2
)

# Training loop
for batch in loader:
    # batch is a torch.Tensor
    pass
```

**Distributed Training Support:**

```python
from synadb.torch import SynaDataset, create_distributed_loader

dataset = SynaDataset("data.db", pattern="train/*")
loader, sampler = create_distributed_loader(dataset, batch_size=32, num_workers=4)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for proper shuffling
    for batch in loader:
        pass
```

### TensorFlow Integration

```python
from synadb.tensorflow import syna_dataset, SynaDataset
import tensorflow as tf

# Create tf.data.Dataset from SynaDB
dataset = syna_dataset(
    path="data.db",
    pattern="train/*",
    batch_size=32
).prefetch(tf.data.AUTOTUNE)

# Use in training
for batch in dataset:
    pass
```

**Distributed Training with tf.distribute:**

```python
from synadb.tensorflow import create_distributed_dataset
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    dataset = create_distributed_dataset(
        path="data.db",
        pattern="train/*",
        batch_size=32
    )
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    model.fit(dist_dataset, epochs=10)
```

---

## GPU Operations

```python
from synadb.gpu import get_tensor_cuda, prefetch_to_gpu, is_gpu_available, get_gpu_info

# Check GPU availability
if is_gpu_available():
    info = get_gpu_info(0)
    print(f"GPU: {info['name']}, Memory: {info['total_memory'] / 1e9:.1f} GB")

# Load tensor directly to GPU
tensor = get_tensor_cuda("data.db", "train/*", device=0)
# tensor is already on cuda:0

# Prefetch data for faster access
prefetch_to_gpu("data.db", "train/*", device=0)
```

---

## Experience Collector (Reinforcement Learning)

```python
from synadb import ExperienceCollector

# Create collector with machine ID for multi-machine sync
collector = ExperienceCollector("experiences.db", machine_id="gpu_server_1")

# Log transitions
with collector.session(model="Qwen/Qwen3-4B") as session:
    session.log(
        state=(0, 1, 2, 0.5),
        action="analyze_weights",
        reward=0.75,
        next_state=(0, 1, 3, 0.6)
    )

# Get rewards as tensor for training
rewards = collector.get_rewards_tensor("default")

# Merge experiences from multiple machines
ExperienceCollector.merge(
    ["machine1/exp.db", "machine2/exp.db"],
    "master/exp.db"
)

# Export for sharing
collector.export_jsonl("experiences.jsonl")
```

---

## Syna Studio (Web UI)

Syna Studio is a web-based interface for exploring and managing SynaDB databases.

```bash
cd demos/python/synadb

# Launch with test data
python run_ui.py --test

# Launch with HuggingFace embeddings
python run_ui.py --test --use-hf --samples 200

# Open existing database
python run_ui.py path/to/database.db
```

**Features:**
- Keys Explorer with search and type filtering
- Model Registry dashboard
- 3D Embedding Clusters visualization (PCA)
- Statistics with customizable widgets
- Integrations scanner
- Custom Suite (compact, export, integrity check)

Access at `http://localhost:8501`. See [STUDIO_DOCS.md](synadb/STUDIO_DOCS.md) for full documentation.

---

## Data Export & Import

SynaDB supports multiple formats for interoperability.

**Export Formats:**

| Format | Method | Dependencies | Best For |
|--------|--------|--------------|----------|
| JSON | `export_json()` | None | Human-readable, config files |
| JSON Lines | `export_jsonl()` | None | Streaming, log processing |
| CSV | `export_csv()` | None | Spreadsheets, simple analysis |
| Pickle | `export_pickle()` | None | Python-only workflows |
| Parquet | `export_parquet()` | `pyarrow` | ML pipelines, Spark, DuckDB |
| Arrow | `export_arrow()` | `pyarrow` | High-performance data exchange |
| MessagePack | `export_msgpack()` | `msgpack` | Compact binary, cross-language |

```python
from synadb import SynaDB

with SynaDB("my_data.db") as db:
    # Export to various formats
    db.export_json("data.json")
    db.export_parquet("data.parquet")  # Requires pyarrow
    
    # Filter by key pattern
    db.export_parquet("sensors.parquet", key_pattern="sensor/*")

# Import from various formats
with SynaDB("new_data.db") as db:
    db.import_json("data.json")
    db.import_parquet("data.parquet", key_prefix="imported/")
```

---

## Performance

SynaDB is optimized for AI/ML workloads:

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Vector insert (VectorStore) | Baseline | 768-dim, sync_on_write=False |
| Vector insert (MmapVectorStore) | 7x faster | 768-dim, batch insert |
| Vector search (1M) | <10ms | Top-10, HNSW index |
| Tensor load | 1+ GB/s | NVMe SSD |
| Experiment log | <100μs | Single metric |

---

## API Reference

### SynaDB (Core Database)

```python
from synadb import SynaDB

db = SynaDB("path.db")

# Write
db.put_float(key, value) -> int
db.put_int(key, value) -> int
db.put_text(key, value) -> int
db.put_bytes(key, value) -> int

# Read
db.get_float(key) -> Optional[float]
db.get_int(key) -> Optional[int]
db.get_text(key) -> Optional[str]
db.get_bytes(key) -> Optional[bytes]
db.get_history_tensor(key) -> np.ndarray

# Operations
db.delete(key)
db.exists(key) -> bool
db.keys() -> List[str]
db.compact()
db.close()

# Export/Import
db.export_json(path, key_pattern=None) -> int
db.export_parquet(path, key_pattern=None) -> int
db.import_json(path, key_prefix="") -> int
db.import_parquet(path, key_prefix="") -> int
```

### VectorStore

```python
from synadb import VectorStore

store = VectorStore(path, dimensions, metric="cosine", sync_on_write=True)

store.insert(key, vector: np.ndarray)
store.search(query: np.ndarray, k=10) -> List[SearchResult]
store.get(key) -> Optional[np.ndarray]
store.delete(key)
store.build_index()  # Build HNSW for large datasets
store.flush()  # Save index without closing
store.close()  # Close and save
len(store) -> int

# Context manager support
with VectorStore(path, dimensions) as store:
    store.insert(key, vector)
# Auto-saved on exit
```

### MmapVectorStore

```python
from synadb import MmapVectorStore

store = MmapVectorStore(path, dimensions, initial_capacity=10000)

store.insert(key, vector: np.ndarray)
store.insert_batch(keys: List[str], vectors: np.ndarray)  # 490K/sec
store.search(query: np.ndarray, k=10) -> List[SearchResult]
store.get(key) -> Optional[np.ndarray]
store.build_index()
store.checkpoint()  # Save to disk
len(store) -> int
```

### GravityWellIndex

```python
from synadb import GravityWellIndex

gwi = GravityWellIndex(path, dimensions)

gwi.initialize(sample_vectors: np.ndarray)  # Required before insert
gwi.insert(key, vector: np.ndarray)
gwi.insert_batch(keys: List[str], vectors: np.ndarray)  # 46K/sec
gwi.search(query: np.ndarray, k=10, nprobe=50) -> List[SearchResult]
gwi.get(key) -> Optional[np.ndarray]
gwi.save()
len(gwi) -> int
```

### CascadeIndex (Experimental)

```python
from synadb import CascadeIndex

# Create with preset
index = CascadeIndex(path, dimensions, preset="large")

# Or custom config
index = CascadeIndex(path, dimensions, num_hyperplanes=16, 
                     bucket_capacity=128, nprobe=8)

index.insert(key, vector: np.ndarray)
index.insert_batch(keys: List[str], vectors: np.ndarray)
index.search(query: np.ndarray, k=10) -> List[SearchResult]
index.get(key) -> Optional[np.ndarray]
index.save()
index.close()
len(index) -> int
```

### SparseVectorStore

```python
from synadb import SparseVectorStore, SparseSearchResult, SparseIndexStats

# Create new store
store = SparseVectorStore(path, vocab_size=30522)

# Open existing store
store = SparseVectorStore.open(path)

# Index sparse vectors (from any encoder: SPLADE, BM25, TF-IDF)
store.index(key, indices: List[int], values: List[float])

# Search
results = store.search(query_indices: List[int], query_values: List[float], k=10) -> List[SparseSearchResult]
# SparseSearchResult has: key, score

# Operations
store.delete(key) -> bool
store.contains(key) -> bool
stats = store.stats() -> SparseIndexStats  # num_vectors, vocab_size, total_postings, avg_nnz

# Persistence
store.save()
store.close()
len(store) -> int
```

### HybridVectorStore

```python
from synadb import HybridVectorStore

store = HybridVectorStore(hot_path, cold_path, dimensions)

store.initialize_hot(sample_vectors: np.ndarray)  # Required before ingest
store.ingest(key, vector: np.ndarray)  # Insert to hot layer
store.ingest_batch(keys: List[str], vectors: np.ndarray) -> int
store.search(query: np.ndarray, k=10) -> List[HybridSearchResult]  # Both layers
store.search_hot(query: np.ndarray, k=10) -> List[SearchResult]  # Hot only
store.search_cold(query: np.ndarray, k=10) -> List[SearchResult]  # Cold only
store.promote_to_cold() -> int  # Move hot data to cold layer
store.hot_count() -> int
store.cold_count() -> int
store.flush_hot()  # Persist hot layer
store.save_cold()  # Persist cold layer
len(store) -> int  # Total across both layers
```

### TensorEngine

```python
from synadb import TensorEngine

engine = TensorEngine(path)

# Store tensor (prefix generates auto-keys: prefix/0000, prefix/0001, ...)
count = engine.put_tensor(prefix, tensor: np.ndarray) -> int
chunks = engine.put_tensor_chunked(prefix, tensor: np.ndarray, chunk_size=1000) -> int

# Load tensor (pattern uses glob matching)
data = engine.get_tensor(pattern, shape=None, dtype=np.float32) -> np.ndarray
data = engine.get_tensor_chunked(pattern, dtype=np.float32, shape=None) -> np.ndarray

# Framework integration
tensor = engine.get_tensor_torch(pattern, shape=None, device="cpu") -> torch.Tensor
tensor = engine.get_tensor_tf(pattern, shape=None) -> tf.Tensor

# Streaming
for batch in engine.stream(pattern, batch_size=32, dtype=np.float32):
    ...

# Utilities
keys = engine.keys(pattern="*") -> List[str]
count = engine.delete(pattern) -> int
```

### ModelRegistry

```python
from synadb import ModelRegistry

registry = ModelRegistry(path)

version = registry.save(name, model, metadata) -> ModelVersion  # Auto-versions
model = registry.load(name, version=None)  # Latest if version=None
versions = registry.list(name) -> List[ModelVersion]
registry.promote(name, version, stage)  # "development", "staging", "production", "archived"
prod = registry.get_production_model(name) -> Optional[Any]
info = registry.get_version(name, version) -> Optional[ModelVersion]
registry.delete(name, version=None)  # Delete specific or all versions
```

### Experiment

```python
from synadb import Experiment, Run

exp = Experiment(name, path)

# Start run with context manager
with exp.start_run(tags=[]) as run:
    run.log_param(key, value)
    run.log_params({key: value, ...})
    run.log_metric(key, value, step=None)
    run.log_metrics({key: value, ...}, step=None)
    run.log_artifact(name, data)
    # Run ends automatically

# Query runs
runs = exp.list_runs() -> List[Run]
run = exp.get_run(run_id) -> Run
best = exp.get_best_run(metric, minimize=True) -> Run
```

### ExperienceCollector

```python
from synadb import ExperienceCollector

collector = ExperienceCollector(path, machine_id=None)

key = collector.log_transition(state, action, reward, next_state, metadata=None)
rewards = collector.get_rewards_tensor(session_id)
collector.export_jsonl(output_path)
collector.import_jsonl(input_path)
ExperienceCollector.merge(sources, dest)
```

### GPU Operations

```python
from synadb.gpu import (
    get_tensor_cuda,
    prefetch_to_gpu,
    is_gpu_available,
    get_gpu_count,
    get_gpu_info
)

tensor = get_tensor_cuda(db_path, pattern, device=0) -> torch.Tensor
prefetch_to_gpu(db_path, pattern, device=0)
is_gpu_available() -> bool
get_gpu_count() -> int
get_gpu_info(device=0) -> Optional[dict]
```

---

## Requirements

- Python 3.8+
- NumPy 1.21+
- The native library is bundled with the package

## Links

- [GitHub Repository](https://github.com/gtava5813/SynaDB)
- [Documentation](https://github.com/gtava5813/SynaDB/wiki)
- [Rust Crate](https://crates.io/crates/synadb)

## License

SynaDB License - Free for personal use and companies under $10M ARR / 1M MAUs. See [LICENSE](https://github.com/gtava5813/SynaDB/blob/main/LICENSE) for details.

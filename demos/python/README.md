# SynaDB

[![CI](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml/badge.svg)](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synadb.svg)](https://pypi.org/project/synadb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AI-native embedded database for Python

SynaDB is an embedded database designed for AI/ML workloads. It combines the simplicity of SQLite with native support for vectors, tensors, model versioning, and experiment tracking.

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
| **HNSW Index** | O(log N) approximate nearest neighbor search for large-scale vectors |
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
| Vector insert | 100K+ vectors/sec | 768-dim, float32 |
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

data, shape = engine.get_tensor(pattern, dtype)
count = engine.put_tensor(prefix, data, shape, dtype)
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

### Experiment

```python
from synadb import Experiment, Run

exp = Experiment(name, path)

# Start run with context manager
with exp.start_run(tags=[]) as run:
    run.log_param(key, value)
    run.log_params({key: value, ...})
    run.log_metric(key, value, step=None)
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

MIT License

# SynaDB Wiki

Welcome to the SynaDB wiki! SynaDB is an AI-native embedded database designed for ML/AI applications.

## Quick Links

- [Getting Started](Getting-Started)
- [Roadmap](Roadmap)
- [Migration Guide](Migration-Guide)
- [Architecture](Architecture)
- [API Reference](API-Reference)
- [Python Guide](Python-Guide)
- [Rust Guide](Rust-Guide)
- [Contributing](Contributing)

## What is SynaDB?

SynaDB is an embedded, log-structured, columnar-mapped database engine written in Rust. It combines:

- **SQLite's simplicity** - Single file, zero config, embedded
- **DuckDB's analytics** - Columnar history, tensor extraction
- **MongoDB's flexibility** - Schema-free Atom type

## Current Version

**v1.0.6** - Latest Release (January 2026)
- Fixed: GravityWellIndex persistence bug - data now correctly persists across close/reopen
- Fixed: CascadeIndex Python import error
- Updated: Comprehensive Rust documentation

**v1.0.4-v1.0.5** - Performance Releases
- MmapVectorStore - Ultra-high-throughput vector storage (7x faster than VectorStore)
- Gravity Well Index (GWI) - O(N) build time, faster than HNSW
- Cascade Index - Three-stage hybrid index (Experimental)
- HNSW recall fix - improved from 0-20% to 100%

**v1.0.0** - Production Release
- Append-only log storage with schema-free data types
- Vector Store with HNSW index for similarity search
- Tensor Engine for batch ML data operations
- Model Registry with checksum verification
- Experiment Tracking for ML workflows
- LangChain, LlamaIndex, Haystack integrations
- PyTorch and TensorFlow integrations
- GPU Direct memory access (optional)
- FAISS integration for billion-scale search (optional)
- Syna CLI and Studio Web UI
- LZ4 and delta compression
- C-ABI for Python/C++ integration

## Installation

```bash
# Python
pip install synadb

# Rust
cargo add synadb
```

## Quick Example

```python
from synadb import SynaDB

with SynaDB("my_data.db") as db:
    db.put_float("temperature", 23.5)
    db.put_float("temperature", 24.1)
    
    # Get history as numpy array for ML
    history = db.get_history_tensor("temperature")
    print(history)  # [23.5, 24.1]
```

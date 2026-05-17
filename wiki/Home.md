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

**v1.4.0** - Feature Store (May 2026)

- Embedded ML feature management (typed schemas, PIT queries, <1ms serving)
- 12 new Rust modules, 8 FFI functions, Python wrapper
- Point-in-time correctness by construction (no data leakage)
- Online serving benchmarked at p99 = 6μs

**v1.3.1** - Syna Query Language (May 2026)

- SQL-like (EQL) and MongoDB-like (EMQ) query interfaces
- Aggregations, temporal joins, anomaly detection, pattern matching, predictions, correlation
- CLI: `syna query mydb.db "SELECT * FROM 'sensor/*' WHERE value > 30"`
- 21 submodules, 339 tests, zero clippy warnings

**v1.2.1** - Security Hardening + DAVO Persistence (May 2026)

- Attribution: project owned by Mindoval, Inc
- Security: bumped pytest (CVE fix), added workflow permissions, fixed path injection + stack trace exposure
- DAVO: persistence (`save()` / `load()`) for `FreshnessIndexV2` and `DecayPredictor`

**v1.2.0** - DAVO: Decay-Aware Value Optimization (May 2026)

- New optional feature (`--features davo`) adding decay-aware storage
- Every value carries a decay rate λ, freshness degrades as `e^(-λ × age)`
- Bayesian learning of optimal decay rates with Thompson Sampling
- O(k + log N) staleness queries via Forward Decay deadline index
- 14 FFI functions, Python wrappers, 6 property tests, real-data demo

**v1.1.2** - Internal Audit: Safety & Correctness Hardening (March 2026)

- Security: Updated 3 Python dependencies to fix critical vulnerabilities
- Documentation: Fixed markdown linting issues across 10 files

**v1.1.0** - Sparse Vector Store (January 2026)

- New: SparseVectorStore (SVS) for lexical embeddings (SPLADE, BM25, TF-IDF)
- New: Hybrid RAG example with Amazon ESCI dataset
- New: Benchmark script with scale testing (100K-500K)
- GWI 13-29x faster than Cascade for inserts (scales better)

**v1.0.6** - GWI Persistence Fix (January 2026)

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

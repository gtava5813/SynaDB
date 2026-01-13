# SynaDB Roadmap

## Vision

Make SynaDB the **default database for AI/ML applications** - the SQLite of the AI era.

## Release Status

| Version | Focus | Released | Status |
|---------|-------|----------|--------|
| v0.1.0 | Core Database | Dec 7, 2025 | ✅ Complete |
| v0.2.0 | Vector Store | Dec 8, 2025 | ✅ Complete |
| v0.5.0 | AI Platform | Dec 2025 | ✅ Complete |
| v1.0.0 | Production Release | Jan 2026 | ✅ Complete |
| v1.0.4 | MmapVectorStore, GWI | Jan 2026 | ✅ Complete |
| v1.0.5 | Cascade Index | Jan 2026 | ✅ Complete |
| v1.0.6 | GWI Persistence Fix | Jan 2026 | ✅ Complete |
| v1.1.0 | Sparse Vector Store | Jan 2026 | ✅ Current |
| v1.2.0 | Query Language | - | 📋 Planned |
| v1.3.0 | Feature Store | - | 📋 Planned |
| v1.4.0 | Distributed Mode | - | 📋 Planned |

---

## v1.0.0 - Production Release ✅

**Status:** Released January 2026

The first production-ready release with full AI/ML ecosystem integration.

### LLM Framework Integrations

- ✅ LangChain - VectorStore, ChatMessageHistory, Loader
- ✅ LlamaIndex - VectorStore, ChatStore
- ✅ Haystack - DocumentStore

### ML Framework Integrations

- ✅ PyTorch - Dataset, DataLoader, DistributedSampler support
- ✅ TensorFlow - tf.data.Dataset integration

### Native Tools

- ✅ Syna CLI - Command-line database inspection and management
- ✅ Syna Studio - Web UI for exploring vectors, experiments, and models

### Performance Features

- ✅ FAISS Integration - Billion-scale vector search (optional feature)
- ✅ GPU Direct - Pinned memory and CUDA stream support (optional feature)
- ✅ Memory-mapped tensor access for zero-copy reads

---

## v1.0.4 - MmapVectorStore & GWI ✅

**Status:** Released January 2026

Major performance and reliability improvements.

### New Features

- ✅ MmapVectorStore - Ultra-high-throughput vector storage (7x faster than VectorStore)
- ✅ Gravity Well Index (GWI) - Novel append-only indexing (faster build than HNSW)
- ✅ HNSW Recall Fix - Fixed critical bug causing 0-20% recall → now 100%
- ✅ HNSW Auto-Build - Index now auto-builds when threshold exceeded
- ✅ sync_on_write - Configurable sync for 456x throughput improvement

---

## v1.0.5 - Cascade Index ✅

**Status:** Released January 2026

### New Features

- ✅ Cascade Index (Experimental) - Three-stage hybrid index (LSH + bucket tree + graph)
- Sub-linear search with tunable recall/latency trade-off
- O(N) build time, 95%+ recall with default settings

---

## v1.0.6 - GWI Persistence Fix ✅ CURRENT

**Status:** Released January 2026

Bug fixes and documentation improvements.

### Fixes

- ✅ GWI Persistence Fix - Critical bug fix for GravityWellIndex data persistence
- ✅ CascadeIndex Import Fix - Fixed Python import error
- ✅ Documentation Updates - Comprehensive Rust docs, Architecture Philosophy section

See [Changelog](Changelog) for full details.

---

## v1.1.0 - Query Language 📋

**Status:** Planned

### Goals

- SQL-like syntax (EQL)
- MongoDB-like syntax (EMQ)
- Aggregations
- Time-series operations

---

## Future

### v1.2.0 - Feature Store 📋

- Feature schema definition
- Point-in-time queries
- Online serving (<1ms)
- Training data generation

### v1.3.0 - Distributed Mode 📋

- Multi-node replication
- Sharding support
- Consensus protocol

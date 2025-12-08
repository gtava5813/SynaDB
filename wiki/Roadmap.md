# SynaDB Roadmap

## Vision

Make SynaDB the **default database for AI/ML applications** - the SQLite of the AI era.

## Release Status

| Version | Focus | Released | Status |
|---------|-------|----------|--------|
| v0.1.0 | Core Database | Dec 7, 2025 | ✅ Complete |
| v0.2.0 | Vector Store | Dec 8, 2025 | ✅ Complete |
| v0.5.0 | AI Platform | Dec 2025 | ✅ Current |
| v0.6.0 | LLM Integrations | - | 📋 Planned |
| v0.7.0 | Query Language | - | 📋 Planned |

---

## v0.5.0 - AI Platform ✅ CURRENT

**Status:** Released

### Features
- ✅ HNSW Index - O(log N) approximate nearest neighbor search
- ✅ Tensor Engine - Batch tensor operations with chunked storage
- ✅ Model Registry - Version models with SHA-256 checksum verification
- ✅ Experiment Tracking - Log params, metrics, artifacts
- ✅ HNSW Persistence - Save/load index to files

---

## v0.6.0 - LLM Framework Integrations 📋

**Status:** Planned

### Goals
- LangChain VectorStore integration
- LlamaIndex integration
- PyTorch DataLoader
- TensorFlow tf.data.Dataset

---

## v0.7.0 - Query Language 📋

**Status:** Planned

### Goals
- SQL-like syntax (EQL)
- MongoDB-like syntax (EMQ)
- Aggregations
- Time-series operations

---

## Future

### v0.8.0 - Feature Store
- Feature schema definition
- Point-in-time queries
- Online serving

### v0.9.0 - Performance
- GPU direct memory access
- Memory-mapped tensor access

### v1.0.0 - Production Ready
- Distributed mode
- Security audit
- Performance guarantees

# SynaDB Wiki

Welcome to the SynaDB wiki! SynaDB is an AI-native embedded database designed for ML/AI applications.

## Quick Links

- [Getting Started](Getting-Started)
- [Roadmap](Roadmap)
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

**v0.1.0** - Core Database
- Append-only log storage
- Schema-free data types (Float, Int, Text, Bytes)
- History/tensor extraction for ML
- LZ4 and delta compression
- C-ABI for Python/C++ integration
- Thread-safe concurrent access

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

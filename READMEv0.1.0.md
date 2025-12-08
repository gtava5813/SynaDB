# SynaDB

[![CI](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml/badge.svg)](https://github.com/gtava5813/SynaDB/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synadb.svg)](https://pypi.org/project/synadb/)
[![Crates.io](https://img.shields.io/crates/v/synadb.svg)](https://crates.io/crates/synadb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An AI-native embedded database.

An embedded, log-structured, columnar-mapped database engine written in Rust. Syna combines the embedded simplicity of SQLite, the columnar analytical speed of DuckDB, and the schema flexibility of MongoDB.

## Features

- **Append-only log structure** - Fast sequential writes, immutable history
- **Schema-free** - Store heterogeneous data types without migrations
- **AI/ML optimized** - Extract time-series data as contiguous tensors for PyTorch/TensorFlow
- **C-ABI interface** - Use from Python, Node.js, C++, or any FFI-capable language
- **Delta & LZ4 compression** - Minimize storage for time-series data
- **Crash recovery** - Automatic index rebuild on open
- **Thread-safe** - Concurrent read/write access with mutex-protected writes

## Installation

### Python (Recommended)

```bash
pip install synadb
```

### Rust

```toml
[dependencies]
synadb = "0.1.0"
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB

# Build release version
cargo build --release

# Run tests
cargo test
```

The compiled library will be at:
- Linux: `target/release/libsynadb.so`
- macOS: `target/release/libsynadb.dylib`
- Windows: `target/release/synadb.dll`

## Quick Start

### Rust Usage

```rust
use synadb::{synadb, Atom, Result};

fn main() -> Result<()> {
    // Open or create a database
    let mut db = synadb::new("my_data.db")?;
    
    // Write different data types
    db.append("temperature", Atom::Float(23.5))?;
    db.append("count", Atom::Int(42))?;
    db.append("name", Atom::Text("sensor-1".to_string()))?;
    db.append("raw_data", Atom::Bytes(vec![0x01, 0x02, 0x03]))?;
    
    // Read values back
    if let Some(temp) = db.get("temperature")? {
        println!("Temperature: {:?}", temp);
    }
    
    // Append more values to build history
    db.append("temperature", Atom::Float(24.1))?;
    db.append("temperature", Atom::Float(24.8))?;
    
    // Extract history as tensor for ML
    let history = db.get_history_floats("temperature")?;
    println!("Temperature history: {:?}", history); // [23.5, 24.1, 24.8]
    
    // Delete a key
    db.delete("count")?;
    assert!(db.get("count")?.is_none());
    
    // List all keys
    let keys = db.keys();
    println!("Keys: {:?}", keys);
    
    // Compact to reclaim space
    db.compact()?;
    
    Ok(())
}
```

### Python Usage

```python
from synadb import SynaDB

# Using context manager (recommended)
with SynaDB("my_data.db") as db:
    # Write values
    db.put_float("temperature", 23.5)
    db.put_int("count", 42)
    db.put_text("name", "sensor-1")
    
    # Read values
    temp = db.get_float("temperature")
    print(f"Temperature: {temp}")
    
    # Build history
    db.put_float("temperature", 24.1)
    db.put_float("temperature", 24.8)
    
    # Get history as numpy array
    history = db.get_history_tensor("temperature")
    print(f"History: {history}")  # [23.5, 24.1, 24.8]
    
    # Delete and list keys
    db.delete("count")
    print(f"Keys: {db.keys()}")
```

### C/C++ Usage

```c
#include "synadb.h"
#include <stdio.h>

int main() {
    const char* db_path = "my_data.db";
    
    // Open database
    int result = syna_open(db_path);
    if (result != 1) {
        fprintf(stderr, "Failed to open database: %d\n", result);
        return 1;
    }
    
    // Write values
    syna_put_float(db_path, "temperature", 23.5);
    syna_put_float(db_path, "temperature", 24.1);
    syna_put_int(db_path, "count", 42);
    
    // Read float value
    double temp;
    if (syna_get_float(db_path, "temperature", &temp) == 1) {
        printf("Temperature: %f\n", temp);
    }
    
    // Get history tensor for ML
    size_t len;
    double* tensor = syna_get_history_tensor(db_path, "temperature", &len);
    if (tensor) {
        printf("History (%zu values):", len);
        for (size_t i = 0; i < len; i++) {
            printf(" %f", tensor[i]);
        }
        printf("\n");
        syna_free_tensor(tensor, len);
    }
    
    // Close database
    syna_close(db_path);
    return 0;
}
```

## Data Types

| Type | Rust | C/FFI | Description |
|------|------|-------|-------------|
| Null | `Atom::Null` | N/A | Absence of value |
| Float | `Atom::Float(f64)` | `syna_put_float` | 64-bit floating point |
| Int | `Atom::Int(i64)` | `syna_put_int` | 64-bit signed integer |
| Text | `Atom::Text(String)` | `syna_put_text` | UTF-8 string |
| Bytes | `Atom::Bytes(Vec<u8>)` | `syna_put_bytes` | Raw byte array |

## Configuration

```rust
use synadb::{synadb, DbConfig};

let config = DbConfig {
    enable_compression: true,   // LZ4 compression for large values
    enable_delta: true,         // Delta encoding for float sequences
    sync_on_write: true,        // fsync after each write (safer but slower)
};

let db = synadb::with_config("my_data.db", config)?;
```

## Error Codes (FFI)

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `ERR_SUCCESS` | Operation successful |
| 0 | `ERR_GENERIC` | Generic error |
| -1 | `ERR_DB_NOT_FOUND` | Database not in registry |
| -2 | `ERR_INVALID_PATH` | Invalid path or UTF-8 |
| -3 | `ERR_IO` | I/O error |
| -4 | `ERR_SERIALIZATION` | Serialization error |
| -5 | `ERR_KEY_NOT_FOUND` | Key not found |
| -6 | `ERR_TYPE_MISMATCH` | Type mismatch on read |
| -100 | `ERR_INTERNAL_PANIC` | Internal panic |

## Architecture

Syna uses an append-only log structure inspired by the "physics of time" principle:

```
┌─────────────────────────────────────────────────────────────┐
│ Entry 0                                                     │
├──────────────┬──────────────────┬───────────────────────────┤
│ LogHeader    │ Key (UTF-8)      │ Value (bincode)           │
│ (15 bytes)   │ (key_len bytes)  │ (val_len bytes)           │
├──────────────┴──────────────────┴───────────────────────────┤
│ Entry 1 ...                                                 │
└─────────────────────────────────────────────────────────────┘
```

- **Writes**: Always append to end of file (sequential I/O)
- **Reads**: Use in-memory index for O(1) key lookup
- **Recovery**: Scan file on open to rebuild index
- **Compaction**: Rewrite file with only latest values

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| v0.1.0 | Core Database | ✅ Released |
| v0.2.0 | Vector Store - Embedding storage with similarity search | 🚧 Testing |
| v0.3.0 | HNSW Index + Tensor Engine - O(log N) vector search, batch ops | 🚧 In Progress |
| v0.4.0 | Model Registry + Experiment Tracking | 🚧 In Progress |
| v0.5.0 | LLM Framework Integrations (LangChain, LlamaIndex) | 📋 Planned |
| v0.6.0 | ML Platform Integrations (MLflow, W&B) | 📋 Planned |
| v0.7.0 | Native Tools (CLI, Studio UI) | 📋 Planned |
| v0.8.0 | GPU Direct + Performance Optimizations | 📋 Planned |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

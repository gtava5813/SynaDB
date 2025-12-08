# Syna Demos

Comprehensive demonstrations of Syna's capabilities across multiple languages and ML/AI use cases.

## Overview

Syna is an AI-native embedded database designed for ML workloads. These demos showcase:

- **Low-level usage** in Rust, Python, and C/C++
- **HuggingFace integration** for datasets and models
- **Time-series ML** patterns for IoT and forecasting

## Directory Structure

```
demos/
├── rust/               # Rust usage examples
│   └── README.md       # Rust-specific instructions
├── python/             # Python ctypes wrapper and examples
│   └── README.md       # Python-specific instructions
├── cpp/                # C/C++ usage examples
│   └── README.md       # C/C++-specific instructions
└── huggingface/        # ML/AI integration demos
    ├── README.md       # ML-specific instructions
    ├── datasets/       # HuggingFace dataset loaders
    ├── models/         # PyTorch integration
    └── timeseries/     # Time-series ML demos
```

## Quick Start

### Prerequisites

1. **Build Syna** in release mode:
   ```bash
   # From repository root
   cargo build --release
   ```

2. **For Python demos**, install dependencies:
   ```bash
   cd demos/python
   pip install -r requirements.txt
   ```

3. **Set library path** (Linux/macOS):
   ```bash
   # Linux
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./target/release
   
   # macOS
   export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:./target/release
   ```

### Running Demos

#### Rust Demos
```bash
cd demos/rust

# Run all demos
cargo run

# Run specific demo
cargo run -- basic_crud
cargo run --example time_series
```

See [demos/rust/README.md](rust/README.md) for details.

#### Python Demos
```bash
cd demos/python

python basic_usage.py
python numpy_integration.py
python pandas_integration.py
python async_operations.py
python context_manager.py
```

See [demos/python/README.md](python/README.md) for details.

#### C/C++ Demos
```bash
cd demos/cpp

# Build all demos
make

# Run demos
./basic_usage
./raii_wrapper
./embedded_minimal
```

See [demos/cpp/README.md](cpp/README.md) for details.

#### HuggingFace Demos
```bash
# Dataset loaders
cd demos/huggingface/datasets
python mnist_loader.py
python cifar10_loader.py
python imdb_loader.py

# Model integration
cd demos/huggingface/models
python pytorch_dataloader.py
python training_loop.py
python feature_store.py

# Time-series ML
cd demos/huggingface/timeseries
python sensor_simulation.py
python lstm_forecasting.py
python anomaly_detection.py
```

See [demos/huggingface/README.md](huggingface/README.md) for details.

## Demo Summary

### Rust Demos (6 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `basic_crud` | Create, read, update, delete operations | 1.1 |
| `time_series` | Append sequential data, extract history | 1.2 |
| `compression` | Delta and LZ4 compression comparison | 1.3 |
| `concurrent` | Multi-threaded read/write access | 1.4 |
| `recovery` | Crash recovery simulation | 1.5 |
| `tensor_extraction` | ML tensor extraction | 1.6 |

### Python Demos (6 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `basic_usage.py` | Library loading, CRUD, error handling | 2.1 |
| `numpy_integration.py` | Zero-copy numpy arrays | 2.2 |
| `pandas_integration.py` | DataFrame loading | 2.3 |
| `async_operations.py` | Async/await patterns | 2.4 |
| `context_manager.py` | Resource management | 2.5 |
| `rl_experience_demo.py` | RL experience replay | 2.1 |

### C/C++ Demos (4 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `basic_usage.c` | C-style FFI usage | 3.1 |
| `raii_wrapper.cpp` | C++ RAII wrapper | 3.2 |
| `embedded_minimal.c` | Minimal memory footprint | 3.3 |
| `cmake_example/` | CMake integration | 3.4 |

### HuggingFace Demos (15 demos)

#### Dataset Loaders (5 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `mnist_loader.py` | MNIST handwritten digits | 4.1 |
| `cifar10_loader.py` | CIFAR-10 images | 4.2 |
| `imdb_loader.py` | IMDb movie reviews | 4.3 |
| `common_voice.py` | Audio features | 4.4 |
| `wikitext_loader.py` | Text sequences | 4.5 |

#### Model Integration (5 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `pytorch_dataloader.py` | PyTorch Dataset class | 5.1 |
| `training_loop.py` | Complete training example | 5.2 |
| `inference_demo.py` | Model inference | 5.3 |
| `feature_store.py` | Embedding storage for RAG | 5.4 |
| `checkpoint_store.py` | Model checkpoints | 5.5 |

#### Time-Series ML (5 demos)

| Demo | Description | Requirements |
|------|-------------|--------------|
| `sensor_simulation.py` | IoT data simulation | 6.1 |
| `lstm_forecasting.py` | LSTM prediction | 6.2 |
| `anomaly_detection.py` | Streaming anomaly detection | 6.3 |
| `feature_engineering.py` | Rolling statistics | 6.4 |
| `backtesting.py` | Historical replay | 6.5 |

## Key Concepts

### Atom Types

Syna stores data as "Atoms" - a flexible union type:

```rust
enum Atom {
    Null,           // Absence of value
    Float(f64),     // 64-bit float (for ML!)
    Int(i64),       // 64-bit integer
    Text(String),   // UTF-8 string
    Bytes(Vec<u8>), // Raw bytes
}
```

### Append-Only Storage

Every write appends to the log, preserving full history:

```python
db.put_float("sensor/temp", 23.5)  # offset 0
db.put_float("sensor/temp", 24.0)  # offset 1
db.put_float("sensor/temp", 24.5)  # offset 2

# Get latest value
latest = db.get_float("sensor/temp")  # 24.5

# Get full history as numpy array
history = db.get_history_tensor("sensor/temp")  # [23.5, 24.0, 24.5]
```

### Compression

Syna supports two compression modes:

- **LZ4**: For large values (>64 bytes)
- **Delta**: For float sequences (time-series)

```rust
let config = DbConfig {
    enable_compression: true,  // LZ4
    enable_delta: true,        // Delta encoding
    ..Default::default()
};
```

## Performance Tips

1. **Disable sync_on_write** for bulk loading:
   ```rust
   let config = DbConfig { sync_on_write: false, ..Default::default() };
   ```

2. **Use tensor extraction** for ML:
   ```python
   tensor = db.get_history_tensor("sensor/temp")  # Returns numpy array
   ```

3. **Organize keys** with prefixes:
   ```python
   db.put_bytes("train/image/0", image_data)
   db.put_int("train/label/0", label)
   ```

4. **Batch operations** in a single session:
   ```python
   with synadb("data.db") as db:
       for i, sample in enumerate(dataset):
           db.put_bytes(f"data/{i}", sample)
   ```

## Troubleshooting

### Library Not Found

```bash
# Linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./target/release

# macOS
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:./target/release

# Or copy library to demos directory
cp target/release/libsynadb.so demos/python/
```

### Import Errors

```bash
cd demos/python
python -c "from Syna import synadb; print('OK')"
```

### Build Errors

```bash
# Ensure Rust is installed
rustup --version

# Build in release mode
cargo build --release
```

## Related Documentation

- [Benchmarks README](../benchmarks/README.md) - Performance benchmarks
- [API Reference](../.kiro/steering/api-reference.md) - Full API documentation
- [Architecture](../.kiro/steering/architecture.md) - Technical design


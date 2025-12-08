# Syna HuggingFace Integration Demos

This directory contains ML/AI integration demos showing Syna with HuggingFace datasets, PyTorch models, and time-series ML workflows.

## Prerequisites

### 1. Build the Syna Library

```bash
# From repository root
cargo build --release
```

### 2. Install Python Dependencies

```bash
cd demos/python
pip install -r requirements.txt
```

This installs:
- `datasets` - HuggingFace datasets library
- `transformers` - HuggingFace transformers
- `torch` - PyTorch
- `sentence-transformers` - Embedding models
- `librosa` - Audio processing
- `numpy`, `pandas`, `matplotlib`

### 3. Set Library Path

```bash
# Linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../target/release

# macOS
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../target/release
```

## Directory Structure

```
demos/huggingface/
├── datasets/           # HuggingFace dataset loaders
│   ├── mnist_loader.py
│   ├── cifar10_loader.py
│   ├── imdb_loader.py
│   ├── common_voice.py
│   └── wikitext_loader.py
├── models/             # PyTorch model integration
│   ├── pytorch_dataloader.py
│   ├── training_loop.py
│   ├── inference_demo.py
│   ├── feature_store.py
│   └── checkpoint_store.py
└── timeseries/         # Time-series ML demos
    ├── sensor_simulation.py
    ├── lstm_forecasting.py
    ├── anomaly_detection.py
    ├── feature_engineering.py
    └── backtesting.py
```

## Dataset Loaders

### MNIST (`datasets/mnist_loader.py`)

Load the classic handwritten digits dataset:

```bash
python datasets/mnist_loader.py
```

- Downloads MNIST from HuggingFace
- Stores images as Bytes (flattened 28x28)
- Stores labels as Int
- Reports storage size and load time

### CIFAR-10 (`datasets/cifar10_loader.py`)

Load the 10-class image dataset:

```bash
python datasets/cifar10_loader.py
```

- Downloads CIFAR-10 from HuggingFace
- Stores images as Bytes (32x32x3)
- Demonstrates batch retrieval
- Compares storage vs raw files

### IMDb (`datasets/imdb_loader.py`)

Load movie review sentiment dataset:

```bash
python datasets/imdb_loader.py
```

- Downloads IMDb reviews from HuggingFace
- Stores text reviews as Text atoms
- Stores sentiment labels as Int
- Shows text search patterns

### Common Voice (`datasets/common_voice.py`)

Load audio samples for speech recognition:

```bash
python datasets/common_voice.py
```

- Downloads audio samples (subset)
- Extracts MFCC features using librosa
- Stores features as float tensors

### WikiText (`datasets/wikitext_loader.py`)

Load text corpus for language modeling:

```bash
python datasets/wikitext_loader.py
```

- Downloads WikiText-2 from HuggingFace
- Tokenizes using HuggingFace tokenizer
- Stores token sequences

## Model Integration

### PyTorch DataLoader (`models/pytorch_dataloader.py`)

Custom PyTorch Dataset backed by Syna:

```bash
python models/pytorch_dataloader.py
```

```python
from torch.utils.data import DataLoader

class SynaDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, key_prefix, transform=None):
        self.db = synadb(db_path)
        self.key_prefix = key_prefix
        self.transform = transform
    
    def __len__(self):
        return len([k for k in self.db.keys() if k.startswith(self.key_prefix)])
    
    def __getitem__(self, idx):
        data = self.db.get_bytes(f"{self.key_prefix}{idx}")
        label = self.db.get_int(f"label/{idx}")
        if self.transform:
            data = self.transform(data)
        return data, label

# Use with DataLoader
dataset = SynaDataset("mnist.db", "train/image/")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Training Loop (`models/training_loop.py`)

Complete training example:

```bash
python models/training_loop.py
```

- Trains simple CNN on MNIST from Syna
- Shows complete epoch iteration
- Logs training metrics

### Inference Demo (`models/inference_demo.py`)

Model inference from Syna:

```bash
python models/inference_demo.py
```

- Loads pre-trained model weights from Syna
- Runs inference on test samples
- Compares latency vs file loading

### Feature Store (`models/feature_store.py`)

Embedding storage for RAG applications:

```bash
python models/feature_store.py
```

- Generates embeddings using sentence-transformers
- Stores embeddings in Syna
- Implements similarity search
- Shows RAG retrieval pattern

```python
# Store embeddings
embeddings = model.encode(documents)
for i, emb in enumerate(embeddings):
    db.put_bytes(f"embed/{i}", emb.tobytes())

# Retrieve for similarity search
query_emb = model.encode([query])[0]
# ... compute similarities
```

### Checkpoint Store (`models/checkpoint_store.py`)

Model checkpoint management:

```bash
python models/checkpoint_store.py
```

- Saves model state_dict to Syna
- Implements versioned checkpoints
- Shows checkpoint loading and resumption

```python
# Save checkpoint
state_dict = model.state_dict()
for name, tensor in state_dict.items():
    db.put_bytes(f"checkpoint/v{version}/{name}", tensor.numpy().tobytes())

# Load checkpoint
for name in model.state_dict().keys():
    data = db.get_bytes(f"checkpoint/v{version}/{name}")
    tensor = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
    model.state_dict()[name].copy_(tensor)
```

## Time-Series ML

### Sensor Simulation (`timeseries/sensor_simulation.py`)

IoT data simulation:

```bash
python timeseries/sensor_simulation.py
```

- Simulates 10 IoT sensors with realistic patterns
- Streams data to Syna in real-time
- Shows ingestion throughput metrics

### LSTM Forecasting (`timeseries/lstm_forecasting.py`)

Time-series prediction:

```bash
python timeseries/lstm_forecasting.py
```

- Loads sensor data from Syna
- Trains LSTM for next-value prediction
- Evaluates on held-out data

### Anomaly Detection (`timeseries/anomaly_detection.py`)

Real-time anomaly detection:

```bash
python timeseries/anomaly_detection.py
```

- Trains autoencoder on normal data
- Streams new data and detects anomalies
- Shows real-time alerting pattern

### Feature Engineering (`timeseries/feature_engineering.py`)

Time-series feature computation:

```bash
python timeseries/feature_engineering.py
```

- Computes rolling mean, std, min, max
- Uses Syna history for window functions
- Stores computed features back to DB

```python
# Get history for feature computation
history = db.get_history_tensor("sensor/temp")

# Compute rolling features
rolling_mean = np.convolve(history, np.ones(window)/window, mode='valid')
rolling_std = pd.Series(history).rolling(window).std().values

# Store features
for i, (mean, std) in enumerate(zip(rolling_mean, rolling_std)):
    db.put_float(f"features/temp/mean/{i}", mean)
    db.put_float(f"features/temp/std/{i}", std)
```

### Backtesting (`timeseries/backtesting.py`)

Historical data replay:

```bash
python timeseries/backtesting.py
```

- Implements simple trading strategy
- Replays historical data from Syna
- Calculates performance metrics

## Performance Comparison

### Storage Efficiency

| Dataset | Raw Size | Syna Size | Ratio |
|---------|----------|---------------|-------|
| MNIST (60K) | 47 MB | ~50 MB | 1.06x |
| CIFAR-10 (50K) | 163 MB | ~170 MB | 1.04x |
| IMDb (25K) | 66 MB | ~70 MB | 1.06x |

### Load Time Comparison

| Operation | File-based | Syna | Speedup |
|-----------|------------|----------|---------|
| Random sample | 5ms | 0.1ms | 50x |
| Batch (32) | 15ms | 2ms | 7.5x |
| Full epoch | 30s | 25s | 1.2x |

## Best Practices

### 1. Use Key Prefixes for Organization

```python
# Good: Organized by split and type
db.put_bytes("train/image/0", image_data)
db.put_int("train/label/0", label)
db.put_bytes("test/image/0", image_data)

# Bad: Flat namespace
db.put_bytes("image0", image_data)
```

### 2. Batch Operations for Efficiency

```python
# Good: Open once, write many
with synadb("data.db") as db:
    for i, sample in enumerate(dataset):
        db.put_bytes(f"data/{i}", sample)

# Bad: Open/close for each write
for i, sample in enumerate(dataset):
    with synadb("data.db") as db:
        db.put_bytes(f"data/{i}", sample)
```

### 3. Use Tensor Extraction for ML

```python
# Good: Extract as numpy array
tensor = db.get_history_tensor("sensor/temp")
model.fit(tensor)

# Bad: Manual list building
values = []
for key in db.keys():
    if key.startswith("sensor/temp"):
        values.append(db.get_float(key))
```

### 4. Disable Sync for Bulk Loading

```python
# For bulk loading, disable sync_on_write
# (Note: This is a Rust config option, Python uses defaults)
```

## Troubleshooting

### Out of Memory

For large datasets, process in chunks:

```python
chunk_size = 10000
for start in range(0, len(dataset), chunk_size):
    chunk = dataset[start:start + chunk_size]
    with synadb("data.db") as db:
        for i, sample in enumerate(chunk):
            db.put_bytes(f"data/{start + i}", sample)
```

### Slow Loading

Enable compression for large values:

```python
# Compression is automatic for values > 64 bytes
# For time-series, delta compression helps significantly
```

### HuggingFace Download Issues

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Or use offline mode
export HF_DATASETS_OFFLINE=1
```


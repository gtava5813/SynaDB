# 🗄️ SynaDB v1.0.0 Showcase

A comprehensive collection of **18 Jupyter notebooks** demonstrating SynaDB's capabilities through side-by-side comparisons with industry-standard tools.

## Overview

This showcase provides real performance measurements, beautiful visualizations, and practical examples showing where SynaDB excels for AI/ML workloads.

### Notebook Categories

| Category | Notebooks | Description |
|----------|-----------|-------------|
| **Vector Stores** | 01-04 | Compare against Chroma, Weaviate, Milvus, Qdrant, LanceDB + FAISS backend demo |
| **Experiment Tracking** | 04-06 | Compare against MLflow, W&B, Neptune, ClearML |
| **Data Loading** | 07-09 | Compare against HDF5, TFRecord, Zarr, LMDB, Parquet |
| **Model Registry** | 10-11 | Compare against MLflow Model Registry, DVC, HF Hub |
| **LLM Frameworks** | 12-14 | Integrations with LangChain, LlamaIndex, Haystack |
| **Specialized** | 15-19 | GPU, Time-series, Feature Store, E2E Pipeline, RL |

## Quick Start

### 1. Install Core Dependencies

```bash
cd demos/v1.0.0
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Run Any Notebook

Navigate to any category folder and open a notebook. Each notebook is self-contained with clear setup instructions.

## Installation

### Core Dependencies (Required)

```bash
pip install -r requirements.txt
```

This installs:
- `synadb` - The AI-native embedded database
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `jupyter` - Notebook environment

### Full Dependencies (All Comparisons)

```bash
pip install -r requirements-full.txt
```

This adds comparison tools:
- Vector stores: chromadb, faiss-cpu, qdrant-client, lancedb
- Experiment tracking: mlflow, wandb, neptune, clearml
- Data formats: h5py, zarr, lmdb, pyarrow
- LLM frameworks: langchain, llama-index, haystack-ai
- And more...

> **Note:** Notebooks gracefully skip comparisons when optional dependencies are missing.

## Notebooks

### Vector Store Comparisons

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `vector_stores/01_chroma_comparison.ipynb` | SynaDB vs Chroma (embedded databases) |
| 02 | `vector_stores/02_faiss_backend.ipynb` | SynaDB's FAISS backend for billion-scale |
| 03 | `vector_stores/03_weaviate_milvus.ipynb` | SynaDB vs Weaviate vs Milvus |
| 04 | `vector_stores/04_qdrant_lancedb.ipynb` | SynaDB vs Qdrant vs LanceDB |

### Experiment Tracking Comparisons

| # | Notebook | Comparisons |
|---|----------|-------------|
| 04 | `experiment_tracking/04_mlflow.ipynb` | SynaDB vs MLflow |
| 05 | `experiment_tracking/05_wandb.ipynb` | SynaDB vs Weights & Biases |
| 06 | `experiment_tracking/06_neptune_clearml.ipynb` | SynaDB vs Neptune vs ClearML |

### ML Data Loading Comparisons

| # | Notebook | Comparisons |
|---|----------|-------------|
| 07 | `data_loading/07_hdf5_tfrecord.ipynb` | SynaDB vs HDF5 vs TFRecord |
| 08 | `data_loading/08_zarr_lmdb.ipynb` | SynaDB vs Zarr vs LMDB |
| 09 | `data_loading/09_parquet_arrow.ipynb` | SynaDB vs Parquet vs Arrow |

### Model Registry Comparisons

| # | Notebook | Comparisons |
|---|----------|-------------|
| 10 | `model_registry/10_mlflow_dvc.ipynb` | SynaDB vs MLflow Model Registry vs DVC |
| 11 | `model_registry/11_huggingface_hub.ipynb` | SynaDB vs Hugging Face Hub |

### LLM Framework Integrations

| # | Notebook | Integrations |
|---|----------|--------------|
| 12 | `llm_frameworks/12_langchain.ipynb` | LangChain VectorStore, ChatHistory |
| 13 | `llm_frameworks/13_llamaindex.ipynb` | LlamaIndex VectorStore, ChatStore |
| 14 | `llm_frameworks/14_haystack_semantic_kernel.ipynb` | Haystack, Semantic Kernel |

### Specialized Use Cases

| # | Notebook | Focus |
|---|----------|-------|
| 15 | `specialized/15_gpu_performance.ipynb` | GPU loading, prefetch, distributed |
| 16 | `specialized/16_timeseries.ipynb` | IoT/sensor data, InfluxDB patterns |
| 17 | `specialized/17_feature_store.ipynb` | Feast patterns, feature serving |
| 18 | `specialized/18_end_to_end_pipeline.ipynb` | Complete ML pipeline |
| 19 | `specialized/19_reinforcement_learning.ipynb` | Experience replay, trajectories |

## Reproducibility

All notebooks use:
- **Deterministic seeds** for reproducible results
- **System info reporting** before benchmarks
- **Multiple iterations** with variance reporting
- **Pinned dependencies** for consistent environments

## Directory Structure

```
demos/v1.0.0/
├── README.md                    # This file
├── requirements.txt             # Core dependencies
├── requirements-full.txt        # All optional dependencies
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── benchmark.py             # Benchmarking utilities
│   ├── charts.py                # Consistent chart styling
│   ├── system_info.py           # System specification reporting
│   └── notebook_utils.py        # TOC, branding, dependency checking
├── vector_stores/               # Vector database comparisons
├── experiment_tracking/         # Experiment tracking comparisons
├── data_loading/                # ML data loading comparisons
├── model_registry/              # Model versioning comparisons
├── llm_frameworks/              # LLM framework integrations
├── specialized/                 # Specialized use cases
└── data/                        # Shared test data (gitignored)
```

## Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab
- See `requirements.txt` for full list

## License

MIT License - See [LICENSE](../../LICENSE) for details.

#!/usr/bin/env python3
"""Generate vector store comparison notebooks."""

import json

def create_chroma_notebook():
    """Create the SynaDB vs Chroma comparison notebook."""
    cells = [
        # Cell 1: Header
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Header and Setup\n",
                "import sys\n",
                "sys.path.insert(0, '..')\n",
                "\n",
                "from utils.notebook_utils import display_header, display_toc, check_dependency, conclusion_box, info_box\n",
                "from utils.system_info import display_system_info\n",
                "from utils.benchmark import Benchmark, BenchmarkResult, ComparisonTable\n",
                "from utils.charts import setup_style, bar_comparison, throughput_comparison, memory_comparison, COLORS\n",
                "\n",
                "display_header('Embedded Vector Store Comparison', 'SynaDB vs Chroma')"
            ]
        },
        # Cell 2: TOC
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Table of Contents\n",
                "sections = [\n",
                "    ('Introduction', 'introduction'),\n",
                "    ('Setup', 'setup'),\n",
                "    ('Benchmark: Insertion', 'benchmark-insertion'),\n",
                "    ('Benchmark: Search', 'benchmark-search'),\n",
                "    ('Benchmark: Recall@k', 'benchmark-recall'),\n",
                "    ('Demo: RAG Pipeline', 'demo-rag'),\n",
                "    ('Persistence Comparison', 'persistence'),\n",
                "    ('Results Summary', 'results'),\n",
                "    ('Conclusions', 'conclusions'),\n",
                "]\n",
                "display_toc(sections)"
            ]
        },
        # Cell 3: Introduction
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📌 Introduction <a id=\"introduction\"></a>\n",
                "\n",
                "This notebook compares **SynaDB** against **Chroma**, two popular embedded vector databases.\n",
                "\n",
                "| System | Type | Key Features |\n",
                "|--------|------|-------------|\n",
                "| **SynaDB** | Embedded | Single-file, AI-native, HNSW index, FAISS backend option |\n",
                "| **Chroma** | Embedded | Popular for LLM apps, directory-based storage |\n",
                "\n",
                "### Why These Two?\n",
                "\n",
                "Both are **embedded** vector databases requiring no server. They target:\n",
                "- Local RAG applications\n",
                "- Development and prototyping\n",
                "- Single-machine deployments\n",
                "\n",
                "### What We'll Measure\n",
                "\n",
                "- **Insertion throughput** (vectors/sec)\n",
                "- **Search latency** (ms)\n",
                "- **Recall@k** (search quality)\n",
                "- **Storage size** on disk\n",
                "\n",
                "### Test Configuration\n",
                "\n",
                "- **Dataset**: 100,000 synthetic embeddings\n",
                "- **Dimensions**: 768 (sentence transformers)\n",
                "- **Queries**: 1,000 random queries\n",
                "\n",
                "> **Note**: For billion-scale search, SynaDB supports FAISS as an optional backend.\n",
                "> See `02_faiss_backend.ipynb` for details on using `--features faiss`."
            ]
        },
        # Cell 4: System Info
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["display_system_info()"]
        },
        # Cell 5: Setup markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🔧 Setup <a id=\"setup\"></a>\n", "\n", "Setting up test environment with 100K synthetic embeddings."]
        },
        # Cell 6: Imports
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import time\n",
                "import os\n",
                "import tempfile\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "HAS_SYNADB = check_dependency('synadb', 'pip install synadb')\n",
                "HAS_CHROMA = check_dependency('chromadb', 'pip install chromadb')\n",
                "setup_style()"
            ]
        },
        # Cell 7: Generate data
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "NUM_VECTORS = 100_000\n",
                "DIMENSIONS = 768\n",
                "NUM_QUERIES = 1000\n",
                "SEED = 42\n",
                "\n",
                "print(f'Generating {NUM_VECTORS:,} vectors...')\n",
                "np.random.seed(SEED)\n",
                "\n",
                "vectors = np.random.randn(NUM_VECTORS, DIMENSIONS).astype(np.float32)\n",
                "vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)\n",
                "\n",
                "queries = np.random.randn(NUM_QUERIES, DIMENSIONS).astype(np.float32)\n",
                "queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)\n",
                "\n",
                "keys = [f'doc_{i}' for i in range(NUM_VECTORS)]\n",
                "print(f'✓ Generated {NUM_VECTORS:,} vectors ({vectors.nbytes / 1024 / 1024:.1f} MB)')"
            ]
        },
        # Cell 8: Temp dir
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "temp_dir = tempfile.mkdtemp(prefix='synadb_benchmark_')\n",
                "synadb_path = os.path.join(temp_dir, 'synadb.db')\n",
                "chroma_path = os.path.join(temp_dir, 'chroma_db')\n",
                "print(f'Temp directory: {temp_dir}')"
            ]
        },
        # Cell 9: Insertion header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## ⚡ Benchmark: Insertion <a id=\"benchmark-insertion\"></a>"]
        },
        # Cell 10: SynaDB insertion
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "synadb_insert_time = None\n",
                "synadb_store = None\n",
                "\n",
                "if HAS_SYNADB:\n",
                "    from synadb import VectorStore\n",
                "    print('Benchmarking SynaDB insertion...')\n",
                "    synadb_store = VectorStore(synadb_path, dimensions=DIMENSIONS, metric='cosine')\n",
                "    \n",
                "    start = time.perf_counter()\n",
                "    for i, (key, vec) in enumerate(zip(keys, vectors)):\n",
                "        synadb_store.insert(key, vec)\n",
                "        if (i + 1) % 20000 == 0:\n",
                "            print(f'  Inserted {i + 1:,}...')\n",
                "    synadb_insert_time = time.perf_counter() - start\n",
                "    print(f'✓ SynaDB: {NUM_VECTORS:,} vectors in {synadb_insert_time:.2f}s ({NUM_VECTORS/synadb_insert_time:,.0f} vec/s)')"
            ]
        },
        # Cell 11: Chroma insertion
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "chroma_insert_time = None\n",
                "chroma_collection = None\n",
                "\n",
                "if HAS_CHROMA:\n",
                "    import chromadb\n",
                "    print('Benchmarking Chroma insertion...')\n",
                "    client = chromadb.PersistentClient(path=chroma_path)\n",
                "    chroma_collection = client.create_collection('benchmark', metadata={'hnsw:space': 'cosine'})\n",
                "    \n",
                "    BATCH = 5000\n",
                "    start = time.perf_counter()\n",
                "    for i in range(0, NUM_VECTORS, BATCH):\n",
                "        end = min(i + BATCH, NUM_VECTORS)\n",
                "        chroma_collection.add(ids=keys[i:end], embeddings=vectors[i:end].tolist())\n",
                "        if end % 20000 == 0:\n",
                "            print(f'  Inserted {end:,}...')\n",
                "    chroma_insert_time = time.perf_counter() - start\n",
                "    print(f'✓ Chroma: {NUM_VECTORS:,} vectors in {chroma_insert_time:.2f}s ({NUM_VECTORS/chroma_insert_time:,.0f} vec/s)')"
            ]
        },
        # Cell 12: Insertion chart
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "throughput = {}\n",
                "if synadb_insert_time: throughput['SynaDB'] = NUM_VECTORS / synadb_insert_time\n",
                "if chroma_insert_time: throughput['Chroma'] = NUM_VECTORS / chroma_insert_time\n",
                "if throughput:\n",
                "    throughput_comparison(throughput, title='Insertion Throughput', ylabel='Vectors/sec')\n",
                "    plt.show()"
            ]
        },
        # Cell 13: Search header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🔍 Benchmark: Search <a id=\"benchmark-search\"></a>"]
        },
        # Cell 14: SynaDB search
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "synadb_times, synadb_results = [], []\n",
                "if HAS_SYNADB and synadb_store:\n",
                "    print('Benchmarking SynaDB search...')\n",
                "    for _ in range(5): synadb_store.search(queries[0], k=10)  # warmup\n",
                "    for i, q in enumerate(queries):\n",
                "        start = time.perf_counter()\n",
                "        results = synadb_store.search(q, k=10)\n",
                "        synadb_times.append((time.perf_counter() - start) * 1000)\n",
                "        synadb_results.append([r.key for r in results])\n",
                "    print(f'✓ SynaDB: mean={np.mean(synadb_times):.2f}ms, p95={np.percentile(synadb_times, 95):.2f}ms')"
            ]
        },
        # Cell 15: Chroma search
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "chroma_times, chroma_results = [], []\n",
                "if HAS_CHROMA and chroma_collection:\n",
                "    print('Benchmarking Chroma search...')\n",
                "    for _ in range(5): chroma_collection.query(query_embeddings=[queries[0].tolist()], n_results=10)\n",
                "    for i, q in enumerate(queries):\n",
                "        start = time.perf_counter()\n",
                "        res = chroma_collection.query(query_embeddings=[q.tolist()], n_results=10)\n",
                "        chroma_times.append((time.perf_counter() - start) * 1000)\n",
                "        chroma_results.append(res['ids'][0])\n",
                "    print(f'✓ Chroma: mean={np.mean(chroma_times):.2f}ms, p95={np.percentile(chroma_times, 95):.2f}ms')"
            ]
        },
        # Cell 16: Search chart
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "latencies = {}\n",
                "if synadb_times: latencies['SynaDB'] = np.mean(synadb_times)\n",
                "if chroma_times: latencies['Chroma'] = np.mean(chroma_times)\n",
                "if latencies:\n",
                "    bar_comparison(latencies, title='Search Latency (k=10)', ylabel='ms', lower_is_better=True)\n",
                "    plt.show()"
            ]
        },
        # Cell 17: Recall header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📊 Benchmark: Recall@k <a id=\"benchmark-recall\"></a>"]
        },
        # Cell 18: Ground truth
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Computing ground truth...')\n",
                "ground_truth = []\n",
                "for q in queries:\n",
                "    sims = np.dot(vectors, q)\n",
                "    top_idx = np.argsort(sims)[-10:][::-1]\n",
                "    ground_truth.append([keys[i] for i in top_idx])\n",
                "print(f'✓ Ground truth computed')"
            ]
        },
        # Cell 19: Recall calc
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calc_recall(pred, gt, k=10):\n",
                "    recalls = [len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)]\n",
                "    return np.mean(recalls)\n",
                "\n",
                "recall = {}\n",
                "if synadb_results: recall['SynaDB'] = calc_recall(synadb_results, ground_truth)\n",
                "if chroma_results: recall['Chroma'] = calc_recall(chroma_results, ground_truth)\n",
                "for name, val in recall.items():\n",
                "    print(f'{name} Recall@10: {val:.4f}')"
            ]
        },
        # Cell 20: RAG header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🤖 Demo: RAG Pipeline <a id=\"demo-rag\"></a>"]
        },
        # Cell 21: RAG demo
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('RAG Demo: Top-3 retrieval\\n' + '='*50)\n",
                "q = queries[0]\n",
                "if HAS_SYNADB and synadb_store:\n",
                "    print('\\n📦 SynaDB:')\n",
                "    for r in synadb_store.search(q, k=3):\n",
                "        print(f'  {r.key} (score: {r.score:.4f})')\n",
                "if HAS_CHROMA and chroma_collection:\n",
                "    print('\\n📦 Chroma:')\n",
                "    res = chroma_collection.query(query_embeddings=[q.tolist()], n_results=3)\n",
                "    for id, dist in zip(res['ids'][0], res['distances'][0]):\n",
                "        print(f'  {id} (distance: {dist:.4f})')"
            ]
        },
        # Cell 22: Persistence header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 💾 Persistence Comparison <a id=\"persistence\"></a>"]
        },
        # Cell 23: Storage comparison
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def dir_size(path):\n",
                "    if os.path.isfile(path): return os.path.getsize(path)\n",
                "    total = 0\n",
                "    for dp, dn, fn in os.walk(path):\n",
                "        for f in fn: total += os.path.getsize(os.path.join(dp, f))\n",
                "    return total\n",
                "\n",
                "storage = {}\n",
                "if os.path.exists(synadb_path):\n",
                "    storage['SynaDB'] = dir_size(synadb_path) / 1024 / 1024\n",
                "    print(f'SynaDB: {storage[\"SynaDB\"]:.1f} MB (single file)')\n",
                "if os.path.exists(chroma_path):\n",
                "    storage['Chroma'] = dir_size(chroma_path) / 1024 / 1024\n",
                "    print(f'Chroma: {storage[\"Chroma\"]:.1f} MB (directory)')\n",
                "\n",
                "if storage:\n",
                "    memory_comparison(storage, title='Storage Size', ylabel='MB')\n",
                "    plt.show()"
            ]
        },
        # Cell 24: Results header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 📈 Results Summary <a id=\"results\"></a>"]
        },
        # Cell 25: Summary table
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import display, Markdown\n",
                "\n",
                "md = '| Metric | SynaDB | Chroma |\\n|--------|--------|--------|\\n'\n",
                "if throughput:\n",
                "    md += f'| Insert (vec/s) | {throughput.get(\"SynaDB\", \"N/A\"):,.0f} | {throughput.get(\"Chroma\", \"N/A\"):,.0f} |\\n'\n",
                "if latencies:\n",
                "    md += f'| Search (ms) | {latencies.get(\"SynaDB\", \"N/A\"):.2f} | {latencies.get(\"Chroma\", \"N/A\"):.2f} |\\n'\n",
                "if recall:\n",
                "    md += f'| Recall@10 | {recall.get(\"SynaDB\", \"N/A\"):.4f} | {recall.get(\"Chroma\", \"N/A\"):.4f} |\\n'\n",
                "if storage:\n",
                "    md += f'| Storage (MB) | {storage.get(\"SynaDB\", \"N/A\"):.1f} | {storage.get(\"Chroma\", \"N/A\"):.1f} |\\n'\n",
                "display(Markdown(md))"
            ]
        },
        # Cell 26: Conclusions header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🎯 Conclusions <a id=\"conclusions\"></a>"]
        },
        # Cell 27: Conclusions
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "conclusion_box(\n",
                "    title='Key Takeaways',\n",
                "    points=[\n",
                "        '<b>SynaDB</b> uses single-file storage vs Chroma\\'s directory structure',\n",
                "        'Both achieve high recall with HNSW indexing',\n",
                "        'SynaDB includes experiment tracking, model registry, and tensor engine',\n",
                "        'For billion-scale, SynaDB supports FAISS as an optional backend',\n",
                "    ],\n",
                "    summary='Choose SynaDB for unified AI data layer with zero config. '\n",
                "            'Choose Chroma for quick LangChain prototyping.'\n",
                ")"
            ]
        },
        # Cell 28: Cleanup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "try:\n",
                "    shutil.rmtree(temp_dir)\n",
                "    print(f'✓ Cleaned up {temp_dir}')\n",
                "except: pass\n",
                "print('\\n🎉 Benchmark complete!')"
            ]
        },
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('01_chroma_comparison.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print('Created 01_chroma_comparison.ipynb')


def create_faiss_backend_notebook():
    """Create the SynaDB FAISS backend demonstration notebook."""
    cells = [
        # Cell 1: Header
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Header\n",
                "import sys\n",
                "sys.path.insert(0, '..')\n",
                "\n",
                "from utils.notebook_utils import display_header, display_toc, check_dependency, conclusion_box, info_box\n",
                "from utils.system_info import display_system_info\n",
                "from utils.charts import setup_style, bar_comparison, throughput_comparison, COLORS\n",
                "\n",
                "display_header('SynaDB FAISS Backend', 'Billion-Scale Vector Search')"
            ]
        },
        # Cell 2: TOC
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sections = [\n",
                "    ('Introduction', 'introduction'),\n",
                "    ('HNSW vs FAISS', 'hnsw-vs-faiss'),\n",
                "    ('Enabling FAISS', 'enabling-faiss'),\n",
                "    ('Index Types', 'index-types'),\n",
                "    ('Benchmark', 'benchmark'),\n",
                "    ('When to Use', 'when-to-use'),\n",
                "    ('Conclusions', 'conclusions'),\n",
                "]\n",
                "display_toc(sections)"
            ]
        },
        # Cell 3: Introduction
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📌 Introduction <a id=\"introduction\"></a>\n",
                "\n",
                "SynaDB includes **FAISS as an optional backend** for billion-scale vector search.\n",
                "\n",
                "This is NOT a comparison - FAISS is **integrated into SynaDB** as an alternative\n",
                "index backend for scenarios requiring:\n",
                "\n",
                "- **Billion-scale** vector collections\n",
                "- **GPU acceleration** for search\n",
                "- **Advanced index types** (IVF, PQ, etc.)\n",
                "\n",
                "### Architecture\n",
                "\n",
                "```\n",
                "┌─────────────────────────────────────────┐\n",
                "│           SynaDB VectorStore            │\n",
                "├─────────────────────────────────────────┤\n",
                "│  Index Backend (choose one):            │\n",
                "│  ├── HNSW (default) - O(log N) search   │\n",
                "│  └── FAISS (optional) - billion-scale   │\n",
                "├─────────────────────────────────────────┤\n",
                "│           Storage Layer                 │\n",
                "└─────────────────────────────────────────┘\n",
                "```"
            ]
        },
        # Cell 4: System info
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["display_system_info()"]
        },
        # Cell 5: HNSW vs FAISS
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚖️ HNSW vs FAISS <a id=\"hnsw-vs-faiss\"></a>\n",
                "\n",
                "| Aspect | HNSW (Default) | FAISS Backend |\n",
                "|--------|----------------|---------------|\n",
                "| **Scale** | Up to ~10M vectors | Billions of vectors |\n",
                "| **Search** | O(log N) | O(1) to O(log N) |\n",
                "| **Memory** | Higher (graph structure) | Lower (quantization) |\n",
                "| **GPU** | CPU only | GPU acceleration |\n",
                "| **Setup** | Zero config | Requires `--features faiss` |\n",
                "| **Index Types** | HNSW only | Flat, IVF, PQ, HNSW, etc. |\n",
                "\n",
                "### When to Use Each\n",
                "\n",
                "**Use HNSW (default):**\n",
                "- Collections under 10M vectors\n",
                "- Development and prototyping\n",
                "- Simple deployment (no extra deps)\n",
                "\n",
                "**Use FAISS backend:**\n",
                "- Billion-scale collections\n",
                "- GPU-accelerated search needed\n",
                "- Memory-constrained environments (PQ)\n",
                "- Production with specific latency SLAs"
            ]
        },
        # Cell 6: Enabling FAISS
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 Enabling FAISS <a id=\"enabling-faiss\"></a>\n",
                "\n",
                "### Rust (Cargo.toml)\n",
                "\n",
                "```toml\n",
                "[dependencies]\n",
                "synadb = { version = \"1.0\", features = [\"faiss\"] }\n",
                "\n",
                "# For GPU support:\n",
                "synadb = { version = \"1.0\", features = [\"faiss-gpu\"] }\n",
                "```\n",
                "\n",
                "### Building from Source\n",
                "\n",
                "```bash\n",
                "# CPU only\n",
                "cargo build --release --features faiss\n",
                "\n",
                "# With GPU support (requires CUDA)\n",
                "cargo build --release --features faiss-gpu\n",
                "```\n",
                "\n",
                "### Python\n",
                "\n",
                "```python\n",
                "from synadb import VectorStore\n",
                "\n",
                "# Use FAISS backend\n",
                "store = VectorStore(\n",
                "    'vectors.db',\n",
                "    dimensions=768,\n",
                "    backend='faiss',  # 'hnsw' is default\n",
                "    faiss_index_type='IVF1024,PQ32'\n",
                ")\n",
                "```"
            ]
        },
        # Cell 7: Index Types
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📚 Index Types <a id=\"index-types\"></a>\n",
                "\n",
                "FAISS supports multiple index types for different trade-offs:\n",
                "\n",
                "| Index | Use Case | Memory | Speed | Recall |\n",
                "|-------|----------|--------|-------|--------|\n",
                "| `Flat` | Exact search, small datasets | High | Slow | 100% |\n",
                "| `IVF` | Medium datasets | Medium | Fast | ~95% |\n",
                "| `PQ` | Memory-constrained | Low | Fast | ~90% |\n",
                "| `IVF,PQ` | Billion-scale | Very Low | Very Fast | ~85% |\n",
                "| `HNSW` | General purpose | High | Very Fast | ~95% |\n",
                "\n",
                "### Index Factory Strings\n",
                "\n",
                "```python\n",
                "# Exact search (brute force)\n",
                "faiss_index_type='Flat'\n",
                "\n",
                "# IVF with 1024 centroids\n",
                "faiss_index_type='IVF1024,Flat'\n",
                "\n",
                "# Product Quantization (32 subquantizers)\n",
                "faiss_index_type='PQ32'\n",
                "\n",
                "# IVF + PQ for billion-scale\n",
                "faiss_index_type='IVF4096,PQ64'\n",
                "\n",
                "# HNSW via FAISS\n",
                "faiss_index_type='HNSW32'\n",
                "```"
            ]
        },
        # Cell 8: Benchmark header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## ⚡ Benchmark <a id=\"benchmark\"></a>\n", "\n", "Comparing SynaDB's native HNSW vs FAISS backend performance."]
        },
        # Cell 9: Benchmark info
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "info_box(\n",
                "    'FAISS Benchmark',\n",
                "    'To run FAISS benchmarks, build SynaDB with the faiss feature:\\n\\n'\n",
                "    '```bash\\n'\n",
                "    'cargo run --release --features faiss -- faiss --quick\\n'\n",
                "    '```\\n\\n'\n",
                "    'This notebook shows expected results. Run the CLI for actual measurements.'\n",
                ")"
            ]
        },
        # Cell 10: Expected results
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import display, Markdown\n",
                "\n",
                "results = '''\n",
                "### Expected Results (100K vectors, 768 dims)\n",
                "\n",
                "| Backend | Insert (vec/s) | Search (ms) | Memory (MB) | Recall@10 |\n",
                "|---------|----------------|-------------|-------------|----------|\n",
                "| HNSW (default) | 50,000 | 0.5 | 80 | 95% |\n",
                "| FAISS-Flat | 100,000 | 10.0 | 60 | 100% |\n",
                "| FAISS-IVF1024 | 80,000 | 1.0 | 65 | 92% |\n",
                "| FAISS-PQ32 | 90,000 | 0.8 | 25 | 88% |\n",
                "\n",
                "*Results vary by hardware. Run benchmarks on your system.*\n",
                "'''\n",
                "display(Markdown(results))"
            ]
        },
        # Cell 11: When to use
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 When to Use <a id=\"when-to-use\"></a>\n",
                "\n",
                "### Use Default HNSW\n",
                "\n",
                "✅ **Recommended for most users**\n",
                "\n",
                "- Collections under 10M vectors\n",
                "- Development and prototyping\n",
                "- Simple deployment\n",
                "- No extra dependencies\n",
                "\n",
                "### Use FAISS Backend\n",
                "\n",
                "🚀 **For scale and performance**\n",
                "\n",
                "- Billion-scale vector collections\n",
                "- GPU-accelerated search\n",
                "- Memory-constrained environments\n",
                "- Specific latency requirements\n",
                "- Production with tuned indexes"
            ]
        },
        # Cell 12: Conclusions header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 🎯 Conclusions <a id=\"conclusions\"></a>"]
        },
        # Cell 13: Conclusions
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "conclusion_box(\n",
                "    title='Key Takeaways',\n",
                "    points=[\n",
                "        'FAISS is an <b>optional backend</b> for SynaDB, not a competitor',\n",
                "        'Default HNSW works great for most use cases (up to 10M vectors)',\n",
                "        'Enable FAISS with <code>--features faiss</code> for billion-scale',\n",
                "        'Choose index type based on memory/speed/recall trade-offs',\n",
                "        'GPU support available with <code>--features faiss-gpu</code>',\n",
                "    ],\n",
                "    summary='Start with default HNSW. Switch to FAISS backend when you need '\n",
                "            'billion-scale search, GPU acceleration, or specific index types.'\n",
                ")"
            ]
        },
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('02_faiss_backend.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print('Created 02_faiss_backend.ipynb')


if __name__ == '__main__':
    create_chroma_notebook()
    create_faiss_backend_notebook()
    print('\\nDone! Generated notebooks:')
    print('  - 01_chroma_comparison.ipynb')
    print('  - 02_faiss_backend.ipynb')

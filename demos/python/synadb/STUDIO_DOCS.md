# Syna Studio V2 - User Guide

Syna Studio is a web interface for exploring, visualizing, and managing SynaDB databases. V2 introduces advanced AI workflows including model registry management, vector clustering, and deep system inspection.

## Quick Start

### Launch with Test Data

The easiest way to try Studio is with the included launcher:

```bash
cd demos/python/synadb

# Create test database and launch
python run_ui.py --test

# With real HuggingFace embeddings (requires: pip install datasets sentence-transformers)
python run_ui.py --test --use-hf --samples 200

# Custom port
python run_ui.py --test --port 8080
```

### Launch with Existing Database

```bash
# Open any existing database
python run_ui.py path/to/my_database.db

# With debug mode
python run_ui.py my_database.db --debug
```

### Launch Programmatically

```python
from synadb import studio
# Launch the studio on port 8501
studio.launch("path/to/my_database.db", port=8501)
```

Access the dashboard at `http://localhost:8501`.

## Key Features

### 1. Keys Explorer (The Core)
*   **Search & Filter**: Real-time filtering by key name or data type (Float, Int, Text, Bytes, Vector).
*   **Deep Inspection**: Click any key to view its raw value.
    *   **Hex Viewer**: For binary (`bytes`) data, inspect the raw hex dump and ASCII representation side-by-side.
    *   **Smart Preview**: Automatically formatted values for large texts or arrays.

### 2. Model Registry Dashboard
*   **Version Control**: View all ML models stored in the database.
*   **Lineage**: See version history (v1, v2...) along with creation timestamps and stages (Dev, Staging, Prod).
*   **Metadata**: Inspect training metrics (accuracy, loss) and hyperparameters directly from the UI.

### 3. 3D Embedding Clusters
*   **Visualization**: Projects high-dimensional vectors (from `put_vector` or JSON arrays) into 3D space using PCA (Principal Component Analysis).
*   **Interactive**: Rotate, zoom, and hover over points to see their key names. Ideal for debugging embedding clusters (e.g., verifying that "dog" and "cat" vectors are close).

### 4. Statistics
The **Statistics** page provides deep insights into your database's performance and composition.
*   **Default Metrics**: View Storage Usage (Treemap) and Key Type Distribution (Pie Chart).
*   **Dynamic Dashboard**: Use the **"+ Add Plot"** button to create custom widgets such as:
    *   **Key Length Distribution**: Analyze the spread of key lengths.
    *   **Write Operations**: Visualize write throughput over time (Simulated).
    *   **Memory Usage**: Monitor memory consumption trends.
*   **Customization**: You can remove widgets to tailor the dashboard to your needs.

### 5. Custom Suite (Administrative Actions)
Execute maintenance tasks and run custom scripts directly from the UI.
*   **Compact Database**: Reclaims unused space.
*   **Export to JSON**: Dumps the database content for backup.
*   **Integrity Check**: Verifies data consistency.
*   **Clear Cache**: Frees up internal memory caches.
*   **Pattern Scanner**: Quickly search for keys matching specific patterns (e.g., `user/*`).

### 6. Integrations
Seamlessly connect with modern AI frameworks. The Integrations tab scans your project for available modules.
*   **Auto-Discovery**: Automatically lists integration scripts found in the `integrations/` directory (e.g., `mlflow.py`, `langchain.py`).
*   **Health Check**: View module size and status directly from the UI.

### 7. Database Switcher
*   **Hot-Swap**: Switch between different `.db` files instantly from the sidebar without restarting the server.
*   **Recent Files**: Remembers your recently accessed databases.

## Advanced Usage

### Embedding Vectors
To see data in the "Clusters" tab, store vectors using the Python API. If native vector support is unavailable, JSON-formatted lists in text keys are automatically detected:

```python
embedding = model.encode("hello world")
# Native
# db.put_vector("embeddings/hello", embedding)

# Fallback (JSON)
import json
db.put_text("vector/hello", json.dumps(embedding))
```

### Saving Models
To see models in the "AI Models" tab, use the `ModelRegistry`:

```python
from synadb.models import ModelRegistry
registry = ModelRegistry("db_path")
registry.save("my_classifier", model_obj, metadata={"accuracy": "0.98"})
```

### Hot Keys
*   `Esc`: Close modals.
*   `Ctrl/Cmd + F`: Focus search bar.

---
*Built for High-Performance AI Engineering.*

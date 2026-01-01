#!/usr/bin/env python3
"""Generate the End-to-End ML Pipeline notebook for SynaDB v1.0.0 Showcase."""

import json

def create_notebook():
    """Create the 18_end_to_end_pipeline.ipynb notebook."""
    
    cells = []
    
    # Cell 1: Header and Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 1: Header and Setup\n",
            "import sys\n",
            "sys.path.insert(0, '..')\n",
            "\n",
            "from utils.notebook_utils import display_header, display_toc, check_dependency, conclusion_box, info_box, warning_box\n",
            "from utils.system_info import display_system_info\n",
            "from utils.benchmark import Benchmark, BenchmarkResult, ComparisonTable\n",
            "from utils.charts import setup_style, bar_comparison, throughput_comparison, COLORS\n",
            "\n",
            "display_header('End-to-End ML Pipeline', 'SynaDB as Unified Data Layer')"
        ]
    })
    
    # Cell 2: Table of Contents
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 2: Table of Contents\n",
            "sections = [\n",
            "    ('Introduction', 'introduction'),\n",
            "    ('Setup', 'setup'),\n",
            "    ('Data Ingestion', 'data-ingestion'),\n",
            "    ('Feature Engineering', 'feature-engineering'),\n",
            "    ('Model Training', 'model-training'),\n",
            "    ('Experiment Tracking', 'experiment-tracking'),\n",
            "    ('Model Registry', 'model-registry'),\n",
            "    ('Inference', 'inference'),\n",
            "    ('Unified Data Layer', 'unified-data'),\n",
            "    ('Results Summary', 'results'),\n",
            "    ('Conclusions', 'conclusions'),\n",
            "]\n",
            "display_toc(sections)"
        ]
    })
    
    # Cell 3: Introduction (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📌 Introduction <a id=\"introduction\"></a>\n",
            "\n",
            "This notebook demonstrates **SynaDB as a unified data layer** for the complete ML lifecycle.\n",
            "\n",
            "### The ML Pipeline Challenge\n",
            "\n",
            "Traditional ML pipelines require multiple tools:\n",
            "\n",
            "```\n",
            "┌─────────────┐   ┌─────────────┐   ┌─────────────┐\n",
            "│ Data Lake   │ → │ Feature     │ → │ Training    │\n",
            "│ (S3/HDFS)   │   │ Store       │   │ (PyTorch)   │\n",
            "└─────────────┘   │ (Feast)     │   └─────────────┘\n",
            "                  └─────────────┘          ↓\n",
            "┌─────────────┐   ┌─────────────┐   ┌─────────────┐\n",
            "│ Inference   │ ← │ Model       │ ← │ Experiment  │\n",
            "│ Service     │   │ Registry    │   │ Tracking    │\n",
            "└─────────────┘   │ (MLflow)    │   │ (W&B)       │\n",
            "                  └─────────────┘   └─────────────┘\n",
            "```\n",
            "\n",
            "### SynaDB Unified Approach\n",
            "\n",
            "```\n",
            "┌─────────────────────────────────────────────────┐\n",
            "│                    SynaDB                        │\n",
            "├─────────────────────────────────────────────────┤\n",
            "│ Raw Data │ Features │ Experiments │ Models      │\n",
            "│          │          │             │             │\n",
            "│ Vectors  │ Tensors  │ Metrics     │ Artifacts   │\n",
            "└─────────────────────────────────────────────────┘\n",
            "```\n",
            "\n",
            "### What We'll Build\n",
            "\n",
            "A complete ML pipeline for image classification:\n",
            "\n",
            "1. **Data Ingestion** - Load and store training data\n",
            "2. **Feature Engineering** - Extract and store features\n",
            "3. **Model Training** - Train with SynaDB DataLoader\n",
            "4. **Experiment Tracking** - Log metrics and parameters\n",
            "5. **Model Registry** - Version and store models\n",
            "6. **Inference** - Load model and make predictions"
        ]
    })
    
    # Cell 4: System Info
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 4: System Info\n",
            "display_system_info()"
        ]
    })
    
    return cells

def create_notebook_part2(cells):
    """Continue creating notebook cells - Setup and data ingestion."""
    
    # Cell 5: Setup Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Setup <a id=\"setup\"></a>\n",
            "\n",
            "Let's set up our environment for the end-to-end pipeline."
        ]
    })
    
    # Cell 6: Setup Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 6: Setup\n",
            "import numpy as np\n",
            "import time\n",
            "import os\n",
            "import tempfile\n",
            "import hashlib\n",
            "from datetime import datetime\n",
            "from dataclasses import dataclass\n",
            "from typing import List, Dict, Any, Optional, Tuple\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Check dependencies\n",
            "HAS_SYNADB = check_dependency('synadb', 'pip install synadb')\n",
            "HAS_TORCH = check_dependency('torch', 'pip install torch', required=False)\n",
            "HAS_SKLEARN = check_dependency('sklearn', 'pip install scikit-learn', required=False)\n",
            "\n",
            "# Apply consistent styling\n",
            "setup_style()\n",
            "\n",
            "# Create temp directory\n",
            "temp_dir = tempfile.mkdtemp(prefix='synadb_pipeline_')\n",
            "print(f'Using temp directory: {temp_dir}')\n",
            "\n",
            "# Pipeline configuration\n",
            "NUM_SAMPLES = 1000\n",
            "NUM_FEATURES = 64\n",
            "NUM_CLASSES = 10\n",
            "TRAIN_SPLIT = 0.8\n",
            "\n",
            "print(f\"\\n✓ Setup complete\")\n",
            "print(f\"  Samples: {NUM_SAMPLES:,}\")\n",
            "print(f\"  Features: {NUM_FEATURES}\")\n",
            "print(f\"  Classes: {NUM_CLASSES}\")\n",
            "print(f\"  Train/Test split: {TRAIN_SPLIT}/{1-TRAIN_SPLIT}\")"
        ]
    })
    
    # Cell 7: Data Ingestion Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📥 Data Ingestion <a id=\"data-ingestion\"></a>\n",
            "\n",
            "The first stage loads raw data into SynaDB.\n",
            "\n",
            "### Data Schema\n",
            "\n",
            "| Key Pattern | Type | Description |\n",
            "|-------------|------|-------------|\n",
            "| `raw/sample/{id}/features` | Vector | Raw feature vector |\n",
            "| `raw/sample/{id}/label` | Int | Class label |\n",
            "| `raw/metadata/num_samples` | Int | Total samples |\n",
            "| `raw/metadata/num_features` | Int | Feature dimensions |"
        ]
    })
    
    # Cell 8: Generate and Ingest Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 8: Generate and Ingest Data\n",
            "np.random.seed(42)\n",
            "\n",
            "# Generate synthetic classification data\n",
            "def generate_synthetic_data(n_samples, n_features, n_classes):\n",
            "    \"\"\"Generate synthetic classification data with cluster structure.\"\"\"\n",
            "    X = []\n",
            "    y = []\n",
            "    \n",
            "    samples_per_class = n_samples // n_classes\n",
            "    \n",
            "    for class_id in range(n_classes):\n",
            "        # Each class has a different center\n",
            "        center = np.random.randn(n_features) * 2\n",
            "        \n",
            "        for _ in range(samples_per_class):\n",
            "            # Add noise around the center\n",
            "            sample = center + np.random.randn(n_features) * 0.5\n",
            "            X.append(sample.astype(np.float32))\n",
            "            y.append(class_id)\n",
            "    \n",
            "    return np.array(X), np.array(y)\n",
            "\n",
            "X_raw, y_raw = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES)\n",
            "print(f\"Generated synthetic data:\")\n",
            "print(f\"  X shape: {X_raw.shape}\")\n",
            "print(f\"  y shape: {y_raw.shape}\")\n",
            "print(f\"  Class distribution: {np.bincount(y_raw)}\")"
        ]
    })
    
    # Cell 9: Store Raw Data in SynaDB
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 9: Store Raw Data in SynaDB\n",
            "if HAS_SYNADB:\n",
            "    from synadb import SynaDB, VectorStore\n",
            "    \n",
            "    db_path = os.path.join(temp_dir, 'pipeline.db')\n",
            "    db = SynaDB(db_path)\n",
            "    \n",
            "    print(\"Ingesting raw data into SynaDB...\\n\")\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    \n",
            "    # Store metadata\n",
            "    db.put_int('raw/metadata/num_samples', NUM_SAMPLES)\n",
            "    db.put_int('raw/metadata/num_features', NUM_FEATURES)\n",
            "    db.put_int('raw/metadata/num_classes', NUM_CLASSES)\n",
            "    db.put_text('raw/metadata/created_at', datetime.now().isoformat())\n",
            "    \n",
            "    # Store samples\n",
            "    for i, (features, label) in enumerate(zip(X_raw, y_raw)):\n",
            "        # Store features as floats (one per dimension for simplicity)\n",
            "        for j, val in enumerate(features):\n",
            "            db.put_float(f'raw/sample/{i:04d}/feature/{j:02d}', float(val))\n",
            "        db.put_int(f'raw/sample/{i:04d}/label', int(label))\n",
            "    \n",
            "    ingestion_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    print(f\"✓ Ingested {NUM_SAMPLES:,} samples in {ingestion_time:.1f}ms\")\n",
            "    print(f\"  Throughput: {NUM_SAMPLES / (ingestion_time / 1000):.0f} samples/sec\")\n",
            "    print(f\"  File size: {os.path.getsize(db_path) / 1024:.1f} KB\")\n",
            "else:\n",
            "    warning_box(\"SynaDB not installed - skipping data ingestion\")\n",
            "    db = None"
        ]
    })
    
    return cells


def create_notebook_part3(cells):
    """Continue creating notebook cells - Feature engineering."""
    
    # Cell 10: Feature Engineering Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Feature Engineering <a id=\"feature-engineering\"></a>\n",
            "\n",
            "Transform raw data into ML-ready features.\n",
            "\n",
            "### Feature Transformations\n",
            "\n",
            "| Transformation | Description |\n",
            "|----------------|-------------|\n",
            "| Normalization | Scale to zero mean, unit variance |\n",
            "| PCA | Dimensionality reduction |\n",
            "| Feature Selection | Keep top-k features |"
        ]
    })
    
    # Cell 11: Feature Engineering Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 11: Feature Engineering\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Performing feature engineering...\\n\")\n",
            "    \n",
            "    # Load raw data from SynaDB\n",
            "    start = time.perf_counter()\n",
            "    \n",
            "    X_loaded = []\n",
            "    y_loaded = []\n",
            "    \n",
            "    for i in range(NUM_SAMPLES):\n",
            "        features = []\n",
            "        for j in range(NUM_FEATURES):\n",
            "            val = db.get_float(f'raw/sample/{i:04d}/feature/{j:02d}')\n",
            "            features.append(val if val is not None else 0.0)\n",
            "        X_loaded.append(features)\n",
            "        y_loaded.append(db.get_int(f'raw/sample/{i:04d}/label'))\n",
            "    \n",
            "    X_loaded = np.array(X_loaded, dtype=np.float32)\n",
            "    y_loaded = np.array(y_loaded)\n",
            "    \n",
            "    load_time = (time.perf_counter() - start) * 1000\n",
            "    print(f\"Loaded raw data in {load_time:.1f}ms\")\n",
            "    \n",
            "    # Normalize features\n",
            "    mean = X_loaded.mean(axis=0)\n",
            "    std = X_loaded.std(axis=0) + 1e-8\n",
            "    X_normalized = (X_loaded - mean) / std\n",
            "    \n",
            "    # Store normalization parameters\n",
            "    for j in range(NUM_FEATURES):\n",
            "        db.put_float(f'features/norm/mean/{j:02d}', float(mean[j]))\n",
            "        db.put_float(f'features/norm/std/{j:02d}', float(std[j]))\n",
            "    \n",
            "    # Store normalized features\n",
            "    start = time.perf_counter()\n",
            "    \n",
            "    for i, (features, label) in enumerate(zip(X_normalized, y_loaded)):\n",
            "        for j, val in enumerate(features):\n",
            "            db.put_float(f'features/sample/{i:04d}/normalized/{j:02d}', float(val))\n",
            "    \n",
            "    store_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    print(f\"\\n✓ Feature engineering complete\")\n",
            "    print(f\"  Normalization: mean={mean.mean():.4f}, std={std.mean():.4f}\")\n",
            "    print(f\"  Stored normalized features in {store_time:.1f}ms\")\n",
            "else:\n",
            "    print(\"Database not available for feature engineering\")"
        ]
    })
    
    # Cell 12: Train/Test Split
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 12: Train/Test Split\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Creating train/test split...\\n\")\n",
            "    \n",
            "    # Shuffle indices\n",
            "    indices = np.random.permutation(NUM_SAMPLES)\n",
            "    split_idx = int(NUM_SAMPLES * TRAIN_SPLIT)\n",
            "    \n",
            "    train_indices = indices[:split_idx]\n",
            "    test_indices = indices[split_idx:]\n",
            "    \n",
            "    # Store split information\n",
            "    db.put_int('split/train_size', len(train_indices))\n",
            "    db.put_int('split/test_size', len(test_indices))\n",
            "    \n",
            "    for i, idx in enumerate(train_indices):\n",
            "        db.put_int(f'split/train/{i:04d}', int(idx))\n",
            "    \n",
            "    for i, idx in enumerate(test_indices):\n",
            "        db.put_int(f'split/test/{i:04d}', int(idx))\n",
            "    \n",
            "    print(f\"✓ Split created\")\n",
            "    print(f\"  Train samples: {len(train_indices):,}\")\n",
            "    print(f\"  Test samples: {len(test_indices):,}\")\n",
            "    \n",
            "    # Prepare numpy arrays for training\n",
            "    X_train = X_normalized[train_indices]\n",
            "    y_train = y_loaded[train_indices]\n",
            "    X_test = X_normalized[test_indices]\n",
            "    y_test = y_loaded[test_indices]\n",
            "    \n",
            "    print(f\"  X_train shape: {X_train.shape}\")\n",
            "    print(f\"  X_test shape: {X_test.shape}\")\n",
            "else:\n",
            "    print(\"Database not available for split\")"
        ]
    })
    
    return cells


def create_notebook_part4(cells):
    """Continue creating notebook cells - Model training."""
    
    # Cell 13: Model Training Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Model Training <a id=\"model-training\"></a>\n",
            "\n",
            "Train a classifier using data from SynaDB.\n",
            "\n",
            "### Training Options\n",
            "\n",
            "| Framework | Integration |\n",
            "|-----------|-------------|\n",
            "| PyTorch | SynaDataset + DataLoader |\n",
            "| scikit-learn | NumPy arrays from SynaDB |\n",
            "| TensorFlow | tf.data.Dataset |"
        ]
    })
    
    # Cell 14: Simple Classifier Training
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 14: Simple Classifier Training\n",
            "training_metrics = []\n",
            "\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Training classifier...\\n\")\n",
            "    \n",
            "    # Simple logistic regression-style classifier\n",
            "    class SimpleClassifier:\n",
            "        def __init__(self, n_features, n_classes, lr=0.01):\n",
            "            self.W = np.random.randn(n_features, n_classes) * 0.01\n",
            "            self.b = np.zeros(n_classes)\n",
            "            self.lr = lr\n",
            "        \n",
            "        def softmax(self, x):\n",
            "            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))\n",
            "            return exp_x / exp_x.sum(axis=1, keepdims=True)\n",
            "        \n",
            "        def forward(self, X):\n",
            "            return self.softmax(X @ self.W + self.b)\n",
            "        \n",
            "        def loss(self, X, y):\n",
            "            probs = self.forward(X)\n",
            "            n = len(y)\n",
            "            return -np.log(probs[np.arange(n), y] + 1e-8).mean()\n",
            "        \n",
            "        def accuracy(self, X, y):\n",
            "            preds = self.forward(X).argmax(axis=1)\n",
            "            return (preds == y).mean()\n",
            "        \n",
            "        def train_step(self, X, y):\n",
            "            n = len(y)\n",
            "            probs = self.forward(X)\n",
            "            \n",
            "            # Gradient\n",
            "            grad = probs.copy()\n",
            "            grad[np.arange(n), y] -= 1\n",
            "            grad /= n\n",
            "            \n",
            "            # Update\n",
            "            self.W -= self.lr * (X.T @ grad)\n",
            "            self.b -= self.lr * grad.sum(axis=0)\n",
            "    \n",
            "    # Train\n",
            "    model = SimpleClassifier(NUM_FEATURES, NUM_CLASSES, lr=0.1)\n",
            "    n_epochs = 50\n",
            "    batch_size = 32\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    \n",
            "    for epoch in range(n_epochs):\n",
            "        # Shuffle training data\n",
            "        perm = np.random.permutation(len(X_train))\n",
            "        X_shuffled = X_train[perm]\n",
            "        y_shuffled = y_train[perm]\n",
            "        \n",
            "        # Mini-batch training\n",
            "        for i in range(0, len(X_train), batch_size):\n",
            "            X_batch = X_shuffled[i:i+batch_size]\n",
            "            y_batch = y_shuffled[i:i+batch_size]\n",
            "            model.train_step(X_batch, y_batch)\n",
            "        \n",
            "        # Log metrics\n",
            "        train_loss = model.loss(X_train, y_train)\n",
            "        train_acc = model.accuracy(X_train, y_train)\n",
            "        test_acc = model.accuracy(X_test, y_test)\n",
            "        \n",
            "        training_metrics.append({\n",
            "            'epoch': epoch,\n",
            "            'train_loss': train_loss,\n",
            "            'train_acc': train_acc,\n",
            "            'test_acc': test_acc\n",
            "        })\n",
            "        \n",
            "        if epoch % 10 == 0:\n",
            "            print(f\"Epoch {epoch:3d}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}\")\n",
            "    \n",
            "    training_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    final_acc = model.accuracy(X_test, y_test)\n",
            "    print(f\"\\n✓ Training complete in {training_time:.1f}ms\")\n",
            "    print(f\"  Final test accuracy: {final_acc:.4f}\")\n",
            "else:\n",
            "    print(\"Database not available for training\")"
        ]
    })
    
    # Cell 15: Visualize Training
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15: Visualize Training\n",
            "if training_metrics:\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "    \n",
            "    epochs = [m['epoch'] for m in training_metrics]\n",
            "    \n",
            "    # Loss curve\n",
            "    axes[0].plot(epochs, [m['train_loss'] for m in training_metrics], \n",
            "                 color=COLORS['synadb'], linewidth=2, label='Train Loss')\n",
            "    axes[0].set_xlabel('Epoch')\n",
            "    axes[0].set_ylabel('Loss')\n",
            "    axes[0].set_title('Training Loss', fontweight='bold')\n",
            "    axes[0].grid(True, alpha=0.3)\n",
            "    axes[0].legend()\n",
            "    \n",
            "    # Accuracy curves\n",
            "    axes[1].plot(epochs, [m['train_acc'] for m in training_metrics], \n",
            "                 color=COLORS['synadb'], linewidth=2, label='Train Acc')\n",
            "    axes[1].plot(epochs, [m['test_acc'] for m in training_metrics], \n",
            "                 color=COLORS['competitor'], linewidth=2, label='Test Acc')\n",
            "    axes[1].set_xlabel('Epoch')\n",
            "    axes[1].set_ylabel('Accuracy')\n",
            "    axes[1].set_title('Training & Test Accuracy', fontweight='bold')\n",
            "    axes[1].grid(True, alpha=0.3)\n",
            "    axes[1].legend()\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"No training metrics to visualize\")"
        ]
    })
    
    return cells


def create_notebook_part5(cells):
    """Continue creating notebook cells - Experiment tracking."""
    
    # Cell 16: Experiment Tracking Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Experiment Tracking <a id=\"experiment-tracking\"></a>\n",
            "\n",
            "Log all training metrics and parameters to SynaDB.\n",
            "\n",
            "### What We Track\n",
            "\n",
            "| Category | Items |\n",
            "|----------|-------|\n",
            "| **Parameters** | learning_rate, batch_size, epochs |\n",
            "| **Metrics** | loss, train_acc, test_acc per epoch |\n",
            "| **Artifacts** | Model weights, normalization params |"
        ]
    })
    
    # Cell 17: Log Experiment
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17: Log Experiment\n",
            "if HAS_SYNADB and db and training_metrics:\n",
            "    print(\"Logging experiment to SynaDB...\\n\")\n",
            "    \n",
            "    import uuid\n",
            "    run_id = str(uuid.uuid4())[:8]\n",
            "    \n",
            "    # Log parameters\n",
            "    params = {\n",
            "        'learning_rate': 0.1,\n",
            "        'batch_size': 32,\n",
            "        'epochs': 50,\n",
            "        'n_features': NUM_FEATURES,\n",
            "        'n_classes': NUM_CLASSES,\n",
            "        'train_size': len(X_train),\n",
            "        'test_size': len(X_test),\n",
            "    }\n",
            "    \n",
            "    for key, value in params.items():\n",
            "        if isinstance(value, float):\n",
            "            db.put_float(f'experiments/{run_id}/params/{key}', value)\n",
            "        else:\n",
            "            db.put_int(f'experiments/{run_id}/params/{key}', value)\n",
            "    \n",
            "    # Log metrics\n",
            "    for m in training_metrics:\n",
            "        epoch = m['epoch']\n",
            "        db.put_float(f'experiments/{run_id}/metrics/train_loss/{epoch:03d}', m['train_loss'])\n",
            "        db.put_float(f'experiments/{run_id}/metrics/train_acc/{epoch:03d}', m['train_acc'])\n",
            "        db.put_float(f'experiments/{run_id}/metrics/test_acc/{epoch:03d}', m['test_acc'])\n",
            "    \n",
            "    # Log final metrics\n",
            "    final = training_metrics[-1]\n",
            "    db.put_float(f'experiments/{run_id}/final/train_loss', final['train_loss'])\n",
            "    db.put_float(f'experiments/{run_id}/final/train_acc', final['train_acc'])\n",
            "    db.put_float(f'experiments/{run_id}/final/test_acc', final['test_acc'])\n",
            "    \n",
            "    # Log metadata\n",
            "    db.put_text(f'experiments/{run_id}/metadata/created_at', datetime.now().isoformat())\n",
            "    db.put_text(f'experiments/{run_id}/metadata/status', 'completed')\n",
            "    \n",
            "    print(f\"✓ Experiment logged\")\n",
            "    print(f\"  Run ID: {run_id}\")\n",
            "    print(f\"  Parameters: {len(params)}\")\n",
            "    print(f\"  Metric series: 3 (train_loss, train_acc, test_acc)\")\n",
            "    print(f\"  Epochs logged: {len(training_metrics)}\")\n",
            "else:\n",
            "    run_id = None\n",
            "    print(\"Database not available for experiment logging\")"
        ]
    })
    
    # Cell 18: Model Registry Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Model Registry <a id=\"model-registry\"></a>\n",
            "\n",
            "Version and store trained models with checksums.\n",
            "\n",
            "### Model Versioning\n",
            "\n",
            "| Field | Description |\n",
            "|-------|-------------|\n",
            "| **version** | Auto-incrementing version number |\n",
            "| **checksum** | SHA-256 hash for integrity |\n",
            "| **stage** | Development → Staging → Production |\n",
            "| **metrics** | Associated performance metrics |"
        ]
    })
    
    # Cell 19: Save Model to Registry
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 19: Save Model to Registry\n",
            "if HAS_SYNADB and db and 'model' in dir():\n",
            "    print(\"Saving model to registry...\\n\")\n",
            "    \n",
            "    model_name = 'simple_classifier'\n",
            "    \n",
            "    # Get current version\n",
            "    version_key = f'models/{model_name}/latest_version'\n",
            "    current_version = db.get_int(version_key)\n",
            "    new_version = (current_version or 0) + 1\n",
            "    \n",
            "    # Serialize model weights\n",
            "    model_bytes = model.W.tobytes() + model.b.tobytes()\n",
            "    checksum = hashlib.sha256(model_bytes).hexdigest()\n",
            "    \n",
            "    # Store model\n",
            "    version_prefix = f'models/{model_name}/v{new_version:03d}'\n",
            "    \n",
            "    # Store weights as individual floats (for demo)\n",
            "    for i in range(model.W.shape[0]):\n",
            "        for j in range(model.W.shape[1]):\n",
            "            db.put_float(f'{version_prefix}/weights/W/{i:02d}/{j:02d}', float(model.W[i, j]))\n",
            "    \n",
            "    for j in range(len(model.b)):\n",
            "        db.put_float(f'{version_prefix}/weights/b/{j:02d}', float(model.b[j]))\n",
            "    \n",
            "    # Store metadata\n",
            "    db.put_text(f'{version_prefix}/checksum', checksum)\n",
            "    db.put_text(f'{version_prefix}/stage', 'development')\n",
            "    db.put_text(f'{version_prefix}/created_at', datetime.now().isoformat())\n",
            "    db.put_float(f'{version_prefix}/metrics/test_acc', final['test_acc'])\n",
            "    \n",
            "    if run_id:\n",
            "        db.put_text(f'{version_prefix}/experiment_id', run_id)\n",
            "    \n",
            "    # Update latest version\n",
            "    db.put_int(version_key, new_version)\n",
            "    \n",
            "    print(f\"✓ Model saved to registry\")\n",
            "    print(f\"  Name: {model_name}\")\n",
            "    print(f\"  Version: {new_version}\")\n",
            "    print(f\"  Checksum: {checksum[:16]}...\")\n",
            "    print(f\"  Stage: development\")\n",
            "    print(f\"  Test accuracy: {final['test_acc']:.4f}\")\n",
            "else:\n",
            "    print(\"Model not available for registry\")"
        ]
    })
    
    # Cell 20: Promote Model
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 20: Promote Model\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Promoting model to production...\\n\")\n",
            "    \n",
            "    # Update stage\n",
            "    db.put_text(f'{version_prefix}/stage', 'production')\n",
            "    \n",
            "    # Set as production model\n",
            "    db.put_int(f'models/{model_name}/production_version', new_version)\n",
            "    \n",
            "    print(f\"✓ Model promoted\")\n",
            "    print(f\"  {model_name} v{new_version} → production\")\n",
            "else:\n",
            "    print(\"Database not available for promotion\")"
        ]
    })
    
    return cells


def create_notebook_part6(cells):
    """Continue creating notebook cells - Inference and summary."""
    
    # Cell 21: Inference Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔮 Inference <a id=\"inference\"></a>\n",
            "\n",
            "Load the production model and make predictions."
        ]
    })
    
    # Cell 22: Load and Inference
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 22: Load and Inference\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Loading production model for inference...\\n\")\n",
            "    \n",
            "    # Get production version\n",
            "    prod_version = db.get_int(f'models/{model_name}/production_version')\n",
            "    version_prefix = f'models/{model_name}/v{prod_version:03d}'\n",
            "    \n",
            "    # Load weights\n",
            "    W_loaded = np.zeros((NUM_FEATURES, NUM_CLASSES))\n",
            "    b_loaded = np.zeros(NUM_CLASSES)\n",
            "    \n",
            "    for i in range(NUM_FEATURES):\n",
            "        for j in range(NUM_CLASSES):\n",
            "            W_loaded[i, j] = db.get_float(f'{version_prefix}/weights/W/{i:02d}/{j:02d}')\n",
            "    \n",
            "    for j in range(NUM_CLASSES):\n",
            "        b_loaded[j] = db.get_float(f'{version_prefix}/weights/b/{j:02d}')\n",
            "    \n",
            "    # Verify checksum\n",
            "    loaded_bytes = W_loaded.tobytes() + b_loaded.tobytes()\n",
            "    loaded_checksum = hashlib.sha256(loaded_bytes).hexdigest()\n",
            "    stored_checksum = db.get_text(f'{version_prefix}/checksum')\n",
            "    \n",
            "    print(f\"Model loaded: {model_name} v{prod_version}\")\n",
            "    print(f\"Checksum verification: {'✓ PASSED' if loaded_checksum == stored_checksum else '✗ FAILED'}\")\n",
            "    \n",
            "    # Create inference model\n",
            "    inference_model = SimpleClassifier(NUM_FEATURES, NUM_CLASSES)\n",
            "    inference_model.W = W_loaded\n",
            "    inference_model.b = b_loaded\n",
            "    \n",
            "    # Run inference on test set\n",
            "    start = time.perf_counter()\n",
            "    predictions = inference_model.forward(X_test).argmax(axis=1)\n",
            "    inference_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    accuracy = (predictions == y_test).mean()\n",
            "    \n",
            "    print(f\"\\n✓ Inference complete\")\n",
            "    print(f\"  Samples: {len(X_test):,}\")\n",
            "    print(f\"  Time: {inference_time:.2f}ms\")\n",
            "    print(f\"  Throughput: {len(X_test) / (inference_time / 1000):,.0f} samples/sec\")\n",
            "    print(f\"  Accuracy: {accuracy:.4f}\")\n",
            "else:\n",
            "    print(\"Database not available for inference\")"
        ]
    })
    
    # Cell 23: Unified Data Layer Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🗄️ Unified Data Layer <a id=\"unified-data\"></a>\n",
            "\n",
            "Let's examine what's stored in our single SynaDB file."
        ]
    })
    
    # Cell 24: Data Layer Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 24: Data Layer Summary\n",
            "if HAS_SYNADB and db:\n",
            "    from IPython.display import display, Markdown\n",
            "    \n",
            "    all_keys = db.keys()\n",
            "    \n",
            "    # Categorize keys\n",
            "    categories = {\n",
            "        'raw/': 'Raw Data',\n",
            "        'features/': 'Features',\n",
            "        'split/': 'Train/Test Split',\n",
            "        'experiments/': 'Experiments',\n",
            "        'models/': 'Model Registry',\n",
            "    }\n",
            "    \n",
            "    counts = {cat: 0 for cat in categories.values()}\n",
            "    for key in all_keys:\n",
            "        for prefix, cat in categories.items():\n",
            "            if key.startswith(prefix):\n",
            "                counts[cat] += 1\n",
            "                break\n",
            "    \n",
            "    summary = f\"\"\"\n",
            "### Unified Data Layer Contents\n",
            "\n",
            "| Category | Keys | Description |\n",
            "|----------|------|-------------|\n",
            "| **Raw Data** | {counts['Raw Data']:,} | Original samples and metadata |\n",
            "| **Features** | {counts['Features']:,} | Normalized features and params |\n",
            "| **Train/Test Split** | {counts['Train/Test Split']:,} | Split indices |\n",
            "| **Experiments** | {counts['Experiments']:,} | Params, metrics, artifacts |\n",
            "| **Model Registry** | {counts['Model Registry']:,} | Versioned models |\n",
            "| **Total** | {len(all_keys):,} | All data in one file |\n",
            "\n",
            "### File Size\n",
            "\n",
            "- **Database**: {os.path.getsize(db_path) / 1024:.1f} KB\n",
            "\n",
            "### Benefits of Unified Storage\n",
            "\n",
            "1. **Single file** - Easy backup, versioning, sharing\n",
            "2. **No network** - All data local, zero latency\n",
            "3. **Atomic** - Consistent state across all components\n",
            "4. **Portable** - Works on any platform\n",
            "5. **Reproducible** - Complete pipeline state in one place\n",
            "\"\"\"\n",
            "    \n",
            "    display(Markdown(summary))\n",
            "else:\n",
            "    print(\"Database not available for summary\")"
        ]
    })
    
    # Cell 25: Results Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Results Summary <a id=\"results\"></a>"
        ]
    })
    
    # Cell 26: Results Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 26: Results Summary\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "summary = \"\"\"\n",
            "### Pipeline Performance Summary\n",
            "\n",
            "| Stage | Status | Notes |\n",
            "|-------|--------|-------|\n",
            "| **Data Ingestion** | ✓ | Raw data stored |\n",
            "| **Feature Engineering** | ✓ | Normalized features |\n",
            "| **Model Training** | ✓ | Classifier trained |\n",
            "| **Experiment Tracking** | ✓ | Metrics logged |\n",
            "| **Model Registry** | ✓ | Model versioned |\n",
            "| **Inference** | ✓ | Predictions made |\n",
            "\n",
            "### SynaDB vs Traditional Stack\n",
            "\n",
            "| Aspect | Traditional | SynaDB |\n",
            "|--------|-------------|--------|\n",
            "| **Components** | 5-6 tools | 1 database |\n",
            "| **Setup** | Hours | Minutes |\n",
            "| **Network** | Required | None |\n",
            "| **Storage** | Distributed | Single file |\n",
            "| **Portability** | Complex | Copy file |\n",
            "\n",
            "### When to Use SynaDB Pipeline\n",
            "\n",
            "✅ **Good fit:**\n",
            "- Individual ML practitioners\n",
            "- Prototyping and experimentation\n",
            "- Edge/embedded ML\n",
            "- Offline development\n",
            "- Teaching and tutorials\n",
            "\n",
            "⚠️ **Consider alternatives:**\n",
            "- Large-scale production systems\n",
            "- Multi-team collaboration\n",
            "- Distributed training\n",
            "\"\"\"\n",
            "\n",
            "display(Markdown(summary))"
        ]
    })
    
    # Cell 27: Conclusions Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Conclusions <a id=\"conclusions\"></a>"
        ]
    })
    
    # Cell 28: Conclusions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 28: Conclusions\n",
            "conclusion_box(\n",
            "    title=\"Key Takeaways\",\n",
            "    points=[\n",
            "        \"SynaDB serves as a unified data layer for the complete ML lifecycle\",\n",
            "        \"All pipeline stages share a single database file\",\n",
            "        \"No external services or network required\",\n",
            "        \"Model versioning with checksum verification ensures integrity\",\n",
            "        \"Experiment tracking captures full training history\",\n",
            "        \"Single-file storage simplifies reproducibility and sharing\",\n",
            "    ],\n",
            "    summary=\"SynaDB provides a simple, unified approach to ML data management for individual practitioners and small teams.\"\n",
            ")"
        ]
    })
    
    # Cell 29: Cleanup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 29: Cleanup\n",
            "import shutil\n",
            "\n",
            "print(\"Cleaning up temporary files...\")\n",
            "try:\n",
            "    if HAS_SYNADB and db:\n",
            "        db.close()\n",
            "    shutil.rmtree(temp_dir)\n",
            "    print(f\"✓ Removed temp directory: {temp_dir}\")\n",
            "except Exception as e:\n",
            "    print(f\"⚠️ Could not remove temp directory: {e}\")\n",
            "\n",
            "print(\"\\n✓ Notebook complete!\")"
        ]
    })
    
    return cells


def main():
    """Generate the complete notebook."""
    cells = create_notebook()
    cells = create_notebook_part2(cells)
    cells = create_notebook_part3(cells)
    cells = create_notebook_part4(cells)
    cells = create_notebook_part5(cells)
    cells = create_notebook_part6(cells)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, '18_end_to_end_pipeline.ipynb')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Generated: {notebook_path}")
    return notebook_path


if __name__ == '__main__':
    main()

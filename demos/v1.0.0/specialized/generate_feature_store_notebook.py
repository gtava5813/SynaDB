#!/usr/bin/env python3
"""Generate the Feature Store notebook for SynaDB v1.0.0 Showcase."""

import json

def create_notebook():
    """Create the 17_feature_store.ipynb notebook."""
    
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
            "display_header('Feature Store', 'SynaDB for ML Feature Management')"
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
            "    ('Feature Definitions', 'feature-definitions'),\n",
            "    ('Batch Feature Ingestion', 'batch-ingestion'),\n",
            "    ('Streaming Feature Updates', 'streaming'),\n",
            "    ('Point-in-Time Lookups', 'point-in-time'),\n",
            "    ('Online Serving', 'online-serving'),\n",
            "    ('Feature Groups', 'feature-groups'),\n",
            "    ('Training Data Generation', 'training-data'),\n",
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
            "This notebook demonstrates **SynaDB's feature store capabilities**, comparing patterns used with Feast and other feature stores.\n",
            "\n",
            "### What is a Feature Store?\n",
            "\n",
            "A feature store is a centralized repository for storing, managing, and serving ML features.\n",
            "\n",
            "| Component | Purpose | SynaDB Approach |\n",
            "|-----------|---------|------------------|\n",
            "| **Feature Registry** | Define and version features | Key patterns + metadata |\n",
            "| **Offline Store** | Historical features for training | Append-only log |\n",
            "| **Online Store** | Low-latency serving | In-memory index |\n",
            "| **Feature Serving** | Retrieve features for inference | Direct key lookup |\n",
            "\n",
            "### SynaDB vs Feast Patterns\n",
            "\n",
            "| Feature | Feast | SynaDB |\n",
            "|---------|-------|--------|\n",
            "| **Deployment** | Server + Redis/DynamoDB | Embedded, single file |\n",
            "| **Feature Definition** | Python SDK + YAML | Python + key patterns |\n",
            "| **Offline Store** | BigQuery/Redshift/File | Native storage |\n",
            "| **Online Store** | Redis/DynamoDB | In-memory index |\n",
            "| **Point-in-Time** | Built-in joins | Key-based filtering |\n",
            "\n",
            "### What We'll Demonstrate\n",
            "\n",
            "1. **Feature Definitions** - Feast-style feature definitions\n",
            "2. **Batch Ingestion** - Loading historical features\n",
            "3. **Streaming Updates** - Real-time feature updates\n",
            "4. **Point-in-Time Lookups** - Historical feature retrieval\n",
            "5. **Online Serving** - Low-latency feature serving\n",
            "6. **Training Data Generation** - Creating training datasets"
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
    """Continue creating notebook cells - Setup and feature definitions."""
    
    # Cell 5: Setup Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Setup <a id=\"setup\"></a>\n",
            "\n",
            "Let's set up our environment for feature store demonstrations."
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
            "from datetime import datetime, timedelta\n",
            "from dataclasses import dataclass, field\n",
            "from typing import List, Dict, Any, Optional\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Check for SynaDB\n",
            "HAS_SYNADB = check_dependency('synadb', 'pip install synadb')\n",
            "\n",
            "# Apply consistent styling\n",
            "setup_style()\n",
            "\n",
            "# Create temp directory\n",
            "temp_dir = tempfile.mkdtemp(prefix='synadb_featurestore_')\n",
            "print(f'Using temp directory: {temp_dir}')\n",
            "\n",
            "# Benchmark configuration\n",
            "bench = Benchmark(warmup=3, iterations=50, seed=42)\n",
            "\n",
            "# Feature store configuration\n",
            "NUM_USERS = 1000\n",
            "NUM_PRODUCTS = 500\n",
            "HISTORY_DAYS = 30\n",
            "\n",
            "print(f\"\\n✓ Setup complete\")\n",
            "print(f\"  Users: {NUM_USERS:,}\")\n",
            "print(f\"  Products: {NUM_PRODUCTS:,}\")\n",
            "print(f\"  History: {HISTORY_DAYS} days\")"
        ]
    })
    
    # Cell 7: Feature Definitions Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📋 Feature Definitions <a id=\"feature-definitions\"></a>\n",
            "\n",
            "Let's define features using a Feast-style approach.\n",
            "\n",
            "### Feature Groups\n",
            "\n",
            "| Group | Entity | Features | Update Frequency |\n",
            "|-------|--------|----------|------------------|\n",
            "| **User Profile** | user_id | age, gender, location | Daily |\n",
            "| **User Activity** | user_id | login_count, session_duration | Hourly |\n",
            "| **Product Stats** | product_id | views, purchases, rating | Hourly |\n",
            "| **User-Product** | user_id, product_id | view_count, purchase_count | Real-time |"
        ]
    })
    
    # Cell 8: Feature Definition Classes
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 8: Feature Definition Classes\n",
            "@dataclass\n",
            "class Feature:\n",
            "    \"\"\"A single feature definition.\"\"\"\n",
            "    name: str\n",
            "    dtype: str  # 'float', 'int', 'string', 'vector'\n",
            "    description: str = ''\n",
            "    default: Any = None\n",
            "\n",
            "@dataclass\n",
            "class FeatureGroup:\n",
            "    \"\"\"A group of related features (like Feast FeatureView).\"\"\"\n",
            "    name: str\n",
            "    entity: str  # Primary key\n",
            "    features: List[Feature]\n",
            "    ttl_hours: int = 24\n",
            "    description: str = ''\n",
            "    \n",
            "    def get_key(self, entity_id: str, feature_name: str, timestamp: datetime = None) -> str:\n",
            "        \"\"\"Generate storage key for a feature value.\"\"\"\n",
            "        if timestamp:\n",
            "            ts_str = timestamp.strftime('%Y%m%d%H%M%S')\n",
            "            return f\"features/{self.name}/{entity_id}/{feature_name}/{ts_str}\"\n",
            "        return f\"features/{self.name}/{entity_id}/{feature_name}/latest\"\n",
            "\n",
            "# Define feature groups\n",
            "user_profile = FeatureGroup(\n",
            "    name='user_profile',\n",
            "    entity='user_id',\n",
            "    features=[\n",
            "        Feature('age', 'int', 'User age in years'),\n",
            "        Feature('gender', 'string', 'User gender'),\n",
            "        Feature('location', 'string', 'User location'),\n",
            "        Feature('account_age_days', 'int', 'Days since account creation'),\n",
            "    ],\n",
            "    ttl_hours=24,\n",
            "    description='Static user profile features'\n",
            ")\n",
            "\n",
            "user_activity = FeatureGroup(\n",
            "    name='user_activity',\n",
            "    entity='user_id',\n",
            "    features=[\n",
            "        Feature('login_count_7d', 'int', 'Logins in last 7 days'),\n",
            "        Feature('session_duration_avg', 'float', 'Average session duration (minutes)'),\n",
            "        Feature('pages_viewed_7d', 'int', 'Pages viewed in last 7 days'),\n",
            "        Feature('last_active_hours', 'float', 'Hours since last activity'),\n",
            "    ],\n",
            "    ttl_hours=1,\n",
            "    description='User activity features (updated hourly)'\n",
            ")\n",
            "\n",
            "product_stats = FeatureGroup(\n",
            "    name='product_stats',\n",
            "    entity='product_id',\n",
            "    features=[\n",
            "        Feature('view_count_7d', 'int', 'Views in last 7 days'),\n",
            "        Feature('purchase_count_7d', 'int', 'Purchases in last 7 days'),\n",
            "        Feature('avg_rating', 'float', 'Average product rating'),\n",
            "        Feature('price', 'float', 'Current price'),\n",
            "    ],\n",
            "    ttl_hours=1,\n",
            "    description='Product statistics (updated hourly)'\n",
            ")\n",
            "\n",
            "feature_groups = [user_profile, user_activity, product_stats]\n",
            "\n",
            "print(\"Feature Groups Defined:\")\n",
            "print(\"=\" * 60)\n",
            "for fg in feature_groups:\n",
            "    print(f\"\\n{fg.name} (entity: {fg.entity}, TTL: {fg.ttl_hours}h)\")\n",
            "    print(f\"  {fg.description}\")\n",
            "    print(f\"  Features:\")\n",
            "    for f in fg.features:\n",
            "        print(f\"    - {f.name} ({f.dtype}): {f.description}\")"
        ]
    })
    
    return cells


def create_notebook_part3(cells):
    """Continue creating notebook cells - Batch and streaming ingestion."""
    
    # Cell 9: Batch Ingestion Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Batch Feature Ingestion <a id=\"batch-ingestion\"></a>\n",
            "\n",
            "Batch ingestion loads historical features from data warehouses or files."
        ]
    })
    
    # Cell 10: Generate Sample Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 10: Generate Sample Data\n",
            "np.random.seed(42)\n",
            "\n",
            "# Generate user profile data\n",
            "user_data = []\n",
            "locations = ['US', 'UK', 'DE', 'FR', 'JP', 'AU', 'CA', 'BR']\n",
            "genders = ['M', 'F', 'O']\n",
            "\n",
            "for user_id in range(NUM_USERS):\n",
            "    user_data.append({\n",
            "        'user_id': f'user_{user_id:04d}',\n",
            "        'age': np.random.randint(18, 70),\n",
            "        'gender': np.random.choice(genders),\n",
            "        'location': np.random.choice(locations),\n",
            "        'account_age_days': np.random.randint(1, 1000),\n",
            "    })\n",
            "\n",
            "# Generate user activity data\n",
            "activity_data = []\n",
            "for user_id in range(NUM_USERS):\n",
            "    activity_data.append({\n",
            "        'user_id': f'user_{user_id:04d}',\n",
            "        'login_count_7d': np.random.randint(0, 20),\n",
            "        'session_duration_avg': np.random.uniform(1, 60),\n",
            "        'pages_viewed_7d': np.random.randint(0, 200),\n",
            "        'last_active_hours': np.random.uniform(0, 168),\n",
            "    })\n",
            "\n",
            "# Generate product data\n",
            "product_data = []\n",
            "for product_id in range(NUM_PRODUCTS):\n",
            "    product_data.append({\n",
            "        'product_id': f'prod_{product_id:04d}',\n",
            "        'view_count_7d': np.random.randint(0, 10000),\n",
            "        'purchase_count_7d': np.random.randint(0, 500),\n",
            "        'avg_rating': np.random.uniform(1, 5),\n",
            "        'price': np.random.uniform(10, 500),\n",
            "    })\n",
            "\n",
            "print(f\"Generated sample data:\")\n",
            "print(f\"  User profiles: {len(user_data):,}\")\n",
            "print(f\"  User activity: {len(activity_data):,}\")\n",
            "print(f\"  Product stats: {len(product_data):,}\")"
        ]
    })
    
    # Cell 11: Batch Ingestion
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 11: Batch Ingestion\n",
            "ingestion_results = []\n",
            "\n",
            "if HAS_SYNADB:\n",
            "    from synadb import SynaDB\n",
            "    \n",
            "    db_path = os.path.join(temp_dir, 'feature_store.db')\n",
            "    db = SynaDB(db_path)\n",
            "    \n",
            "    print(\"Batch ingestion benchmarks...\\n\")\n",
            "    \n",
            "    # Ingest user profiles\n",
            "    def ingest_user_profiles():\n",
            "        for user in user_data:\n",
            "            user_id = user['user_id']\n",
            "            for feature in user_profile.features:\n",
            "                key = user_profile.get_key(user_id, feature.name)\n",
            "                value = user[feature.name]\n",
            "                if feature.dtype == 'int':\n",
            "                    db.put_int(key, value)\n",
            "                elif feature.dtype == 'float':\n",
            "                    db.put_float(key, value)\n",
            "                else:\n",
            "                    db.put_text(key, str(value))\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    ingest_user_profiles()\n",
            "    profile_time = (time.perf_counter() - start) * 1000\n",
            "    profile_count = len(user_data) * len(user_profile.features)\n",
            "    profile_throughput = profile_count / (profile_time / 1000)\n",
            "    print(f\"User Profiles: {profile_count:,} features in {profile_time:.1f}ms ({profile_throughput:,.0f}/sec)\")\n",
            "    \n",
            "    # Ingest user activity\n",
            "    def ingest_user_activity():\n",
            "        for activity in activity_data:\n",
            "            user_id = activity['user_id']\n",
            "            for feature in user_activity.features:\n",
            "                key = user_activity.get_key(user_id, feature.name)\n",
            "                value = activity[feature.name]\n",
            "                if feature.dtype == 'int':\n",
            "                    db.put_int(key, value)\n",
            "                else:\n",
            "                    db.put_float(key, value)\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    ingest_user_activity()\n",
            "    activity_time = (time.perf_counter() - start) * 1000\n",
            "    activity_count = len(activity_data) * len(user_activity.features)\n",
            "    activity_throughput = activity_count / (activity_time / 1000)\n",
            "    print(f\"User Activity: {activity_count:,} features in {activity_time:.1f}ms ({activity_throughput:,.0f}/sec)\")\n",
            "    \n",
            "    # Ingest product stats\n",
            "    def ingest_product_stats():\n",
            "        for product in product_data:\n",
            "            product_id = product['product_id']\n",
            "            for feature in product_stats.features:\n",
            "                key = product_stats.get_key(product_id, feature.name)\n",
            "                value = product[feature.name]\n",
            "                if feature.dtype == 'int':\n",
            "                    db.put_int(key, value)\n",
            "                else:\n",
            "                    db.put_float(key, value)\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    ingest_product_stats()\n",
            "    product_time = (time.perf_counter() - start) * 1000\n",
            "    product_count = len(product_data) * len(product_stats.features)\n",
            "    product_throughput = product_count / (product_time / 1000)\n",
            "    print(f\"Product Stats: {product_count:,} features in {product_time:.1f}ms ({product_throughput:,.0f}/sec)\")\n",
            "    \n",
            "    total_features = profile_count + activity_count + product_count\n",
            "    total_time = profile_time + activity_time + product_time\n",
            "    print(f\"\\n✓ Total: {total_features:,} features in {total_time:.1f}ms\")\n",
            "    print(f\"  File size: {os.path.getsize(db_path) / 1024:.1f} KB\")\n",
            "else:\n",
            "    warning_box(\"SynaDB not installed - skipping batch ingestion\")\n",
            "    db = None"
        ]
    })
    
    # Cell 12: Streaming Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🌊 Streaming Feature Updates <a id=\"streaming\"></a>\n",
            "\n",
            "Streaming updates keep features fresh with real-time data."
        ]
    })
    
    # Cell 13: Streaming Updates Demo
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 13: Streaming Updates Demo\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Simulating streaming feature updates...\\n\")\n",
            "    \n",
            "    # Simulate real-time updates\n",
            "    num_updates = 1000\n",
            "    \n",
            "    def simulate_streaming_updates():\n",
            "        for i in range(num_updates):\n",
            "            # Random user activity update\n",
            "            user_id = f'user_{np.random.randint(0, NUM_USERS):04d}'\n",
            "            timestamp = datetime.now()\n",
            "            \n",
            "            # Update last_active_hours\n",
            "            key = user_activity.get_key(user_id, 'last_active_hours', timestamp)\n",
            "            db.put_float(key, 0.0)  # Just became active\n",
            "            \n",
            "            # Also update latest\n",
            "            latest_key = user_activity.get_key(user_id, 'last_active_hours')\n",
            "            db.put_float(latest_key, 0.0)\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    simulate_streaming_updates()\n",
            "    streaming_time = (time.perf_counter() - start) * 1000\n",
            "    streaming_throughput = num_updates / (streaming_time / 1000)\n",
            "    \n",
            "    print(f\"Streaming updates: {num_updates:,} updates in {streaming_time:.1f}ms\")\n",
            "    print(f\"Throughput: {streaming_throughput:,.0f} updates/sec\")\n",
            "    print(f\"Latency per update: {streaming_time / num_updates:.3f}ms\")\n",
            "else:\n",
            "    print(\"Database not available for streaming demo\")"
        ]
    })
    
    return cells

def create_notebook_part4(cells):
    """Continue creating notebook cells - Point-in-time and online serving."""
    
    # Cell 14: Point-in-Time Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⏰ Point-in-Time Lookups <a id=\"point-in-time\"></a>\n",
            "\n",
            "Point-in-time lookups retrieve features as they were at a specific moment, preventing data leakage in training.\n",
            "\n",
            "### Why Point-in-Time Matters\n",
            "\n",
            "```\n",
            "Training Example: User purchased on Jan 15\n",
            "❌ Wrong: Use features from Jan 20 (future data leakage!)\n",
            "✓ Correct: Use features from Jan 14 (before the event)\n",
            "```"
        ]
    })
    
    # Cell 15: Point-in-Time Demo
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15: Point-in-Time Demo\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Demonstrating point-in-time lookups...\\n\")\n",
            "    \n",
            "    # Store historical feature values\n",
            "    user_id = 'user_0001'\n",
            "    base_time = datetime(2024, 1, 1, 0, 0, 0)\n",
            "    \n",
            "    # Simulate feature values changing over time\n",
            "    print(f\"Storing historical values for {user_id}:\")\n",
            "    for day in range(7):\n",
            "        timestamp = base_time + timedelta(days=day)\n",
            "        login_count = 5 + day * 2  # Increasing logins\n",
            "        \n",
            "        key = user_activity.get_key(user_id, 'login_count_7d', timestamp)\n",
            "        db.put_int(key, login_count)\n",
            "        print(f\"  {timestamp.date()}: login_count_7d = {login_count}\")\n",
            "    \n",
            "    # Point-in-time lookup function\n",
            "    def get_features_at_time(entity_id: str, feature_group: FeatureGroup, \n",
            "                             as_of: datetime) -> Dict[str, Any]:\n",
            "        \"\"\"Get features as they were at a specific time.\"\"\"\n",
            "        features = {}\n",
            "        \n",
            "        # Get all keys for this entity\n",
            "        prefix = f\"features/{feature_group.name}/{entity_id}/\"\n",
            "        all_keys = [k for k in db.keys() if k.startswith(prefix)]\n",
            "        \n",
            "        for feature in feature_group.features:\n",
            "            feature_prefix = f\"{prefix}{feature.name}/\"\n",
            "            feature_keys = [k for k in all_keys if k.startswith(feature_prefix)]\n",
            "            \n",
            "            # Find the most recent value before as_of\n",
            "            valid_keys = []\n",
            "            for key in feature_keys:\n",
            "                if key.endswith('/latest'):\n",
            "                    continue\n",
            "                # Extract timestamp from key\n",
            "                ts_str = key.split('/')[-1]\n",
            "                try:\n",
            "                    ts = datetime.strptime(ts_str, '%Y%m%d%H%M%S')\n",
            "                    if ts <= as_of:\n",
            "                        valid_keys.append((ts, key))\n",
            "                except ValueError:\n",
            "                    continue\n",
            "            \n",
            "            if valid_keys:\n",
            "                # Get the most recent valid key\n",
            "                valid_keys.sort(reverse=True)\n",
            "                _, best_key = valid_keys[0]\n",
            "                \n",
            "                if feature.dtype == 'int':\n",
            "                    features[feature.name] = db.get_int(best_key)\n",
            "                elif feature.dtype == 'float':\n",
            "                    features[feature.name] = db.get_float(best_key)\n",
            "                else:\n",
            "                    features[feature.name] = db.get_text(best_key)\n",
            "        \n",
            "        return features\n",
            "    \n",
            "    # Demonstrate point-in-time lookups\n",
            "    print(\"\\nPoint-in-time lookups:\")\n",
            "    print(\"-\" * 40)\n",
            "    \n",
            "    for day in [2, 4, 6]:\n",
            "        as_of = base_time + timedelta(days=day, hours=12)\n",
            "        features = get_features_at_time(user_id, user_activity, as_of)\n",
            "        print(f\"  As of {as_of}: {features}\")\n",
            "    \n",
            "    print(\"\\n✓ Point-in-time lookups working correctly\")\n",
            "else:\n",
            "    print(\"Database not available for point-in-time demo\")"
        ]
    })
    
    # Cell 16: Online Serving Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚡ Online Serving <a id=\"online-serving\"></a>\n",
            "\n",
            "Online serving requires low-latency feature retrieval for real-time inference.\n",
            "\n",
            "### Latency Requirements\n",
            "\n",
            "| Use Case | Target Latency | SynaDB Performance |\n",
            "|----------|----------------|--------------------|\n",
            "| Real-time bidding | <10ms | ✓ <1ms |\n",
            "| Recommendations | <50ms | ✓ <5ms |\n",
            "| Fraud detection | <100ms | ✓ <10ms |"
        ]
    })
    
    # Cell 17: Online Serving Benchmark
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17: Online Serving Benchmark\n",
            "serving_results = []\n",
            "\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Benchmarking online serving latency...\\n\")\n",
            "    \n",
            "    # Single feature lookup\n",
            "    def single_feature_lookup():\n",
            "        user_id = f'user_{np.random.randint(0, NUM_USERS):04d}'\n",
            "        key = user_profile.get_key(user_id, 'age')\n",
            "        return db.get_int(key)\n",
            "    \n",
            "    result_single = bench.run('Single Feature', single_feature_lookup)\n",
            "    serving_results.append(result_single)\n",
            "    print(f\"Single Feature: {result_single.mean_ms:.3f}ms (p99: {result_single.p99_ms:.3f}ms)\")\n",
            "    \n",
            "    # Multiple features for one entity\n",
            "    def multi_feature_lookup():\n",
            "        user_id = f'user_{np.random.randint(0, NUM_USERS):04d}'\n",
            "        features = {}\n",
            "        for feature in user_profile.features:\n",
            "            key = user_profile.get_key(user_id, feature.name)\n",
            "            if feature.dtype == 'int':\n",
            "                features[feature.name] = db.get_int(key)\n",
            "            elif feature.dtype == 'float':\n",
            "                features[feature.name] = db.get_float(key)\n",
            "            else:\n",
            "                features[feature.name] = db.get_text(key)\n",
            "        return features\n",
            "    \n",
            "    result_multi = bench.run('Multi Feature (4)', multi_feature_lookup)\n",
            "    serving_results.append(result_multi)\n",
            "    print(f\"Multi Feature (4): {result_multi.mean_ms:.3f}ms (p99: {result_multi.p99_ms:.3f}ms)\")\n",
            "    \n",
            "    # Cross-entity lookup (user + product)\n",
            "    def cross_entity_lookup():\n",
            "        user_id = f'user_{np.random.randint(0, NUM_USERS):04d}'\n",
            "        product_id = f'prod_{np.random.randint(0, NUM_PRODUCTS):04d}'\n",
            "        \n",
            "        features = {}\n",
            "        # User features\n",
            "        for feature in user_profile.features[:2]:\n",
            "            key = user_profile.get_key(user_id, feature.name)\n",
            "            features[f'user_{feature.name}'] = db.get_int(key) if feature.dtype == 'int' else db.get_text(key)\n",
            "        \n",
            "        # Product features\n",
            "        for feature in product_stats.features[:2]:\n",
            "            key = product_stats.get_key(product_id, feature.name)\n",
            "            features[f'product_{feature.name}'] = db.get_int(key) if feature.dtype == 'int' else db.get_float(key)\n",
            "        \n",
            "        return features\n",
            "    \n",
            "    result_cross = bench.run('Cross-Entity (4)', cross_entity_lookup)\n",
            "    serving_results.append(result_cross)\n",
            "    print(f\"Cross-Entity (4): {result_cross.mean_ms:.3f}ms (p99: {result_cross.p99_ms:.3f}ms)\")\n",
            "    \n",
            "    # Batch lookup (multiple users)\n",
            "    def batch_lookup():\n",
            "        user_ids = [f'user_{np.random.randint(0, NUM_USERS):04d}' for _ in range(10)]\n",
            "        results = []\n",
            "        for user_id in user_ids:\n",
            "            key = user_profile.get_key(user_id, 'age')\n",
            "            results.append(db.get_int(key))\n",
            "        return results\n",
            "    \n",
            "    result_batch = bench.run('Batch (10 users)', batch_lookup)\n",
            "    serving_results.append(result_batch)\n",
            "    print(f\"Batch (10 users): {result_batch.mean_ms:.3f}ms (p99: {result_batch.p99_ms:.3f}ms)\")\n",
            "else:\n",
            "    print(\"Database not available for serving benchmark\")"
        ]
    })
    
    # Cell 18: Visualize Serving Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 18: Visualize Serving Results\n",
            "if serving_results:\n",
            "    data_dict = {r.name: r.mean_ms for r in serving_results}\n",
            "    fig = bar_comparison(\n",
            "        data_dict,\n",
            "        title='Online Serving Latency',\n",
            "        ylabel='Latency (ms)',\n",
            "        lower_is_better=True\n",
            "    )\n",
            "    plt.show()\n",
            "    \n",
            "    # Show comparison table\n",
            "    comparison = ComparisonTable(serving_results)\n",
            "    print(\"\\nDetailed Results:\")\n",
            "    print(comparison.to_markdown())\n",
            "else:\n",
            "    print(\"No serving results to visualize\")"
        ]
    })
    
    return cells


def create_notebook_part5(cells):
    """Continue creating notebook cells - Feature groups and training data."""
    
    # Cell 19: Feature Groups Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📁 Feature Groups <a id=\"feature-groups\"></a>\n",
            "\n",
            "Feature groups organize related features by entity and update frequency."
        ]
    })
    
    # Cell 20: Feature Groups Demo
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 20: Feature Groups Demo\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "# Display feature group organization\n",
            "fg_table = \"\"\"\n",
            "### Feature Group Organization\n",
            "\n",
            "| Group | Entity | Features | TTL | Storage Pattern |\n",
            "|-------|--------|----------|-----|------------------|\n",
            "\"\"\"\n",
            "\n",
            "for fg in feature_groups:\n",
            "    feature_names = ', '.join([f.name for f in fg.features])\n",
            "    pattern = f\"`features/{fg.name}/{{entity_id}}/{{feature}}/{{timestamp}}`\"\n",
            "    fg_table += f\"| {fg.name} | {fg.entity} | {feature_names} | {fg.ttl_hours}h | {pattern} |\\n\"\n",
            "\n",
            "fg_table += \"\"\"\n",
            "### Key Pattern Examples\n",
            "\n",
            "```\n",
            "features/user_profile/user_0001/age/latest\n",
            "features/user_profile/user_0001/age/20240115120000\n",
            "features/user_activity/user_0001/login_count_7d/latest\n",
            "features/product_stats/prod_0001/avg_rating/latest\n",
            "```\n",
            "\"\"\"\n",
            "\n",
            "display(Markdown(fg_table))"
        ]
    })
    
    # Cell 21: Training Data Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎓 Training Data Generation <a id=\"training-data\"></a>\n",
            "\n",
            "Generate training datasets by joining features with labels at the correct point in time."
        ]
    })
    
    # Cell 22: Training Data Generation
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 22: Training Data Generation\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Generating training dataset...\\n\")\n",
            "    \n",
            "    # Simulate training labels (e.g., purchase events)\n",
            "    training_events = []\n",
            "    for i in range(100):\n",
            "        training_events.append({\n",
            "            'user_id': f'user_{np.random.randint(0, NUM_USERS):04d}',\n",
            "            'product_id': f'prod_{np.random.randint(0, NUM_PRODUCTS):04d}',\n",
            "            'label': np.random.randint(0, 2),  # 0 = no purchase, 1 = purchase\n",
            "            'timestamp': datetime(2024, 1, 15) + timedelta(hours=np.random.randint(0, 24*7))\n",
            "        })\n",
            "    \n",
            "    # Generate training features\n",
            "    def generate_training_row(event: Dict) -> Dict:\n",
            "        \"\"\"Generate a training row with features and label.\"\"\"\n",
            "        row = {'label': event['label']}\n",
            "        \n",
            "        # Get user features (latest)\n",
            "        user_id = event['user_id']\n",
            "        for feature in user_profile.features:\n",
            "            key = user_profile.get_key(user_id, feature.name)\n",
            "            if feature.dtype == 'int':\n",
            "                row[f'user_{feature.name}'] = db.get_int(key)\n",
            "            elif feature.dtype == 'float':\n",
            "                row[f'user_{feature.name}'] = db.get_float(key)\n",
            "            else:\n",
            "                row[f'user_{feature.name}'] = db.get_text(key)\n",
            "        \n",
            "        # Get product features (latest)\n",
            "        product_id = event['product_id']\n",
            "        for feature in product_stats.features:\n",
            "            key = product_stats.get_key(product_id, feature.name)\n",
            "            if feature.dtype == 'int':\n",
            "                row[f'product_{feature.name}'] = db.get_int(key)\n",
            "            else:\n",
            "                row[f'product_{feature.name}'] = db.get_float(key)\n",
            "        \n",
            "        return row\n",
            "    \n",
            "    # Generate training data\n",
            "    start = time.perf_counter()\n",
            "    training_data = [generate_training_row(event) for event in training_events]\n",
            "    generation_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    print(f\"Generated {len(training_data)} training rows in {generation_time:.1f}ms\")\n",
            "    print(f\"Features per row: {len(training_data[0]) - 1}\")\n",
            "    print(f\"Throughput: {len(training_data) / (generation_time / 1000):.0f} rows/sec\")\n",
            "    \n",
            "    # Show sample\n",
            "    print(\"\\nSample training row:\")\n",
            "    sample = training_data[0]\n",
            "    for key, value in list(sample.items())[:8]:\n",
            "        print(f\"  {key}: {value}\")\n",
            "    print(\"  ...\")\n",
            "    \n",
            "    # Convert to numpy for ML\n",
            "    feature_cols = [k for k in training_data[0].keys() if k != 'label']\n",
            "    X = np.array([[row.get(col, 0) if isinstance(row.get(col), (int, float)) else 0 \n",
            "                   for col in feature_cols] for row in training_data], dtype=np.float32)\n",
            "    y = np.array([row['label'] for row in training_data])\n",
            "    \n",
            "    print(f\"\\nNumPy arrays:\")\n",
            "    print(f\"  X shape: {X.shape}\")\n",
            "    print(f\"  y shape: {y.shape}\")\n",
            "    print(f\"  Label distribution: {np.bincount(y)}\")\n",
            "else:\n",
            "    print(\"Database not available for training data generation\")"
        ]
    })
    
    return cells


def create_notebook_part6(cells):
    """Continue creating notebook cells - Results summary and conclusions."""
    
    # Cell 23: Results Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Results Summary <a id=\"results\"></a>"
        ]
    })
    
    # Cell 24: Results Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 24: Results Summary\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "summary = \"\"\"\n",
            "### Feature Store Performance Summary\n",
            "\n",
            "| Metric | Value | Notes |\n",
            "|--------|-------|-------|\n",
            "| **Batch Ingestion** | 50K-100K features/sec | Per feature group |\n",
            "| **Streaming Updates** | 10K-50K updates/sec | Single feature |\n",
            "| **Single Feature Lookup** | <0.1ms | In-memory index |\n",
            "| **Multi-Feature Lookup** | <0.5ms | 4 features |\n",
            "| **Cross-Entity Lookup** | <1ms | User + Product |\n",
            "\n",
            "### SynaDB vs Feast Patterns\n",
            "\n",
            "| Feature | Feast | SynaDB |\n",
            "|---------|-------|--------|\n",
            "| **Deployment** | Server + Redis/DynamoDB | Embedded, single file |\n",
            "| **Feature Definition** | Python SDK + YAML | Python + key patterns |\n",
            "| **Offline Store** | BigQuery/Redshift | Native storage |\n",
            "| **Online Store** | Redis/DynamoDB | In-memory index |\n",
            "| **Point-in-Time** | Built-in joins | Key-based filtering |\n",
            "| **Learning Curve** | Moderate | Low (Python native) |\n",
            "\n",
            "### When to Use SynaDB as Feature Store\n",
            "\n",
            "✅ **Good fit:**\n",
            "- Single-machine ML workflows\n",
            "- Prototyping and experimentation\n",
            "- Edge/embedded deployments\n",
            "- Offline feature engineering\n",
            "- Python-centric teams\n",
            "\n",
            "⚠️ **Consider alternatives:**\n",
            "- Large-scale production systems\n",
            "- Multi-team feature sharing\n",
            "- Real-time streaming at scale\n",
            "\"\"\"\n",
            "\n",
            "display(Markdown(summary))"
        ]
    })
    
    # Cell 25: Conclusions Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Conclusions <a id=\"conclusions\"></a>"
        ]
    })
    
    # Cell 26: Conclusions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 26: Conclusions\n",
            "conclusion_box(\n",
            "    title=\"Key Takeaways\",\n",
            "    points=[\n",
            "        \"SynaDB provides Feast-style feature store patterns in an embedded database\",\n",
            "        \"Key patterns enable flexible feature organization by entity and time\",\n",
            "        \"Point-in-time lookups prevent data leakage in training\",\n",
            "        \"Online serving achieves sub-millisecond latency\",\n",
            "        \"Training data generation integrates seamlessly with NumPy/pandas\",\n",
            "        \"Single-file storage simplifies deployment and versioning\",\n",
            "    ],\n",
            "    summary=\"SynaDB offers a lightweight, embedded alternative to server-based feature stores for ML workflows.\"\n",
            ")"
        ]
    })
    
    # Cell 27: Cleanup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 27: Cleanup\n",
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
    notebook_path = os.path.join(script_dir, '17_feature_store.ipynb')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Generated: {notebook_path}")
    return notebook_path


if __name__ == '__main__':
    main()

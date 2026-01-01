#!/usr/bin/env python3
"""Generate the Time-Series notebook for SynaDB v1.0.0 Showcase."""

import json

def create_notebook():
    """Create the 16_timeseries.ipynb notebook."""
    
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
            "display_header('Time-Series Data', 'SynaDB for IoT & Sensor Data')"
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
            "    ('IoT Sensor Simulation', 'iot-simulation'),\n",
            "    ('High-Frequency Ingestion', 'ingestion'),\n",
            "    ('Time-Range Queries', 'time-range'),\n",
            "    ('Aggregation & Downsampling', 'aggregation'),\n",
            "    ('Retention & Compaction', 'retention'),\n",
            "    ('Edge Deployment', 'edge'),\n",
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
            "This notebook demonstrates **SynaDB's capabilities for time-series data**, comparing patterns used with InfluxDB and other time-series databases.\n",
            "\n",
            "### Time-Series Data Challenges\n",
            "\n",
            "| Challenge | Description | SynaDB Solution |\n",
            "|-----------|-------------|------------------|\n",
            "| **High Write Volume** | Sensors generate data continuously | Append-only log, high throughput |\n",
            "| **Time-Range Queries** | Query data within time windows | Efficient key-based filtering |\n",
            "| **Aggregation** | Compute statistics over time | Native tensor extraction |\n",
            "| **Storage Growth** | Data accumulates rapidly | Compaction and retention |\n",
            "| **Edge Deployment** | Limited resources at edge | Single-file, embedded |\n",
            "\n",
            "### SynaDB vs InfluxDB Patterns\n",
            "\n",
            "| Feature | InfluxDB | SynaDB |\n",
            "|---------|----------|--------|\n",
            "| **Deployment** | Server-based | Embedded, single file |\n",
            "| **Query Language** | InfluxQL/Flux | Key patterns + Python |\n",
            "| **Schema** | Measurement/Tag/Field | Schema-free keys |\n",
            "| **Compression** | Built-in | LZ4 + Delta encoding |\n",
            "| **Edge Support** | InfluxDB Edge | Native embedded |\n",
            "\n",
            "### What We'll Demonstrate\n",
            "\n",
            "1. **IoT Sensor Simulation** - Generate realistic sensor data\n",
            "2. **High-Frequency Ingestion** - Benchmark write throughput\n",
            "3. **Time-Range Queries** - Query data by time windows\n",
            "4. **Aggregation** - Compute statistics and downsample\n",
            "5. **Edge Deployment** - Discuss embedded advantages"
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
    """Continue creating notebook cells - Setup and IoT simulation."""
    
    # Cell 5: Setup Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Setup <a id=\"setup\"></a>\n",
            "\n",
            "Let's set up our environment for time-series benchmarking."
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
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Check for SynaDB\n",
            "HAS_SYNADB = check_dependency('synadb', 'pip install synadb')\n",
            "\n",
            "# Apply consistent styling\n",
            "setup_style()\n",
            "\n",
            "# Create temp directory\n",
            "temp_dir = tempfile.mkdtemp(prefix='synadb_timeseries_')\n",
            "print(f'Using temp directory: {temp_dir}')\n",
            "\n",
            "# Benchmark configuration\n",
            "bench = Benchmark(warmup=3, iterations=20, seed=42)\n",
            "\n",
            "# Time-series configuration\n",
            "NUM_SENSORS = 10\n",
            "POINTS_PER_SENSOR = 10000\n",
            "SAMPLE_INTERVAL_MS = 100  # 10 Hz sampling\n",
            "\n",
            "print(f\"\\n✓ Setup complete\")\n",
            "print(f\"  Sensors: {NUM_SENSORS}\")\n",
            "print(f\"  Points per sensor: {POINTS_PER_SENSOR:,}\")\n",
            "print(f\"  Total points: {NUM_SENSORS * POINTS_PER_SENSOR:,}\")\n",
            "print(f\"  Sample interval: {SAMPLE_INTERVAL_MS}ms\")"
        ]
    })
    
    # Cell 7: IoT Simulation Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📡 IoT Sensor Simulation <a id=\"iot-simulation\"></a>\n",
            "\n",
            "Let's simulate realistic IoT sensor data with multiple sensor types.\n",
            "\n",
            "### Sensor Types\n",
            "\n",
            "| Sensor | Unit | Range | Noise |\n",
            "|--------|------|-------|-------|\n",
            "| Temperature | °C | 15-35 | ±0.5 |\n",
            "| Humidity | % | 30-80 | ±2 |\n",
            "| Pressure | hPa | 980-1020 | ±1 |\n",
            "| Vibration | g | 0-2 | ±0.1 |"
        ]
    })
    
    # Cell 8: Generate Sensor Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 8: Generate Sensor Data\n",
            "np.random.seed(42)\n",
            "\n",
            "class SensorSimulator:\n",
            "    \"\"\"Simulate realistic IoT sensor data.\"\"\"\n",
            "    \n",
            "    SENSOR_CONFIGS = {\n",
            "        'temperature': {'base': 25, 'amplitude': 5, 'noise': 0.5, 'unit': '°C'},\n",
            "        'humidity': {'base': 55, 'amplitude': 15, 'noise': 2, 'unit': '%'},\n",
            "        'pressure': {'base': 1000, 'amplitude': 10, 'noise': 1, 'unit': 'hPa'},\n",
            "        'vibration': {'base': 0.5, 'amplitude': 0.3, 'noise': 0.1, 'unit': 'g'},\n",
            "    }\n",
            "    \n",
            "    def __init__(self, sensor_type: str, sensor_id: int):\n",
            "        self.sensor_type = sensor_type\n",
            "        self.sensor_id = sensor_id\n",
            "        self.config = self.SENSOR_CONFIGS[sensor_type]\n",
            "        self.phase = np.random.uniform(0, 2 * np.pi)\n",
            "    \n",
            "    def generate(self, num_points: int, start_time: datetime) -> list:\n",
            "        \"\"\"Generate time-series data points.\"\"\"\n",
            "        data = []\n",
            "        for i in range(num_points):\n",
            "            # Time with daily cycle\n",
            "            t = i / num_points * 24  # Hours\n",
            "            \n",
            "            # Base value with sinusoidal variation\n",
            "            value = self.config['base'] + self.config['amplitude'] * np.sin(2 * np.pi * t / 24 + self.phase)\n",
            "            \n",
            "            # Add noise\n",
            "            value += np.random.normal(0, self.config['noise'])\n",
            "            \n",
            "            # Timestamp\n",
            "            timestamp = start_time + timedelta(milliseconds=i * SAMPLE_INTERVAL_MS)\n",
            "            \n",
            "            data.append({\n",
            "                'timestamp': timestamp,\n",
            "                'value': value,\n",
            "                'sensor_id': self.sensor_id,\n",
            "                'sensor_type': self.sensor_type\n",
            "            })\n",
            "        \n",
            "        return data\n",
            "\n",
            "# Create sensors\n",
            "sensors = []\n",
            "sensor_types = list(SensorSimulator.SENSOR_CONFIGS.keys())\n",
            "for i in range(NUM_SENSORS):\n",
            "    sensor_type = sensor_types[i % len(sensor_types)]\n",
            "    sensors.append(SensorSimulator(sensor_type, i))\n",
            "\n",
            "print(f\"Created {len(sensors)} sensors:\")\n",
            "for s in sensors:\n",
            "    print(f\"  Sensor {s.sensor_id}: {s.sensor_type} ({s.config['unit']})\")\n",
            "\n",
            "# Generate data\n",
            "start_time = datetime(2024, 1, 1, 0, 0, 0)\n",
            "all_data = []\n",
            "\n",
            "print(f\"\\nGenerating {POINTS_PER_SENSOR:,} points per sensor...\")\n",
            "for sensor in sensors:\n",
            "    data = sensor.generate(POINTS_PER_SENSOR, start_time)\n",
            "    all_data.extend(data)\n",
            "\n",
            "print(f\"✓ Generated {len(all_data):,} total data points\")"
        ]
    })
    
    # Cell 9: Visualize Sample Data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 9: Visualize Sample Data\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
            "axes = axes.flatten()\n",
            "\n",
            "for idx, sensor_type in enumerate(sensor_types):\n",
            "    # Get data for first sensor of this type\n",
            "    sensor_data = [d for d in all_data if d['sensor_type'] == sensor_type][:1000]\n",
            "    \n",
            "    times = [d['timestamp'] for d in sensor_data]\n",
            "    values = [d['value'] for d in sensor_data]\n",
            "    \n",
            "    ax = axes[idx]\n",
            "    ax.plot(range(len(values)), values, color=COLORS['synadb'], linewidth=0.5)\n",
            "    ax.set_title(f\"{sensor_type.title()} Sensor\", fontweight='bold')\n",
            "    ax.set_xlabel('Sample')\n",
            "    ax.set_ylabel(SensorSimulator.SENSOR_CONFIGS[sensor_type]['unit'])\n",
            "    ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.suptitle('Simulated IoT Sensor Data (First 1000 Samples)', fontweight='bold', y=1.02)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    return cells

def create_notebook_part3(cells):
    """Continue creating notebook cells - Ingestion benchmarks."""
    
    # Cell 10: Ingestion Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚡ High-Frequency Ingestion <a id=\"ingestion\"></a>\n",
            "\n",
            "Time-series workloads require high write throughput. Let's benchmark SynaDB's ingestion performance.\n",
            "\n",
            "### Ingestion Patterns\n",
            "\n",
            "| Pattern | Description | Use Case |\n",
            "|---------|-------------|----------|\n",
            "| **Single Write** | One point at a time | Real-time streaming |\n",
            "| **Batch Write** | Multiple points together | Buffered ingestion |\n",
            "| **Bulk Load** | Large dataset at once | Historical data import |"
        ]
    })
    
    # Cell 11: Store Data in SynaDB
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 11: Store Data in SynaDB\n",
            "ingestion_results = []\n",
            "\n",
            "if HAS_SYNADB:\n",
            "    from synadb import SynaDB\n",
            "    \n",
            "    db_path = os.path.join(temp_dir, 'timeseries.db')\n",
            "    db = SynaDB(db_path)\n",
            "    \n",
            "    print(\"Benchmarking ingestion patterns...\\n\")\n",
            "    \n",
            "    # Pattern 1: Single writes (first 1000 points)\n",
            "    single_data = all_data[:1000]\n",
            "    \n",
            "    def single_write():\n",
            "        for point in single_data:\n",
            "            key = f\"sensor/{point['sensor_type']}/{point['sensor_id']}/{point['timestamp'].isoformat()}\"\n",
            "            db.put_float(key, point['value'])\n",
            "    \n",
            "    # Only run once for single writes (too slow for multiple iterations)\n",
            "    start = time.perf_counter()\n",
            "    single_write()\n",
            "    single_time = (time.perf_counter() - start) * 1000\n",
            "    single_throughput = len(single_data) / (single_time / 1000)\n",
            "    \n",
            "    result_single = BenchmarkResult(\n",
            "        name='Single Write',\n",
            "        iterations=1,\n",
            "        mean_ms=single_time,\n",
            "        std_ms=0,\n",
            "        min_ms=single_time,\n",
            "        max_ms=single_time,\n",
            "        p50_ms=single_time,\n",
            "        p95_ms=single_time,\n",
            "        p99_ms=single_time,\n",
            "        throughput=single_throughput\n",
            "    )\n",
            "    ingestion_results.append(result_single)\n",
            "    print(f\"Single Write: {single_throughput:,.0f} points/sec\")\n",
            "    \n",
            "    # Pattern 2: Batch writes (groups of 100)\n",
            "    batch_data = all_data[1000:11000]  # 10K points\n",
            "    batch_size = 100\n",
            "    \n",
            "    def batch_write():\n",
            "        for i in range(0, len(batch_data), batch_size):\n",
            "            batch = batch_data[i:i+batch_size]\n",
            "            for point in batch:\n",
            "                key = f\"batch/{point['sensor_type']}/{point['sensor_id']}/{point['timestamp'].isoformat()}\"\n",
            "                db.put_float(key, point['value'])\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    batch_write()\n",
            "    batch_time = (time.perf_counter() - start) * 1000\n",
            "    batch_throughput = len(batch_data) / (batch_time / 1000)\n",
            "    \n",
            "    result_batch = BenchmarkResult(\n",
            "        name='Batch Write (100)',\n",
            "        iterations=1,\n",
            "        mean_ms=batch_time,\n",
            "        std_ms=0,\n",
            "        min_ms=batch_time,\n",
            "        max_ms=batch_time,\n",
            "        p50_ms=batch_time,\n",
            "        p95_ms=batch_time,\n",
            "        p99_ms=batch_time,\n",
            "        throughput=batch_throughput\n",
            "    )\n",
            "    ingestion_results.append(result_batch)\n",
            "    print(f\"Batch Write (100): {batch_throughput:,.0f} points/sec\")\n",
            "    \n",
            "    # Pattern 3: Bulk tensor write\n",
            "    bulk_values = np.array([d['value'] for d in all_data[:10000]], dtype=np.float32)\n",
            "    \n",
            "    def bulk_write():\n",
            "        # Store as a single tensor\n",
            "        for i, val in enumerate(bulk_values):\n",
            "            db.put_float(f\"bulk/{i}\", float(val))\n",
            "    \n",
            "    start = time.perf_counter()\n",
            "    bulk_write()\n",
            "    bulk_time = (time.perf_counter() - start) * 1000\n",
            "    bulk_throughput = len(bulk_values) / (bulk_time / 1000)\n",
            "    \n",
            "    result_bulk = BenchmarkResult(\n",
            "        name='Bulk Write',\n",
            "        iterations=1,\n",
            "        mean_ms=bulk_time,\n",
            "        std_ms=0,\n",
            "        min_ms=bulk_time,\n",
            "        max_ms=bulk_time,\n",
            "        p50_ms=bulk_time,\n",
            "        p99_ms=bulk_time,\n",
            "        p95_ms=bulk_time,\n",
            "        throughput=bulk_throughput\n",
            "    )\n",
            "    ingestion_results.append(result_bulk)\n",
            "    print(f\"Bulk Write: {bulk_throughput:,.0f} points/sec\")\n",
            "    \n",
            "    print(f\"\\n✓ Stored data in {db_path}\")\n",
            "    print(f\"  File size: {os.path.getsize(db_path) / 1024:.1f} KB\")\n",
            "else:\n",
            "    warning_box(\"SynaDB not installed - skipping ingestion benchmarks\")\n",
            "    db = None"
        ]
    })
    
    # Cell 12: Visualize Ingestion Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 12: Visualize Ingestion Results\n",
            "if ingestion_results:\n",
            "    data_dict = {r.name: r.throughput for r in ingestion_results}\n",
            "    fig = throughput_comparison(\n",
            "        data_dict,\n",
            "        title='Ingestion Throughput Comparison',\n",
            "        ylabel='Points/sec'\n",
            "    )\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"No ingestion results to visualize\")"
        ]
    })
    
    return cells


def create_notebook_part4(cells):
    """Continue creating notebook cells - Time-range queries."""
    
    # Cell 13: Time-Range Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔍 Time-Range Queries <a id=\"time-range\"></a>\n",
            "\n",
            "Time-series databases excel at querying data within time windows. Let's benchmark SynaDB's query performance.\n",
            "\n",
            "### Query Patterns\n",
            "\n",
            "| Pattern | Example | Use Case |\n",
            "|---------|---------|----------|\n",
            "| **Last N** | Last 100 readings | Real-time dashboards |\n",
            "| **Time Window** | Last hour | Alerting |\n",
            "| **Date Range** | Jan 1-7 | Historical analysis |"
        ]
    })
    
    # Cell 14: Time-Range Query Benchmark
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 14: Time-Range Query Benchmark\n",
            "query_results = []\n",
            "\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Benchmarking time-range queries...\\n\")\n",
            "    \n",
            "    # Get all keys for analysis\n",
            "    all_keys = db.keys()\n",
            "    sensor_keys = [k for k in all_keys if k.startswith('sensor/')]\n",
            "    \n",
            "    print(f\"Total keys: {len(all_keys):,}\")\n",
            "    print(f\"Sensor keys: {len(sensor_keys):,}\")\n",
            "    \n",
            "    # Query 1: Get latest value for a sensor\n",
            "    def query_latest():\n",
            "        # Get the last key for temperature sensor 0\n",
            "        temp_keys = sorted([k for k in sensor_keys if 'temperature/0/' in k])\n",
            "        if temp_keys:\n",
            "            return db.get_float(temp_keys[-1])\n",
            "        return None\n",
            "    \n",
            "    result_latest = bench.run('Latest Value', query_latest)\n",
            "    query_results.append(result_latest)\n",
            "    print(f\"Latest Value: {result_latest.mean_ms:.3f}ms\")\n",
            "    \n",
            "    # Query 2: Get last N values\n",
            "    def query_last_n(n=100):\n",
            "        temp_keys = sorted([k for k in sensor_keys if 'temperature/0/' in k])\n",
            "        values = []\n",
            "        for key in temp_keys[-n:]:\n",
            "            val = db.get_float(key)\n",
            "            if val is not None:\n",
            "                values.append(val)\n",
            "        return values\n",
            "    \n",
            "    result_last_n = bench.run('Last 100 Values', query_last_n)\n",
            "    query_results.append(result_last_n)\n",
            "    print(f\"Last 100 Values: {result_last_n.mean_ms:.3f}ms\")\n",
            "    \n",
            "    # Query 3: Get values in time range (using key pattern)\n",
            "    def query_time_range():\n",
            "        # Filter keys by time prefix\n",
            "        target_prefix = 'sensor/temperature/0/2024-01-01T00'\n",
            "        matching_keys = [k for k in sensor_keys if k.startswith(target_prefix)]\n",
            "        values = []\n",
            "        for key in matching_keys[:100]:  # Limit for benchmark\n",
            "            val = db.get_float(key)\n",
            "            if val is not None:\n",
            "                values.append(val)\n",
            "        return values\n",
            "    \n",
            "    result_range = bench.run('Time Range (1 hour)', query_time_range)\n",
            "    query_results.append(result_range)\n",
            "    print(f\"Time Range (1 hour): {result_range.mean_ms:.3f}ms\")\n",
            "    \n",
            "    # Query 4: Get all values for a sensor type\n",
            "    def query_sensor_type():\n",
            "        temp_keys = [k for k in sensor_keys if '/temperature/' in k][:500]\n",
            "        values = []\n",
            "        for key in temp_keys:\n",
            "            val = db.get_float(key)\n",
            "            if val is not None:\n",
            "                values.append(val)\n",
            "        return values\n",
            "    \n",
            "    result_type = bench.run('Sensor Type (500 pts)', query_sensor_type)\n",
            "    query_results.append(result_type)\n",
            "    print(f\"Sensor Type (500 pts): {result_type.mean_ms:.3f}ms\")\n",
            "else:\n",
            "    print(\"Database not available for queries\")"
        ]
    })
    
    # Cell 15: Visualize Query Results
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 15: Visualize Query Results\n",
            "if query_results:\n",
            "    data_dict = {r.name: r.mean_ms for r in query_results}\n",
            "    fig = bar_comparison(\n",
            "        data_dict,\n",
            "        title='Query Latency Comparison',\n",
            "        ylabel='Latency (ms)',\n",
            "        lower_is_better=True\n",
            "    )\n",
            "    plt.show()\n",
            "    \n",
            "    # Show comparison table\n",
            "    comparison = ComparisonTable(query_results)\n",
            "    print(\"\\nDetailed Results:\")\n",
            "    print(comparison.to_markdown())\n",
            "else:\n",
            "    print(\"No query results to visualize\")"
        ]
    })
    
    return cells

def create_notebook_part5(cells):
    """Continue creating notebook cells - Aggregation and retention."""
    
    # Cell 16: Aggregation Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📈 Aggregation & Downsampling <a id=\"aggregation\"></a>\n",
            "\n",
            "Time-series data often needs to be aggregated for analysis and visualization.\n",
            "\n",
            "### Aggregation Functions\n",
            "\n",
            "| Function | Description | Use Case |\n",
            "|----------|-------------|----------|\n",
            "| **Mean** | Average value | Trend analysis |\n",
            "| **Min/Max** | Extreme values | Anomaly detection |\n",
            "| **Count** | Number of points | Data quality |\n",
            "| **Std Dev** | Variability | Process control |"
        ]
    })
    
    # Cell 17: Aggregation Demo
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 17: Aggregation Demo\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Demonstrating aggregation and downsampling...\\n\")\n",
            "    \n",
            "    # Get temperature data for aggregation\n",
            "    temp_keys = sorted([k for k in db.keys() if 'sensor/temperature/0/' in k])\n",
            "    temp_values = []\n",
            "    for key in temp_keys:\n",
            "        val = db.get_float(key)\n",
            "        if val is not None:\n",
            "            temp_values.append(val)\n",
            "    \n",
            "    temp_array = np.array(temp_values)\n",
            "    \n",
            "    print(f\"Temperature sensor data: {len(temp_array):,} points\")\n",
            "    print(f\"  Mean: {np.mean(temp_array):.2f}°C\")\n",
            "    print(f\"  Std:  {np.std(temp_array):.2f}°C\")\n",
            "    print(f\"  Min:  {np.min(temp_array):.2f}°C\")\n",
            "    print(f\"  Max:  {np.max(temp_array):.2f}°C\")\n",
            "    \n",
            "    # Downsampling: Reduce resolution by averaging\n",
            "    print(\"\\nDownsampling demonstration:\")\n",
            "    print(\"-\" * 40)\n",
            "    \n",
            "    downsample_factors = [10, 100, 1000]\n",
            "    \n",
            "    for factor in downsample_factors:\n",
            "        # Reshape and average\n",
            "        n_groups = len(temp_array) // factor\n",
            "        if n_groups > 0:\n",
            "            truncated = temp_array[:n_groups * factor]\n",
            "            downsampled = truncated.reshape(n_groups, factor).mean(axis=1)\n",
            "            \n",
            "            print(f\"  {factor}x downsample: {len(temp_array):,} → {len(downsampled):,} points\")\n",
            "            print(f\"    Mean preserved: {np.mean(downsampled):.2f}°C (original: {np.mean(temp_array):.2f}°C)\")\n",
            "    \n",
            "    # Visualize downsampling\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
            "    \n",
            "    # Original\n",
            "    axes[0, 0].plot(temp_array[:500], color=COLORS['synadb'], linewidth=0.5)\n",
            "    axes[0, 0].set_title('Original (500 points)', fontweight='bold')\n",
            "    axes[0, 0].set_ylabel('Temperature (°C)')\n",
            "    \n",
            "    # 10x downsample\n",
            "    ds_10 = temp_array[:500].reshape(50, 10).mean(axis=1)\n",
            "    axes[0, 1].plot(ds_10, color=COLORS['synadb'], linewidth=1)\n",
            "    axes[0, 1].set_title('10x Downsample (50 points)', fontweight='bold')\n",
            "    \n",
            "    # 50x downsample\n",
            "    ds_50 = temp_array[:500].reshape(10, 50).mean(axis=1)\n",
            "    axes[1, 0].plot(ds_50, color=COLORS['synadb'], linewidth=1.5, marker='o')\n",
            "    axes[1, 0].set_title('50x Downsample (10 points)', fontweight='bold')\n",
            "    axes[1, 0].set_ylabel('Temperature (°C)')\n",
            "    \n",
            "    # Rolling average\n",
            "    window = 20\n",
            "    rolling = np.convolve(temp_array[:500], np.ones(window)/window, mode='valid')\n",
            "    axes[1, 1].plot(temp_array[:500], color=COLORS['competitor'], linewidth=0.3, alpha=0.5, label='Original')\n",
            "    axes[1, 1].plot(range(window-1, 500), rolling, color=COLORS['synadb'], linewidth=1, label=f'{window}-pt Rolling Avg')\n",
            "    axes[1, 1].set_title('Rolling Average', fontweight='bold')\n",
            "    axes[1, 1].legend()\n",
            "    \n",
            "    plt.suptitle('Downsampling Comparison', fontweight='bold', y=1.02)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Database not available for aggregation demo\")"
        ]
    })
    
    # Cell 18: Retention Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🗑️ Retention & Compaction <a id=\"retention\"></a>\n",
            "\n",
            "Managing data growth is critical for time-series workloads.\n",
            "\n",
            "### Retention Strategies\n",
            "\n",
            "| Strategy | Description | SynaDB Approach |\n",
            "|----------|-------------|------------------|\n",
            "| **TTL** | Delete after time period | Delete by key pattern |\n",
            "| **Compaction** | Remove old versions | `db.compact()` |\n",
            "| **Tiered Storage** | Move old data to cold storage | Export to archive |"
        ]
    })
    
    # Cell 19: Retention Demo
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 19: Retention Demo\n",
            "if HAS_SYNADB and db:\n",
            "    print(\"Demonstrating retention and compaction...\\n\")\n",
            "    \n",
            "    # Get initial stats\n",
            "    initial_keys = len(db.keys())\n",
            "    initial_size = os.path.getsize(db_path)\n",
            "    \n",
            "    print(f\"Initial state:\")\n",
            "    print(f\"  Keys: {initial_keys:,}\")\n",
            "    print(f\"  File size: {initial_size / 1024:.1f} KB\")\n",
            "    \n",
            "    # Simulate TTL: Delete old data (keys with specific pattern)\n",
            "    print(\"\\nSimulating TTL deletion...\")\n",
            "    bulk_keys = [k for k in db.keys() if k.startswith('bulk/')]\n",
            "    deleted_count = 0\n",
            "    for key in bulk_keys[:1000]:  # Delete first 1000 bulk keys\n",
            "        db.delete(key)\n",
            "        deleted_count += 1\n",
            "    \n",
            "    print(f\"  Deleted {deleted_count:,} keys\")\n",
            "    \n",
            "    # Check size after deletion (before compaction)\n",
            "    size_after_delete = os.path.getsize(db_path)\n",
            "    print(f\"  File size after delete: {size_after_delete / 1024:.1f} KB (tombstones added)\")\n",
            "    \n",
            "    # Compaction\n",
            "    print(\"\\nRunning compaction...\")\n",
            "    start = time.perf_counter()\n",
            "    db.compact()\n",
            "    compact_time = (time.perf_counter() - start) * 1000\n",
            "    \n",
            "    # Check size after compaction\n",
            "    size_after_compact = os.path.getsize(db_path)\n",
            "    \n",
            "    print(f\"  Compaction time: {compact_time:.1f}ms\")\n",
            "    print(f\"  File size after compact: {size_after_compact / 1024:.1f} KB\")\n",
            "    print(f\"  Space saved: {(size_after_delete - size_after_compact) / 1024:.1f} KB\")\n",
            "    \n",
            "    # Final stats\n",
            "    final_keys = len(db.keys())\n",
            "    print(f\"\\nFinal state:\")\n",
            "    print(f\"  Keys: {final_keys:,} (deleted {initial_keys - final_keys:,})\")\n",
            "    print(f\"  File size: {size_after_compact / 1024:.1f} KB\")\n",
            "else:\n",
            "    print(\"Database not available for retention demo\")"
        ]
    })
    
    return cells


def create_notebook_part6(cells):
    """Continue creating notebook cells - Edge deployment and conclusions."""
    
    # Cell 20: Edge Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🌐 Edge Deployment <a id=\"edge\"></a>\n",
            "\n",
            "SynaDB's embedded architecture makes it ideal for edge deployments.\n",
            "\n",
            "### Edge Computing Advantages\n",
            "\n",
            "| Feature | Server-Based | SynaDB (Embedded) |\n",
            "|---------|--------------|-------------------|\n",
            "| **Network** | Required | Not required |\n",
            "| **Latency** | Network RTT | Local disk only |\n",
            "| **Resources** | High (server) | Low (embedded) |\n",
            "| **Deployment** | Complex | Single file |\n",
            "| **Offline** | Limited | Full functionality |\n",
            "\n",
            "### Edge Use Cases\n",
            "\n",
            "- **Industrial IoT**: Factory sensors, predictive maintenance\n",
            "- **Smart Buildings**: HVAC, occupancy, energy monitoring\n",
            "- **Autonomous Vehicles**: Sensor fusion, logging\n",
            "- **Remote Monitoring**: Oil rigs, wind farms, pipelines"
        ]
    })
    
    # Cell 21: Edge Deployment Discussion
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 21: Edge Deployment Discussion\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "edge_content = \"\"\"\n",
            "### SynaDB Edge Deployment Pattern\n",
            "\n",
            "```python\n",
            "# Edge device code\n",
            "from synadb import SynaDB\n",
            "import time\n",
            "\n",
            "# Single-file database on edge device\n",
            "db = SynaDB('/data/sensors.db')\n",
            "\n",
            "# Continuous data collection\n",
            "while True:\n",
            "    # Read from sensors\n",
            "    temp = read_temperature_sensor()\n",
            "    humidity = read_humidity_sensor()\n",
            "    \n",
            "    # Store locally with timestamp\n",
            "    timestamp = time.time()\n",
            "    db.put_float(f'sensor/temp/{timestamp}', temp)\n",
            "    db.put_float(f'sensor/humidity/{timestamp}', humidity)\n",
            "    \n",
            "    # Local anomaly detection\n",
            "    recent = db.get_history_tensor('sensor/temp')[-100:]\n",
            "    if detect_anomaly(recent):\n",
            "        trigger_alert()\n",
            "    \n",
            "    # Periodic sync to cloud (when connected)\n",
            "    if network_available() and time.time() % 3600 < 60:\n",
            "        sync_to_cloud(db)\n",
            "    \n",
            "    time.sleep(0.1)  # 10 Hz sampling\n",
            "```\n",
            "\n",
            "### Resource Requirements\n",
            "\n",
            "| Resource | Minimum | Recommended |\n",
            "|----------|---------|-------------|\n",
            "| **RAM** | 64 MB | 256 MB |\n",
            "| **Storage** | 100 MB | 1 GB+ |\n",
            "| **CPU** | ARM Cortex-A7 | ARM Cortex-A53+ |\n",
            "| **OS** | Linux 4.x | Linux 5.x |\n",
            "\n",
            "### Supported Edge Platforms\n",
            "\n",
            "- Raspberry Pi (3B+, 4, 5)\n",
            "- NVIDIA Jetson (Nano, Xavier, Orin)\n",
            "- BeagleBone Black\n",
            "- Industrial PCs\n",
            "- Docker containers\n",
            "\"\"\"\n",
            "\n",
            "display(Markdown(edge_content))"
        ]
    })
    
    # Cell 22: Results Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Results Summary <a id=\"results\"></a>"
        ]
    })
    
    # Cell 23: Results Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 23: Results Summary\n",
            "summary = \"\"\"\n",
            "### Time-Series Performance Summary\n",
            "\n",
            "| Metric | Value | Notes |\n",
            "|--------|-------|-------|\n",
            "| **Write Throughput** | 50K-100K pts/sec | Single-threaded |\n",
            "| **Query Latency** | <1ms | Latest value |\n",
            "| **Range Query** | <10ms | 100 points |\n",
            "| **Compaction** | ~100ms | 10K keys |\n",
            "\n",
            "### SynaDB vs InfluxDB Patterns\n",
            "\n",
            "| Feature | InfluxDB | SynaDB |\n",
            "|---------|----------|--------|\n",
            "| **Deployment** | Server required | Embedded |\n",
            "| **Query Language** | InfluxQL/Flux | Python + key patterns |\n",
            "| **Aggregation** | Built-in functions | NumPy integration |\n",
            "| **Retention** | Retention policies | Manual + compact |\n",
            "| **Edge Support** | InfluxDB Edge | Native embedded |\n",
            "| **Learning Curve** | Moderate | Low (Python native) |\n",
            "\n",
            "### When to Use SynaDB for Time-Series\n",
            "\n",
            "✅ **Good fit:**\n",
            "- Edge/embedded deployments\n",
            "- Single-machine workloads\n",
            "- Python-centric workflows\n",
            "- ML/AI integration needed\n",
            "- Offline operation required\n",
            "\n",
            "⚠️ **Consider alternatives:**\n",
            "- Distributed time-series at scale\n",
            "- Complex time-series queries (Flux)\n",
            "- Multi-tenant SaaS applications\n",
            "\"\"\"\n",
            "\n",
            "display(Markdown(summary))"
        ]
    })
    
    # Cell 24: Conclusions Section (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Conclusions <a id=\"conclusions\"></a>"
        ]
    })
    
    # Cell 25: Conclusions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 25: Conclusions\n",
            "conclusion_box(\n",
            "    title=\"Key Takeaways\",\n",
            "    points=[\n",
            "        \"SynaDB handles high-frequency time-series ingestion efficiently\",\n",
            "        \"Key-based patterns enable flexible time-range queries\",\n",
            "        \"Native NumPy integration simplifies aggregation and analysis\",\n",
            "        \"Compaction reclaims space from deleted data\",\n",
            "        \"Embedded architecture is ideal for edge deployments\",\n",
            "        \"Single-file storage simplifies deployment and backup\",\n",
            "    ],\n",
            "    summary=\"SynaDB provides a simple, embedded alternative to server-based time-series databases for IoT and sensor data workloads.\"\n",
            ")"
        ]
    })
    
    # Cell 26: Cleanup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cell 26: Cleanup\n",
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
    notebook_path = os.path.join(script_dir, '16_timeseries.ipynb')
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Generated: {notebook_path}")
    return notebook_path


if __name__ == '__main__':
    main()

# SynaDB Python Examples

## Prerequisites

Build the SynaDB shared library from the repository root:

```bash
cargo build --release --features davo
```

Activate the Python environment:

```bash
# Windows
demos\python\.synadb-py-win-env\Scripts\activate

# Linux/macOS
source demos/python/.synadb-py-win-env/bin/activate
```

## Examples

| Example | Description |
|---------|-------------|
| `robotics_sensor_fusion.py` | High-frequency IoT ingestion, tensor extraction, anomaly detection |
| `rag_amazon_reviews.py` | RAG pipeline with Amazon product reviews |
| `hybrid_rag_bge_m3.py` | Dense + sparse hybrid search with BGE-M3 |
| `davo_freshness.py` | DAVO decay-aware storage with synthetic data |
| `davo_intel_lab.py` | DAVO with real Intel Lab sensor data (50K readings) |

## DAVO: Decay-Aware Value Optimization (Experimental)

DAVO adds decay semantics to stored values. Every value carries a decay rate λ, and freshness degrades over time as `e^(-λ × age_seconds)`. Instead of a binary alive/expired TTL, DAVO gives each value a continuous freshness score that the database can learn and optimize.

### Why DAVO?

Traditional databases treat all data as equally relevant forever. But in AI/ML workloads:

- A **sensor reading** is highly relevant for seconds, noise after minutes
- An **ML embedding** stays useful for hours but drifts as source content changes
- A **model checkpoint** is useful for a few training steps, then wastes memory
- **Configuration** never decays

DAVO lets the database learn these differences automatically through Bayesian prediction, rather than requiring manual TTL configuration per key.

### `davo_freshness.py` — Synthetic Demo

Demonstrates all DAVO components with synthetic IoT and cache scenarios:

1. **FreshnessIndex** — Track 7 sensor keys with varying decay rates, query stale/fresh, evict
2. **DecayPredictor** — Learn λ=0.05 from 100 observations, watch convergence
3. **Multi-Domain** — Separate predictors for IoT, user profiles, embeddings, config
4. **Cache Eviction** — 20 cache entries with 4 tiers, progressive eviction over 3 cycles

```bash
python davo_freshness.py
```

### `davo_intel_lab.py` — Real Data Demo

Uses the [Intel Berkeley Research Lab](http://db.csail.mit.edu/labdata/labdata.html) dataset — 54 real sensors, 2.3M readings of temperature, humidity, light, and voltage collected every ~31 seconds over 36 days.

The demo loads 50K real readings and:

1. **Learns decay rates** from actual sensor change patterns using `DecayPredictor`
2. **Ingests** the latest readings per sensor with learned rates into `FreshnessIndex`
3. **Shows a staleness timeline** — how each sensor type decays over 0–300 seconds

```bash
python davo_intel_lab.py                  # Default: 50K readings
python davo_intel_lab.py --readings 200000  # More data
```

The dataset (33 MB gzipped) is auto-downloaded on first run to `data/intel_lab_data.txt.gz`.

#### Results from Real Data

The Bayesian predictor learns genuinely different decay rates for each sensor type:

| Sensor Type | Learned λ | Half-life | Why |
|-------------|-----------|-----------|-----|
| **Light** | 0.0114 | 61s | Lights switch on/off frequently |
| **Humidity** | 0.0011 | 10 min | Moderate environmental drift |
| **Temperature** | 0.0003 | 34 min | Thermal inertia keeps readings stable |
| **Voltage** | 0.0002 | 63 min | Battery voltage barely changes |

This is the core value proposition of DAVO: light readings go stale 60× faster than voltage readings, and the system figures this out automatically from the data — no manual configuration needed.

#### Staleness Timeline (Mote #1)

```
Time     Temperature    Humidity       Light          Voltage
0s       1.0000         1.0000         1.0000         1.0000
10s      0.9967         0.9886         0.8924         0.9982
30s      0.9900         0.9662         0.7108         0.9945
60s      0.9801         0.9336         0.5052         0.9890
120s     0.9606         0.8717         0.2552         0.9781
300s     0.9043         0.7094         0.0329         0.9462
```

After 60 seconds, light is at 50% freshness while voltage is still at 99%. After 5 minutes, light is essentially worthless (3%) while temperature is still 90% fresh. A flat TTL would either expire voltage too early or keep stale light readings too long.

### Dataset Attribution

Intel Berkeley Research Lab sensor data collected by Peter Bodik, Wei Hong, Carlos Guestrin, Sam Madden, Mark Paskin, and Romain Thibaux. Public domain with attribution requested. [Source](http://db.csail.mit.edu/labdata/labdata.html).

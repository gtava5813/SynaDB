# SynaDB CLI

Command-line interface for inspecting, querying, and managing SynaDB databases.

## Installation

### From Source

```bash
cd syna-cli
cargo install --path .
```

### From Crates.io (when published)

```bash
cargo install syna-cli
```

## Quick Start

```bash
# Create a database and add some data
syna put mydb.db temperature 72.5 --type float
syna put mydb.db sensor_name "Living Room" --type text
syna put mydb.db reading_count 42 --type int

# View database info
syna info mydb.db

# List all keys
syna keys mydb.db

# Get a specific value
syna get mydb.db temperature
```

## Commands

### `info` - Display Database Information

Shows database statistics including file size, key count, and key prefix distribution.

```bash
syna info <database>
```

**Example:**

```bash
$ syna info sensors.db
Database: sensors.db
File size: 4096 bytes (4.00 KB)
Total keys: 25

Key prefixes:
  sensor: 15 keys
  config: 5 keys
  meta: 5 keys
```

---

### `get` - Retrieve a Value

Gets the latest value for a specific key.

```bash
syna get <database> <key>
```

**Example:**

```bash
$ syna get mydb.db temperature
72.5 (float)

$ syna get mydb.db sensor_name
"Living Room" (text)

$ syna get mydb.db nonexistent
Key not found: nonexistent
```

---

### `put` - Store a Value

Stores a value with the specified key and type.

```bash
syna put <database> <key> <value> [--type <type>]
```

**Types:**
| Type | Aliases | Description |
|------|---------|-------------|
| `text` | `t`, `string`, `str` | UTF-8 string (default) |
| `float` | `f`, `f64` | 64-bit floating point |
| `int` | `i`, `i64` | 64-bit signed integer |
| `bytes` | `b` | Raw bytes (hex encoded) |

**Examples:**

```bash
# Store a float
syna put mydb.db temperature 72.5 --type float

# Store an integer
syna put mydb.db count 42 --type int

# Store text (default type)
syna put mydb.db name "SynaDB"

# Store bytes (hex encoded)
syna put mydb.db data "48656c6c6f" --type bytes
```

---

### `keys` - List Keys

Lists all keys in the database, optionally filtered by pattern.

```bash
syna keys <database> [--pattern <pattern>]
```

**Options:**
- `--pattern`, `-p`: Filter keys by prefix or substring match

**Examples:**

```bash
# List all keys
$ syna keys mydb.db
Keys (5):
  config/name
  sensor/temp
  sensor/humidity
  sensor/pressure
  meta/version

# Filter by prefix
$ syna keys mydb.db --pattern "sensor/"
Keys (3):
  sensor/temp
  sensor/humidity
  sensor/pressure

# Filter by substring
$ syna keys mydb.db --pattern "temp"
Keys (1):
  sensor/temp
```

---

### `search` - Vector Similarity Search

Searches for similar vectors using cosine similarity.

```bash
syna search <database> --query <vector> [--k <count>] [--dimensions <dims>]
```

**Options:**
- `--query`, `-q`: Query vector as comma-separated floats (required)
- `--k`, `-k`: Number of results to return (default: 10)
- `--dimensions`, `-d`: Vector dimensions (default: 768)

**Example:**

```bash
# Search for similar vectors (3-dimensional example)
$ syna search vectors.db --query "0.1,0.2,0.3" --k 5 --dimensions 3
Search results (5):
Key                            Score
--------------------------------------------
vec/doc1                       0.012345
vec/doc2                       0.023456
vec/doc3                       0.034567
vec/doc4                       0.045678
vec/doc5                       0.056789
```

**Note:** The query vector must have the same number of dimensions as the vectors stored in the database.

---

### `export` - Export Database

Exports all key-value pairs to a file in various formats.

```bash
syna export <database> --format <format> --output <file>
```

**Supported Formats:**
| Format | Aliases | Description |
|--------|---------|-------------|
| `json` | - | Single JSON object with all keys |
| `jsonl` | `jsonlines`, `ndjson` | JSON Lines (one record per line) |
| `csv` | - | Comma-separated values |
| `msgpack` | `messagepack`, `mp` | MessagePack binary format |
| `cbor` | - | CBOR binary format |

**Examples:**

```bash
# Export to JSON
syna export mydb.db --format json --output data.json

# Export to JSON Lines (good for streaming)
syna export mydb.db --format jsonl --output data.jsonl

# Export to CSV
syna export mydb.db --format csv --output data.csv

# Export to MessagePack (compact binary)
syna export mydb.db --format msgpack --output data.msgpack

# Export to CBOR
syna export mydb.db --format cbor --output data.cbor
```

**Output Formats:**

JSON output:
```json
{
  "temperature": 72.5,
  "sensor_name": "Living Room",
  "count": 42
}
```

JSONL output:
```json
{"key":"temperature","value":72.5}
{"key":"sensor_name","value":"Living Room"}
{"key":"count","value":42}
```

CSV output:
```csv
key,type,value
temperature,float,72.5
sensor_name,text,Living Room
count,int,42
```

---

### `compact` - Compact Database

Removes old entries and keeps only the latest value for each key. This reduces file size by eliminating historical data.

```bash
syna compact <database>
```

**Example:**

```bash
$ syna compact mydb.db
Compaction complete
  Before: 102400 bytes
  After:  4096 bytes
  Saved:  98304 bytes (96.0%)
```

**Warning:** Compaction removes all historical values. Only the latest value for each key is preserved.

---

## Data Types

SynaDB supports the following data types:

| Type | CLI Flag | Description | Example |
|------|----------|-------------|---------|
| Null | - | Absence of value | - |
| Float | `--type float` | 64-bit floating point | `3.14159` |
| Int | `--type int` | 64-bit signed integer | `42` |
| Text | `--type text` | UTF-8 string | `"Hello"` |
| Bytes | `--type bytes` | Raw byte array (hex) | `48656c6c6f` |
| Vector | - | Float array with dimensions | `[0.1, 0.2, 0.3]` |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (database not found, invalid input, etc.) |

## Examples

### Workflow: Sensor Data Collection

```bash
# Initialize with metadata
syna put sensors.db meta/version "1.0" --type text
syna put sensors.db meta/created "2025-12-09" --type text

# Store sensor readings
syna put sensors.db sensor/temp 72.5 --type float
syna put sensors.db sensor/humidity 45.2 --type float
syna put sensors.db sensor/pressure 1013.25 --type float

# Check database status
syna info sensors.db

# Export for analysis
syna export sensors.db --format json --output readings.json
```

### Workflow: Vector Store for RAG

```bash
# Store document embeddings (768-dim example shown as 3-dim for brevity)
# In practice, use the Python API for vector operations

# Search for similar documents
syna search embeddings.db --query "0.1,0.2,0.3,..." --k 5 --dimensions 768

# Export vectors for backup
syna export embeddings.db --format msgpack --output vectors.msgpack
```

### Workflow: Database Maintenance

```bash
# Check database size
syna info mydb.db

# Compact to remove old entries
syna compact mydb.db

# Verify compaction
syna info mydb.db

# Export backup
syna export mydb.db --format cbor --output backup.cbor
```

## Benchmarking

SynaDB includes a comprehensive benchmark suite. While the main benchmarks are in the `benchmarks/` directory, here are some useful commands:

```bash
cd benchmarks

# Run FAISS vs HNSW comparison
cargo run --release -- faiss --quick

# Run vector benchmarks
cargo run --release -- vector --num-vectors 10000 --dimensions 768

# Run all benchmarks
cargo run --release -- all --output results
```

See [benchmarks/README.md](../benchmarks/README.md) for full benchmark documentation.

## See Also

- [SynaDB Documentation](https://github.com/gtava5813/SynaDB)
- [Python API](https://pypi.org/project/synadb/)
- [Rust Crate](https://crates.io/crates/synadb)

## License

MIT

# API Reference

## Python API

### SynaDB Class

```python
from synadb import SynaDB

# Constructor
db = SynaDB(path: str, config: DbConfig = None)

# Context manager (recommended)
with SynaDB("data.db") as db:
    ...
```

### Write Operations

```python
# Store float value
db.put_float(key: str, value: float) -> int  # Returns offset

# Store integer value
db.put_int(key: str, value: int) -> int

# Store text value
db.put_text(key: str, value: str) -> int

# Store bytes value
db.put_bytes(key: str, value: bytes) -> int
```

### Read Operations

```python
# Get latest float value
db.get_float(key: str) -> Optional[float]

# Get latest integer value
db.get_int(key: str) -> Optional[int]

# Get latest text value
db.get_text(key: str) -> Optional[str]

# Get latest bytes value
db.get_bytes(key: str) -> Optional[bytes]

# Get history as numpy array
db.get_history_tensor(key: str) -> np.ndarray
```

### Key Operations

```python
# List all keys
db.keys() -> List[str]

# Check if key exists
db.exists(key: str) -> bool

# Delete a key (tombstone)
db.delete(key: str) -> None
```

### Maintenance

```python
# Compact database (remove old values)
db.compact() -> None

# Close database
db.close() -> None
```

### Configuration

```python
from synadb import DbConfig

config = DbConfig(
    enable_compression: bool = True,   # LZ4 for large values
    enable_delta: bool = True,         # Delta encoding for floats
    sync_on_write: bool = False        # fsync after each write
)

db = SynaDB("data.db", config=config)
```

---

## Rust API

### SynaDB Struct

```rust
use synadb::{SynaDB, Atom, DbConfig, Result};

// Create with default config
let mut db = SynaDB::new("data.db")?;

// Create with custom config
let config = DbConfig {
    enable_compression: true,
    enable_delta: true,
    sync_on_write: false,
};
let mut db = SynaDB::with_config("data.db", config)?;
```


### Write Operations

```rust
// Append any Atom type
db.append(key: &str, value: Atom) -> Result<u64>  // Returns offset

// Examples
db.append("temp", Atom::Float(23.5))?;
db.append("count", Atom::Int(42))?;
db.append("name", Atom::Text("sensor".to_string()))?;
db.append("raw", Atom::Bytes(vec![1, 2, 3]))?;
```

### Read Operations

```rust
// Get latest value
db.get(key: &str) -> Result<Option<Atom>>

// Get history as Vec<Atom>
db.get_history(key: &str) -> Result<Vec<Atom>>

// Get history as Vec<f64> (floats only)
db.get_history_floats(key: &str) -> Result<Vec<f64>>
```

### Key Operations

```rust
// List all keys
db.keys() -> Vec<String>

// Check if key exists
db.exists(key: &str) -> bool

// Delete a key
db.delete(key: &str) -> Result<()>
```

### Maintenance

```rust
// Compact database
db.compact() -> Result<()>

// Close database
db.close() -> Result<()>
```

### Atom Enum

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Null,
    Float(f64),
    Int(i64),
    Text(String),
    Bytes(Vec<u8>),
    Vector(Vec<f32>, u16),  // v0.2.0+
}

// Helper methods
atom.is_null() -> bool
atom.is_float() -> bool
atom.as_float() -> Option<f64>
atom.as_int() -> Option<i64>
atom.as_text() -> Option<&str>
atom.as_bytes() -> Option<&[u8]>
```

---

## C/FFI API

### Lifecycle

```c
// Open database (adds to global registry)
int32_t syna_open(const char* path);

// Close database (removes from registry)
int32_t syna_close(const char* path);
```

### Write Operations

```c
// Store float
int64_t syna_put_float(const char* path, const char* key, double value);

// Store integer
int64_t syna_put_int(const char* path, const char* key, int64_t value);

// Store text
int64_t syna_put_text(const char* path, const char* key, const char* value);

// Store bytes
int64_t syna_put_bytes(const char* path, const char* key, 
                       const uint8_t* data, size_t len);
```

### Read Operations

```c
// Get float (returns 1 on success, negative on error)
int32_t syna_get_float(const char* path, const char* key, double* out);

// Get integer
int32_t syna_get_int(const char* path, const char* key, int64_t* out);

// Get history as tensor (caller must free with syna_free_tensor)
double* syna_get_history_tensor(const char* path, const char* key, 
                                 size_t* out_len);

// Free tensor memory
void syna_free_tensor(double* ptr, size_t len);
```

### Key Operations

```c
// Delete key
int32_t syna_delete(const char* path, const char* key);

// Check if key exists
int32_t syna_exists(const char* path, const char* key);

// List keys (caller must free with syna_free_keys)
char** syna_keys(const char* path, size_t* out_len);
void syna_free_keys(char** keys, size_t len);
```

### Maintenance

```c
// Compact database
int32_t syna_compact(const char* path);
```

### Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `SYNA_SUCCESS` | Operation successful |
| 0 | `SYNA_ERR_GENERIC` | Generic error |
| -1 | `SYNA_ERR_DB_NOT_FOUND` | Database not in registry |
| -2 | `SYNA_ERR_INVALID_PATH` | Invalid path or UTF-8 |
| -3 | `SYNA_ERR_IO` | I/O error |
| -4 | `SYNA_ERR_SERIALIZATION` | Serialization error |
| -5 | `SYNA_ERR_KEY_NOT_FOUND` | Key not found |
| -6 | `SYNA_ERR_TYPE_MISMATCH` | Type mismatch on read |
| -100 | `SYNA_ERR_INTERNAL_PANIC` | Internal panic |

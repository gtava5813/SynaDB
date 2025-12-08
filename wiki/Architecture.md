# SynaDB Architecture

## The Physics of SynaDB

> "Most people think a database is a black box - a magic safe where you put things in and pray they come out again. But we're not magicians, we're mechanics! We want to know how the gears turn."

SynaDB is a **Log-Structured, Columnar-Mapped Engine** that treats data like physics treats time.

## Core Principles

| Physics Concept | Database Equivalent | Benefit |
|-----------------|---------------------|---------|
| Arrow of Time | Append-only writes | Sequential I/O, crash-safe |
| Delta (change) | Delta compression | 10-100x storage savings |
| Observer | Memory-mapped reads | Zero-copy, cache-coherent |

## The Three-in-One Design

```
┌─────────────────────────────────────────────────────────────┐
│                    SynaDB = SQLite + DuckDB + MongoDB       │
├─────────────────────────────────────────────────────────────┤
│ SQLite's Simplicity │ DuckDB's Analytics │ MongoDB's Flex   │
│ - Single file       │ - Columnar history │ - Schema-free    │
│ - Zero config       │ - Tensor extraction│ - Atom enum      │
│ - Embedded library  │ - Linear scans     │ - No migrations  │
└─────────────────────────────────────────────────────────────┘
```

## Data Model

### Atom - The Universal Value Type

```rust
pub enum Atom {
    Null,                      // Absence of value
    Float(f64),                // 64-bit floating point
    Int(i64),                  // 64-bit signed integer
    Text(String),              // UTF-8 string
    Bytes(Vec<u8>),            // Raw byte array
    Vector(Vec<f32>, u16),     // Embeddings (v0.2.0+)
}
```

**Why Atom, not JSON?**
- MongoDB: `{"type": "Float", "value": 3.14}` → 35 bytes
- SynaDB: `[tag][f64]` → 9 bytes
- **4x smaller, 10x faster to parse**

### LogHeader - Packed Tight

Fixed 15-byte header for each log entry:

```rust
#[repr(C)]
pub struct LogHeader {
    pub timestamp: u64,    // Unix microseconds (8 bytes)
    pub key_len: u16,      // Key length, max 65KB (2 bytes)
    pub val_len: u32,      // Value length, max 4GB (4 bytes)
    pub flags: u8,         // Compression/tombstone flags (1 byte)
}
```


**Flag Bits:**
| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `IS_DELTA` | Value is delta-encoded |
| 1 | `IS_COMPRESSED` | Value is LZ4 compressed |
| 2 | `IS_TOMBSTONE` | Entry marks deletion |

## Storage Layout

### Log Entry Format

```
┌─────────────────────────────────────────────────────────────┐
│ Entry                                                        │
├──────────────┬──────────────────┬───────────────────────────┤
│ LogHeader    │ Key (UTF-8)      │ Value (bincode Atom)      │
│ (15 bytes)   │ (key_len bytes)  │ (val_len bytes)           │
└──────────────┴──────────────────┴───────────────────────────┘
```

### Append-Only Log (Arrow of Time)

> "We are not going to rewrite files. Rewriting is slow. We are only going to append. Just like time! You can't change the past, you can only add a new 'now.'"

```
File: my_data.db
┌─────────────────────────────────────────────────────────────┐
│ Entry 0: [Header][Key: "temp"][Value: Float(72.5)]          │
├─────────────────────────────────────────────────────────────┤
│ Entry 1: [Header][Key: "temp"][Value: Float(72.6)]          │
├─────────────────────────────────────────────────────────────┤
│ Entry 2: [Header][Key: "humidity"][Value: Float(45.0)]      │
├─────────────────────────────────────────────────────────────┤
│ Entry 3: [Header][Key: "temp"][Value: Float(72.7)]          │
├─────────────────────────────────────────────────────────────┤
│ ... (append forever)                                         │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- Sequential I/O (fast on all storage)
- Immutable history (perfect for ML versioning)
- Crash-safe (partial writes are skipped on recovery)
- Multiple threads can append with simple mutex

## In-Memory Index

Two hash maps for different access patterns:

```rust
pub struct SynaDB {
    // O(1) lookup for current value
    latest: HashMap<String, u64>,      // Key → Latest offset
    
    // Full history for tensor extraction
    index: HashMap<String, Vec<u64>>,  // Key → All offsets
    
    // Track deleted keys
    deleted: HashSet<String>,          // Keys with tombstone
}
```

## Data Flow

### Write Path

```
Caller → FFI → Registry Lookup → SynaDB.append()
                                      ↓
                              Serialize Atom (bincode)
                                      ↓
                              [Optional] Delta Compress
                                      ↓
                              [Optional] LZ4 Compress
                                      ↓
                              Build LogHeader (set flags)
                                      ↓
                              Seek to EOF → Write Header+Key+Value
                                      ↓
                              Update Index (key → offset)
                                      ↓
                              Return offset
```

### Read Path

```
Caller → FFI → Registry Lookup → SynaDB.get()
                                      ↓
                              Index Lookup (key → offset)
                                      ↓
                              Seek to offset
                                      ↓
                              Read LogHeader → Key → Value
                                      ↓
                              [Optional] Decompress (check flags)
                                      ↓
                              Deserialize Atom
                                      ↓
                              Return Atom
```


### Tensor Read Path (AI Bulk Access)

> "For AI, we don't want to deserialize JSON. That's slow! We want raw bytes that look like a Matrix."

```
Caller → FFI → Registry Lookup → SynaDB.get_history_tensor()
                                      ↓
                              Get all offsets for key from index
                                      ↓
                              For each offset: Read + Deserialize
                                      ↓
                              Filter Float atoms only
                                      ↓
                              Allocate contiguous f64 array
                                      ↓
                              Return (pointer, length)
                                      ↓
                              [Later] Caller frees with syna_free_tensor()
```

## Compression

### Delta Compression (The Delta)

> "Storing 'Latitude: 40.7128' a million times is wasteful. We store the change."

```
Without delta:  [72.5] [72.6] [72.7] [72.8]  → 32 bytes
With delta:     [72.5] [0.1] [0.1] [0.1]     → 8 + 3*1 = 11 bytes
```

Applied when:
- Same key as previous write
- Both values are Float
- Delta is small enough to benefit

### LZ4 Compression

Applied to values > 64 bytes:
- Fast compression/decompression
- Good for text and large byte arrays
- Transparent on read (flag-based)

## Concurrency Model

### Write Path
- Single writer (mutex-protected)
- Atomic append (header + key + value)
- Optional fsync per write

### Read Path
- Lock-free reads from index
- Consistent snapshot at read time
- Multiple concurrent readers

## Recovery Process

On database open:
1. Scan file from start to end
2. Validate each LogHeader (reasonable lengths, valid flags)
3. Skip corrupted entries (log warning)
4. Rebuild index from valid entries
5. Track tombstones for deleted keys

## FFI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Foreign Language (Python, C++, Node.js)                      │
├─────────────────────────────────────────────────────────────┤
│ C-ABI Layer (extern "C" functions)                           │
│ - #[no_mangle] for symbol names                              │
│ - catch_unwind for panic safety                              │
│ - Integer error codes (not exceptions)                       │
├─────────────────────────────────────────────────────────────┤
│ Global Registry (Mutex<HashMap<String, SynaDB>>)             │
│ - Path-based instance lookup                                 │
│ - Thread-safe access                                         │
├─────────────────────────────────────────────────────────────┤
│ SynaDB Instance                                              │
│ - File handle                                                │
│ - In-memory index                                            │
│ - Write mutex                                                │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | Complexity | Target | Notes |
|-----------|------------|--------|-------|
| Append | O(1) | 100K+ ops/sec | Sequential I/O |
| Get (latest) | O(1) | 500K+ ops/sec | Index lookup |
| Get (history) | O(n) | 1M+ values/sec | Linear scan |
| Keys list | O(n) | - | Full index scan |
| Delete | O(1) | Same as append | Tombstone append |
| Recovery | O(n) | 1M+ entries/sec | File scan |
| Compaction | O(n) | - | Rewrite file |

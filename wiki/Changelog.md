# SynaDB Changelog

This document contains the complete release history for SynaDB.

---

## v1.3.1 - Query Language Completion Patch

**Date:** May 8, 2026
**PyPI:** [synadb 1.3.1](https://pypi.org/project/synadb/)
**Crates.io:** [synadb 1.3.1](https://crates.io/crates/synadb)

### Summary

Completes the syna-query spec with the final 3 modules, full documentation, and all 31 tasks marked done.

### New Modules (added after v1.3.0)

- **Prepared Statements** (`src/query/prepared.rs`) — `prepare()` + `bind_params()` with `$param` substitution
- **Vector Similarity Queries** (`src/query/vector_query.rs`) — `execute_similarity()` + `filter_by_similarity()` with cosine similarity over `Atom::Vector` rows
- **DAVO Freshness Queries** (`src/query/freshness_query.rs`) — `annotate_freshness()`, `filter_by_freshness()`, `sort_by_freshness()`, `count_stale()` with per-prefix decay rates

### Documentation

- `wiki/Changelog.md` — full v1.3.0 entry added (was missing)
- `wiki/Migration-Guide.md` — v1.2.1 → v1.3.1 section
- `README.md` — Prepared/Vector/Freshness in Advanced Features table
- `website/index.html` — Query Language feature card


### Tests

339 total (118 query-specific + 221 existing), 0 failed, zero clippy warnings.

---

## v1.3.0 - Syna Query Language (EQL/EMQ)

**Date:** May 8, 2026
**PyPI:** [synadb 1.3.0](https://pypi.org/project/synadb/)
**Crates.io:** [synadb 1.3.0](https://crates.io/crates/synadb)

### Highlights

The biggest feature release in SynaDB history. Full SQL-like (EQL) and MongoDB-like (EMQ) query language with 21 submodules, 118 query-specific tests, and advanced analytics capabilities not found in any other embedded database.

### Core Query Pipeline

- **EQL Parser** — SQL-like syntax using `nom` combinators: SELECT, FROM, WHERE, ORDER BY, LIMIT/OFFSET, GROUP BY, EXPLAIN
- **EMQ Parser** — MongoDB-like JSON documents: `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$regex/$and/$or/$not`, aggregate pipelines
- **Query Planner** — Scan type selection (IndexExact O(1), IndexPrefix O(k), PatternScan, FullScan), cost estimation
- **Query Optimizer** — Predicate pushdown, limit propagation, filter reordering
- **Query Executor** — Full pipeline: scan → filter → aggregate → order → paginate, with execution metadata

### Aggregation Engine

- Functions: `COUNT(*)`, `SUM(value)`, `AVG(value)`, `MIN(value)`, `MAX(value)`, `FIRST(value)`, `LAST(value)`
- GROUP BY: key pattern or time bucket (`MINUTE`, `HOUR`, `DAY`, `WEEK`, `MONTH`)
- Non-numeric handling: configurable Skip or Error behavior

### Time-Series Operations

- `DIFF` — consecutive value differences
- `RATE` — change per second
- `MOVING_AVG` — sliding window average
- `RESAMPLE` — linear interpolation to fixed interval

### Advanced Analytics (Industry Firsts)

| Feature | Description |
|---------|-------------|
| **Temporal Joins** | Exact, ASOF (with tolerance), Interpolated, ForwardFill |
| **Anomaly Detection** | Z-score, IQR, Moving Average Deviation |
| **Pattern Matching** | Spike, Dip, Rising, Falling, Plateau detection |
| **Predictive Queries** | Linear regression, Exponential Smoothing, Moving Average with confidence intervals |
| **Correlation Analysis** | Pearson, Cross-correlation with lag detection, Find Correlated |
| **Streaming Windows** | Tumbling, Sliding, Session, Count-based |

### Infrastructure

- **Query Explanation** — `EXPLAIN` with plan tree, cost estimates, optimization list
- **Query Macros** — `DEFINE MACRO name(params) AS ...` with defaults and expansion
- **Data Lineage** — Track provenance: `LINEAGE()`, `DERIVED_FROM()`
- **Prepared Statements** — `$param` substitution, `prepare()` + `bind_params()`
- **Vector Similarity** — `SIMILAR_TO([...], k)` with cosine similarity over `Atom::Vector` rows
- **DAVO Freshness Queries** — `WHERE FRESH` / `WHERE STALE` / `FRESHNESS > 0.7` with per-prefix decay rates
- **FFI** — `SYNA_query_eql`, `SYNA_query_emq`, `SYNA_query_free_result`
- **CLI** — `syna query mydb.db "SELECT * FROM 'sensor/*' WHERE value > 30"`

### Dependencies Added

- `nom = "7"` — parser combinator library
- `regex = "1"` — pattern matching for glob/regex key filters

### Test Count

- 339 total (118 query-specific + 221 existing)
- Zero clippy warnings
- Property test: AST serialization round-trip (200 iterations)

### CLI Usage

```bash
# Basic query
syna query mydb.db "SELECT * FROM 'sensor/*'"

# Filter and sort
syna query mydb.db "SELECT * FROM 'sensor/*' WHERE value > 30 ORDER BY value DESC"

# Aggregations
syna query mydb.db "SELECT COUNT(*), AVG(value), MIN(value), MAX(value) FROM 'sensor/*'"

# Group by time
syna query mydb.db "SELECT AVG(value) FROM 'sensor/*' GROUP BY HOUR"

# Explain
syna query mydb.db "EXPLAIN SELECT * FROM 'sensor/*' WHERE value > 100 LIMIT 10"
```

---

## v1.2.1 - Security Hardening + DAVO Persistence

**Date:** May 7, 2026
**PyPI:** [synadb 1.2.1](https://pypi.org/project/synadb/)
**Crates.io:** [synadb 1.2.1](https://crates.io/crates/synadb)

### Attribution

SynaDB is owned and maintained by **Mindoval, Inc**. Updated Licensor, Cargo.toml authors, pyproject.toml, and release workflow accordingly.

### Security Fixes

**Dependabot (1 alert):**

- Bumped pytest to `>=9.0.3` in 4 files (fixes tmpdir handling vulnerability)

**CodeQL (13 alerts):**

| Category | Count | Fix |
|----------|-------|-----|
| Workflow permissions | 1 | Added `permissions: contents: read` to `ci.yml` |
| Path injection | 1 | Validated user input against `[a-zA-Z0-9_]+` regex + realpath containment check in `studio.py` |
| Insecure temp file | 1 | Replaced `tempfile.mktemp` with `mkstemp` in `inference_demo.py` |
| Stack trace exposure | 10 | Replaced `str(e)` in API responses with generic messages + proper logging (2 in `studio.py`, 8 in `flask_app.py`) |

### New Feature: DAVO Persistence

DAVO indexes and predictors can now persist to disk.

**Rust API:**

```rust
// Save
index.save("freshness.bin")?;
predictor.save("predictor.bin")?;

// Load
let index = FreshnessIndexV2::load("freshness.bin")?;
let predictor = DecayPredictor::load("predictor.bin")?;
```

**Python API:**

```python
from synadb.davo import FreshnessIndex, DecayPredictor

# Save
idx.save("freshness.bin")
pred.save("predictor.bin")

# Load
idx = FreshnessIndex.load("my_index", "freshness.bin")
pred = DecayPredictor.load("my_pred", "predictor.bin")
```

**FFI Functions (4 new):**

- `SYNA_davo_freshness_index_save`, `SYNA_davo_freshness_index_load`
- `SYNA_davo_predictor_save`, `SYNA_davo_predictor_load`

**Format:** bincode with a version field for future migrations. The BTreeMap deadline index is rebuilt from persisted entries. The PRNG state is NOT persisted (a fresh RNG is seeded on predictor load).

**Tests:** 2 new property tests (Properties 25, 26) for round-trip correctness.

---

## v1.2.0 - DAVO: Decay-Aware Value Optimization

**Date:** May 4, 2026
**PyPI:** [synadb 1.2.0](https://pypi.org/project/synadb/)
**Crates.io:** [synadb 1.2.0](https://crates.io/crates/synadb)

### Highlights

- **DAVO** — New optional feature (`--features davo`) adding decay-aware storage
- **Bayesian Learning** — Automatic decay rate prediction with Thompson Sampling
- **Forward Decay** — O(k + log N) staleness queries instead of O(N)
- **Real-Data Demo** — Intel Berkeley Lab sensor dataset (50K readings)

### New Features

#### DAVO Module (`src/davo/`) (Experimental)

Every value can carry a decay rate λ, and freshness degrades over time as `e^(-λ × age_seconds)`. The system learns optimal decay rates automatically from observed data.

**Core Types:**

- `DAVOAtom` — Values with decay metadata (Static, Decaying, SelfImproving, Thunk)
- `FreshnessIndexV2` — Deadline-based BTreeMap index for scalable staleness queries
- `DecayPredictor` — Gamma conjugate prior with Thompson Sampling for exploration
- `OutcomeTracker` — TP/FP/TN/FN classification with asymmetric loss (FN = 10× FP)
- `Thunk` / `ThunkRegistry` — Lazy evaluation with probation-based garbage collection

**FFI Layer (14 functions):**

- `SYNA_davo_freshness_index_new`, `_insert`, `_get_freshness`, `_query_stale`, `_evict_stale`, `_len`, `_close`, `_free_keys`
- `SYNA_davo_predictor_new`, `_observe`, `_predict`, `_sample`, `_uncertainty`, `_close`

**Python Wrapper:**

```python
from synadb.davo import FreshnessIndex, DecayPredictor

with FreshnessIndex("sensors", threshold=0.5) as idx:
    idx.insert("sensor/temp", decay_rate=0.001)
    print(idx.get_freshness("sensor/temp"))  # ~1.0

with DecayPredictor("learner") as pred:
    for _ in range(100):
        pred.observe(0.05)
    print(pred.predict())  # ~0.05
```

**Tests:** 49 total (19 unit + 8 FFI + 7 property + 2 doc + 13 Python)

**Demos:**

- `demos/rust/davo_freshness.rs` — All DAVO components with synthetic data
- `demos/python/examples/davo_freshness.py` — IoT and cache scenarios
- `demos/python/examples/davo_intel_lab.py` — Real Intel Berkeley Lab sensor data

### Build

- Feature flag: `cargo build --features davo`
- Default builds completely unaffected
- `FreshnessIndex` V1 deprecated in favor of V2

---

## v1.1.2 - Internal Audit: Safety & Correctness Hardening

**Date:** March 18, 2026  
**Scope:** Rust source (`src/`) and integration tests (`tests/`)

A comprehensive internal audit of the Rust codebase targeting three areas: test hygiene, unsafe code elimination, and unwrap removal from library code. No public API changes. All existing tests continue to pass.

### Why This Was Necessary

The codebase had accumulated unjustified `unsafe` blocks (raw pointer writes where safe alternatives exist), `unwrap()` calls in library code that introduce panic paths in production, and test files with warnings or no content. This audit brings the code in line with the project's own coding standards: "Don't use `unwrap()` in library code" and Rust's principle of minimizing `unsafe` surface area.

### Phase 0: Test Hygiene

| Change | File | Reason |
|--------|------|--------|
| Removed 46 unnecessary `unsafe` blocks | `tests/sparse_ffi.rs` | `svs_*` FFI functions are declared as safe `extern "C"` (not `unsafe extern "C"`), so wrapping calls in `unsafe {}` was redundant and produced compiler warnings |
| Deleted empty test file | `tests/sparse_vector_roundtrip.rs` | File contained no tests (0 test runs). Dead code |

### Phase 1: Unsafe Code Elimination

Replaced unjustified `unsafe` raw pointer operations with safe byte-level alternatives. These changes also fix potential undefined behavior from unaligned memory access on ARM architectures.

| File | Change | Reason |
|------|--------|--------|
| `src/gwi.rs` | `ptr::write` / `copy_nonoverlapping` → `copy_from_slice(&val.to_le_bytes())` in `insert()` and `write_attractors()` | Raw pointer writes assumed aligned memory; safe byte-level writes are correct on all architectures |
| `src/mmap_vector.rs` | Same safe byte-level writes in `insert()` / `insert_batch()`; `from_raw_parts` → `align_to` + runtime check in `get_slice()` | Eliminates alignment UB and removes 5 unnecessary `unsafe` blocks |
| `src/cascade/mmap_store.rs` | Safe byte-level writes in `append()` | Same alignment safety concern as above |
| `src/mmap.rs` | `from_raw_parts` → `align_to` + alignment checks in all 6 slice methods | Slice reinterpretation without alignment verification was unsound on non-x86 platforms |
| `src/tensor.rs` | Same `align_to` fix in `MmapTensorRef` 4 slice methods | Consistent with `mmap.rs` fix |

### Phase 2: `unwrap()` Removal from Library Code

All `unwrap()` calls in non-test, non-doc-comment library code were replaced with safe alternatives. Test code and doc examples are exempt per project coding standards.

| File | Change | Reason |
|------|--------|--------|
| `src/experiment.rs` (3 locations) | `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` → `.unwrap_or_default()` | Can fail if system clock is before 1970; should degrade gracefully, not panic |
| `src/model_registry.rs` (1 location) | Same `SystemTime` fix | Same reason |
| `src/hnsw.rs` (2 locations) | `self.entry_point.unwrap()` → `match` with early return | Guard-then-unwrap pattern still has a separate panic path; `match` is zero-panic |
| `src/cascade/lsh.rs` (1 location) | `partial_cmp().unwrap()` → `.unwrap_or(std::cmp::Ordering::Equal)` | `f32::partial_cmp` returns `None` for NaN; unwrap would panic on NaN distances |
| `src/types.rs` (3 locations) | `slice.try_into().unwrap()` → direct array construction `[buf[0], buf[1], ...]` | Infallible in practice but introduces a panic path; direct construction is zero-panic |
| `src/tensor.rs` (2 locations) | Same direct array construction | Same reason |
| `src/mmap_vector.rs` (7 locations) | Same direct array construction | Same reason |
| `src/gwi.rs` (9 locations) | Same direct array construction | Same reason |
| `src/cascade/mmap_store.rs` (6 locations) | Same direct array construction | Same reason |
| `src/cascade/append_graph.rs` (3 locations) | Same direct array construction | Same reason |

### Phase 3: C/C++ and Python Demo Audit

Audited all Python wrapper files and C/C++ demo code for correctness against the actual FFI exports.

| File | Change | Reason |
|------|--------|--------|
| Python wrappers (7 files) | No changes needed | All correctly use `SYNA_*` (uppercase) FFI names |
| `demos/cpp/basic_usage.c` | ~50 `entangle_*`/`ENTANGLE_*` → `SYNA_*` | Old project name ("EntangleDB") never updated after rename; file would not compile |
| `demos/cpp/embedded_minimal.c` | ~35 `entangle_*`/`ENTANGLE_*` → `SYNA_*` | Same old project name issue; file would not compile |
| `demos/cpp/raii_wrapper.cpp` | ~40 lowercase `syna_*` → uppercase `SYNA_*` | Correct header but wrong case; actual FFI exports are uppercase. Would not link |
| `demos/cpp/cmake_example/main.cpp` | ~20 lowercase `syna_*` → uppercase `SYNA_*` | Same wrong-case issue; would not link |
| `demos/cpp/Makefile` | `entangle_db` → `synadb` in all 5 platform sections | Wrong library name; make would fail to find the shared library |
| `demos/cpp/README.md` | All function references updated to `SYNA_*` | Documentation matched neither old nor current API names |

### Verification

- `cargo build` — clean compile, zero errors
- `cargo clippy -- -D warnings` — zero warnings
- `cargo test` — 222 unit tests pass, 32 integration/property test files pass, 119 doc-tests pass
- No public API changes, no behavioral changes

### Known Remaining Items (Low Priority)

1. `ffi_sparse.rs` `cstr_to_str` returns `&'static str` — technically unsound lifetime, works in practice
2. `MmapVectorStore::get_slice` returns `None` on misalignment — callers silently skip instead of falling back to copy
3. `gpu.rs` `unsafe impl Send for GpuTensor` — strong safety claim, low risk (optional feature)
4. Byte-level writes may be marginally slower than bulk `copy_nonoverlapping` for large vectors — compiler likely optimizes identically

---

## v1.1.1 - Security Patch

**Released:** February 14, 2026  
**PyPI:** [synadb 1.1.1](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.1.1](https://crates.io/crates/synadb)

### Highlights

- **Security Fixes** - Updated 3 Python dependencies to address critical vulnerabilities
- **Documentation** - Fixed markdown formatting issues across 10 files

### Security Updates

Updated dependencies in `demos/v1.0.0/requirements-full.txt`:

| Package | Old Version | New Version | Vulnerability |
|---------|-------------|-------------|---------------|
| `semantic-kernel` | `>=0.4.0,<1.0.0` | `>=1.39.3` | Arbitrary File Write via AI Agent Function Calling |
| `mlflow` | `>=2.8.0,<3.0.0` | `>=3.5.0` | Unsafe deserialization + insecure temp permissions |
| `clearml` | `>=1.13.0,<2.0.0` | `>=2.0.2` | Path traversal in `safe_extract` |

### Documentation Fixes

Fixed markdown linting issues (Codacy) in:

- `benchmarks/README.md`
- `demos/cpp/README.md`
- `demos/python/synadb/STUDIO_DOCS.md`
- `demos/huggingface/README.md`
- `demos/README.md`
- `wiki/API-Reference.md`
- `wiki/Architecture.md`
- `wiki/Changelog.md`
- `wiki/Contributing.md`
- `CONTRIBUTING.md`

---

## v1.1.0 - Sparse Vector Store

**Released:** January 10, 2026  
**PyPI:** [synadb 1.1.0](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.1.0](https://crates.io/crates/synadb)

### Highlights

- **SparseVectorStore (SVS)** - New inverted index for lexical embeddings (SPLADE, BM25, TF-IDF)
- **Hybrid RAG Example** - Amazon ESCI dataset with dense+sparse search
- **Benchmark Script** - Scale testing showing GWI 13-29x faster than Cascade

### New Features

#### SparseVectorStore

A new inverted index for lexical/sparse embeddings:

```python
from synadb import SparseVectorStore

store = SparseVectorStore("lexical.svs")
store.index("doc1", {101: 0.8, 2054: 0.5, 3000: 0.3})
results = store.search({101: 1.0, 2054: 0.5}, k=10)
store.save("index.svs")
```

#### Hybrid RAG Example

Updated `hybrid_rag_bge_m3.py` with:

- Amazon ESCI dataset (US locale) for product search
- GWI (real-time dense) + Cascade (historical dense) + SVS (lexical sparse)
- Reciprocal Rank Fusion (RRF) for hybrid search
- BGE-M3 native sparse embeddings via FlagEmbedding

#### Benchmark Results

| Scale | GWI | Cascade | GWI/Cascade |
|-------|-----|---------|-------------|
| 100K | 28,513 vec/sec | 2,179 vec/sec | 13.1x |
| 500K | 29,959 vec/sec | 1,033 vec/sec | 29.0x |

---

## v1.0.6 - GWI Persistence Fix

**Released:** January 2, 2026  
**PyPI:** [synadb 1.0.6](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.6](https://crates.io/crates/synadb)

### Highlights

- **GWI Persistence Fix** - Critical bug fix for GravityWellIndex data persistence
- **CascadeIndex Import Fix** - Fixed Python import error
- **Benchmark Results** - GWI is 388x faster to build than HNSW
- **Interactive Playground** - New web-based playground for validating SynaDB claims

### New Features

#### Interactive Playground (`website/playground.html`)

A new web-based playground for validating SynaDB's performance claims:

- **Claim Validation Cards** - Visual overview of all SynaDB features organized by category (Core Storage, Vector Search, AI/ML)
- **Google Colab Integration** - One-click access to run real benchmarks with the actual `synadb` package at 1M, 10M, and 100M scale
- **PythonAnywhere Live Demo** - Hosted demo at [gtava5813.pythonanywhere.com](https://gtava5813.pythonanywhere.com/) for instant testing without setup
- **Copy-Paste Benchmarks** - Ready-to-run Python and Rust code samples for local verification

**Benchmarks Available:**

- MmapVectorStore vs VectorStore (batch insert speed)
- GWI vs HNSW (index build time)
- HNSW vs Brute Force (search speed)
- Schema-Free Storage, Crash Recovery, Tensor Extraction, Compression

### Bug Fixes

#### GWI Persistence Bug (Critical)

**Issue:** GravityWellIndex data was not persisted correctly. After inserting vectors and closing, reopening the same file showed `len(gwi) = 0`.

**Root Cause:**

1. Missing `SYNA_gwi_open` FFI function - Python couldn't open existing files
2. Python wrapper always called `SYNA_gwi_new` which truncates existing files
3. Unsafe memory alignment in mmap reads caused undefined behavior

**Fix:**

- Added `SYNA_gwi_open` FFI function for opening existing GWI files
- Python wrapper now detects existing files and opens them instead of truncating
- Rewrote header serialization using safe byte-by-byte operations
- Fixed `load_attractors`, `read_entry_at`, `rebuild_key_index` to read floats safely

**Files Changed:**

- `src/ffi.rs` - Added `SYNA_gwi_open` function
- `src/gwi.rs` - Fixed header read/write, fixed mmap float reads
- `demos/python/synadb/gwi.py` - Added open logic, `SYNA_gwi_open` binding

**Verification:**
```python
from synadb import GravityWellIndex
import numpy as np

# Create and populate
gwi = GravityWellIndex("test.gwi", dimensions=128)
sample = np.random.randn(100, 128).astype(np.float32)
gwi.initialize(sample)
gwi.insert_batch([f"item_{i}" for i in range(1000)], 
                 np.random.randn(1000, 128).astype(np.float32))
print(f"Before close: len={len(gwi)}")  # 1000
gwi.close()

# Reopen - NOW WORKS!
gwi2 = GravityWellIndex("test.gwi", dimensions=128)
print(f"After reopen: len={len(gwi2)}")  # 1000 ✓
```

#### CascadeIndex Import Fix

**Issue:** `from synadb import CascadeIndex` failed with `ImportError: cannot import name '_get_lib'`

**Root Cause:** `cascade.py` referenced non-existent `_get_lib` function from `wrapper.py`

**Fix:** Added local `_load_library()` function to `cascade.py`, consistent with other modules

**Files Changed:**

- `demos/python/synadb/cascade.py` - Added `_load_library()` function

### New Tests

| Test | Description |
|------|-------------|
| `test_gwi_persistence_roundtrip` | Verifies data persists across close/reopen |
| `test_gwi_search_after_reopen` | Verifies search works after reopen |
| `test_gwi_empty_persistence` | Verifies empty GWI can be created |

### New FFI Function

| Function | Description |
|----------|-------------|
| `SYNA_gwi_open` | Open existing GravityWellIndex file |

### Benchmark Results (Amazon Fine Food Reviews - 100K vectors)

| Metric | GWI | HNSW | Winner |
|--------|-----|------|--------|
| Total build | 2.94s | 1141.5s | **GWI (388x faster)** |
| Search p50 | 0.40ms | 0.89ms | **GWI (2.2x faster)** |
| Insert rate | 92K/sec | 715K/sec | HNSW |
| File size | 154.9MB | - | - |

---

## v1.0.5 - MmapVectorStore, GWI & Cascade Index

**Released:** January 2026  
**PyPI:** [synadb 1.0.5](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.5](https://crates.io/crates/synadb)

### Highlights

- **MmapVectorStore** - Ultra-high-throughput vector storage (7x faster than VectorStore)
- **Gravity Well Index (GWI)** - Novel append-only vector indexing algorithm
- **Cascade Index** - O(N) build time index with no initialization required
- **Extended Dimensions** - Support for 384-7168 dimensional embeddings
- **Critical Fixes** - HNSW auto-build, index persistence, sync_on_write

### New Features

#### Cascade Index (NEW)

A novel vector index combining LSH + Adaptive Buckets + Sparse Graph for O(N) build time without requiring initialization samples:

```python
from synadb import CascadeIndex
import numpy as np

# Create index - no initialization required!
index = CascadeIndex("vectors.cascade", dimensions=768)

# Insert vectors
index.insert("doc1", embedding)
index.insert_batch(keys, vectors)

# Search with configurable parameters
results = index.search(query, k=10)
results = index.search(query, k=10, num_probes=5, ef_search=100)  # Higher recall
```

**Key Features:**

- **O(N) build time** - No quadratic neighbor search during construction
- **No initialization required** - Unlike GWI, no sample vectors needed
- **Adaptive buckets** - Automatically splits as data grows
- **Multi-probe LSH** - Smart probing for better recall
- **Sparse graph refinement** - Graph-based search for high accuracy
- **Full persistence** - Save/load index to disk

**Configuration Presets:**

```python
from synadb.cascade import CascadeConfig

# Default (balanced)
config = CascadeConfig()

# Small datasets (< 10K vectors)
config = CascadeConfig.small()

# Large datasets (> 100K vectors)
config = CascadeConfig.large()

# High recall (> 99%)
config = CascadeConfig.high_recall()

# Fast search
config = CascadeConfig.fast_search()
```

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                      Cascade Index                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  LSH Layer                           │    │
│  │  Random hyperplane hashing with multi-probe support  │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Adaptive Bucket Tree                    │    │
│  │  Auto-splitting buckets for O(1) amortized lookup    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Sparse Graph Layer                   │    │
│  │  Neighbor connections for search refinement          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**When to use Cascade vs GWI vs HNSW:**

| Aspect | Cascade | GWI | HNSW |
|--------|---------|-----|------|
| Build time | O(N) | O(N) | O(N²) |
| Initialization | None | Requires samples | None |
| Adaptability | Adaptive buckets | Fixed attractors | Fixed structure |
| Best for | General use | Streaming data | Query-heavy |

#### MmapVectorStore

Memory-mapped vector storage for maximum throughput:

```python
from synadb import MmapVectorStore

store = MmapVectorStore("vectors.mmap", dimensions=768, initial_capacity=100_000)
store.insert_batch(keys, vectors)  # 7x faster than VectorStore
store.build_index()
results = store.search(query, k=10)  # 0.6ms
```

**Benchmark Results (10,000 vectors):**

| Model | Dims | Write/sec | Search | Storage |
|-------|------|-----------|--------|---------|
| MiniLM | 384 | 766,642 | 0.3ms | 18.8MB |
| BERT | 768 | 489,733 | 0.6ms | 34.9MB |
| OpenAI ada-002 | 1536 | 278,369 | 1.4ms | 67.2MB |
| DeepSeek-V3 | 7168 | 64,103 | 5.7ms | 303.5MB |

**Trade-offs vs VectorStore:**

| Aspect | VectorStore | MmapVectorStore |
|--------|-------------|-----------------|
| Write speed | ~67K/sec | ~490K/sec |
| Durability | Per-write | Checkpoint |
| Capacity | Dynamic | Pre-allocated |

#### Gravity Well Index (GWI)

A novel vector indexing algorithm designed for append-only, mmap-friendly architecture:

```python
from synadb import GravityWellIndex

gwi = GravityWellIndex("vectors.gwi", dimensions=768)
gwi.initialize(sample_vectors)  # Initialize attractors from sample
gwi.insert_batch(keys, vectors)
results = gwi.search(query, k=10, nprobe=50)  # 98% recall
```

**Performance vs HNSW:**

| Dataset | GWI Build | HNSW Build | Speedup |
|---------|-----------|------------|---------|
| 10K × 384 | 1.0s | 8.8s | 8.6x |
| 10K × 768 | 2.1s | 18.4s | 8.9x |
| 50K × 384 | 1.5s | 272s | 186x |
| 50K × 768 | 3.0s | 504s | 169x |

**Recall vs nprobe:**

| nprobe | Recall@10 | Latency |
|--------|-----------|---------|
| 3 | ~50% | 0.23ms |
| 10 | ~70% | 0.37ms |
| 30 | ~90% | 0.59ms |
| 50 | ~98% | 0.68ms |
| 100 | ~100% | 0.86ms |

**When to use GWI vs HNSW:**

- **GWI:** Index build time critical, streaming/real-time data, append-only required
- **HNSW:** Search latency critical, index built once and queried many times

### Bug Fixes

#### HNSW Recall Bug (Critical)

**Issue:** Both VectorStore and MmapVectorStore had 0-20% recall on 10K+ clustered vectors.

**Root Cause:** Entry point and `max_level` not updated when adding nodes with higher levels. In HNSW, the entry point must always be the node with the highest level.

**Fix:**

- Added `set_max_level()` method to `HnswIndex`
- Correctly update both entry point AND max_level in `add_node_to_index()`
- Fixed in both `VectorStore` and `MmapVectorStore`

**Files Changed:**
- `src/hnsw.rs` - Added `set_max_level()` method
- `src/vector.rs` - Fixed `add_node_to_index()`
- `src/mmap_vector.rs` - Fixed `add_node_to_index()` and `insert_to_hnsw_incremental()`

| Component | Before | After |
|-----------|--------|-------|
| MmapVectorStore | 0% recall | 100% recall |
| VectorStore | 20% recall | 100% recall |

#### HNSW Auto-Build Fix (Critical)

**Issue:** HNSW index was never automatically built during inserts, causing all searches to fall back to O(N) brute-force (11+ seconds per query on 59K vectors).

**Fix:** Added auto-build logic in `insert()` that triggers index building when vector count exceeds `index_threshold`.

| Metric | Before | After |
|--------|--------|-------|
| Search (59K vectors) | 11,000ms | <1ms |

#### HNSW Index Persistence

**Issue:** HNSW index was not saved/loaded on close/open, requiring rebuild on every reopen.

**Fix:**

- Auto-load existing `.hnsw` index files on open
- Auto-save index after `build_index()`
- Added `save_index()` and `flush()` methods

#### VectorStore Close/Flush

**Issue:** FFI global registry prevented proper cleanup, index never saved.

**Fix:** Added explicit `close()` and `flush()` FFI functions with Python context manager support.

```python
with VectorStore("vectors.db", dimensions=768) as store:
    # ... operations ...
# Automatically saved on exit
```

#### sync_on_write Configuration

**Issue:** Default `sync_on_write=True` limited throughput to ~18-100 ops/sec.

**Fix:** Exposed `sync_on_write` parameter in both SynaDB and VectorStore.

```python
# High-throughput mode (456x faster)
store = VectorStore("vectors.db", dimensions=768, sync_on_write=False)
```

| Setting | Throughput |
|---------|------------|
| `sync_on_write=True` | 19 ops/sec |
| `sync_on_write=False` | 8,675 ops/sec |

### New Files

| File | Description |
|------|-------------|
| `src/mmap_vector.rs` | Rust MmapVectorStore implementation |
| `src/gwi.rs` | Gravity Well Index implementation |
| `demos/python/synadb/mmap_vector.py` | Python MmapVectorStore wrapper |
| `demos/python/synadb/gwi.py` | Python GWI wrapper |

### New FFI Functions

| Function | Description |
|----------|-------------|
| `SYNA_vector_store_build_index` | Manually build HNSW index |
| `SYNA_vector_store_has_index` | Check if index exists |
| `SYNA_vector_store_close` | Close and save index |
| `SYNA_vector_store_flush` | Save index without closing |
| `SYNA_vector_store_new_with_config` | Create with sync_on_write option |
| `SYNA_open_with_config` | Open SynaDB with sync_on_write option |

---

## v1.0.3 - PyPI Native Library Fix

**Released:** January 2026  
**PyPI:** [synadb 1.0.3](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.3](https://crates.io/crates/synadb)

### Fixed

#### PyPI Native Library Bundling

- **Fixed:** `pip install synadb` now works on all platforms (Linux, macOS, Windows)
- **Issue:** Previous releases only bundled the Linux x86_64 native library
- **Solution:** Release workflow now copies all platform libraries into the PyPI package

#### Platform Support

| Platform | Library |
|----------|---------|
| Linux x86_64 | `libsynadb.so` |
| macOS x86_64 | `libsynadb-x86_64.dylib` |
| macOS ARM64 (Apple Silicon) | `libsynadb-arm64.dylib` |
| Windows x86_64 | `synadb.dll` |

#### Python Wrapper

- Enhanced `_find_library()` to detect platform AND architecture
- macOS ARM64 (Apple Silicon) now correctly loads ARM-specific library
- Library search now checks inside installed package directory first

---

## v1.0.2 - Bug Fixes

**Released:** January 2026

### Fixed

- Minor bug fixes and stability improvements

---

## v1.0.0 - Production Release

**Released:** January 2026  
**PyPI:** [synadb 1.0.0](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 1.0.0](https://crates.io/crates/synadb)

The first production-ready release of SynaDB with full AI/ML ecosystem integration.

### Highlights

- 🚀 **Production Ready** - Stable API, performance guarantees
- 🔗 **LLM Integrations** - LangChain, LlamaIndex, Haystack
- 🤖 **ML Integrations** - PyTorch Dataset/DataLoader, TensorFlow tf.data
- 🛠️ **Native Tools** - CLI and Studio Web UI
- ⚡ **Performance** - GPU Direct, FAISS, Memory-mapped I/O

### New Features

#### LLM Framework Integrations

##### LangChain

```python
from synadb.integrations.langchain import (
    SynaVectorStore,
    SynaChatMessageHistory,
    SynaDocumentLoader
)

# Vector store for RAG
vectorstore = SynaVectorStore.from_documents(documents, embedding, path="langchain.db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Chat history persistence
history = SynaChatMessageHistory(path="chat.db", session_id="user_123")
```

##### LlamaIndex

```python
from synadb.integrations.llamaindex import SynaVectorStore, SynaChatStore

vector_store = SynaVectorStore(path="index.db", dimensions=1536)
chat_store = SynaChatStore(path="chats.db")
```

##### Haystack

```python
from synadb.integrations.haystack import SynaDocumentStore

store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
```

#### ML Framework Integrations

##### PyTorch

```python
from synadb.torch import SynaDataset, SynaDataLoader, create_distributed_loader

dataset = SynaDataset(path="data.db", pattern="train/*")
loader = SynaDataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Distributed training support
loader, sampler = create_distributed_loader(dataset, batch_size=32)
```

##### TensorFlow

```python
from synadb.tensorflow import syna_dataset, create_distributed_dataset

dataset = syna_dataset(path="data.db", pattern="train/*", batch_size=32)

# Distributed training with tf.distribute
strategy = tf.distribute.MirroredStrategy()
dist_dataset = create_distributed_dataset(path="data.db", pattern="train/*", batch_size=32)
```

#### Native Tools

##### Syna CLI

Command-line interface for database inspection and management:

```bash
syna info mydata.db          # Database statistics
syna keys mydata.db          # List all keys
syna get mydata.db key       # Get value
syna export mydata.db out.json  # Export to JSON
```

##### Syna Studio Web UI

Web-based database explorer with:

- Keys Explorer with search and type filtering
- Model Registry dashboard
- 3D Embedding Clusters visualization (PCA)
- Statistics dashboard with customizable widgets
- Integrations scanner
- Custom Suite (compact, export, integrity check)
- Database switcher

```bash
# Launch with test data
cd demos/python/synadb
python run_ui.py --test

# Launch with HuggingFace embeddings
python run_ui.py --test --use-hf --samples 200

# Open existing database
python run_ui.py path/to/database.db
```

#### Performance Features

##### GPU Direct (Optional)

```python
from synadb.gpu import get_tensor_cuda, is_gpu_available

if is_gpu_available():
    tensor = get_tensor_cuda("data.db", "train/*", device=0)
```

##### FAISS Integration (Optional)

```rust
// Rust API with FAISS feature
let config = FaissConfig {
    index_type: "IVF1024,Flat".to_string(),
    train_size: 10000,
    nprobe: 10,
    use_gpu: false,
};
let mut index = FaissIndex::new(768, DistanceMetric::Cosine, config)?;
```

##### Memory-Mapped I/O

```rust
// Zero-copy tensor access
use synadb::mmap::{MmapReader, MmapTensorRef};

let reader = MmapReader::new("data.db")?;
let tensor_ref = reader.get_tensor_ref("weights")?;
```

### New Rust Modules

| Module | File | Description |
|--------|------|-------------|
| GPU Direct | `gpu.rs` | CUDA bindings for GPU memory access |
| FAISS Index | `faiss_index.rs` | FAISS wrapper for billion-scale search |
| Memory-Mapped I/O | `mmap.rs` | Zero-copy tensor access |

### New Python Modules

| Module | File | Description |
|--------|------|-------------|
| PyTorch | `torch.py` | Dataset, DataLoader, DistributedSampler |
| TensorFlow | `tensorflow.py` | tf.data.Dataset integration |
| Studio | `studio.py` | Flask-based web UI |
| GPU | `gpu.py` | Python GPU wrapper |
| MLflow | `integrations/mlflow.py` | MLflow backend |

---

## v0.5.0 - AI Platform Release

**Released:** December 2025  
**PyPI:** [synadb 0.5.0](https://pypi.org/project/synadb/)  
**Crates.io:** [synadb 0.5.0](https://crates.io/crates/synadb)

This is a major feature release that transforms SynaDB from a vector database into a complete AI/ML platform.

### Highlights

- 🚀 **HNSW Index** - O(log N) approximate nearest neighbor search
- 📊 **Tensor Engine** - Batch tensor operations with chunked storage
- 📦 **Model Registry** - Version models with SHA-256 checksum verification
- 🔬 **Experiment Tracking** - MLflow-style experiment logging
- 🔗 **LLM Integrations** - LangChain, LlamaIndex, and Haystack support

### New Features

#### HNSW Index

The Hierarchical Navigable Small World (HNSW) index provides fast approximate nearest neighbor search:

```python
from synadb import VectorStore

store = VectorStore("vectors.db", dimensions=768)

# Insert vectors (HNSW index builds automatically)
for doc_id, embedding in embeddings:
    store.insert(doc_id, embedding)

# Search is now O(log N) instead of O(N)
results = store.search(query_embedding, k=10)
```

**Features:**
- Multi-layer graph structure for efficient search
- Configurable parameters (M, ef_construction, ef_search)
- Automatic index building when vector count exceeds threshold (default: 1000)
- Save/load persistence to `.hnsw` sidecar files
- 95%+ recall on standard benchmarks

#### Tensor Engine

Batch tensor operations for ML training pipelines:

```python
from synadb import TensorEngine
import numpy as np

engine = TensorEngine("data.db")

# Store large tensors with automatic chunking
X_train = np.random.randn(10000, 768).astype(np.float32)
engine.put_tensor_chunked("train/X", X_train)

# Retrieve with shape preservation
X, shape = engine.get_tensor_chunked("train/X")

# Stream batches for training
for batch in engine.stream_batches("train/*", batch_size=32):
    model.train_step(batch)
```

**Features:**
- Pattern-based key matching (`sensor/*`, `train/*`)
- Chunked blob storage for large tensors (1MB chunks)
- Support for Float32, Float64, Int32, Int64 dtypes
- Memory-mapped access for zero-copy reads
- Direct I/O support for high throughput

#### Model Registry

Version and manage ML models with integrity verification:

```python
from synadb import ModelRegistry

registry = ModelRegistry("models.db")

# Save model with automatic versioning
version = registry.save_model(
    "classifier",
    model_bytes,
    metadata={"accuracy": "0.95", "framework": "pytorch"}
)

# Load with checksum verification
data, info = registry.load_model("classifier")
print(f"Version: {info.version}, Checksum: {info.checksum[:16]}...")

# Stage management
registry.set_stage("classifier", version.version, "Production")
prod_model = registry.get_production("classifier")
```

**Features:**
- Automatic version numbering
- SHA-256 checksum computation and verification
- Stage management (Development → Staging → Production → Archived)
- Metadata storage per version
- Corruption detection on load

#### Experiment Tracking

MLflow-style experiment logging built into the database:

```python
from synadb import ExperimentTracker

tracker = ExperimentTracker("experiments.db")

# Start a run
run_id = tracker.start_run("mnist", tags=["baseline", "cnn"])

# Log parameters and metrics
tracker.log_param(run_id, "learning_rate", "0.001")
tracker.log_param(run_id, "batch_size", "32")

for epoch in range(100):
    loss = train_epoch()
    tracker.log_metric(run_id, "loss", loss, step=epoch)
    tracker.log_metric(run_id, "accuracy", accuracy, step=epoch)

# Log artifacts
tracker.log_artifact(run_id, "model.pt", model_bytes)

# End run
tracker.end_run(run_id, "Completed")

# Query runs
runs = tracker.query_runs("mnist", status="Completed", sort_by="accuracy")
```

**Features:**

- UUID-based run IDs
- Parameter and metric logging with step numbers
- Artifact storage (models, plots, configs)
- Run status management (Running, Completed, Failed, Killed)
- Query and filter runs by experiment, status, tags, parameters

#### LLM Framework Integrations

##### LangChain

```python
from langchain_openai import OpenAIEmbeddings
from synadb.integrations.langchain import SynaVectorStore, SynaChatMessageHistory

# Vector store for RAG
vectorstore = SynaVectorStore.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
    path="langchain.db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Chat history persistence
history = SynaChatMessageHistory(path="chat.db", session_id="user_123")
history.add_user_message("Hello!")
history.add_ai_message("Hi there!")
```

##### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext
from synadb.integrations.llamaindex import SynaVectorStore

vector_store = SynaVectorStore(path="index.db", dimensions=1536)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
```

##### Haystack

```python
from synadb.integrations.haystack import SynaDocumentStore

store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
store.write_documents(documents)
results = store.filter_documents(filters={"category": "tech"})
```

**Note:** LangChain and LlamaIndex VectorStore integrations store document metadata in-memory only. Vectors are fully persisted. See [Known Limitations](#known-limitations) for details.

### Improvements

- VectorStore now supports optional HNSW indexing with automatic fallback to brute-force
- Improved error messages with more context
- Better memory management for large tensor operations
- FFI layer now uses canonicalized paths for consistent registry lookups

### Known Limitations

#### LangChain/LlamaIndex Metadata Persistence

When using `SynaVectorStore` with LangChain or LlamaIndex, document metadata (text content, custom metadata) is stored in-memory only. This means:

- ✅ Vectors persist across application restarts
- ✅ Similarity search works correctly
- ❌ Metadata is lost when the application restarts

**Workaround:** Store metadata in a separate database file:

```python
vectorstore = SynaVectorStore(path="vectors.db", ...)
metadata_db = SynaDB("metadata.db")  # Separate file for metadata
```

This limitation will be addressed in a future release by adding native metadata support to the Rust VectorStore.

---

## v0.2.0 - Vector Store

**Released:** December 8, 2025

This release adds vector embedding storage and similarity search capabilities.

### New Features

#### Vector Store

```python
from synadb import VectorStore
import numpy as np

store = VectorStore("vectors.db", dimensions=768, metric="cosine")

# Insert embeddings
store.insert("doc1", embedding1)
store.insert("doc2", embedding2)

# Search for similar vectors
results = store.search(query_embedding, k=5)
for r in results:
    print(f"{r.key}: {r.score:.4f}")
```

**Features:**

- `Atom::Vector(Vec<f32>, u16)` type for embedding storage
- Brute-force k-NN search (HNSW added in v0.5.0)
- Distance metrics: Cosine, Euclidean, DotProduct
- Dimension validation (64-8192)
- Python `VectorStore` class with NumPy integration

#### FFI Extensions

New C-ABI functions for vector operations:
- `SYNA_put_vector()` - Store vectors
- `SYNA_get_vector()` - Retrieve vectors
- `SYNA_free_vector()` - Free vector memory

### Property Tests

- Property 17: Vector Serialization Round-Trip
- Property 18: Similarity Search Correctness

---

## v0.1.0 - Core Database

**Released:** December 7, 2025

The initial release of SynaDB - an AI-native embedded database.

### Features

#### Core Database

```python
from synadb import SynaDB

with SynaDB("my.db") as db:
    # Write values
    db.put_float("sensor/temp", 72.5)
    db.put_int("counter", 42)
    db.put_text("config/name", "production")
    
    # Read values
    temp = db.get_float("sensor/temp")
    
    # Get history as tensor (for ML)
    tensor = db.get_history_tensor("sensor/temp")
```

**Features:**

- `Atom` enum: Null, Float, Int, Text, Bytes
- Append-only log storage with crash recovery
- In-memory index with O(1) key lookup
- Tombstone-based deletion
- Compaction to reclaim space

#### Compression

- LZ4 compression for values > 64 bytes
- Delta compression for consecutive floats
- Transparent decompression on read

#### FFI Layer

- C-ABI interface with `extern "C"` functions
- Global registry for instance management
- Panic safety with `catch_unwind`
- Integer error codes

#### Python Bindings

- `SynaDB` class with ctypes bindings
- Context manager support (`with` statement)
- NumPy integration for tensor extraction

### Property Tests (16 total)

| # | Property | Description |
|---|----------|-------------|
| 1 | Atom Serialization Round-Trip | Atoms serialize and deserialize correctly |
| 2 | LogHeader Serialization Round-Trip | Headers serialize and deserialize correctly |
| 3 | Write-Read Round-Trip | Written values can be read back |
| 4 | Index Rebuild on Reopen | Index rebuilds correctly after crash |
| 5 | Database Instance Isolation | Multiple databases don't interfere |
| 6 | Tensor Extraction Correctness | Float history extracts correctly |
| 7 | Tensor Filters Non-Float Types | Non-floats are filtered from tensors |
| 8 | Delta Compression Reduces Storage | Delta encoding saves space |
| 9 | LZ4 Compression Round-Trip | Compressed values decompress correctly |
| 10 | Concurrent Writes Preserve All Data | Parallel writes don't lose data |
| 11 | Corruption Recovery Skips Bad Entries | Bad entries are skipped on recovery |
| 12 | Schema-Free Key Acceptance | Any valid UTF-8 key is accepted |
| 13 | Delete Makes Key Unreadable | Deleted keys return None |
| 14 | Delete-Write Resurrection | Writing after delete resurrects key |
| 15 | Compaction Preserves Latest Values | Compaction keeps latest values |
| 16 | History Excludes Post-Tombstone | History stops at tombstone |

---

## Installation

### Python

```bash
pip install synadb
```

### Rust

```toml
[dependencies]
synadb = "1.0.6"
```

### Building from Source

```bash
git clone https://github.com/gtava5813/SynaDB.git
cd SynaDB
cargo build --release
```

---

## Links

- [GitHub Repository](https://github.com/gtava5813/SynaDB)
- [PyPI Package](https://pypi.org/project/synadb/)
- [Crates.io](https://crates.io/crates/synadb)
- [API Reference](API-Reference)
- [Getting Started Guide](Getting-Started)

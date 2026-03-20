# SynaDB C/C++ Demos

This directory contains C and C++ examples demonstrating how to use the SynaDB database.

## Prerequisites

Before building the demos, you must build the SynaDB library:

```bash
# From the repository root
cargo build --release
```

This creates the shared library:

- Linux: `target/release/libsynadb.so`
- macOS: `target/release/libsynadb.dylib`
- Windows: `target/release/synadb.dll`

## Demos

### 1. Basic Usage (C)

**File:** `basic_usage.c`

Demonstrates fundamental SynaDB operations in C:

- Opening and closing databases
- Writing all data types (float, int, text, bytes)
- Reading values back
- Deleting keys
- Listing keys
- Error handling
- Time-series data and tensor extraction
- Database compaction

### 2. RAII Wrapper (C++)

**File:** `raii_wrapper.cpp`

A modern C++ wrapper with:

- RAII semantics (automatic resource management)
- Smart pointers for memory safety
- Exception-safe operations
- `std::optional` for nullable returns
- Move semantics
- Range-based iteration for tensors

### 3. CMake Integration

**Directory:** `cmake_example/`

Shows how to integrate SynaDB into CMake projects:

- Cross-platform CMakeLists.txt
- Library detection and linking
- RPATH configuration
- Windows DLL handling

### 4. Embedded Minimal (C)

**File:** `embedded_minimal.c`

Optimized for constrained environments:

- No dynamic allocation in hot paths
- Fixed-size ring buffers
- Stack-based operations
- Immediate memory cleanup
- Memory usage documentation

## Building with Make

```bash
# Build all demos
make

# Build specific demo
make basic_usage
make raii_wrapper
make embedded_minimal

# Run all demos
make run

# Clean build artifacts
make clean
```

## Building with CMake

```bash
cd cmake_example
mkdir build && cd build
cmake ..
cmake --build .
./syna_cmake_demo
```

## Platform Support

| Platform | Compiler | Status |
|----------|----------|--------|
| Linux | GCC, Clang | ✅ |
| macOS | Clang, GCC | ✅ |
| Windows | MSVC, MinGW | ✅ |

## Memory Management

When using the C API, you must free memory returned by these functions:

| Function | Free With |
|----------|-----------|
| `SYNA_get_text()` | `SYNA_free_text()` |
| `SYNA_get_bytes()` | `SYNA_free_bytes()` |
| `SYNA_get_history_tensor()` | `SYNA_free_tensor()` |
| `SYNA_keys()` | `SYNA_free_keys()` |

The C++ RAII wrapper handles this automatically using smart pointers.

## Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `SYNA_SUCCESS` | Operation succeeded |
| 0 | `SYNA_ERR_GENERIC` | Generic error |
| -1 | `SYNA_ERR_DB_NOT_FOUND` | Database not in registry |
| -2 | `SYNA_ERR_INVALID_PATH` | Invalid path or UTF-8 |
| -3 | `SYNA_ERR_IO` | I/O error |
| -4 | `SYNA_ERR_SERIALIZATION` | Serialization error |
| -5 | `SYNA_ERR_KEY_NOT_FOUND` | Key not found |
| -6 | `SYNA_ERR_TYPE_MISMATCH` | Wrong type for key |
| -7 | `SYNA_ERR_EMPTY_KEY` | Empty key not allowed |
| -8 | `SYNA_ERR_KEY_TOO_LONG` | Key exceeds 65535 bytes |
| -100 | `SYNA_ERR_INTERNAL_PANIC` | Internal error |

## Example: Quick Start

```c
#include "synadb.h"

int main() {
    // Open database
    SYNA_open("my.db");
    
    // Write values
    SYNA_put_float("my.db", "temperature", 23.5);
    SYNA_put_int("my.db", "count", 42);
    
    // Read values
    double temp;
    SYNA_get_float("my.db", "temperature", &temp);
    printf("Temperature: %f\n", temp);
    
    // Close database
    SYNA_close("my.db");
    return 0;
}
```

## Requirements

- C11 compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- C++17 compiler for RAII wrapper (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+ (for CMake example)

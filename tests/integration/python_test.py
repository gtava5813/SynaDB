#!/usr/bin/env python3
"""
End-to-end integration test for syna database using Python ctypes.

This test verifies the C-ABI FFI interface works correctly across the language boundary.
Tests the full workflow: open, put_float, put_text, get_float, get_history_tensor, free_tensor, close.

Requirements: 6.1

Usage:
    1. Build the library: cargo build --release
    2. Run this test: python tests/integration/python_test.py
"""

import ctypes
import os
import sys
import tempfile
import platform
from pathlib import Path


def find_library():
    """Find the compiled synadb library."""
    # Determine library extension based on platform
    if platform.system() == "Windows":
        lib_name = "synadb.dll"
    elif platform.system() == "Darwin":
        lib_name = "libsynadb.dylib"
    else:
        lib_name = "libsynadb.so"
    
    # Search paths
    search_paths = [
        Path("target/release") / lib_name,
        Path("target/debug") / lib_name,
        Path("..") / "target" / "release" / lib_name,
        Path("..") / "target" / "debug" / lib_name,
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path.resolve())
    
    raise FileNotFoundError(
        f"Could not find {lib_name}. Please run 'cargo build --release' first.\n"
        f"Searched: {[str(p) for p in search_paths]}"
    )


def setup_library(lib_path):
    """Load and configure the library with proper function signatures."""
    lib = ctypes.CDLL(lib_path)
    
    # syna_open(path: *const c_char) -> i32
    lib.syna_open.argtypes = [ctypes.c_char_p]
    lib.syna_open.restype = ctypes.c_int32
    
    # syna_close(path: *const c_char) -> i32
    lib.syna_close.argtypes = [ctypes.c_char_p]
    lib.syna_close.restype = ctypes.c_int32
    
    # syna_put_float(path, key, value) -> i64
    lib.syna_put_float.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
    lib.syna_put_float.restype = ctypes.c_int64
    
    # syna_put_int(path, key, value) -> i64
    lib.syna_put_int.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]
    lib.syna_put_int.restype = ctypes.c_int64
    
    # syna_put_text(path, key, value) -> i64
    lib.syna_put_text.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.syna_put_text.restype = ctypes.c_int64
    
    # syna_get_float(path, key, out) -> i32
    lib.syna_get_float.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
    lib.syna_get_float.restype = ctypes.c_int32
    
    # syna_get_int(path, key, out) -> i32
    lib.syna_get_int.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]
    lib.syna_get_int.restype = ctypes.c_int32
    
    # syna_get_history_tensor(path, key, out_len) -> *mut f64
    lib.syna_get_history_tensor.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.syna_get_history_tensor.restype = ctypes.POINTER(ctypes.c_double)
    
    # syna_free_tensor(ptr, len)
    lib.syna_free_tensor.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.syna_free_tensor.restype = None
    
    # syna_delete(path, key) -> i32
    lib.syna_delete.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.syna_delete.restype = ctypes.c_int32
    
    # syna_exists(path, key) -> i32
    lib.syna_exists.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.syna_exists.restype = ctypes.c_int32
    
    # syna_compact(path) -> i32
    lib.syna_compact.argtypes = [ctypes.c_char_p]
    lib.syna_compact.restype = ctypes.c_int32
    
    return lib


# Error codes
ERR_SUCCESS = 1
ERR_GENERIC = 0
ERR_DB_NOT_FOUND = -1
ERR_INVALID_PATH = -2
ERR_KEY_NOT_FOUND = -5
ERR_TYPE_MISMATCH = -6


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def fail(self, name, msg):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  ✗ {name}: {msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} tests passed")
        if self.errors:
            print("\nFailures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


def test_open_close(lib, db_path, results):
    """Test database open and close operations."""
    # Open database
    ret = lib.syna_open(db_path)
    if ret == ERR_SUCCESS:
        results.ok("open database")
    else:
        results.fail("open database", f"returned {ret}")
        return False
    
    # Close database
    ret = lib.syna_close(db_path)
    if ret == ERR_SUCCESS:
        results.ok("close database")
    else:
        results.fail("close database", f"returned {ret}")
        return False
    
    return True


def test_put_get_float(lib, db_path, results):
    """Test writing and reading float values."""
    lib.syna_open(db_path)
    
    key = b"temperature"
    value = 98.6
    
    # Write float
    offset = lib.syna_put_float(db_path, key, value)
    if offset >= 0:
        results.ok("put_float")
    else:
        results.fail("put_float", f"returned {offset}")
        lib.syna_close(db_path)
        return False
    
    # Read float back
    out = ctypes.c_double()
    ret = lib.syna_get_float(db_path, key, ctypes.byref(out))
    if ret == ERR_SUCCESS and abs(out.value - value) < 1e-10:
        results.ok("get_float")
    else:
        results.fail("get_float", f"returned {ret}, value={out.value}, expected={value}")
    
    lib.syna_close(db_path)
    return True


def test_put_get_int(lib, db_path, results):
    """Test writing and reading integer values."""
    lib.syna_open(db_path)
    
    key = b"count"
    value = 42
    
    # Write int
    offset = lib.syna_put_int(db_path, key, value)
    if offset >= 0:
        results.ok("put_int")
    else:
        results.fail("put_int", f"returned {offset}")
        lib.syna_close(db_path)
        return False
    
    # Read int back
    out = ctypes.c_int64()
    ret = lib.syna_get_int(db_path, key, ctypes.byref(out))
    if ret == ERR_SUCCESS and out.value == value:
        results.ok("get_int")
    else:
        results.fail("get_int", f"returned {ret}, value={out.value}, expected={value}")
    
    lib.syna_close(db_path)
    return True


def test_put_text(lib, db_path, results):
    """Test writing text values."""
    lib.syna_open(db_path)
    
    key = b"message"
    value = b"Hello, syna!"
    
    # Write text
    offset = lib.syna_put_text(db_path, key, value)
    if offset >= 0:
        results.ok("put_text")
    else:
        results.fail("put_text", f"returned {offset}")
    
    lib.syna_close(db_path)
    return True


def test_history_tensor(lib, db_path, results):
    """Test tensor extraction for AI workloads."""
    lib.syna_open(db_path)
    
    key = b"sensor_data"
    values = [1.0, 2.5, 3.7, 4.2, 5.9]
    
    # Write multiple float values
    for v in values:
        offset = lib.syna_put_float(db_path, key, v)
        if offset < 0:
            results.fail("tensor write", f"put_float returned {offset}")
            lib.syna_close(db_path)
            return False
    
    results.ok("tensor write (5 values)")
    
    # Get history tensor
    out_len = ctypes.c_size_t()
    ptr = lib.syna_get_history_tensor(db_path, key, ctypes.byref(out_len))
    
    if ptr and out_len.value == len(values):
        # Read values from tensor
        tensor_values = [ptr[i] for i in range(out_len.value)]
        
        # Verify values match
        all_match = all(abs(a - b) < 1e-10 for a, b in zip(tensor_values, values))
        if all_match:
            results.ok("get_history_tensor")
        else:
            results.fail("get_history_tensor", f"values mismatch: {tensor_values} vs {values}")
        
        # Free tensor memory
        lib.syna_free_tensor(ptr, out_len.value)
        results.ok("free_tensor")
    else:
        results.fail("get_history_tensor", f"ptr={ptr}, len={out_len.value}, expected={len(values)}")
    
    lib.syna_close(db_path)
    return True


def test_delete_exists(lib, db_path, results):
    """Test delete and exists operations."""
    lib.syna_open(db_path)
    
    key = b"to_delete"
    
    # Write a value
    lib.syna_put_float(db_path, key, 123.456)
    
    # Check exists
    ret = lib.syna_exists(db_path, key)
    if ret == 1:
        results.ok("exists (before delete)")
    else:
        results.fail("exists (before delete)", f"returned {ret}")
    
    # Delete
    ret = lib.syna_delete(db_path, key)
    if ret == ERR_SUCCESS:
        results.ok("delete")
    else:
        results.fail("delete", f"returned {ret}")
    
    # Check not exists
    ret = lib.syna_exists(db_path, key)
    if ret == 0:
        results.ok("exists (after delete)")
    else:
        results.fail("exists (after delete)", f"returned {ret}, expected 0")
    
    lib.syna_close(db_path)
    return True


def test_compact(lib, db_path, results):
    """Test database compaction."""
    lib.syna_open(db_path)
    
    key = b"compact_test"
    
    # Write multiple values to same key
    for i in range(10):
        lib.syna_put_float(db_path, key, float(i))
    
    # Compact
    ret = lib.syna_compact(db_path)
    if ret == ERR_SUCCESS:
        results.ok("compact")
    else:
        results.fail("compact", f"returned {ret}")
    
    # Verify latest value still readable
    out = ctypes.c_double()
    ret = lib.syna_get_float(db_path, key, ctypes.byref(out))
    if ret == ERR_SUCCESS and abs(out.value - 9.0) < 1e-10:
        results.ok("read after compact")
    else:
        results.fail("read after compact", f"returned {ret}, value={out.value}")
    
    lib.syna_close(db_path)
    return True


def test_error_handling(lib, db_path, results):
    """Test error handling for invalid inputs."""
    # Try to get from unopened database
    out = ctypes.c_double()
    ret = lib.syna_get_float(b"nonexistent.db", b"key", ctypes.byref(out))
    if ret == ERR_DB_NOT_FOUND:
        results.ok("error: db not found")
    else:
        results.fail("error: db not found", f"returned {ret}, expected {ERR_DB_NOT_FOUND}")
    
    # Open database
    lib.syna_open(db_path)
    
    # Try to get nonexistent key
    ret = lib.syna_get_float(db_path, b"nonexistent_key", ctypes.byref(out))
    if ret == ERR_KEY_NOT_FOUND:
        results.ok("error: key not found")
    else:
        results.fail("error: key not found", f"returned {ret}, expected {ERR_KEY_NOT_FOUND}")
    
    # Write int, try to read as float (type mismatch)
    lib.syna_put_int(db_path, b"int_key", 42)
    ret = lib.syna_get_float(db_path, b"int_key", ctypes.byref(out))
    if ret == ERR_TYPE_MISMATCH:
        results.ok("error: type mismatch")
    else:
        results.fail("error: type mismatch", f"returned {ret}, expected {ERR_TYPE_MISMATCH}")
    
    lib.syna_close(db_path)
    return True


def test_data_integrity(lib, db_path, results):
    """Test data integrity across multiple operations."""
    lib.syna_open(db_path)
    
    # Write various data types
    test_data = [
        (b"float1", "float", 3.14159),
        (b"float2", "float", -273.15),
        (b"int1", "int", 9223372036854775807),  # i64 max
        (b"int2", "int", -9223372036854775808),  # i64 min
        (b"text1", "text", b"Unicode test: \xc3\xa9\xc3\xa0\xc3\xbc"),  # UTF-8
    ]
    
    # Write all data
    for key, dtype, value in test_data:
        if dtype == "float":
            lib.syna_put_float(db_path, key, value)
        elif dtype == "int":
            lib.syna_put_int(db_path, key, value)
        elif dtype == "text":
            lib.syna_put_text(db_path, key, value)
    
    # Close and reopen to test persistence
    lib.syna_close(db_path)
    lib.syna_open(db_path)
    
    # Verify all data
    all_ok = True
    for key, dtype, expected in test_data:
        if dtype == "float":
            out = ctypes.c_double()
            ret = lib.syna_get_float(db_path, key, ctypes.byref(out))
            if ret != ERR_SUCCESS or abs(out.value - expected) > 1e-10:
                all_ok = False
                results.fail(f"integrity {key.decode()}", f"got {out.value}, expected {expected}")
        elif dtype == "int":
            out = ctypes.c_int64()
            ret = lib.syna_get_int(db_path, key, ctypes.byref(out))
            if ret != ERR_SUCCESS or out.value != expected:
                all_ok = False
                results.fail(f"integrity {key.decode()}", f"got {out.value}, expected {expected}")
    
    if all_ok:
        results.ok("data integrity after reopen")
    
    lib.syna_close(db_path)
    return True


def main():
    print("syna Database - Python ctypes Integration Test")
    print("=" * 60)
    
    # Find and load library
    try:
        lib_path = find_library()
        print(f"Using library: {lib_path}")
        lib = setup_library(lib_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except OSError as e:
        print(f"ERROR loading library: {e}")
        return 1
    
    results = TestResult()
    
    # Create temporary directory for test databases
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Test directory: {tmpdir}\n")
        
        # Run tests
        print("Test: Open/Close")
        db_path = os.path.join(tmpdir, "test1.db").encode()
        test_open_close(lib, db_path, results)
        
        print("\nTest: Put/Get Float")
        db_path = os.path.join(tmpdir, "test2.db").encode()
        test_put_get_float(lib, db_path, results)
        
        print("\nTest: Put/Get Int")
        db_path = os.path.join(tmpdir, "test3.db").encode()
        test_put_get_int(lib, db_path, results)
        
        print("\nTest: Put Text")
        db_path = os.path.join(tmpdir, "test4.db").encode()
        test_put_text(lib, db_path, results)
        
        print("\nTest: History Tensor")
        db_path = os.path.join(tmpdir, "test5.db").encode()
        test_history_tensor(lib, db_path, results)
        
        print("\nTest: Delete/Exists")
        db_path = os.path.join(tmpdir, "test6.db").encode()
        test_delete_exists(lib, db_path, results)
        
        print("\nTest: Compact")
        db_path = os.path.join(tmpdir, "test7.db").encode()
        test_compact(lib, db_path, results)
        
        print("\nTest: Error Handling")
        db_path = os.path.join(tmpdir, "test8.db").encode()
        test_error_handling(lib, db_path, results)
        
        print("\nTest: Data Integrity")
        db_path = os.path.join(tmpdir, "test9.db").encode()
        test_data_integrity(lib, db_path, results)
    
    # Print summary
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


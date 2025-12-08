#!/usr/bin/env python3
"""
Basic Syna Usage Demo

This demo shows fundamental database operations using the Python wrapper:
- Library loading and database open/close
- Writing values of different types (Float, Int, Text, Bytes)
- Reading values back
- Deleting keys
- Listing keys
- Error handling patterns

Requirements: 2.1 - WHEN a developer views the Python basic demo THEN the demo 
SHALL show loading the library, opening DB, and CRUD operations

Run with: python basic_usage.py
"""

import os
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Syna import SynaDB, SynaError


def demo_library_loading():
    """Demonstrate library loading and path discovery."""
    print("=" * 60)
    print("1. LIBRARY LOADING")
    print("=" * 60)
    
    # The library is loaded automatically when SynaDB is instantiated
    # Show the library path that was found
    print("\nThe Syna library is loaded via ctypes from the shared library.")
    print("Supported platforms:")
    print("  - Linux:   libsynadb.so")
    print("  - macOS:   libsynadb.dylib")
    print("  - Windows: synadb.dll")
    print("\nSearch paths (in order):")
    print("  1. target/release/ (relative to package)")
    print("  2. target/debug/   (relative to package)")
    print("  3. Current working directory")
    print("  4. System library path")
    
    # Show actual library path after loading
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SynaDB(db_path) as db:
            print(f"\n✓ Library loaded from: {db._lib_path}")
    
    print()


def demo_database_open_close():
    """Demonstrate database open and close operations."""
    print("=" * 60)
    print("2. DATABASE OPEN/CLOSE")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        # Method 1: Context manager (recommended)
        print("\nMethod 1: Context Manager (recommended)")
        print("-" * 40)
        print("""
with SynaDB("my.db") as db:
    # Database is open here
    db.put_float("key", 3.14)
# Database is automatically closed here
""")
        
        with SynaDB(db_path) as db:
            db.put_float("test", 1.0)
            print("✓ Database opened with context manager")
        print("✓ Database automatically closed on exit")
        
        # Method 2: Manual open/close
        print("\nMethod 2: Manual Open/Close")
        print("-" * 40)
        print("""
db = SynaDB("my.db")
try:
    db.put_float("key", 3.14)
finally:
    db.close()
""")
        
        db = SynaDB(db_path)
        try:
            db.put_float("test2", 2.0)
            print("✓ Database opened manually")
        finally:
            db.close()
            print("✓ Database closed manually")
    
    print()


def demo_write_operations():
    """Demonstrate all write operations."""
    print("=" * 60)
    print("3. WRITE OPERATIONS (PUT)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        with SynaDB(db_path) as db:
            # Float
            print("\nput_float(key, value) -> offset")
            print("-" * 40)
            offset = db.put_float("sensor/temperature", 23.5)
            print(f"db.put_float('sensor/temperature', 23.5)")
            print(f"✓ Written at offset: {offset}")
            
            # Integer
            print("\nput_int(key, value) -> offset")
            print("-" * 40)
            offset = db.put_int("counter/visits", 42)
            print(f"db.put_int('counter/visits', 42)")
            print(f"✓ Written at offset: {offset}")
            
            # Text
            print("\nput_text(key, value) -> offset")
            print("-" * 40)
            offset = db.put_text("config/name", "Syna Demo")
            print(f"db.put_text('config/name', 'Syna Demo')")
            print(f"✓ Written at offset: {offset}")
            
            # Bytes
            print("\nput_bytes(key, value) -> offset")
            print("-" * 40)
            data = b"\x00\x01\x02\x03\xff"
            offset = db.put_bytes("binary/data", data)
            print(f"db.put_bytes('binary/data', b'\\x00\\x01\\x02\\x03\\xff')")
            print(f"✓ Written at offset: {offset}")
            
            # Appending multiple values (history)
            print("\nAppending Multiple Values (History)")
            print("-" * 40)
            print("Each put() appends a new value, preserving history:")
            for i, temp in enumerate([24.0, 24.5, 25.0, 25.5]):
                offset = db.put_float("sensor/temperature", temp)
                print(f"  put_float('sensor/temperature', {temp}) -> offset {offset}")
    
    print()


def demo_read_operations():
    """Demonstrate all read operations."""
    print("=" * 60)
    print("4. READ OPERATIONS (GET)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        with SynaDB(db_path) as db:
            # Setup test data
            db.put_float("sensor/temp", 23.5)
            db.put_float("sensor/temp", 24.0)
            db.put_float("sensor/temp", 24.5)
            db.put_int("counter", 100)
            db.put_text("message", "Hello, Syna!")
            db.put_bytes("binary", b"\xde\xad\xbe\xef")
            
            # get_float
            print("\nget_float(key) -> Optional[float]")
            print("-" * 40)
            value = db.get_float("sensor/temp")
            print(f"db.get_float('sensor/temp') = {value}")
            print("✓ Returns the LATEST value for the key")
            
            # get_int
            print("\nget_int(key) -> Optional[int]")
            print("-" * 40)
            value = db.get_int("counter")
            print(f"db.get_int('counter') = {value}")
            
            # get_text
            print("\nget_text(key) -> Optional[str]")
            print("-" * 40)
            value = db.get_text("message")
            print(f"db.get_text('message') = '{value}'")
            
            # get_bytes
            print("\nget_bytes(key) -> Optional[bytes]")
            print("-" * 40)
            value = db.get_bytes("binary")
            print(f"db.get_bytes('binary') = {value}")
            
            # get_history_tensor (for ML)
            print("\nget_history_tensor(key) -> np.ndarray")
            print("-" * 40)
            history = db.get_history_tensor("sensor/temp")
            print(f"db.get_history_tensor('sensor/temp') = {history}")
            print(f"✓ Shape: {history.shape}, dtype: {history.dtype}")
            print("✓ Returns ALL float values in chronological order")
            
            # Non-existent key
            print("\nReading Non-Existent Key")
            print("-" * 40)
            value = db.get_float("nonexistent")
            print(f"db.get_float('nonexistent') = {value}")
            print("✓ Returns None (not an error)")
    
    print()


def demo_delete_operations():
    """Demonstrate delete operations."""
    print("=" * 60)
    print("5. DELETE OPERATIONS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        with SynaDB(db_path) as db:
            # Setup
            db.put_float("to_delete", 1.0)
            db.put_float("to_keep", 2.0)
            
            print("\nBefore Delete:")
            print(f"  exists('to_delete') = {db.exists('to_delete')}")
            print(f"  exists('to_keep') = {db.exists('to_keep')}")
            
            # Delete
            print("\ndelete(key)")
            print("-" * 40)
            db.delete("to_delete")
            print("db.delete('to_delete')")
            
            print("\nAfter Delete:")
            print(f"  exists('to_delete') = {db.exists('to_delete')}")
            print(f"  get_float('to_delete') = {db.get_float('to_delete')}")
            print("✓ Deleted keys return None and exists() returns False")
            
            # exists()
            print("\nexists(key) -> bool")
            print("-" * 40)
            print(f"db.exists('to_keep') = {db.exists('to_keep')}")
            print(f"db.exists('to_delete') = {db.exists('to_delete')}")
            print(f"db.exists('never_existed') = {db.exists('never_existed')}")
    
    print()


def demo_key_listing():
    """Demonstrate key listing."""
    print("=" * 60)
    print("6. KEY LISTING")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        with SynaDB(db_path) as db:
            # Setup hierarchical keys
            db.put_float("sensor/temp/room1", 22.0)
            db.put_float("sensor/temp/room2", 23.0)
            db.put_float("sensor/humidity/room1", 45.0)
            db.put_int("counter/visits", 100)
            db.put_text("config/name", "Demo")
            
            # Delete one key
            db.delete("counter/visits")
            
            print("\nkeys() -> List[str]")
            print("-" * 40)
            keys = db.keys()
            print("db.keys() =")
            for key in sorted(keys):
                print(f"  - {key}")
            print(f"\n✓ Total: {len(keys)} keys")
            print("✓ Deleted keys are NOT included")
    
    print()


def demo_error_handling():
    """Demonstrate error handling patterns."""
    print("=" * 60)
    print("7. ERROR HANDLING PATTERNS")
    print("=" * 60)
    
    print("\nSynaError Exception")
    print("-" * 40)
    print("""
SynaError has two attributes:
  - code: Integer error code
  - message: Human-readable description

Error codes:
   0: Generic error
  -1: Database not found in registry
  -2: Invalid path or UTF-8
  -3: I/O error
  -4: Serialization error
  -5: Key not found
  -6: Type mismatch
  -7: Empty key not allowed
  -8: Key too long
-100: Internal panic
""")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        # Pattern 1: Try/except for specific errors
        print("\nPattern 1: Try/Except")
        print("-" * 40)
        print("""
try:
    with SynaDB("my.db") as db:
        db.put_float("key", value)
except SynaError as e:
    print(f"Error {e.code}: {e.message}")
""")
        
        try:
            with SynaDB(db_path) as db:
                # Try to use empty key (will fail)
                db.put_float("", 1.0)
        except SynaError as e:
            print(f"✓ Caught error: code={e.code}, message='{e.message}'")
        
        # Pattern 2: Check for None on reads
        print("\nPattern 2: Check for None on Reads")
        print("-" * 40)
        print("""
value = db.get_float("key")
if value is None:
    print("Key not found")
else:
    print(f"Value: {value}")
""")
        
        with SynaDB(db_path) as db:
            value = db.get_float("nonexistent")
            if value is None:
                print("✓ Key not found (returned None)")
            else:
                print(f"Value: {value}")
        
        # Pattern 3: Check existence before operations
        print("\nPattern 3: Check Existence First")
        print("-" * 40)
        print("""
if db.exists("key"):
    value = db.get_float("key")
    # Safe to use value
""")
        
        with SynaDB(db_path) as db:
            db.put_float("existing", 42.0)
            
            if db.exists("existing"):
                value = db.get_float("existing")
                print(f"✓ Key exists, value = {value}")
            
            if not db.exists("missing"):
                print("✓ Key does not exist, skipping read")
        
        # Pattern 4: Using closed database
        print("\nPattern 4: Handling Closed Database")
        print("-" * 40)
        
        db = SynaDB(db_path)
        db.close()
        
        try:
            db.put_float("key", 1.0)
        except SynaError as e:
            print(f"✓ Caught error on closed DB: '{e.message}'")
    
    print()


def demo_compaction():
    """Demonstrate database compaction."""
    print("=" * 60)
    print("8. DATABASE COMPACTION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        with SynaDB(db_path) as db:
            # Write many values
            print("\nWriting 100 values to same key...")
            for i in range(100):
                db.put_float("sensor", float(i))
            
            # Check file size before
            size_before = os.path.getsize(db_path)
            print(f"File size before compaction: {size_before} bytes")
            
            # Get history
            history = db.get_history_tensor("sensor")
            print(f"History length: {len(history)} values")
            
            # Compact
            print("\ncompact()")
            print("-" * 40)
            db.compact()
            print("db.compact()")
            
            # Check file size after
            size_after = os.path.getsize(db_path)
            print(f"File size after compaction: {size_after} bytes")
            print(f"Space saved: {size_before - size_after} bytes")
            
            # History after compaction
            history = db.get_history_tensor("sensor")
            print(f"History length after compaction: {len(history)} value(s)")
            print("✓ Compaction keeps only the latest value per key")
    
    print()


def demo_persistence():
    """Demonstrate data persistence across sessions."""
    print("=" * 60)
    print("9. DATA PERSISTENCE")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        
        # Session 1: Write data
        print("\nSession 1: Writing data")
        print("-" * 40)
        with SynaDB(db_path) as db:
            db.put_float("persistent/value", 42.0)
            db.put_text("persistent/message", "Hello from session 1!")
            print("✓ Data written and database closed")
        
        # Session 2: Read data back
        print("\nSession 2: Reading data back")
        print("-" * 40)
        with SynaDB(db_path) as db:
            value = db.get_float("persistent/value")
            message = db.get_text("persistent/message")
            print(f"persistent/value = {value}")
            print(f"persistent/message = '{message}'")
            print("✓ Data persisted across sessions!")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   Syna PYTHON BASIC USAGE DEMO")
    print("   Requirements: 2.1")
    print("=" * 60 + "\n")
    
    try:
        demo_library_loading()
        demo_database_open_close()
        demo_write_operations()
        demo_read_operations()
        demo_delete_operations()
        demo_key_listing()
        demo_error_handling()
        demo_compaction()
        demo_persistence()
        
        print("=" * 60)
        print("   ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


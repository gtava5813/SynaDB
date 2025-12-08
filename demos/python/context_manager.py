#!/usr/bin/env python3
"""
Syna Context Manager Demo

This demo shows proper resource management with context managers:
- Using `with` statement for automatic cleanup
- Automatic cleanup on exception
- Nested context managers for multiple DBs

Requirements: 2.5 - WHEN a developer views the Python context manager demo THEN 
the demo SHALL show proper resource management with `with` statements

Run with: python context_manager.py
"""

import os
import sys
import tempfile
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Syna import SynaDB, SynaError


def demo_basic_context_manager():
    """Demonstrate basic context manager usage."""
    print("=" * 60)
    print("1. BASIC CONTEXT MANAGER USAGE")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "context_demo.db")
        
        print("\nThe `with` Statement")
        print("-" * 40)
        print("""
# Recommended pattern - automatic cleanup
with SynaDB("my.db") as db:
    db.put_float("key", 3.14)
    value = db.get_float("key")
# Database is automatically closed here, even if an exception occurs
""")
        
        # Demonstrate with statement
        print("Executing:")
        with SynaDB(db_path) as db:
            print("  ✓ Database opened")
            db.put_float("test", 42.0)
            print("  ✓ Data written")
            value = db.get_float("test")
            print(f"  ✓ Data read: {value}")
        print("  ✓ Database automatically closed")
        
        # Verify it's closed by reopening
        print("\nVerifying closure by reopening:")
        with SynaDB(db_path) as db:
            value = db.get_float("test")
            print(f"  ✓ Reopened and read: {value}")
        print("  ✓ Closed again")
    
    print()


def demo_manual_vs_context_manager():
    """Compare manual management vs context manager."""
    print("=" * 60)
    print("2. MANUAL VS CONTEXT MANAGER")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "context_demo.db")
        
        # Manual management (not recommended)
        print("\nManual Management (NOT Recommended)")
        print("-" * 40)
        print("""
# Risk: If an exception occurs, close() may not be called
db = SynaDB("my.db")
try:
    db.put_float("key", 3.14)
finally:
    db.close()  # Must remember to close!
""")
        
        db = SynaDB(db_path)
        try:
            db.put_float("manual", 1.0)
            print("  ✓ Manual: Data written")
        finally:
            db.close()
            print("  ✓ Manual: Database closed in finally block")
        
        # Context manager (recommended)
        print("\nContext Manager (Recommended)")
        print("-" * 40)
        print("""
# Safe: Database is always closed, even on exception
with SynaDB("my.db") as db:
    db.put_float("key", 3.14)
# Automatically closed here
""")
        
        with SynaDB(db_path) as db:
            db.put_float("context", 2.0)
            print("  ✓ Context: Data written")
        print("  ✓ Context: Database automatically closed")
        
        # Why context manager is better
        print("\nWhy Context Manager is Better:")
        print("-" * 40)
        print("""
1. Automatic cleanup - no need to remember close()
2. Exception-safe - closes even if exception occurs
3. Cleaner code - less boilerplate
4. Prevents resource leaks
5. Works with nested resources
""")
    
    print()


def demo_exception_cleanup():
    """Demonstrate automatic cleanup on exception."""
    print("=" * 60)
    print("3. AUTOMATIC CLEANUP ON EXCEPTION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "context_demo.db")
        
        print("\nException During Operation")
        print("-" * 40)
        print("""
# Even if an exception occurs, the database is properly closed
try:
    with SynaDB("my.db") as db:
        db.put_float("before_error", 1.0)
        raise ValueError("Simulated error!")
        db.put_float("after_error", 2.0)  # Never executed
except ValueError as e:
    print(f"Caught: {e}")
# Database is still properly closed!
""")
        
        print("\nExecuting:")
        try:
            with SynaDB(db_path) as db:
                db.put_float("before_error", 1.0)
                print("  ✓ Wrote 'before_error'")
                
                # Simulate an error
                raise ValueError("Simulated error!")
                
                # This line is never reached
                db.put_float("after_error", 2.0)
                
        except ValueError as e:
            print(f"  ✓ Caught exception: {e}")
        
        print("  ✓ Database was closed despite exception")
        
        # Verify data integrity
        print("\nVerifying Data Integrity:")
        with SynaDB(db_path) as db:
            before = db.get_float("before_error")
            after = db.get_float("after_error")
            print(f"  ✓ 'before_error' = {before} (written before exception)")
            print(f"  ✓ 'after_error' = {after} (None - never written)")
        
        # Multiple exception types
        print("\nHandling Different Exception Types:")
        print("-" * 40)
        
        exceptions_to_test = [
            ("KeyError", KeyError("missing")),
            ("TypeError", TypeError("wrong type")),
            ("RuntimeError", RuntimeError("runtime issue")),
        ]
        
        for name, exc in exceptions_to_test:
            try:
                with SynaDB(db_path) as db:
                    db.put_float(f"test_{name}", 1.0)
                    raise exc
            except Exception as e:
                print(f"  ✓ {name}: Caught and cleaned up")
    
    print()


def demo_nested_context_managers():
    """Demonstrate nested context managers for multiple DBs."""
    print("=" * 60)
    print("4. NESTED CONTEXT MANAGERS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db1_path = os.path.join(tmpdir, "db1.db")
        db2_path = os.path.join(tmpdir, "db2.db")
        db3_path = os.path.join(tmpdir, "db3.db")
        
        # Nested with statements
        print("\nNested `with` Statements")
        print("-" * 40)
        print("""
# Multiple databases open simultaneously
with SynaDB("db1.db") as db1:
    with SynaDB("db2.db") as db2:
        # Both databases are open
        db1.put_float("key", 1.0)
        db2.put_float("key", 2.0)
    # db2 is closed here
# db1 is closed here
""")
        
        print("\nExecuting nested context managers:")
        with SynaDB(db1_path) as db1:
            print("  ✓ db1 opened")
            with SynaDB(db2_path) as db2:
                print("  ✓ db2 opened")
                
                db1.put_float("value", 1.0)
                db2.put_float("value", 2.0)
                print("  ✓ Data written to both")
                
            print("  ✓ db2 closed")
        print("  ✓ db1 closed")
        
        # Multiple context managers on one line
        print("\nMultiple Context Managers (Single Line)")
        print("-" * 40)
        print("""
# Python 3.9+ syntax for multiple context managers
with (
    SynaDB("db1.db") as db1,
    SynaDB("db2.db") as db2,
    SynaDB("db3.db") as db3
):
    # All three databases are open
    pass
# All three are closed
""")
        
        print("\nExecuting multiple context managers:")
        with SynaDB(db1_path) as db1, \
             SynaDB(db2_path) as db2, \
             SynaDB(db3_path) as db3:
            
            print("  ✓ All three databases opened")
            
            db1.put_float("multi", 1.0)
            db2.put_float("multi", 2.0)
            db3.put_float("multi", 3.0)
            
            print("  ✓ Data written to all three")
        
        print("  ✓ All three databases closed")
        
        # Verify all data persisted
        print("\nVerifying all data persisted:")
        with SynaDB(db1_path) as db1, \
             SynaDB(db2_path) as db2, \
             SynaDB(db3_path) as db3:
            
            print(f"  ✓ db1: {db1.get_float('multi')}")
            print(f"  ✓ db2: {db2.get_float('multi')}")
            print(f"  ✓ db3: {db3.get_float('multi')}")
    
    print()


def demo_data_transfer_between_dbs():
    """Demonstrate data transfer between databases."""
    print("=" * 60)
    print("5. DATA TRANSFER BETWEEN DATABASES")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.db")
        dest_path = os.path.join(tmpdir, "dest.db")
        
        print("\nTransferring Data Between Databases")
        print("-" * 40)
        print("""
# Copy data from source to destination
with SynaDB("source.db") as source:
    with SynaDB("dest.db") as dest:
        for key in source.keys():
            value = source.get_float(key)
            if value is not None:
                dest.put_float(key, value)
""")
        
        # Setup source database
        print("\nSetup: Creating source database")
        with SynaDB(source_path) as source:
            for i in range(10):
                source.put_float(f"data/{i}", float(i * 10))
            print(f"  ✓ Created {len(source.keys())} keys in source")
        
        # Transfer data
        print("\nTransferring data:")
        with SynaDB(source_path) as source:
            with SynaDB(dest_path) as dest:
                keys = source.keys()
                for key in keys:
                    value = source.get_float(key)
                    if value is not None:
                        dest.put_float(key, value)
                print(f"  ✓ Transferred {len(keys)} keys")
        
        # Verify transfer
        print("\nVerifying transfer:")
        with SynaDB(dest_path) as dest:
            keys = dest.keys()
            print(f"  ✓ Destination has {len(keys)} keys")
            sample = dest.get_float("data/5")
            print(f"  ✓ Sample value data/5 = {sample}")
    
    print()


def demo_context_manager_with_tempfile():
    """Demonstrate context manager with tempfile."""
    print("=" * 60)
    print("6. CONTEXT MANAGER WITH TEMPFILE")
    print("=" * 60)
    
    print("\nCombining with tempfile.TemporaryDirectory")
    print("-" * 40)
    print("""
# Perfect for testing - automatic cleanup of both DB and directory
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, "test.db")
    with SynaDB(db_path) as db:
        db.put_float("test", 42.0)
        # ... do testing ...
# Both database and directory are cleaned up
""")
    
    print("\nExecuting:")
    
    # The directory and database are automatically cleaned up
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        print(f"  ✓ Temp directory: {tmpdir}")
        
        with SynaDB(db_path) as db:
            db.put_float("test", 42.0)
            print(f"  ✓ Database created: {db_path}")
            print(f"  ✓ File exists: {os.path.exists(db_path)}")
        
        print(f"  ✓ Database closed, file still exists: {os.path.exists(db_path)}")
    
    print(f"  ✓ Temp directory cleaned up")
    
    print()


def demo_error_recovery_pattern():
    """Demonstrate error recovery pattern."""
    print("=" * 60)
    print("7. ERROR RECOVERY PATTERN")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "recovery.db")
        
        print("\nRobust Error Recovery Pattern")
        print("-" * 40)
        print("""
def safe_operation(db_path, key, value):
    '''Safely write to database with error recovery.'''
    try:
        with SynaDB(db_path) as db:
            db.put_float(key, value)
            return True
    except SynaError as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
""")
        
        def safe_operation(db_path: str, key: str, value: float) -> bool:
            """Safely write to database with error recovery."""
            try:
                with SynaDB(db_path) as db:
                    db.put_float(key, value)
                    return True
            except SynaError as e:
                print(f"  Database error: {e}")
                return False
            except Exception as e:
                print(f"  Unexpected error: {e}")
                return False
        
        print("\nExecuting safe operations:")
        
        # Successful operation
        result = safe_operation(db_path, "good_key", 42.0)
        print(f"  ✓ safe_operation('good_key', 42.0) = {result}")
        
        # Operation with empty key (will fail)
        result = safe_operation(db_path, "", 42.0)
        print(f"  ✓ safe_operation('', 42.0) = {result}")
        
        # Verify good data was saved
        with SynaDB(db_path) as db:
            value = db.get_float("good_key")
            print(f"  ✓ Verified: good_key = {value}")
    
    print()


def demo_context_manager_protocol():
    """Explain the context manager protocol."""
    print("=" * 60)
    print("8. CONTEXT MANAGER PROTOCOL")
    print("=" * 60)
    
    print("\nHow Context Managers Work")
    print("-" * 40)
    print("""
The `with` statement uses two special methods:

class SynaDB:
    def __enter__(self):
        '''Called when entering the `with` block.'''
        return self  # Returns the database instance
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        '''Called when exiting the `with` block.'''
        self.close()  # Always close, even on exception
        return False  # Don't suppress exceptions

When you write:
    with SynaDB("my.db") as db:
        db.put_float("key", 1.0)

Python translates it to:
    db = SynaDB("my.db")
    db.__enter__()
    try:
        db.put_float("key", 1.0)
    finally:
        db.__exit__(None, None, None)
""")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "protocol.db")
        
        print("\nDemonstrating the protocol manually:")
        
        # Manual protocol demonstration
        db = SynaDB(db_path)
        print("  1. Created SynaDB instance")
        
        result = db.__enter__()
        print(f"  2. Called __enter__(), returned: {result is db}")
        
        try:
            db.put_float("manual", 123.0)
            print("  3. Performed operations")
        finally:
            db.__exit__(None, None, None)
            print("  4. Called __exit__(), database closed")
        
        # Verify
        with SynaDB(db_path) as db:
            value = db.get_float("manual")
            print(f"  ✓ Verified: manual = {value}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   Syna CONTEXT MANAGER DEMO")
    print("   Requirements: 2.5")
    print("=" * 60 + "\n")
    
    try:
        demo_basic_context_manager()
        demo_manual_vs_context_manager()
        demo_exception_cleanup()
        demo_nested_context_managers()
        demo_data_transfer_between_dbs()
        demo_context_manager_with_tempfile()
        demo_error_recovery_pattern()
        demo_context_manager_protocol()
        
        print("=" * 60)
        print("   ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


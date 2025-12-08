"""
Unit tests for Syna Python wrapper.

Tests cover:
- All CRUD operations (put/get for float, int, text, bytes)
- Numpy array extraction (get_history_tensor)
- Context manager cleanup
- Error handling

Requirements: 2.1, 2.2, 2.5
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Syna import SynaDB, SynaError


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


class TestCRUDOperations:
    """Test basic CRUD operations - Requirements 2.1"""
    
    def test_put_get_float(self, temp_db):
        """Test writing and reading float values."""
        with SynaDB(temp_db) as db:
            offset = db.put_float("temp", 23.5)
            assert offset >= 0
            
            value = db.get_float("temp")
            assert value == 23.5
    
    def test_put_get_int(self, temp_db):
        """Test writing and reading integer values."""
        with SynaDB(temp_db) as db:
            offset = db.put_int("count", 42)
            assert offset >= 0
            
            value = db.get_int("count")
            assert value == 42
    
    def test_put_get_text(self, temp_db):
        """Test writing text values."""
        with SynaDB(temp_db) as db:
            offset = db.put_text("name", "Syna")
            assert offset >= 0
            # Note: get_text not implemented in FFI yet
    
    def test_put_get_bytes(self, temp_db):
        """Test writing bytes values."""
        with SynaDB(temp_db) as db:
            data = b"Hello, World!"
            offset = db.put_bytes("data", data)
            assert offset >= 0
            # Note: get_bytes not implemented in FFI yet
    
    def test_get_nonexistent_key(self, temp_db):
        """Test reading a key that doesn't exist returns None."""
        with SynaDB(temp_db) as db:
            value = db.get_float("nonexistent")
            assert value is None
            
            value = db.get_int("nonexistent")
            assert value is None
    
    def test_delete_key(self, temp_db):
        """Test deleting a key."""
        with SynaDB(temp_db) as db:
            db.put_float("temp", 23.5)
            assert db.exists("temp") is True
            
            db.delete("temp")
            assert db.exists("temp") is False
            assert db.get_float("temp") is None
    
    def test_exists(self, temp_db):
        """Test checking key existence."""
        with SynaDB(temp_db) as db:
            assert db.exists("key") is False
            
            db.put_int("key", 1)
            assert db.exists("key") is True
    
    def test_keys(self, temp_db):
        """Test listing all keys."""
        with SynaDB(temp_db) as db:
            db.put_float("a", 1.0)
            db.put_int("b", 2)
            db.put_text("c", "three")
            
            keys = db.keys()
            assert len(keys) == 3
            assert set(keys) == {"a", "b", "c"}
    
    def test_compact(self, temp_db):
        """Test database compaction."""
        with SynaDB(temp_db) as db:
            # Write multiple values for same key
            db.put_float("temp", 1.0)
            db.put_float("temp", 2.0)
            db.put_float("temp", 3.0)
            
            # Compact should preserve latest value
            db.compact()
            
            value = db.get_float("temp")
            assert value == 3.0


class TestNumpyIntegration:
    """Test numpy array extraction - Requirements 2.2"""
    
    def test_get_history_tensor_single_value(self, temp_db):
        """Test extracting history with single value."""
        with SynaDB(temp_db) as db:
            db.put_float("sensor", 23.5)
            
            tensor = db.get_history_tensor("sensor")
            assert isinstance(tensor, np.ndarray)
            assert tensor.dtype == np.float64
            assert len(tensor) == 1
            assert tensor[0] == 23.5
    
    def test_get_history_tensor_multiple_values(self, temp_db):
        """Test extracting history with multiple values."""
        with SynaDB(temp_db) as db:
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            for v in values:
                db.put_float("sensor", v)
            
            tensor = db.get_history_tensor("sensor")
            assert len(tensor) == 5
            np.testing.assert_array_equal(tensor, values)
    
    def test_get_history_tensor_empty(self, temp_db):
        """Test extracting history for nonexistent key."""
        with SynaDB(temp_db) as db:
            tensor = db.get_history_tensor("nonexistent")
            assert isinstance(tensor, np.ndarray)
            assert len(tensor) == 0
    
    def test_get_history_tensor_filters_non_floats(self, temp_db):
        """Test that history tensor only includes float values."""
        with SynaDB(temp_db) as db:
            db.put_float("mixed", 1.0)
            db.put_int("mixed", 2)  # This should be filtered out
            db.put_float("mixed", 3.0)
            
            tensor = db.get_history_tensor("mixed")
            # Only float values should be included
            assert len(tensor) == 2
            assert tensor[0] == 1.0
            assert tensor[1] == 3.0


class TestContextManager:
    """Test context manager and cleanup - Requirements 2.5"""
    
    def test_context_manager_opens_and_closes(self, temp_db):
        """Test that context manager properly opens and closes database."""
        with SynaDB(temp_db) as db:
            db.put_float("test", 1.0)
            assert db.get_float("test") == 1.0
        
        # After context exit, database should be closed
        # Reopening should work and data should persist
        with SynaDB(temp_db) as db:
            assert db.get_float("test") == 1.0
    
    def test_context_manager_cleanup_on_exception(self, temp_db):
        """Test that context manager cleans up on exception."""
        try:
            with SynaDB(temp_db) as db:
                db.put_float("test", 1.0)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Database should still be usable after exception
        with SynaDB(temp_db) as db:
            assert db.get_float("test") == 1.0
    
    def test_manual_close(self, temp_db):
        """Test manual close without context manager."""
        db = SynaDB(temp_db)
        db.put_float("test", 1.0)
        db.close()
        
        # Reopening should work
        with SynaDB(temp_db) as db2:
            assert db2.get_float("test") == 1.0
    
    def test_operations_after_close_raise_error(self, temp_db):
        """Test that operations after close raise an error."""
        db = SynaDB(temp_db)
        db.put_float("test", 1.0)
        db.close()
        
        with pytest.raises(SynaError):
            db.put_float("test2", 2.0)


class TestErrorHandling:
    """Test error handling - Requirements 2.1, 2.5"""
    
    def test_type_mismatch_float_to_int(self, temp_db):
        """Test reading float as int raises type mismatch error."""
        with SynaDB(temp_db) as db:
            db.put_float("value", 3.14)
            
            # Reading float as int should raise type mismatch error
            with pytest.raises(SynaError) as exc_info:
                db.get_int("value")
            assert exc_info.value.code == -6  # TYPE_MISMATCH
    
    def test_type_mismatch_int_to_float(self, temp_db):
        """Test reading int as float raises type mismatch error."""
        with SynaDB(temp_db) as db:
            db.put_int("value", 42)
            
            # Reading int as float should raise type mismatch error
            with pytest.raises(SynaError) as exc_info:
                db.get_float("value")
            assert exc_info.value.code == -6  # TYPE_MISMATCH


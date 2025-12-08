"""Tests for TensorEngine class."""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import TensorEngine


class TestTensorEngine:
    """Test cases for TensorEngine."""
    
    def test_put_get_tensor_basic(self):
        """Test basic put and get tensor operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Create test data
                X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
                
                # Store tensor
                count = engine.put_tensor('test/', X)
                assert count == 5
                
                # Retrieve tensor
                result = engine.get_tensor('test/*', dtype=np.float32)
                assert len(result) == 5
                assert np.allclose(X, result)
    
    def test_put_get_tensor_with_shape(self):
        """Test tensor operations with reshape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Create 2D test data
                X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                
                # Store tensor (flattened)
                count = engine.put_tensor('matrix/', X)
                assert count == 4
                
                # Retrieve with shape
                result = engine.get_tensor('matrix/*', shape=(2, 2), dtype=np.float32)
                assert result.shape == (2, 2)
                assert np.allclose(X, result)
    
    def test_stream_batches(self):
        """Test streaming data in batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Create test data
                X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
                engine.put_tensor('data/', X)
                
                # Stream in batches of 2
                batches = list(engine.stream('data/*', batch_size=2))
                
                # Should have 3 batches: [1,2], [3,4], [5]
                assert len(batches) == 3
                assert len(batches[0]) == 2
                assert len(batches[1]) == 2
                assert len(batches[2]) == 1
    
    def test_empty_pattern(self):
        """Test getting tensor with no matching keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Get with pattern that matches nothing
                result = engine.get_tensor('nonexistent/*', dtype=np.float32)
                assert len(result) == 0
    
    def test_keys_method(self):
        """Test listing keys with pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Store some data
                engine.put_tensor('train/', np.array([1.0, 2.0]))
                engine.put_tensor('test/', np.array([3.0, 4.0]))
                
                # Get all keys
                all_keys = engine.keys('*')
                assert len(all_keys) == 4  # 2 train + 2 test
                
                # Get only train keys
                train_keys = engine.keys('train/*')
                assert len(train_keys) == 2
    
    def test_delete_pattern(self):
        """Test deleting keys by pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Store some data
                engine.put_tensor('train/', np.array([1.0, 2.0]))
                engine.put_tensor('test/', np.array([3.0, 4.0]))
                
                # Delete train keys
                deleted = engine.delete('train/*')
                assert deleted == 2
                
                # Verify train keys are gone
                train_keys = engine.keys('train/*')
                assert len(train_keys) == 0
                
                # Test keys should still exist
                test_keys = engine.keys('test/*')
                assert len(test_keys) == 2
    
    def test_dtype_conversion(self):
        """Test dtype conversion on retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Store float data
                X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                engine.put_tensor('data/', X)
                
                # Retrieve as float64
                result = engine.get_tensor('data/*', dtype=np.float64)
                assert result.dtype == np.float64
                
                # Retrieve as int32
                result_int = engine.get_tensor('data/*', dtype=np.int32)
                assert result_int.dtype == np.int32
                assert list(result_int) == [1, 2, 3]
    
    def test_context_manager(self):
        """Test context manager properly closes database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Use context manager
            with TensorEngine(db_path) as engine:
                engine.put_tensor('test/', np.array([1.0]))
            
            # Should be able to reopen
            with TensorEngine(db_path) as engine:
                result = engine.get_tensor('test/*')
                assert len(result) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

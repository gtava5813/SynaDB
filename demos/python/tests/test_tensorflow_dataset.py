"""
Tests for TensorFlow Dataset integration.

Tests cover:
- syna_dataset function
- SynaDataset class
- Pattern matching for keys
- Batching functionality

Requirements: 13.3
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import SynaDB

# Skip all tests if tensorflow is not available
tf = pytest.importorskip("tensorflow")

from synadb.tensorflow import (
    syna_dataset,
    SynaDataset,
    create_distributed_dataset,
    is_tensorflow_available,
    TF_AVAILABLE,
)


class TestSynaDatasetFunction:
    """Test cases for syna_dataset function."""
    
    def test_tensorflow_available(self):
        """Test that tensorflow availability check works."""
        assert is_tensorflow_available() is True
        assert TF_AVAILABLE is True
    
    def test_syna_dataset_basic(self):
        """Test creating a basic tf.data.Dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create some test data
            with SynaDB(db_path) as db:
                db.put_float("train/0", 1.0)
                db.put_float("train/1", 2.0)
                db.put_float("train/2", 3.0)
            
            # Create dataset
            dataset = syna_dataset(db_path, pattern="train/*", batch_size=2)
            
            # Should be a tf.data.Dataset
            assert isinstance(dataset, tf.data.Dataset)
            
            # Iterate and collect batches
            batches = list(dataset)
            
            # With 3 items and batch_size=2, we should have 2 batches
            assert len(batches) == 2
    
    def test_syna_dataset_all_keys(self):
        """Test dataset with all keys pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("a", 1.0)
                db.put_float("b", 2.0)
                db.put_float("c", 3.0)
            
            dataset = syna_dataset(db_path, pattern="*", batch_size=3)
            
            batches = list(dataset)
            assert len(batches) == 1
            assert batches[0].shape[0] == 3
    
    def test_syna_dataset_values(self):
        """Test that dataset returns correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("data/0", 1.0)
                db.put_float("data/1", 2.0)
                db.put_float("data/2", 3.0)
            
            dataset = syna_dataset(db_path, pattern="data/*", batch_size=3)
            
            batches = list(dataset)
            assert len(batches) == 1
            
            # Get all values
            batch = batches[0]
            values = set()
            for item in batch:
                values.add(float(item.numpy()[0]))
            
            assert values == {1.0, 2.0, 3.0}
    
    def test_syna_dataset_history(self):
        """Test dataset with history (multiple values per key)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                # Write multiple values to same key
                db.put_float("sensor", 1.0)
                db.put_float("sensor", 2.0)
                db.put_float("sensor", 3.0)
            
            dataset = syna_dataset(db_path, pattern="sensor", batch_size=1)
            
            batches = list(dataset)
            assert len(batches) == 1
            
            # The batch should contain the history
            batch = batches[0]
            assert batch.shape[0] == 1  # One item in batch
            assert batch.shape[1] == 3  # Three values in history
    
    def test_syna_dataset_dtype(self):
        """Test dataset with different dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("x", 1.5)
            
            # Default dtype (float32)
            dataset = syna_dataset(db_path, pattern="*", batch_size=1)
            batch = list(dataset)[0]
            assert batch.dtype == tf.float32
            
            # Explicit float64
            dataset64 = syna_dataset(db_path, pattern="*", batch_size=1, dtype=tf.float64)
            batch64 = list(dataset64)[0]
            assert batch64.dtype == tf.float64
    
    def test_syna_dataset_pattern_prefix(self):
        """Test pattern matching with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("train/0", 1.0)
                db.put_float("train/1", 2.0)
                db.put_float("test/0", 3.0)
                db.put_float("val/0", 4.0)
            
            # Only train keys
            train_dataset = syna_dataset(db_path, pattern="train/*", batch_size=10)
            train_batches = list(train_dataset)
            assert len(train_batches) == 1
            assert train_batches[0].shape[0] == 2
            
            # Only test keys
            test_dataset = syna_dataset(db_path, pattern="test/*", batch_size=10)
            test_batches = list(test_dataset)
            assert len(test_batches) == 1
            assert test_batches[0].shape[0] == 1
    
    def test_syna_dataset_prefetch(self):
        """Test that prefetch works with the dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(10):
                    db.put_float(f"data/{i}", float(i))
            
            dataset = syna_dataset(db_path, pattern="data/*", batch_size=2)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            batches = list(dataset)
            assert len(batches) == 5


class TestSynaDatasetClass:
    """Test cases for SynaDataset class."""
    
    def test_dataset_class_creation(self):
        """Test creating a SynaDataset instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("train/0", 1.0)
                db.put_float("train/1", 2.0)
                db.put_float("train/2", 3.0)
            
            ds = SynaDataset(db_path, pattern="train/*")
            
            assert len(ds) == 3
            assert ds.path == db_path
            assert ds.pattern == "train/*"
            assert len(ds.keys) == 3
            
            ds.close()
    
    def test_dataset_class_to_tf_dataset(self):
        """Test converting SynaDataset to tf.data.Dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(6):
                    db.put_float(f"data/{i}", float(i))
            
            ds = SynaDataset(db_path, pattern="data/*")
            tf_dataset = ds.to_tf_dataset(batch_size=2)
            
            assert isinstance(tf_dataset, tf.data.Dataset)
            
            batches = list(tf_dataset)
            assert len(batches) == 3
            
            ds.close()
    
    def test_dataset_class_shuffle(self):
        """Test SynaDataset with shuffle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(20):
                    db.put_float(f"data/{i:02d}", float(i))
            
            ds = SynaDataset(db_path, pattern="data/*")
            tf_dataset = ds.to_tf_dataset(batch_size=5, shuffle=True, buffer_size=20)
            
            # Collect all values
            all_values = []
            for batch in tf_dataset:
                for item in batch:
                    all_values.append(float(item.numpy()[0]))
            
            # Should have all 20 values
            assert len(all_values) == 20
            
            ds.close()


class TestDistributedDataset:
    """Test cases for distributed dataset creation."""
    
    def test_create_distributed_dataset(self):
        """Test creating a distributed dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(10):
                    db.put_float(f"data/{i}", float(i))
            
            dataset = create_distributed_dataset(
                path=db_path,
                pattern="data/*",
                batch_size=2,
            )
            
            assert isinstance(dataset, tf.data.Dataset)
            
            batches = list(dataset)
            assert len(batches) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for PyTorch Dataset integration.

Tests cover:
- SynaDataset creation and basic operations
- Pattern matching for keys
- Transform support
- SynaDataLoader functionality

Requirements: 13.1
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import SynaDB

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from synadb.torch import SynaDataset, SynaDataLoader, is_torch_available, TORCH_AVAILABLE


class TestSynaDataset:
    """Test cases for SynaDataset."""
    
    def test_torch_available(self):
        """Test that torch availability check works."""
        assert is_torch_available() is True
        assert TORCH_AVAILABLE is True
    
    def test_dataset_creation(self):
        """Test creating a SynaDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            # Create some test data
            with SynaDB(db_path) as db:
                db.put_float("train/0", 1.0)
                db.put_float("train/1", 2.0)
                db.put_float("train/2", 3.0)
            
            # Create dataset
            dataset = SynaDataset(db_path, pattern="train/*")
            
            assert len(dataset) == 3
            dataset.close()
    
    def test_dataset_all_keys(self):
        """Test dataset with all keys pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("a", 1.0)
                db.put_float("b", 2.0)
                db.put_float("c", 3.0)
            
            dataset = SynaDataset(db_path, pattern="*")
            
            assert len(dataset) == 3
            dataset.close()
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("data/0", 1.0)
                db.put_float("data/1", 2.0)
                db.put_float("data/2", 3.0)
            
            dataset = SynaDataset(db_path, pattern="data/*")
            
            # Get items
            item0 = dataset[0]
            item1 = dataset[1]
            item2 = dataset[2]
            
            assert isinstance(item0, torch.Tensor)
            assert isinstance(item1, torch.Tensor)
            assert isinstance(item2, torch.Tensor)
            
            # Values should be in sorted key order
            assert item0.item() == 1.0
            assert item1.item() == 2.0
            assert item2.item() == 3.0
            
            dataset.close()
    
    def test_dataset_getitem_history(self):
        """Test getting items with history (multiple values per key)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                # Write multiple values to same key
                db.put_float("sensor", 1.0)
                db.put_float("sensor", 2.0)
                db.put_float("sensor", 3.0)
            
            dataset = SynaDataset(db_path, pattern="sensor")
            
            assert len(dataset) == 1
            
            item = dataset[0]
            assert isinstance(item, torch.Tensor)
            assert len(item) == 3
            assert torch.allclose(item, torch.tensor([1.0, 2.0, 3.0]))
            
            dataset.close()
    
    def test_dataset_with_transform(self):
        """Test dataset with transform function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("x", 2.0)
            
            # Transform that doubles the value
            def double_transform(x):
                return x * 2
            
            dataset = SynaDataset(db_path, pattern="*", transform=double_transform)
            
            item = dataset[0]
            assert item.item() == 4.0  # 2.0 * 2
            
            dataset.close()
    
    def test_dataset_get_key(self):
        """Test getting the key for an index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("alpha", 1.0)
                db.put_float("beta", 2.0)
            
            dataset = SynaDataset(db_path, pattern="*")
            
            # Keys should be sorted
            assert dataset.get_key(0) == "alpha"
            assert dataset.get_key(1) == "beta"
            
            dataset.close()
    
    def test_dataset_index_error(self):
        """Test that out of bounds index raises IndexError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("x", 1.0)
            
            dataset = SynaDataset(db_path, pattern="*")
            
            with pytest.raises(IndexError):
                _ = dataset[10]
            
            with pytest.raises(IndexError):
                _ = dataset[-10]
            
            dataset.close()
    
    def test_dataset_pattern_prefix(self):
        """Test pattern matching with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                db.put_float("train/0", 1.0)
                db.put_float("train/1", 2.0)
                db.put_float("test/0", 3.0)
                db.put_float("val/0", 4.0)
            
            # Only train keys
            train_dataset = SynaDataset(db_path, pattern="train/*")
            assert len(train_dataset) == 2
            train_dataset.close()
            
            # Only test keys
            test_dataset = SynaDataset(db_path, pattern="test/*")
            assert len(test_dataset) == 1
            test_dataset.close()


class TestSynaDataLoader:
    """Test cases for SynaDataLoader."""
    
    def test_dataloader_basic(self):
        """Test basic DataLoader functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(10):
                    db.put_float(f"data/{i}", float(i))
            
            dataset = SynaDataset(db_path, pattern="data/*")
            loader = SynaDataLoader(dataset, batch_size=2, shuffle=False)
            
            batches = list(loader)
            
            assert len(batches) == 5  # 10 items / batch_size 2
            
            # First batch should have first two items
            assert batches[0].shape[0] == 2
            
            dataset.close()
    
    def test_dataloader_shuffle(self):
        """Test DataLoader with shuffle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(20):
                    db.put_float(f"data/{i:02d}", float(i))
            
            dataset = SynaDataset(db_path, pattern="data/*")
            
            # Get two iterations with shuffle
            loader = SynaDataLoader(dataset, batch_size=5, shuffle=True)
            
            first_run = [batch.clone() for batch in loader]
            second_run = [batch.clone() for batch in loader]
            
            # With shuffle, order should likely be different
            # (Note: there's a small chance they could be the same)
            all_first = torch.cat(first_run).flatten()
            all_second = torch.cat(second_run).flatten()
            
            # Both should contain same values (just different order)
            assert set(all_first.tolist()) == set(all_second.tolist())
            
            dataset.close()
    
    def test_dataloader_iteration(self):
        """Test iterating through DataLoader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with SynaDB(db_path) as db:
                for i in range(6):
                    db.put_float(f"x/{i}", float(i))
            
            dataset = SynaDataset(db_path, pattern="x/*")
            loader = SynaDataLoader(dataset, batch_size=2)
            
            total_items = 0
            for batch in loader:
                total_items += batch.shape[0]
            
            assert total_items == 6
            
            dataset.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

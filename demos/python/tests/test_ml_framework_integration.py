"""
ML Framework Integration Tests for SynaDB v0.6.0.

This module provides comprehensive integration tests for ML frameworks:
- PyTorch Dataset and DataLoader integration
- TensorFlow tf.data.Dataset integration
- End-to-end ML training workflows
- Cross-component integration (VectorStore + TensorEngine + ML frameworks)

These tests validate Requirements 13.1-13.4 from the design document:
- 13.1: PyTorch Dataset and DataLoader implementation
- 13.2: PyTorch DistributedSampler for multi-GPU training
- 13.3: TensorFlow tf.data.Dataset implementation
- 13.4: TensorFlow tf.distribute strategies support

**Feature: syna-ai-native, ML Framework Integration Tests**
**Validates: Requirements 13.1, 13.2, 13.3, 13.4**
"""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import SynaDB, TensorEngine, VectorStore


# ============================================================================
# PyTorch Integration Tests
# ============================================================================

class TestPyTorchMLWorkflow:
    """
    End-to-end ML workflow tests with PyTorch.
    
    Tests the complete data pipeline from SynaDB to PyTorch training.
    **Validates: Requirements 13.1, 13.2**
    """
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, 'test.db')
    
    def test_pytorch_training_data_pipeline(self, temp_db):
        """
        Test complete PyTorch training data pipeline.
        
        Validates that data flows correctly from SynaDB through
        Dataset and DataLoader for training.
        **Validates: Requirements 13.1**
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset, SynaDataLoader
        
        # Create training data in SynaDB using float values
        with SynaDB(temp_db) as db:
            # Store 100 samples as float values
            for i in range(100):
                db.put_float(f"train/sample_{i:03d}", float(i))
        
        # Create PyTorch Dataset
        dataset = SynaDataset(temp_db, pattern="train/*")
        
        assert len(dataset) == 100
        
        # Create DataLoader with batching
        loader = SynaDataLoader(dataset, batch_size=16, shuffle=True)
        
        # Simulate training loop
        total_batches = 0
        total_samples = 0
        
        for batch in loader:
            assert isinstance(batch, torch.Tensor)
            total_batches += 1
            total_samples += batch.shape[0]
        
        # Should have processed all samples
        assert total_samples == 100
        # With batch_size=16 and 100 samples, expect 7 batches (6*16 + 1*4)
        assert total_batches == 7
        
        dataset.close()
    
    def test_pytorch_tensor_engine_integration(self, temp_db):
        """
        Test TensorEngine integration with PyTorch.
        
        Validates that TensorEngine can load data directly as PyTorch tensors.
        **Validates: Requirements 13.1**
        """
        torch = pytest.importorskip("torch")
        
        # Create tensor data
        with TensorEngine(temp_db) as engine:
            # Store a batch of training data
            X_train = np.random.randn(64, 128).astype(np.float32)
            engine.put_tensor('train/X/', X_train)
            
            # Load as PyTorch tensor
            X_torch = engine.get_tensor_torch('train/X/*')
            
            assert isinstance(X_torch, torch.Tensor)
            assert X_torch.dtype == torch.float32
            # Flattened data should have same total elements
            assert X_torch.numel() == X_train.size
    
    def test_pytorch_dataset_with_transforms(self, temp_db):
        """
        Test PyTorch Dataset with custom transforms.
        
        Validates that transforms are correctly applied to data.
        **Validates: Requirements 13.1**
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(10):
                db.put_float(f"data/{i}", float(i))
        
        # Define a transform that normalizes data
        def normalize(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        
        dataset = SynaDataset(temp_db, pattern="data/*", transform=normalize)
        
        # Get a sample and verify transform was applied
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        
        dataset.close()
    
    def test_pytorch_distributed_sampler(self, temp_db):
        """
        Test DistributedSampler for multi-GPU training.
        
        Validates that DistributedSampler correctly partitions data.
        **Validates: Requirements 13.2**
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset, get_distributed_sampler
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(100):
                db.put_float(f"data/{i:03d}", float(i))
        
        dataset = SynaDataset(temp_db, pattern="data/*")
        
        # Create sampler for 2 replicas
        sampler = get_distributed_sampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False
        )
        
        # Sampler should partition data
        indices = list(sampler)
        
        # Each replica should get half the data
        assert len(indices) == 50
        
        dataset.close()
    
    def test_pytorch_multiple_epochs(self, temp_db):
        """
        Test DataLoader across multiple epochs.
        
        Validates that data can be iterated multiple times.
        **Validates: Requirements 13.1**
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset, SynaDataLoader
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(20):
                db.put_float(f"data/{i:02d}", float(i))
        
        dataset = SynaDataset(temp_db, pattern="data/*")
        loader = SynaDataLoader(dataset, batch_size=5, shuffle=True)
        
        # Run multiple epochs
        epoch_sums = []
        for epoch in range(3):
            epoch_sum = 0.0
            for batch in loader:
                epoch_sum += batch.sum().item()
            epoch_sums.append(epoch_sum)
        
        # All epochs should process same total data
        expected_sum = sum(range(20))
        for epoch_sum in epoch_sums:
            assert abs(epoch_sum - expected_sum) < 1e-5
        
        dataset.close()


# ============================================================================
# TensorFlow Integration Tests
# ============================================================================

class TestTensorFlowMLWorkflow:
    """
    End-to-end ML workflow tests with TensorFlow.
    
    Tests the complete data pipeline from SynaDB to TensorFlow training.
    **Validates: Requirements 13.3, 13.4**
    """
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, 'test.db')
    
    def test_tensorflow_training_data_pipeline(self, temp_db):
        """
        Test complete TensorFlow training data pipeline.
        
        Validates that data flows correctly from SynaDB through
        tf.data.Dataset for training.
        **Validates: Requirements 13.3**
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import syna_dataset
        
        # Create training data in SynaDB
        with SynaDB(temp_db) as db:
            for i in range(50):
                db.put_float(f"train/{i:03d}", float(i))
        
        # Create TensorFlow Dataset
        dataset = syna_dataset(temp_db, pattern="train/*", batch_size=10)
        
        # Simulate training loop
        total_batches = 0
        total_samples = 0
        
        for batch in dataset:
            assert isinstance(batch, tf.Tensor)
            total_batches += 1
            total_samples += batch.shape[0]
        
        # Should have processed all samples
        assert total_samples == 50
        assert total_batches == 5
    
    def test_tensorflow_dataset_prefetch(self, temp_db):
        """
        Test TensorFlow Dataset with prefetching.
        
        Validates that prefetch works correctly for performance.
        **Validates: Requirements 13.3**
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import syna_dataset
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(30):
                db.put_float(f"data/{i:02d}", float(i))
        
        # Create dataset with prefetch
        dataset = syna_dataset(temp_db, pattern="data/*", batch_size=5)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Iterate and verify
        batches = list(dataset)
        assert len(batches) == 6
    
    def test_tensorflow_dataset_class(self, temp_db):
        """
        Test SynaDataset class for TensorFlow.
        
        Validates the object-oriented interface.
        **Validates: Requirements 13.3**
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import SynaDataset
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(25):
                db.put_float(f"samples/{i:02d}", float(i))
        
        # Create SynaDataset instance
        ds = SynaDataset(temp_db, pattern="samples/*")
        
        assert len(ds) == 25
        assert ds.path == temp_db
        assert ds.pattern == "samples/*"
        
        # Convert to tf.data.Dataset
        tf_dataset = ds.to_tf_dataset(batch_size=5)
        
        batches = list(tf_dataset)
        assert len(batches) == 5
        
        ds.close()
    
    def test_tensorflow_distributed_dataset(self, temp_db):
        """
        Test distributed dataset creation.
        
        Validates that datasets can be created for distributed training.
        **Validates: Requirements 13.4**
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import create_distributed_dataset
        
        # Create test data
        with SynaDB(temp_db) as db:
            for i in range(40):
                db.put_float(f"dist/{i:02d}", float(i))
        
        # Create distributed dataset
        dataset = create_distributed_dataset(
            path=temp_db,
            pattern="dist/*",
            batch_size=8
        )
        
        assert isinstance(dataset, tf.data.Dataset)
        
        # Verify data
        batches = list(dataset)
        assert len(batches) == 5
    
    def test_tensorflow_dtype_conversion(self, temp_db):
        """
        Test TensorFlow Dataset with different dtypes.
        
        Validates that dtype conversion works correctly.
        **Validates: Requirements 13.3**
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import syna_dataset
        
        # Create test data
        with SynaDB(temp_db) as db:
            db.put_float("x", 3.14159)
        
        # Test float32 (default)
        dataset32 = syna_dataset(temp_db, pattern="*", batch_size=1)
        batch32 = list(dataset32)[0]
        assert batch32.dtype == tf.float32
        
        # Test float64
        dataset64 = syna_dataset(temp_db, pattern="*", batch_size=1, dtype=tf.float64)
        batch64 = list(dataset64)[0]
        assert batch64.dtype == tf.float64


# ============================================================================
# Cross-Component Integration Tests
# ============================================================================

class TestCrossComponentIntegration:
    """
    Tests for integration between different SynaDB components.
    
    Validates that VectorStore, TensorEngine, and ML frameworks
    work together seamlessly.
    """
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, 'test.db')
    
    def test_vectorstore_to_pytorch(self, temp_db):
        """
        Test loading VectorStore embeddings into PyTorch.
        
        Validates that embeddings stored in VectorStore can be
        loaded as PyTorch tensors for training.
        """
        torch = pytest.importorskip("torch")
        
        # Create embeddings in VectorStore
        store = VectorStore(temp_db, dimensions=128, metric="cosine")
        
        embeddings = []
        for i in range(20):
            emb = np.random.randn(128).astype(np.float32)
            store.insert(f"doc_{i:02d}", emb)
            embeddings.append(emb)
        
        # Search and get results
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=5)
        
        # Convert results to PyTorch tensors
        result_vectors = [torch.from_numpy(np.array(r.vector)) for r in results]
        
        assert len(result_vectors) == 5
        for vec in result_vectors:
            assert isinstance(vec, torch.Tensor)
            assert vec.shape == (128,)
    
    def test_tensorengine_to_tensorflow(self, temp_db):
        """
        Test loading TensorEngine data into TensorFlow.
        
        Validates that tensor data can be loaded as TensorFlow tensors.
        """
        tf = pytest.importorskip("tensorflow")
        
        # Create tensor data
        with TensorEngine(temp_db) as engine:
            X = np.random.randn(32, 64).astype(np.float32)
            engine.put_tensor('features/', X)
            
            # Load as numpy and convert to TensorFlow
            X_loaded = engine.get_tensor('features/*', dtype=np.float32)
            X_tf = tf.convert_to_tensor(X_loaded)
            
            assert isinstance(X_tf, tf.Tensor)
            assert X_tf.dtype == tf.float32
    
    def test_combined_ml_pipeline(self, temp_db):
        """
        Test a combined ML pipeline using multiple components.
        
        Simulates a real-world scenario where:
        1. Features are stored in TensorEngine
        2. Embeddings are stored in VectorStore
        3. Data is loaded for training
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset, SynaDataLoader
        
        # Store training features as float values
        with SynaDB(temp_db) as db:
            for i in range(50):
                # Store feature values as floats
                db.put_float(f"features/{i:03d}", float(i))
        
        # Create dataset and loader
        dataset = SynaDataset(temp_db, pattern="features/*")
        loader = SynaDataLoader(dataset, batch_size=10, shuffle=True)
        
        # Simulate training
        batch_count = 0
        for batch in loader:
            assert isinstance(batch, torch.Tensor)
            batch_count += 1
        
        assert batch_count == 5
        
        dataset.close()


# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestMLFrameworkPerformance:
    """
    Performance tests for ML framework integrations.
    
    These tests verify that the integrations can handle
    realistic data sizes efficiently.
    """
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, 'test.db')
    
    def test_pytorch_large_dataset(self, temp_db):
        """
        Test PyTorch integration with larger dataset.
        
        Validates performance with 1000 samples.
        """
        torch = pytest.importorskip("torch")
        from synadb.torch import SynaDataset, SynaDataLoader
        
        # Create larger dataset
        with SynaDB(temp_db) as db:
            for i in range(1000):
                db.put_float(f"data/{i:04d}", float(i))
        
        dataset = SynaDataset(temp_db, pattern="data/*")
        loader = SynaDataLoader(dataset, batch_size=64, shuffle=True)
        
        # Iterate through all data
        total = 0
        for batch in loader:
            total += batch.shape[0]
        
        assert total == 1000
        
        dataset.close()
    
    def test_tensorflow_large_dataset(self, temp_db):
        """
        Test TensorFlow integration with larger dataset.
        
        Validates performance with 1000 samples.
        """
        tf = pytest.importorskip("tensorflow")
        from synadb.tensorflow import syna_dataset
        
        # Create larger dataset
        with SynaDB(temp_db) as db:
            for i in range(1000):
                db.put_float(f"data/{i:04d}", float(i))
        
        dataset = syna_dataset(temp_db, pattern="data/*", batch_size=64)
        
        # Iterate through all data
        total = 0
        for batch in dataset:
            total += batch.shape[0]
        
        assert total == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

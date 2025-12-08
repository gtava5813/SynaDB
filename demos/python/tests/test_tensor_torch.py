"""Tests for TensorEngine PyTorch integration."""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import TensorEngine

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")


class TestTensorEngineTorch:
    """Test cases for TensorEngine PyTorch integration."""
    
    def test_get_tensor_torch_basic(self):
        """Test loading tensor directly as PyTorch tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Create test data
                X = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
                engine.put_tensor('test/', X)
                
                # Get as PyTorch tensor
                result = engine.get_tensor_torch('test/*')
                
                assert isinstance(result, torch.Tensor)
                assert result.dtype == torch.float32
                assert len(result) == 5
                assert torch.allclose(result, torch.tensor(X))
    
    def test_get_tensor_torch_with_shape(self):
        """Test loading tensor with reshape as PyTorch tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                # Create 2D test data
                X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                engine.put_tensor('matrix/', X)
                
                # Get as PyTorch tensor with shape
                result = engine.get_tensor_torch('matrix/*', shape=(2, 2))
                
                assert result.shape == (2, 2)
                assert torch.allclose(result, torch.tensor(X))
    
    def test_get_tensor_torch_device_cpu(self):
        """Test loading tensor to CPU device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with TensorEngine(db_path) as engine:
                X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                engine.put_tensor('test/', X)
                
                result = engine.get_tensor_torch('test/*', device='cpu')
                
                assert result.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

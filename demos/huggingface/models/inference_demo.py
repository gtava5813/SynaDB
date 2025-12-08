#!/usr/bin/env python3
"""
Model Inference Demo with Syna

This demo shows how to:
- Load pre-trained model weights from Syna
- Run inference on test samples
- Compare latency vs file loading

Requirements: 5.3 - WHEN running the inference demo THEN the demo 
SHALL show loading model weights stored in Syna

Run with: python inference_demo.py
"""

import os
import sys
import time
import tempfile
import numpy as np
from typing import Dict, Any
import io

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def save_model_to_Syna(db: SynaDB, model: nn.Module, model_name: str = "model"):
    """
    Save a PyTorch model's state_dict to Syna.
    
    Each parameter tensor is stored as a separate key for efficient
    partial loading and inspection.
    
    Args:
        db: SynaDB instance
        model: PyTorch model to save
        model_name: Prefix for model keys
    """
    state_dict = model.state_dict()
    
    # Store metadata
    param_names = list(state_dict.keys())
    db.put_text(f"{model_name}/metadata/param_names", ",".join(param_names))
    db.put_int(f"{model_name}/metadata/num_params", len(param_names))
    
    # Store each parameter
    for name, tensor in state_dict.items():
        # Convert tensor to bytes
        buffer = io.BytesIO()
        torch.save({
            'data': tensor.cpu().numpy(),
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }, buffer)
        
        db.put_bytes(f"{model_name}/params/{name}", buffer.getvalue())


def load_model_from_Syna(db: SynaDB, model: nn.Module, model_name: str = "model"):
    """
    Load a PyTorch model's state_dict from Syna.
    
    Args:
        db: SynaDB instance
        model: PyTorch model to load weights into
        model_name: Prefix for model keys
        
    Returns:
        The model with loaded weights
    """
    state_dict = {}
    
    # Get parameter names
    param_names_str = db.get_text(f"{model_name}/metadata/param_names")
    if not param_names_str:
        raise ValueError(f"No model found with name '{model_name}'")
    
    param_names = param_names_str.split(",")
    
    # Load each parameter
    for name in param_names:
        param_bytes = db.get_bytes(f"{model_name}/params/{name}")
        if param_bytes is None:
            raise ValueError(f"Missing parameter: {name}")
        
        # Deserialize
        buffer = io.BytesIO(param_bytes)
        data = torch.load(buffer, weights_only=False)
        tensor = torch.from_numpy(data['data'])
        state_dict[name] = tensor
    
    model.load_state_dict(state_dict)
    return model


def save_model_to_file(model: nn.Module, filepath: str):
    """Save model to a file using torch.save."""
    torch.save(model.state_dict(), filepath)


def load_model_from_file(model: nn.Module, filepath: str):
    """Load model from a file using torch.load."""
    state_dict = torch.load(filepath, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def create_test_samples(num_samples: int = 100) -> tuple:
    """Create synthetic test samples."""
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    return images, labels


def benchmark_inference(model: nn.Module, images: torch.Tensor, num_runs: int = 10) -> dict:
    """Benchmark inference performance."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(images[:10])
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(images)
            times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'samples_per_sec': len(images) / np.mean(times)
    }


def benchmark_model_loading(load_fn, num_runs: int = 10, name: str = "") -> dict:
    """Benchmark model loading performance."""
    times = []
    
    for i in range(num_runs):
        start = time.time()
        load_fn()
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def main():
    print("=" * 60)
    print("Model Inference Demo with Syna")
    print("Requirement 5.3: Load model weights from Syna")
    print("=" * 60 + "\n")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        return 1
    
    # Configuration
    DB_PATH = os.path.abspath("inference_demo.db")
    MODEL_FILE = tempfile.mktemp(suffix=".pt")
    NUM_TEST_SAMPLES = 100
    NUM_BENCHMARK_RUNS = 3  # Reduced for faster demo
    
    try:
        # 1. Create and initialize model
        print("1. Creating model")
        print("-" * 40)
        model = SimpleCNN()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model: SimpleCNN")
        print(f"   Parameters: {num_params:,}")
        
        # Initialize with random weights (simulating trained model)
        for param in model.parameters():
            nn.init.normal_(param, mean=0, std=0.1)
        print("   ✓ Initialized with random weights")
        print()
        
        # 2. Save model to both Syna and file
        print("2. Saving model")
        print("-" * 40)
        
        # Save to Syna
        start = time.time()
        with SynaDB(DB_PATH) as db:
            save_model_to_Syna(db, model, "cnn_v1")
        syna_save_time = time.time() - start
        syna_size = os.path.getsize(DB_PATH)
        print(f"   Syna: {syna_save_time * 1000:.2f}ms, {syna_size / 1024:.1f} KB")
        
        # Save to file
        start = time.time()
        save_model_to_file(model, MODEL_FILE)
        file_save_time = time.time() - start
        file_size = os.path.getsize(MODEL_FILE)
        print(f"   File:     {file_save_time * 1000:.2f}ms, {file_size / 1024:.1f} KB")
        print()
        
        # 3. Benchmark model loading
        print("3. Benchmarking model loading")
        print("-" * 40)
        
        # Syna loading - single measurement (opening DB is slow)
        start = time.time()
        syna_model = SimpleCNN()
        with SynaDB(DB_PATH) as db:
            load_model_from_Syna(db, syna_model, "cnn_v1")
        syna_load_time = (time.time() - start) * 1000
        print(f"   Syna loading: {syna_load_time:.2f}ms")
        
        # File loading - single measurement
        start = time.time()
        file_model = SimpleCNN()
        load_model_from_file(file_model, MODEL_FILE)
        file_load_time = (time.time() - start) * 1000
        print(f"   File loading: {file_load_time:.2f}ms")
        
        # Comparison
        if syna_load_time > 0:
            speedup = file_load_time / syna_load_time
            print(f"\n   Loading comparison: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} with Syna")
        print()
        
        # 4. Load model from Syna and verify
        print("4. Loading and verifying model from Syna")
        print("-" * 40)
        
        loaded_model = SimpleCNN()
        with SynaDB(DB_PATH) as db:
            load_model_from_Syna(db, loaded_model, "cnn_v1")
        
        # Verify weights match
        original_params = list(model.parameters())
        loaded_params = list(loaded_model.parameters())
        
        all_match = True
        for orig, loaded in zip(original_params, loaded_params):
            if not torch.allclose(orig, loaded):
                all_match = False
                break
        
        if all_match:
            print("   ✓ All parameters match original model")
        else:
            print("   ✗ Parameter mismatch detected!")
        print()
        
        # 5. Run inference
        print("5. Running inference")
        print("-" * 40)
        
        # Create test samples
        test_images, test_labels = create_test_samples(NUM_TEST_SAMPLES)
        print(f"   Test samples: {NUM_TEST_SAMPLES}")
        print(f"   Input shape: {tuple(test_images.shape)}")
        
        # Run inference
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(test_images)
            predictions = outputs.argmax(dim=1)
        
        print(f"   Output shape: {tuple(outputs.shape)}")
        print(f"   Predictions: {predictions[:10].tolist()}...")
        print()
        
        # 6. Benchmark inference
        print("6. Benchmarking inference")
        print("-" * 40)
        
        inference_results = benchmark_inference(loaded_model, test_images, NUM_BENCHMARK_RUNS)
        print(f"   Batch size: {NUM_TEST_SAMPLES}")
        print(f"   Mean latency: {inference_results['mean_ms']:.2f}ms")
        print(f"   Std: {inference_results['std_ms']:.2f}ms")
        print(f"   Throughput: {inference_results['samples_per_sec']:.0f} samples/sec")
        
        # Per-sample latency
        per_sample_ms = inference_results['mean_ms'] / NUM_TEST_SAMPLES
        print(f"   Per-sample latency: {per_sample_ms * 1000:.2f}μs")
        print()
        
        # 7. Demonstrate partial loading (advanced feature)
        print("7. Demonstrating partial parameter inspection")
        print("-" * 40)
        
        with SynaDB(DB_PATH) as db:
            # List stored parameters
            keys = db.keys()
            param_keys = [k for k in keys if k.startswith("cnn_v1/params/")]
            print(f"   Stored parameters: {len(param_keys)}")
            
            # Show parameter names
            param_names = db.get_text("cnn_v1/metadata/param_names")
            print(f"   Parameter names:")
            for name in param_names.split(",")[:5]:
                print(f"      - {name}")
            if len(param_names.split(",")) > 5:
                print(f"      ... and {len(param_names.split(',')) - 5} more")
        print()
        
        # 8. Summary
        print("8. Summary")
        print("-" * 40)
        print(f"   Model storage:")
        print(f"      Syna: {syna_size / 1024:.1f} KB")
        print(f"      File:     {file_size / 1024:.1f} KB")
        print(f"   Loading time:")
        print(f"      Syna: {syna_load_time:.2f}ms")
        print(f"      File:     {file_load_time:.2f}ms")
        print(f"   Inference throughput: {inference_results['samples_per_sec']:.0f} samples/sec")
        print()
        
        print("=" * 60)
        print("Inference Demo Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


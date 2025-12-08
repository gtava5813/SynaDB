#!/usr/bin/env python3
"""
Syna NumPy Integration Demo

This demo shows how to use Syna with NumPy for ML workloads:
- Storing numpy arrays as bytes
- Extracting history as numpy arrays
- Zero-copy tensor access patterns
- Memory efficiency comparison

Requirements: 2.2 - WHEN a developer views the Python numpy demo THEN the demo 
SHALL show zero-copy tensor extraction into numpy arrays

Run with: python numpy_integration.py
"""

import os
import sys
import time
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from Syna import SynaDB


def demo_store_numpy_arrays():
    """Demonstrate storing numpy arrays as bytes."""
    print("=" * 60)
    print("1. STORING NUMPY ARRAYS AS BYTES")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "numpy_demo.db")
        
        with SynaDB(db_path) as db:
            # Store 1D array
            print("\n1D Array Storage")
            print("-" * 40)
            arr_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
            print(f"Original: {arr_1d}")
            print(f"Shape: {arr_1d.shape}, dtype: {arr_1d.dtype}")
            
            # Store as bytes
            db.put_bytes("array/1d", arr_1d.tobytes())
            print("✓ Stored with db.put_bytes('array/1d', arr.tobytes())")
            
            # Retrieve
            data = db.get_bytes("array/1d")
            recovered = np.frombuffer(data, dtype=np.float64)
            print(f"Recovered: {recovered}")
            print(f"✓ Arrays match: {np.array_equal(arr_1d, recovered)}")
            
            # Store 2D array
            print("\n2D Array Storage")
            print("-" * 40)
            arr_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            print(f"Original:\n{arr_2d}")
            print(f"Shape: {arr_2d.shape}, dtype: {arr_2d.dtype}")
            
            # Store with shape metadata
            db.put_bytes("array/2d/data", arr_2d.tobytes())
            db.put_text("array/2d/shape", str(arr_2d.shape))
            db.put_text("array/2d/dtype", str(arr_2d.dtype))
            print("✓ Stored data, shape, and dtype separately")
            
            # Retrieve with shape
            data = db.get_bytes("array/2d/data")
            shape = eval(db.get_text("array/2d/shape"))
            dtype = np.dtype(db.get_text("array/2d/dtype"))
            recovered_2d = np.frombuffer(data, dtype=dtype).reshape(shape)
            print(f"Recovered:\n{recovered_2d}")
            print(f"✓ Arrays match: {np.array_equal(arr_2d, recovered_2d)}")
            
            # Using put_numpy helper
            print("\nUsing put_numpy() Helper")
            print("-" * 40)
            arr = np.random.randn(10, 5).astype(np.float64)
            db.put_numpy("array/helper", arr)
            print(f"db.put_numpy('array/helper', arr)  # shape={arr.shape}")
            
            recovered = db.get_numpy("array/helper", dtype=np.float64, shape=(10, 5))
            print(f"db.get_numpy('array/helper', dtype=np.float64, shape=(10, 5))")
            print(f"✓ Arrays match: {np.allclose(arr, recovered)}")
    
    print()


def demo_history_tensor_extraction():
    """Demonstrate extracting history as numpy array."""
    print("=" * 60)
    print("2. HISTORY TENSOR EXTRACTION")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "numpy_demo.db")
        
        with SynaDB(db_path) as db:
            # Simulate time-series data
            print("\nSimulating Sensor Time-Series")
            print("-" * 40)
            
            # Generate synthetic temperature data
            np.random.seed(42)
            n_samples = 1000
            base_temp = 20.0
            noise = np.random.randn(n_samples) * 0.5
            trend = np.linspace(0, 2, n_samples)
            temperatures = base_temp + trend + noise
            
            print(f"Generating {n_samples} temperature readings...")
            
            # Store each reading (simulating real-time ingestion)
            start = time.time()
            for temp in temperatures:
                db.put_float("sensor/temperature", float(temp))
            write_time = time.time() - start
            print(f"✓ Wrote {n_samples} values in {write_time:.3f}s")
            print(f"  Throughput: {n_samples/write_time:.0f} writes/sec")
            
            # Extract as tensor
            print("\nExtracting History as NumPy Array")
            print("-" * 40)
            
            start = time.time()
            tensor = db.get_history_tensor("sensor/temperature")
            read_time = time.time() - start
            
            print(f"tensor = db.get_history_tensor('sensor/temperature')")
            print(f"✓ Shape: {tensor.shape}")
            print(f"✓ dtype: {tensor.dtype}")
            print(f"✓ Extraction time: {read_time*1000:.2f}ms")
            print(f"✓ First 5 values: {tensor[:5]}")
            print(f"✓ Last 5 values: {tensor[-5:]}")
            
            # Verify data integrity
            print("\nData Integrity Check")
            print("-" * 40)
            print(f"Original mean: {temperatures.mean():.4f}")
            print(f"Extracted mean: {tensor.mean():.4f}")
            print(f"✓ Values match: {np.allclose(temperatures, tensor)}")
    
    print()


def demo_zero_copy_access():
    """Demonstrate zero-copy tensor access patterns."""
    print("=" * 60)
    print("3. ZERO-COPY TENSOR ACCESS")
    print("=" * 60)
    
    print("""
Note: The current Python wrapper copies data from the FFI boundary
for safety. True zero-copy would require memory-mapped access.

However, we can demonstrate efficient patterns that minimize copies:
""")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "numpy_demo.db")
        
        with SynaDB(db_path) as db:
            # Store large tensor
            n_values = 100000
            print(f"\nStoring {n_values} float values...")
            
            for i in range(n_values):
                db.put_float("large/tensor", float(i) * 0.001)
            
            # Pattern 1: Direct extraction (single copy)
            print("\nPattern 1: Direct Extraction")
            print("-" * 40)
            print("""
# Best: Single extraction, single copy
tensor = db.get_history_tensor("key")
# Use tensor directly - no additional copies
""")
            
            start = time.time()
            tensor = db.get_history_tensor("large/tensor")
            time1 = time.time() - start
            print(f"✓ Extracted {len(tensor)} values in {time1*1000:.2f}ms")
            
            # Pattern 2: Avoid repeated extractions
            print("\nPattern 2: Cache Extracted Tensors")
            print("-" * 40)
            print("""
# Good: Extract once, use many times
tensor = db.get_history_tensor("key")
mean = tensor.mean()
std = tensor.std()
max_val = tensor.max()

# Bad: Extract multiple times
mean = db.get_history_tensor("key").mean()  # Extracts again!
std = db.get_history_tensor("key").std()    # Extracts again!
""")
            
            # Good pattern
            start = time.time()
            tensor = db.get_history_tensor("large/tensor")
            mean = tensor.mean()
            std = tensor.std()
            max_val = tensor.max()
            good_time = time.time() - start
            print(f"✓ Good pattern: {good_time*1000:.2f}ms")
            
            # Bad pattern (for comparison)
            start = time.time()
            mean = db.get_history_tensor("large/tensor").mean()
            std = db.get_history_tensor("large/tensor").std()
            max_val = db.get_history_tensor("large/tensor").max()
            bad_time = time.time() - start
            print(f"✗ Bad pattern: {bad_time*1000:.2f}ms ({bad_time/good_time:.1f}x slower)")
            
            # Pattern 3: Slice after extraction
            print("\nPattern 3: Slice After Extraction")
            print("-" * 40)
            print("""
# Extract full tensor, then slice in numpy (fast)
tensor = db.get_history_tensor("key")
recent = tensor[-100:]  # Last 100 values
window = tensor[500:600]  # Specific window
""")
            
            tensor = db.get_history_tensor("large/tensor")
            recent = tensor[-100:]
            window = tensor[500:600]
            print(f"✓ recent[-100:] shape: {recent.shape}")
            print(f"✓ window[500:600] shape: {window.shape}")
    
    print()


def demo_memory_efficiency():
    """Demonstrate memory efficiency comparison."""
    print("=" * 60)
    print("4. MEMORY EFFICIENCY COMPARISON")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "numpy_demo.db")
        
        # Test different storage methods
        n_values = 10000
        values = np.random.randn(n_values).astype(np.float64)
        
        print(f"\nStoring {n_values} float64 values")
        print("-" * 40)
        
        # Method 1: Individual floats (with history)
        with SynaDB(db_path) as db:
            start = time.time()
            for v in values:
                db.put_float("method1/individual", float(v))
            time1 = time.time() - start
        
        size1 = os.path.getsize(db_path)
        os.remove(db_path)
        
        # Method 2: Single bytes blob
        with SynaDB(db_path) as db:
            start = time.time()
            db.put_bytes("method2/blob", values.tobytes())
            time2 = time.time() - start
        
        size2 = os.path.getsize(db_path)
        os.remove(db_path)
        
        # Method 3: Chunked storage
        chunk_size = 1000
        with SynaDB(db_path) as db:
            start = time.time()
            for i in range(0, n_values, chunk_size):
                chunk = values[i:i+chunk_size]
                db.put_bytes(f"method3/chunk/{i}", chunk.tobytes())
            time3 = time.time() - start
        
        size3 = os.path.getsize(db_path)
        os.remove(db_path)
        
        # Raw data size
        raw_size = values.nbytes
        
        print(f"\nRaw data size: {raw_size:,} bytes ({raw_size/1024:.1f} KB)")
        print()
        
        print("Method 1: Individual put_float() calls")
        print(f"  Storage: {size1:,} bytes ({size1/raw_size:.2f}x raw)")
        print(f"  Time: {time1*1000:.1f}ms")
        print(f"  ✓ Preserves full history, supports get_history_tensor()")
        print()
        
        print("Method 2: Single put_bytes() blob")
        print(f"  Storage: {size2:,} bytes ({size2/raw_size:.2f}x raw)")
        print(f"  Time: {time2*1000:.1f}ms")
        print(f"  ✓ Most compact, fastest write")
        print()
        
        print("Method 3: Chunked put_bytes()")
        print(f"  Storage: {size3:,} bytes ({size3/raw_size:.2f}x raw)")
        print(f"  Time: {time3*1000:.1f}ms")
        print(f"  ✓ Balance of compactness and flexibility")
        print()
        
        print("Recommendation:")
        print("-" * 40)
        print("""
- Use put_float() for time-series that need history extraction
- Use put_bytes() for large arrays that are read as a whole
- Use chunked storage for very large datasets with partial reads
""")
    
    print()


def demo_ml_workflow():
    """Demonstrate a typical ML workflow with numpy."""
    print("=" * 60)
    print("5. ML WORKFLOW EXAMPLE")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ml_demo.db")
        
        with SynaDB(db_path) as db:
            # Simulate feature collection
            print("\nStep 1: Collect Features")
            print("-" * 40)
            
            np.random.seed(42)
            n_samples = 500
            
            # Simulate multiple sensors
            for i in range(n_samples):
                db.put_float("feature/temp", 20 + np.random.randn() * 2)
                db.put_float("feature/humidity", 50 + np.random.randn() * 10)
                db.put_float("feature/pressure", 1013 + np.random.randn() * 5)
                db.put_float("label/anomaly", float(np.random.rand() > 0.9))
            
            print(f"✓ Collected {n_samples} samples with 3 features + label")
            
            # Extract as training data
            print("\nStep 2: Extract Training Data")
            print("-" * 40)
            
            temp = db.get_history_tensor("feature/temp")
            humidity = db.get_history_tensor("feature/humidity")
            pressure = db.get_history_tensor("feature/pressure")
            labels = db.get_history_tensor("label/anomaly")
            
            # Stack into feature matrix
            X = np.column_stack([temp, humidity, pressure])
            y = labels
            
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"Feature means: {X.mean(axis=0)}")
            print(f"Label distribution: {y.mean():.2%} positive")
            
            # Normalize features
            print("\nStep 3: Normalize Features")
            print("-" * 40)
            
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_normalized = (X - X_mean) / X_std
            
            print(f"Normalized means: {X_normalized.mean(axis=0)}")
            print(f"Normalized stds: {X_normalized.std(axis=0)}")
            
            # Store normalization parameters
            db.put_numpy("model/X_mean", X_mean)
            db.put_numpy("model/X_std", X_std)
            print("✓ Stored normalization parameters")
            
            # Train/test split
            print("\nStep 4: Train/Test Split")
            print("-" * 40)
            
            split_idx = int(0.8 * len(X))
            X_train, X_test = X_normalized[:split_idx], X_normalized[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"Train: {len(X_train)} samples")
            print(f"Test: {len(X_test)} samples")
            
            # Store splits
            db.put_numpy("data/X_train", X_train)
            db.put_numpy("data/X_test", X_test)
            db.put_numpy("data/y_train", y_train)
            db.put_numpy("data/y_test", y_test)
            print("✓ Stored train/test splits")
            
            # Verify retrieval
            print("\nStep 5: Verify Data Retrieval")
            print("-" * 40)
            
            X_train_loaded = db.get_numpy("data/X_train", shape=X_train.shape)
            print(f"✓ X_train loaded: {X_train_loaded.shape}")
            print(f"✓ Data matches: {np.allclose(X_train, X_train_loaded)}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   Syna NUMPY INTEGRATION DEMO")
    print("   Requirements: 2.2")
    print("=" * 60 + "\n")
    
    try:
        demo_store_numpy_arrays()
        demo_history_tensor_extraction()
        demo_zero_copy_access()
        demo_memory_efficiency()
        demo_ml_workflow()
        
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


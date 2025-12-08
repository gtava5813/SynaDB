#!/usr/bin/env python3
"""
MNIST Dataset Loader Demo

This demo shows how to:
- Download MNIST from HuggingFace datasets
- Store images and labels in Syna
- Retrieve data for ML training

Run with: python mnist_loader.py
"""

import os
import sys
import time
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


def main():
    print("=== MNIST Dataset Loader Demo ===\n")
    
    # Check if datasets library is available
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return 1
    
    db_path = "mnist_Syna.db"
    
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 1. Load MNIST from HuggingFace
        print("1. Loading MNIST from HuggingFace...")
        start = time.time()
        dataset = load_dataset("mnist", split="train[:1000]")  # First 1000 for demo
        load_time = time.time() - start
        print(f"   ✓ Loaded {len(dataset)} samples in {load_time:.2f}s\n")
        
        # 2. Store in Syna
        print("2. Storing in Syna database...")
        start = time.time()
        
        with SynaDB(db_path) as db:
            for i, sample in enumerate(dataset):
                # Get image as numpy array and flatten
                image = np.array(sample['image']).flatten().astype(np.uint8)
                label = sample['label']
                
                # Store image as bytes
                db.put_bytes(f"train/image/{i}", image.tobytes())
                
                # Store label as int
                db.put_int(f"train/label/{i}", label)
                
                if (i + 1) % 200 == 0:
                    print(f"   Stored {i + 1}/{len(dataset)} samples...")
        
        store_time = time.time() - start
        print(f"   ✓ Stored {len(dataset)} samples in {store_time:.2f}s\n")
        
        # 3. Check storage size
        file_size = os.path.getsize(db_path)
        print("3. Storage statistics:")
        print(f"   Database size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Bytes per sample: {file_size / len(dataset):.0f}")
        print(f"   Raw image size: {28 * 28} bytes (784 pixels)\n")
        
        # 4. Retrieve and verify data
        print("4. Retrieving and verifying data...")
        
        with SynaDB(db_path) as db:
            # Get a random sample
            idx = 42
            
            # Note: We need get_bytes which isn't implemented yet
            # For now, verify labels work
            label = db.get_int(f"train/label/{idx}")
            print(f"   Sample {idx} label: {label}")
            
            # Count keys
            keys = db.keys()
            image_keys = [k for k in keys if k.startswith("train/image/")]
            label_keys = [k for k in keys if k.startswith("train/label/")]
            
            print(f"   Image keys: {len(image_keys)}")
            print(f"   Label keys: {len(label_keys)}\n")
        
        # 5. Benchmark retrieval
        print("5. Benchmarking retrieval...")
        
        with SynaDB(db_path) as db:
            start = time.time()
            for i in range(100):
                db.get_int(f"train/label/{i}")
            read_time = time.time() - start
            
            print(f"   100 label reads: {read_time * 1000:.2f}ms")
            print(f"   Throughput: {100 / read_time:.0f} reads/sec\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    print("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())


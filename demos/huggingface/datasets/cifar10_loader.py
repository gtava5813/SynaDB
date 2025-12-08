#!/usr/bin/env python3
"""
CIFAR-10 Dataset Loader Demo

This demo shows how to:
- Download CIFAR-10 from HuggingFace datasets
- Store images (32x32x3) and labels in Syna
- Demonstrate batch retrieval
- Compare storage vs raw files

Run with: python cifar10_loader.py
"""

import os
import sys
import time
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    print("=== CIFAR-10 Dataset Loader Demo ===\n")
    
    # Check if datasets library is available
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return 1
    
    db_path = "cifar10_Syna.db"
    
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    try:
        # 1. Load CIFAR-10 from HuggingFace
        print("1. Loading CIFAR-10 from HuggingFace...")
        start = time.time()
        dataset = load_dataset("cifar10", split="train[:1000]")  # First 1000 for demo
        load_time = time.time() - start
        print(f"   ✓ Loaded {len(dataset)} samples in {load_time:.2f}s\n")
        
        # 2. Store in Syna
        print("2. Storing in Syna database...")
        start = time.time()
        
        raw_size = 0
        with SynaDB(db_path) as db:
            for i, sample in enumerate(dataset):
                # Get image as numpy array (32x32x3)
                image = np.array(sample['img']).astype(np.uint8)
                label = sample['label']
                
                # Store image as bytes (flattened)
                image_bytes = image.tobytes()
                raw_size += len(image_bytes)
                db.put_bytes(f"train/image/{i}", image_bytes)
                
                # Store label as int
                db.put_int(f"train/label/{i}", label)
                
                if (i + 1) % 200 == 0:
                    print(f"   Stored {i + 1}/{len(dataset)} samples...")
        
        store_time = time.time() - start
        print(f"   ✓ Stored {len(dataset)} samples in {store_time:.2f}s\n")

        # 3. Check storage size and compare
        file_size = os.path.getsize(db_path)
        print("3. Storage statistics:")
        print(f"   Database size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Raw image data: {raw_size / 1024 / 1024:.2f} MB")
        print(f"   Bytes per sample: {file_size / len(dataset):.0f}")
        print(f"   Raw image size: {32 * 32 * 3} bytes (3072 pixels)")
        print(f"   Storage overhead: {(file_size - raw_size) / raw_size * 100:.1f}%\n")
        
        # 4. Demonstrate batch retrieval
        print("4. Batch retrieval demo...")
        
        with SynaDB(db_path) as db:
            batch_size = 32
            batch_indices = list(range(batch_size))
            
            start = time.time()
            images = []
            labels = []
            
            for idx in batch_indices:
                # Retrieve image bytes and reshape
                img_bytes = db.get_bytes(f"train/image/{idx}")
                if img_bytes:
                    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(32, 32, 3)
                    images.append(img)
                
                # Retrieve label
                label = db.get_int(f"train/label/{idx}")
                labels.append(label)
            
            batch_time = time.time() - start
            
            # Stack into batch tensor
            batch_images = np.stack(images)
            batch_labels = np.array(labels)
            
            print(f"   Batch shape: {batch_images.shape}")
            print(f"   Labels shape: {batch_labels.shape}")
            print(f"   Batch load time: {batch_time * 1000:.2f}ms")
            print(f"   Throughput: {batch_size / batch_time:.0f} samples/sec\n")
            
            # Show sample labels
            print("   Sample labels from batch:")
            for i in range(min(5, len(labels))):
                print(f"     [{i}] {CIFAR10_CLASSES[labels[i]]}")
            print()
        
        # 5. Verify data integrity
        print("5. Verifying data integrity...")
        
        with SynaDB(db_path) as db:
            # Check a few random samples
            test_indices = [0, 42, 100, 500, 999]
            all_match = True
            
            for idx in test_indices:
                # Get stored data
                stored_bytes = db.get_bytes(f"train/image/{idx}")
                stored_label = db.get_int(f"train/label/{idx}")
                
                # Get original data
                original_img = np.array(dataset[idx]['img']).astype(np.uint8)
                original_label = dataset[idx]['label']
                
                # Compare
                stored_img = np.frombuffer(stored_bytes, dtype=np.uint8).reshape(32, 32, 3)
                
                if not np.array_equal(stored_img, original_img):
                    print(f"   ✗ Image mismatch at index {idx}")
                    all_match = False
                if stored_label != original_label:
                    print(f"   ✗ Label mismatch at index {idx}")
                    all_match = False
            
            if all_match:
                print(f"   ✓ All {len(test_indices)} samples verified correctly\n")
        
        # 6. Benchmark sequential vs random access
        print("6. Access pattern benchmarks...")
        
        with SynaDB(db_path) as db:
            # Sequential access
            start = time.time()
            for i in range(100):
                db.get_bytes(f"train/image/{i}")
            seq_time = time.time() - start
            
            # Random access
            np.random.seed(42)
            random_indices = np.random.randint(0, len(dataset), 100)
            start = time.time()
            for idx in random_indices:
                db.get_bytes(f"train/image/{idx}")
            rand_time = time.time() - start
            
            print(f"   Sequential (100 reads): {seq_time * 1000:.2f}ms")
            print(f"   Random (100 reads): {rand_time * 1000:.2f}ms")
            print(f"   Sequential throughput: {100 / seq_time:.0f} reads/sec")
            print(f"   Random throughput: {100 / rand_time:.0f} reads/sec\n")
        
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


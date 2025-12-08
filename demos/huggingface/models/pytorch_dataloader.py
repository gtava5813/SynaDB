#!/usr/bin/env python3
"""
PyTorch DataLoader Integration Demo

This demo shows how to:
- Create a PyTorch Dataset backed by Syna
- Use DataLoader for batched training
- Compare performance vs file-based loading

Requirements: 5.1 - WHEN running the PyTorch DataLoader demo THEN the demo 
SHALL implement a custom Dataset class backed by Syna

Run with: python pytorch_dataloader.py
"""

import os
import sys
import time
import tempfile
import numpy as np
from typing import Optional, Tuple, Callable, Any

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB

# Check for PyTorch availability
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy classes for type hints
    class Dataset:
        pass


class SynaDataset(Dataset):
    """
    PyTorch Dataset backed by Syna database.
    
    This class implements the full PyTorch Dataset interface, allowing
    Syna to be used as a data source for training loops with DataLoader.
    
    Features:
    - Supports __len__ and __getitem__ for PyTorch compatibility
    - Optional transforms for data augmentation
    - Efficient batch loading from Syna storage
    - Automatic type conversion to PyTorch tensors
    
    Example:
        >>> dataset = SynaDataset("mnist.db", key_prefix="train")
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for images, labels in dataloader:
        ...     # Training loop
        ...     pass
    """
    
    def __init__(
        self, 
        db_path: str, 
        key_prefix: str = "train",
        image_shape: Tuple[int, ...] = (1, 28, 28),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize the Syna-backed dataset.
        
        Args:
            db_path: Path to the Syna database file
            key_prefix: Prefix for data keys (e.g., "train" or "test")
            image_shape: Shape to reshape image data into (C, H, W)
            transform: Optional transform to apply to image data
            target_transform: Optional transform to apply to labels
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SynaDataset")
        
        self.db_path = db_path
        self.key_prefix = key_prefix
        self.image_shape = image_shape
        self.transform = transform
        self.target_transform = target_transform
        
        # Open database connection
        self._db = SynaDB(db_path)
        
        # Build index of available samples
        self._build_index()
    
    def _build_index(self):
        """Build an index of available samples in the database."""
        all_keys = self._db.keys()
        
        # Find all label keys to determine available indices
        label_prefix = f"{self.key_prefix}/label/"
        self._indices = []
        
        for key in all_keys:
            if key.startswith(label_prefix):
                try:
                    idx = int(key[len(label_prefix):])
                    self._indices.append(idx)
                except ValueError:
                    continue
        
        # Sort indices for consistent ordering
        self._indices.sort()
        self._length = len(self._indices)
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample (0 to len-1)
            
        Returns:
            Tuple of (image_tensor, label)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")
        
        # Map to actual database index
        actual_idx = self._indices[idx]
        
        # Load image data
        image_bytes = self._db.get_bytes(f"{self.key_prefix}/image/{actual_idx}")
        if image_bytes is not None:
            # Convert bytes to numpy array and reshape
            image = np.frombuffer(image_bytes, dtype=np.uint8).astype(np.float32)
            image = image.reshape(self.image_shape) / 255.0  # Normalize to [0, 1]
        else:
            # Return zeros if image not found
            image = np.zeros(self.image_shape, dtype=np.float32)
        
        # Load label
        label = self._db.get_int(f"{self.key_prefix}/label/{actual_idx}")
        if label is None:
            label = 0
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, '_db') and self._db is not None:
            self._db.close()
            self._db = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class FileBasedDataset(Dataset):
    """
    Simple file-based dataset for performance comparison.
    
    Stores each sample as a separate .npy file.
    """
    
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Count available samples
        self._length = 0
        while os.path.exists(os.path.join(data_dir, f"image_{self._length}.npy")):
            self._length += 1
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image = np.load(os.path.join(self.data_dir, f"image_{idx}.npy"))
        label = int(np.load(os.path.join(self.data_dir, f"label_{idx}.npy")))
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)
        
        return image, label


def create_sample_database(db_path: str, num_samples: int = 1000) -> float:
    """
    Create a sample database with MNIST-like data.
    
    Returns:
        Time taken to create the database
    """
    print(f"   Creating Syna database with {num_samples} samples...")
    start = time.time()
    
    with SynaDB(db_path) as db:
        for i in range(num_samples):
            # Create random MNIST-like image (28x28 grayscale)
            image = np.random.randint(0, 256, size=(28 * 28,), dtype=np.uint8)
            label = i % 10  # Labels 0-9
            
            # Store in Syna
            db.put_bytes(f"train/image/{i}", image.tobytes())
            db.put_int(f"train/label/{i}", label)
    
    elapsed = time.time() - start
    print(f"   ✓ Created in {elapsed:.2f}s ({num_samples / elapsed:.0f} samples/sec)")
    return elapsed


def create_file_based_dataset(data_dir: str, num_samples: int = 1000) -> float:
    """
    Create a file-based dataset for comparison.
    
    Returns:
        Time taken to create the dataset
    """
    print(f"   Creating file-based dataset with {num_samples} samples...")
    os.makedirs(data_dir, exist_ok=True)
    start = time.time()
    
    for i in range(num_samples):
        # Create random MNIST-like image
        image = np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8).astype(np.float32) / 255.0
        label = np.array(i % 10)
        
        # Save as numpy files
        np.save(os.path.join(data_dir, f"image_{i}.npy"), image)
        np.save(os.path.join(data_dir, f"label_{i}.npy"), label)
    
    elapsed = time.time() - start
    print(f"   ✓ Created in {elapsed:.2f}s ({num_samples / elapsed:.0f} samples/sec)")
    return elapsed


def benchmark_dataset(dataset: Dataset, name: str, num_iterations: int = 100) -> dict:
    """
    Benchmark a dataset's loading performance.
    
    Returns:
        Dictionary with benchmark results
    """
    # Sequential access benchmark
    start = time.time()
    for i in range(min(num_iterations, len(dataset))):
        _ = dataset[i]
    seq_time = time.time() - start
    
    # Random access benchmark
    indices = np.random.randint(0, len(dataset), size=num_iterations)
    start = time.time()
    for idx in indices:
        _ = dataset[idx]
    rand_time = time.time() - start
    
    return {
        'name': name,
        'sequential_time_ms': seq_time * 1000,
        'random_time_ms': rand_time * 1000,
        'sequential_throughput': num_iterations / seq_time,
        'random_throughput': num_iterations / rand_time,
    }


def benchmark_dataloader(dataset: Dataset, name: str, batch_size: int = 32, num_batches: int = 10) -> dict:
    """
    Benchmark DataLoader performance.
    
    Returns:
        Dictionary with benchmark results
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded for fair comparison
        pin_memory=False
    )
    
    start = time.time()
    total_samples = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        total_samples += len(labels)
        if batch_idx >= num_batches - 1:
            break
    elapsed = time.time() - start
    
    return {
        'name': name,
        'batch_size': batch_size,
        'num_batches': min(num_batches, len(dataloader)),
        'total_samples': total_samples,
        'time_ms': elapsed * 1000,
        'throughput': total_samples / elapsed,
    }


def main():
    print("=" * 60)
    print("PyTorch DataLoader Integration Demo")
    print("Requirement 5.1: Custom Dataset class backed by Syna")
    print("=" * 60 + "\n")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        print("\nRunning in limited mode without DataLoader tests...\n")
    
    # Configuration
    NUM_SAMPLES = 500
    BATCH_SIZE = 32
    
    # Create temporary directories - use absolute paths
    db_path = os.path.abspath("pytorch_dataloader_demo.db")
    file_dir = tempfile.mkdtemp(prefix="syna_file_dataset_")
    
    try:
        # 1. Create datasets
        print("1. Creating test datasets")
        print("-" * 40)
        syna_time = create_sample_database(db_path, NUM_SAMPLES)
        
        if HAS_TORCH:
            file_time = create_file_based_dataset(file_dir, NUM_SAMPLES)
            print(f"\n   Write speedup: {file_time / syna_time:.2f}x faster with Syna\n")
        print()
        
        # 2. Test SynaDataset interface
        print("2. Testing SynaDataset interface")
        print("-" * 40)
        
        if HAS_TORCH:
            dataset = SynaDataset(
                db_path, 
                key_prefix="train",
                image_shape=(1, 28, 28)
            )
            
            print(f"   Dataset length: {len(dataset)}")
            
            # Test __getitem__
            image, label = dataset[0]
            print(f"   Sample 0: shape={tuple(image.shape)}, dtype={image.dtype}, label={label}")
            
            image, label = dataset[42]
            print(f"   Sample 42: shape={tuple(image.shape)}, dtype={image.dtype}, label={label}")
            
            # Verify tensor type
            print(f"   Is torch.Tensor: {isinstance(image, torch.Tensor)}")
            print()
        else:
            print("   Skipped (PyTorch not available)\n")
        
        # 3. Benchmark single-sample access
        print("3. Benchmarking single-sample access")
        print("-" * 40)
        
        if HAS_TORCH:
            syna_results = benchmark_dataset(dataset, "Syna", num_iterations=100)
            print(f"   Syna sequential: {syna_results['sequential_time_ms']:.2f}ms "
                  f"({syna_results['sequential_throughput']:.0f} samples/sec)")
            print(f"   Syna random:     {syna_results['random_time_ms']:.2f}ms "
                  f"({syna_results['random_throughput']:.0f} samples/sec)")
            
            file_dataset = FileBasedDataset(file_dir)
            file_results = benchmark_dataset(file_dataset, "File-based", num_iterations=100)
            print(f"   File-based sequential: {file_results['sequential_time_ms']:.2f}ms "
                  f"({file_results['sequential_throughput']:.0f} samples/sec)")
            print(f"   File-based random:     {file_results['random_time_ms']:.2f}ms "
                  f"({file_results['random_throughput']:.0f} samples/sec)")
            
            # Calculate speedup
            seq_speedup = file_results['sequential_time_ms'] / syna_results['sequential_time_ms']
            rand_speedup = file_results['random_time_ms'] / syna_results['random_time_ms']
            print(f"\n   Speedup: {seq_speedup:.2f}x (sequential), {rand_speedup:.2f}x (random)")
            print()
        else:
            print("   Skipped (PyTorch not available)\n")
        
        # 4. Benchmark DataLoader batching
        print("4. Benchmarking DataLoader batching")
        print("-" * 40)
        
        if HAS_TORCH:
            syna_dl_results = benchmark_dataloader(
                dataset, "Syna", batch_size=BATCH_SIZE, num_batches=10
            )
            print(f"   Syna DataLoader:")
            print(f"      Batch size: {syna_dl_results['batch_size']}")
            print(f"      Batches: {syna_dl_results['num_batches']}")
            print(f"      Total samples: {syna_dl_results['total_samples']}")
            print(f"      Time: {syna_dl_results['time_ms']:.2f}ms")
            print(f"      Throughput: {syna_dl_results['throughput']:.0f} samples/sec")
            
            file_dl_results = benchmark_dataloader(
                file_dataset, "File-based", batch_size=BATCH_SIZE, num_batches=10
            )
            print(f"\n   File-based DataLoader:")
            print(f"      Batch size: {file_dl_results['batch_size']}")
            print(f"      Batches: {file_dl_results['num_batches']}")
            print(f"      Total samples: {file_dl_results['total_samples']}")
            print(f"      Time: {file_dl_results['time_ms']:.2f}ms")
            print(f"      Throughput: {file_dl_results['throughput']:.0f} samples/sec")
            
            dl_speedup = file_dl_results['time_ms'] / syna_dl_results['time_ms']
            print(f"\n   DataLoader speedup: {dl_speedup:.2f}x with Syna")
            print()
        else:
            print("   Skipped (PyTorch not available)\n")
        
        # 5. Demonstrate training loop usage
        print("5. Demonstrating training loop usage")
        print("-" * 40)
        
        if HAS_TORCH:
            print("   Example training loop code:")
            print("""
   # Create dataset and dataloader
   dataset = SynaDataset("mnist.db", key_prefix="train")
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
   # Training loop
   model = SimpleCNN()
   optimizer = torch.optim.Adam(model.parameters())
   criterion = torch.nn.CrossEntropyLoss()
   
   for epoch in range(num_epochs):
       for images, labels in dataloader:
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
""")
            
            # Actually iterate through a few batches
            print("   Running actual iteration test...")
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            batch_count = 0
            sample_count = 0
            for images, labels in dataloader:
                batch_count += 1
                sample_count += len(labels)
                if batch_count >= 5:
                    break
            print(f"   ✓ Successfully iterated {batch_count} batches ({sample_count} samples)")
            print()
        else:
            print("   Skipped (PyTorch not available)\n")
        
        # Cleanup
        if HAS_TORCH:
            dataset.close()
        
        print("=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup temporary files
        if os.path.exists(db_path):
            os.remove(db_path)
        
        import shutil
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


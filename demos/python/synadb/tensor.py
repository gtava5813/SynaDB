"""
Tensor engine for batch operations.

This module provides a high-level Python interface for batch tensor operations
using the Syna database. It enables efficient storage and retrieval of tensors
for ML data loading workflows.

Example:
    >>> from synadb import TensorEngine
    >>> engine = TensorEngine("data.db")
    >>> # Store training data
    >>> engine.put_tensor("train/", X_train)
    >>> # Load as tensor
    >>> X = engine.get_tensor("train/*", dtype=np.float32)
"""

import fnmatch
import numpy as np
from typing import Tuple, Optional, Iterator, List

from .wrapper import SynaDB, SynaError


class TensorEngine:
    """
    Batch tensor operations for ML data loading.
    
    Provides efficient storage and retrieval of tensors for training data,
    with support for pattern-based key matching and automatic type conversion.
    
    Example:
        >>> engine = TensorEngine("data.db")
        >>> # Store training data
        >>> engine.put_tensor("train/", X_train)
        >>> # Load as tensor
        >>> X = engine.get_tensor("train/*", dtype=np.float32)
        >>> # Stream in batches
        >>> for batch in engine.stream("train/*", batch_size=32):
        ...     model.train_step(batch)
    
    Attributes:
        path: Path to the database file.
    """
    
    def __init__(self, path: str):
        """
        Open database for tensor operations.
        
        Args:
            path: Path to the database file. Will be created if it doesn't exist.
        
        Raises:
            SynaError: If the database cannot be opened.
        """
        self._db = SynaDB(path)
        self._path = path
    
    def get_tensor(
        self,
        pattern: str,
        shape: Tuple[int, ...] = None,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Load matching keys as a single tensor.
        
        Retrieves all values matching the given key pattern and combines them
        into a single contiguous numpy array. Keys are sorted alphabetically
        before concatenation to ensure consistent ordering.
        
        Args:
            pattern: Glob pattern for keys (e.g., "train/*", "data/batch_*").
                     Supports standard glob wildcards: * matches any characters,
                     ? matches single character.
            shape: Optional shape to reshape result. If None, returns a 1D array.
                   The total number of elements must match the data size.
            dtype: NumPy dtype for output (default: np.float32).
                   Supported types: np.float32, np.float64, np.int32, np.int64.
        
        Returns:
            NumPy array with loaded data. If shape is provided, the array is
            reshaped accordingly. Otherwise, returns a 1D array.
        
        Raises:
            SynaError: If the database operation fails.
            ValueError: If shape doesn't match the data size.
        
        Example:
            >>> engine = TensorEngine("data.db")
            >>> # Load all training samples
            >>> X = engine.get_tensor("train/*")
            >>> # Load with specific shape
            >>> X = engine.get_tensor("images/*", shape=(100, 28, 28))
        
        _Requirements: 2.5, 8.2_
        """
        # Get all keys matching the pattern
        all_keys = self._db.keys()
        matching_keys = sorted([k for k in all_keys if fnmatch.fnmatch(k, pattern)])
        
        if not matching_keys:
            # Return empty array with correct dtype
            result = np.array([], dtype=dtype)
            if shape is not None:
                # Validate shape allows empty array
                if np.prod(shape) != 0:
                    raise ValueError(f"Cannot reshape empty array to shape {shape}")
                result = result.reshape(shape)
            return result
        
        # Collect all values
        values: List[np.ndarray] = []
        
        for key in matching_keys:
            # Try to get as float history first (most common for ML)
            history = self._db.get_history_tensor(key)
            if len(history) > 0:
                values.append(history)
            else:
                # Try to get as bytes (for stored numpy arrays)
                data = self._db.get_bytes(key)
                if data is not None:
                    arr = np.frombuffer(data, dtype=dtype)
                    values.append(arr)
                else:
                    # Try to get as single float
                    val = self._db.get_float(key)
                    if val is not None:
                        values.append(np.array([val], dtype=np.float64))
                    else:
                        # Try to get as single int
                        val = self._db.get_int(key)
                        if val is not None:
                            values.append(np.array([val], dtype=np.int64))
        
        if not values:
            result = np.array([], dtype=dtype)
            if shape is not None:
                if np.prod(shape) != 0:
                    raise ValueError(f"Cannot reshape empty array to shape {shape}")
                result = result.reshape(shape)
            return result
        
        # Concatenate all values
        result = np.concatenate(values)
        
        # Convert to requested dtype
        result = result.astype(dtype)
        
        # Reshape if requested
        if shape is not None:
            expected_size = np.prod(shape)
            if result.size != expected_size:
                raise ValueError(
                    f"Cannot reshape array of size {result.size} to shape {shape} "
                    f"(expected {expected_size} elements)"
                )
            result = result.reshape(shape)
        
        return result
    
    def put_tensor(
        self,
        key_prefix: str,
        tensor: np.ndarray,
    ) -> int:
        """
        Store tensor with auto-generated keys.
        
        Flattens the tensor and stores each element with an auto-generated key
        based on the provided prefix. Keys are generated as "{prefix}{index}"
        where index is zero-padded for consistent sorting.
        
        Args:
            key_prefix: Prefix for generated keys (e.g., "train/", "data/batch_").
                        Each element will be stored with key "{prefix}{index}".
            tensor: NumPy array to store. Will be flattened before storage.
        
        Returns:
            Number of entries written.
        
        Raises:
            SynaError: If the database operation fails.
        
        Example:
            >>> engine = TensorEngine("data.db")
            >>> X = np.random.randn(100, 784)  # 100 samples, 784 features
            >>> count = engine.put_tensor("train/", X)
            >>> print(f"Stored {count} values")
        
        _Requirements: 2.5, 8.2_
        """
        # Flatten the tensor
        flat = tensor.flatten()
        count = 0
        
        # Calculate padding width for consistent key sorting
        num_digits = len(str(len(flat) - 1)) if len(flat) > 0 else 1
        
        for i, value in enumerate(flat):
            # Generate zero-padded key for consistent sorting
            key = f"{key_prefix}{i:0{num_digits}d}"
            
            # Store based on dtype
            if np.issubdtype(tensor.dtype, np.floating):
                self._db.put_float(key, float(value))
            elif np.issubdtype(tensor.dtype, np.integer):
                self._db.put_int(key, int(value))
            else:
                # Default to float
                self._db.put_float(key, float(value))
            
            count += 1
        
        return count
    
    def put_tensor_chunked(
        self,
        key_prefix: str,
        tensor: np.ndarray,
        chunk_size: int = 1000,
    ) -> int:
        """
        Store tensor in chunks for more efficient storage.
        
        Instead of storing each element individually, stores the tensor
        in binary chunks. This is more efficient for large tensors.
        
        Args:
            key_prefix: Prefix for generated keys.
            tensor: NumPy array to store.
            chunk_size: Number of elements per chunk (default: 1000).
        
        Returns:
            Number of chunks written.
        
        Example:
            >>> engine = TensorEngine("data.db")
            >>> X = np.random.randn(10000, 784)
            >>> chunks = engine.put_tensor_chunked("train/", X, chunk_size=10000)
        """
        flat = tensor.flatten()
        num_chunks = (len(flat) + chunk_size - 1) // chunk_size
        num_digits = len(str(num_chunks - 1)) if num_chunks > 0 else 1
        
        count = 0
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]
            key = f"{key_prefix}chunk_{count:0{num_digits}d}"
            self._db.put_bytes(key, chunk.tobytes())
            count += 1
        
        return count
    
    def get_tensor_chunked(
        self,
        pattern: str,
        dtype=np.float32,
        shape: Tuple[int, ...] = None,
    ) -> np.ndarray:
        """
        Load tensor stored in chunks.
        
        Args:
            pattern: Glob pattern for chunk keys (e.g., "train/chunk_*").
            dtype: NumPy dtype for output.
            shape: Optional shape to reshape result.
        
        Returns:
            NumPy array with loaded data.
        """
        all_keys = self._db.keys()
        matching_keys = sorted([k for k in all_keys if fnmatch.fnmatch(k, pattern)])
        
        if not matching_keys:
            result = np.array([], dtype=dtype)
            if shape is not None:
                result = result.reshape(shape)
            return result
        
        chunks = []
        for key in matching_keys:
            data = self._db.get_bytes(key)
            if data is not None:
                chunk = np.frombuffer(data, dtype=dtype)
                chunks.append(chunk)
        
        if not chunks:
            result = np.array([], dtype=dtype)
        else:
            result = np.concatenate(chunks)
        
        if shape is not None:
            result = result.reshape(shape)
        
        return result
    
    def get_tensor_torch(
        self,
        pattern: str,
        shape: Tuple[int, ...] = None,
        device: str = "cpu",
    ) -> "torch.Tensor":
        """
        Load directly as PyTorch tensor.
        
        Convenience method that loads data and converts it to a PyTorch tensor
        in a single call. Requires PyTorch to be installed.
        
        Args:
            pattern: Glob pattern for keys (e.g., "train/*").
            shape: Optional shape to reshape result.
            device: PyTorch device to place tensor on (default: "cpu").
                    Can be "cpu", "cuda", "cuda:0", etc.
        
        Returns:
            PyTorch tensor with loaded data.
        
        Raises:
            ImportError: If PyTorch is not installed.
            SynaError: If the database operation fails.
        
        Example:
            >>> engine = TensorEngine("data.db")
            >>> X = engine.get_tensor_torch("train/*", device="cuda")
            >>> model(X)
        
        _Requirements: 2.5, 8.2_
        """
        import torch
        np_array = self.get_tensor(pattern, shape, dtype=np.float32)
        return torch.from_numpy(np_array).to(device)
    
    def get_tensor_tf(
        self,
        pattern: str,
        shape: Tuple[int, ...] = None,
    ) -> "tf.Tensor":
        """
        Load directly as TensorFlow tensor.
        
        Convenience method that loads data and converts it to a TensorFlow tensor
        in a single call. Requires TensorFlow to be installed.
        
        Args:
            pattern: Glob pattern for keys (e.g., "train/*").
            shape: Optional shape to reshape result.
        
        Returns:
            TensorFlow tensor with loaded data.
        
        Raises:
            ImportError: If TensorFlow is not installed.
            SynaError: If the database operation fails.
        """
        import tensorflow as tf
        np_array = self.get_tensor(pattern, shape, dtype=np.float32)
        return tf.convert_to_tensor(np_array)
    
    def stream(
        self,
        pattern: str,
        batch_size: int = 32,
        dtype=np.float32,
    ) -> Iterator[np.ndarray]:
        """
        Stream data as batches.
        
        Generator that yields batches of data matching the pattern.
        Useful for training loops where you want to process data in chunks.
        
        Args:
            pattern: Glob pattern for keys (e.g., "train/*").
            batch_size: Number of elements per batch (default: 32).
            dtype: NumPy dtype for output.
        
        Yields:
            NumPy arrays of size batch_size (last batch may be smaller).
        
        Example:
            >>> engine = TensorEngine("data.db")
            >>> for batch in engine.stream("train/*", batch_size=64):
            ...     loss = model.train_step(batch)
        """
        # Get all matching keys
        all_keys = self._db.keys()
        matching_keys = sorted([k for k in all_keys if fnmatch.fnmatch(k, pattern)])
        
        if not matching_keys:
            return
        
        # Collect values in batches
        batch = []
        
        for key in matching_keys:
            # Get value for this key
            history = self._db.get_history_tensor(key)
            if len(history) > 0:
                for val in history:
                    batch.append(val)
                    if len(batch) >= batch_size:
                        yield np.array(batch, dtype=dtype)
                        batch = []
            else:
                val = self._db.get_float(key)
                if val is not None:
                    batch.append(val)
                    if len(batch) >= batch_size:
                        yield np.array(batch, dtype=dtype)
                        batch = []
        
        # Yield remaining elements
        if batch:
            yield np.array(batch, dtype=dtype)
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        List keys matching a pattern.
        
        Args:
            pattern: Glob pattern for keys (default: "*" matches all).
        
        Returns:
            List of matching keys, sorted alphabetically.
        """
        all_keys = self._db.keys()
        return sorted([k for k in all_keys if fnmatch.fnmatch(k, pattern)])
    
    def delete(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Glob pattern for keys to delete.
        
        Returns:
            Number of keys deleted.
        """
        matching_keys = self.keys(pattern)
        count = 0
        for key in matching_keys:
            try:
                self._db.delete(key)
                count += 1
            except SynaError:
                pass
        return count
    
    def close(self) -> None:
        """Close the database."""
        self._db.close()
    
    def __enter__(self) -> 'TensorEngine':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass
    
    @property
    def path(self) -> str:
        """Return the database path."""
        return self._path

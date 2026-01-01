"""
TensorFlow integration for SynaDB.

Provides tf.data.Dataset implementations backed by SynaDB
for efficient ML training data loading.

Example:
    >>> from synadb.tensorflow import syna_dataset
    >>> 
    >>> dataset = syna_dataset(
    ...     path="data.db",
    ...     pattern="train/*",
    ...     batch_size=32
    ... ).prefetch(tf.data.AUTOTUNE)
    >>> 
    >>> for batch in dataset:
    ...     # train with batch

Requirements: 13.3
"""

from typing import Callable, List, Optional, Tuple, Any
import fnmatch
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _match_keys(db, pattern: str) -> List[str]:
    """
    Match database keys against a glob pattern.
    
    Args:
        db: SynaDB instance.
        pattern: Glob pattern to match. Supports:
                 - "*" matches all keys
                 - "prefix/*" matches keys starting with "prefix/"
                 - Standard glob patterns via fnmatch
    
    Returns:
        Sorted list of matching keys.
    """
    keys = db.keys()
    
    if pattern == "*":
        return sorted(keys)
    
    # Handle "prefix/*" pattern efficiently
    if pattern.endswith("/*"):
        prefix = pattern[:-1]  # Keep the trailing slash
        return sorted([k for k in keys if k.startswith(prefix)])
    
    # Use fnmatch for general glob patterns
    return sorted([k for k in keys if fnmatch.fnmatch(k, pattern)])


def syna_dataset(
    path: str,
    pattern: str = "*",
    batch_size: int = 32,
    dtype: "tf.DType" = None,
) -> "tf.data.Dataset":
    """
    Create a tf.data.Dataset from SynaDB.
    
    This function creates a TensorFlow Dataset that reads data from a SynaDB
    database. It supports pattern-based key matching and automatic batching.
    
    Example:
        >>> dataset = syna_dataset(
        ...     path="data.db",
        ...     pattern="train/*",
        ...     batch_size=32
        ... ).prefetch(tf.data.AUTOTUNE)
        >>> 
        >>> for batch in dataset:
        ...     # train with batch
    
    Args:
        path: Path to the SynaDB database file.
        pattern: Glob pattern to match keys. Use "*" for all keys,
                 "prefix/*" for keys starting with prefix.
        batch_size: Number of samples per batch (default: 32).
        dtype: TensorFlow dtype for output tensors (default: tf.float32).
    
    Returns:
        A tf.data.Dataset that yields batched tensors.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        
    Note:
        The dataset uses a generator internally, which means it can handle
        databases larger than memory. For best performance, chain with
        .prefetch(tf.data.AUTOTUNE).
        
    Requirements: 13.3
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for syna_dataset. "
            "Install it with: pip install tensorflow"
        )
    
    # Set default dtype if not provided
    if dtype is None:
        dtype = tf.float32
    
    from .wrapper import SynaDB
    db = SynaDB(path)
    
    # Get matching keys
    keys = _match_keys(db, pattern)
    
    def generator():
        """Generator function that yields data from SynaDB."""
        for key in keys:
            # Try to get as tensor first (history of floats)
            data = db.get_history_tensor(key)
            
            if data is not None and len(data) > 0:
                yield data.astype(np.float32)
            else:
                # Fall back to single float value
                value = db.get_float(key)
                if value is not None:
                    yield np.array([value], dtype=np.float32)
                else:
                    # Try to get as bytes and interpret as numpy array
                    bytes_data = db.get_bytes(key)
                    if bytes_data is not None:
                        # Assume float32 array stored as bytes
                        arr = np.frombuffer(bytes_data, dtype=np.float32)
                        yield arr.copy()
                    else:
                        # Return empty array as fallback
                        yield np.array([0.0], dtype=np.float32)
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=dtype),
    )
    
    return dataset.batch(batch_size)


class SynaDataset:
    """
    TensorFlow Dataset wrapper for SynaDB.
    
    This class provides a more object-oriented interface for creating
    TensorFlow datasets from SynaDB, with additional configuration options.
    
    Example:
        >>> ds = SynaDataset("data.db", pattern="train/*")
        >>> tf_dataset = ds.to_tf_dataset(batch_size=32)
        >>> for batch in tf_dataset:
        ...     # train with batch
    
    Args:
        path: Path to the SynaDB database file.
        pattern: Glob pattern to match keys (default: "*" for all keys).
        
    Attributes:
        path: The database path.
        pattern: The key matching pattern.
        keys: List of matched keys.
    """
    
    def __init__(
        self,
        path: str,
        pattern: str = "*",
    ):
        """
        Initialize the SynaDataset.
        
        Args:
            path: Path to the SynaDB database file.
            pattern: Glob pattern to match keys (default: "*" for all keys).
            
        Raises:
            ImportError: If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for SynaDataset. "
                "Install it with: pip install tensorflow"
            )
        
        from .wrapper import SynaDB
        self._db = SynaDB(path)
        self._path = path
        self._pattern = pattern
        
        # Get matching keys
        self._keys = _match_keys(self._db, pattern)
    
    @property
    def path(self) -> str:
        """Return the database path."""
        return self._path
    
    @property
    def pattern(self) -> str:
        """Return the key matching pattern."""
        return self._pattern
    
    @property
    def keys(self) -> List[str]:
        """Return the list of matched keys."""
        return self._keys.copy()
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._keys)
    
    def to_tf_dataset(
        self,
        batch_size: int = 32,
        dtype: "tf.DType" = None,
        shuffle: bool = False,
        buffer_size: int = 1000,
    ) -> "tf.data.Dataset":
        """
        Convert to a tf.data.Dataset.
        
        Args:
            batch_size: Number of samples per batch (default: 32).
            dtype: TensorFlow dtype for output tensors (default: tf.float32).
            shuffle: Whether to shuffle the dataset (default: False).
            buffer_size: Buffer size for shuffling (default: 1000).
        
        Returns:
            A tf.data.Dataset that yields batched tensors.
        """
        if dtype is None:
            dtype = tf.float32
        
        keys = self._keys
        db = self._db
        
        def generator():
            """Generator function that yields data from SynaDB."""
            for key in keys:
                # Try to get as tensor first (history of floats)
                data = db.get_history_tensor(key)
                
                if data is not None and len(data) > 0:
                    yield data.astype(np.float32)
                else:
                    # Fall back to single float value
                    value = db.get_float(key)
                    if value is not None:
                        yield np.array([value], dtype=np.float32)
                    else:
                        # Try to get as bytes and interpret as numpy array
                        bytes_data = db.get_bytes(key)
                        if bytes_data is not None:
                            arr = np.frombuffer(bytes_data, dtype=np.float32)
                            yield arr.copy()
                        else:
                            yield np.array([0.0], dtype=np.float32)
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(None,), dtype=dtype),
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        
        return dataset.batch(batch_size)
    
    def close(self) -> None:
        """Close the underlying database connection."""
        if hasattr(self, '_db') and self._db is not None:
            self._db.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def create_distributed_dataset(
    path: str,
    pattern: str = "*",
    batch_size: int = 32,
    dtype: "tf.DType" = None,
) -> "tf.data.Dataset":
    """
    Create a tf.data.Dataset configured for distributed training.
    
    This function creates a dataset that works with tf.distribute strategies
    for multi-GPU or multi-worker training.
    
    Example:
        >>> strategy = tf.distribute.MirroredStrategy()
        >>> with strategy.scope():
        ...     dataset = create_distributed_dataset(
        ...         path="data.db",
        ...         pattern="train/*",
        ...         batch_size=32
        ...     )
        ...     dist_dataset = strategy.experimental_distribute_dataset(dataset)
        >>> for batch in dist_dataset:
        ...     # train with distributed batch
    
    Args:
        path: Path to the SynaDB database file.
        pattern: Glob pattern to match keys.
        batch_size: Global batch size (will be divided among replicas).
        dtype: TensorFlow dtype for output tensors (default: tf.float32).
    
    Returns:
        A tf.data.Dataset suitable for distribution.
        
    Raises:
        ImportError: If TensorFlow is not installed.
        
    Requirements: 13.4
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for create_distributed_dataset. "
            "Install it with: pip install tensorflow"
        )
    
    # Create base dataset
    dataset = syna_dataset(
        path=path,
        pattern=pattern,
        batch_size=batch_size,
        dtype=dtype,
    )
    
    # Add prefetching for better performance in distributed setting
    return dataset.prefetch(tf.data.AUTOTUNE)


# Convenience function to check if TensorFlow is available
def is_tensorflow_available() -> bool:
    """
    Check if TensorFlow is available.
    
    Returns:
        True if TensorFlow is installed and can be imported.
    """
    return TF_AVAILABLE


__all__ = [
    "syna_dataset",
    "SynaDataset",
    "create_distributed_dataset",
    "is_tensorflow_available",
    "TF_AVAILABLE",
]

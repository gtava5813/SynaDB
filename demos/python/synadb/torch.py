"""
PyTorch integration for SynaDB.

Provides PyTorch Dataset and DataLoader implementations backed by SynaDB
for efficient ML training data loading.

Example:
    >>> from synadb.torch import SynaDataset, SynaDataLoader
    >>> from torchvision import transforms
    >>> 
    >>> dataset = SynaDataset(
    ...     path="mnist.db",
    ...     pattern="train/*",
    ...     transform=transforms.ToTensor()
    ... )
    >>> loader = SynaDataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     # train
"""

from typing import Callable, List, Optional, Tuple, Any
import fnmatch
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder classes for type hints when torch is not available
    class Dataset:
        pass
    class DataLoader:
        pass
    class DistributedSampler:
        pass


class SynaDataset(Dataset):
    """
    PyTorch Dataset backed by SynaDB.
    
    This dataset loads data from a SynaDB database, supporting pattern-based
    key matching and optional transforms.
    
    Example:
        >>> dataset = SynaDataset(
        ...     path="mnist.db",
        ...     pattern="train/*",
        ...     transform=transforms.ToTensor()
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # train
    
    Args:
        path: Path to the SynaDB database file.
        pattern: Glob pattern to match keys. Use "*" for all keys,
                 "prefix/*" for keys starting with prefix.
        transform: Optional transform to apply to each sample.
        target_transform: Optional transform to apply to targets (if using
                          labeled data with separate target keys).
    
    Attributes:
        path: The database path.
        pattern: The key matching pattern.
        transform: The sample transform function.
        target_transform: The target transform function.
    """
    
    def __init__(
        self,
        path: str,
        pattern: str = "*",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the SynaDataset.
        
        Args:
            path: Path to the SynaDB database file.
            pattern: Glob pattern to match keys (default: "*" for all keys).
            transform: Optional callable to transform samples.
            target_transform: Optional callable to transform targets.
            
        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SynaDataset. "
                "Install it with: pip install torch"
            )
        
        from .wrapper import SynaDB
        self._db = SynaDB(path)
        self._path = path
        self._pattern = pattern
        self._transform = transform
        self._target_transform = target_transform
        
        # Get matching keys
        self._keys = self._match_keys(pattern)
    
    def _match_keys(self, pattern: str) -> List[str]:
        """
        Match database keys against a glob pattern.
        
        Args:
            pattern: Glob pattern to match. Supports:
                     - "*" matches all keys
                     - "prefix/*" matches keys starting with "prefix/"
                     - Standard glob patterns via fnmatch
        
        Returns:
            Sorted list of matching keys.
        """
        keys = self._db.keys()
        
        if pattern == "*":
            return sorted(keys)
        
        # Handle "prefix/*" pattern efficiently
        if pattern.endswith("/*"):
            prefix = pattern[:-1]  # Keep the trailing slash
            return sorted([k for k in keys if k.startswith(prefix)])
        
        # Use fnmatch for general glob patterns
        return sorted([k for k in keys if fnmatch.fnmatch(k, pattern)])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._keys)
    
    def __getitem__(self, idx: int) -> "torch.Tensor":
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            PyTorch tensor containing the sample data. If a transform
            is specified, the transformed tensor is returned.
        """
        if idx < 0 or idx >= len(self._keys):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._keys)}")
        
        key = self._keys[idx]
        
        # Try to get as tensor first (history of floats)
        data = self._db.get_history_tensor(key)
        
        if data is not None and len(data) > 0:
            tensor = torch.from_numpy(data.astype(np.float32))
        else:
            # Fall back to single float value
            value = self._db.get_float(key)
            if value is not None:
                tensor = torch.tensor([value], dtype=torch.float32)
            else:
                # Try to get as bytes and interpret as numpy array
                bytes_data = self._db.get_bytes(key)
                if bytes_data is not None:
                    # Assume float32 array stored as bytes
                    arr = np.frombuffer(bytes_data, dtype=np.float32)
                    tensor = torch.from_numpy(arr.copy())
                else:
                    # Return empty tensor as fallback
                    tensor = torch.tensor([0.0], dtype=torch.float32)
        
        if self._transform:
            tensor = self._transform(tensor)
        
        return tensor
    
    def get_key(self, idx: int) -> str:
        """
        Get the database key for a given index.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            The database key string.
        """
        return self._keys[idx]
    
    def close(self) -> None:
        """Close the underlying database connection."""
        if hasattr(self, '_db') and self._db is not None:
            self._db.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class SynaDataLoader(DataLoader):
    """
    DataLoader with SynaDB-specific optimizations.
    
    This DataLoader extends PyTorch's DataLoader with features optimized
    for SynaDB access patterns:
    
    - Prefetching for reduced I/O latency
    - Support for distributed training via DistributedSampler
    
    Example:
        >>> dataset = SynaDataset("data.db", pattern="train/*")
        >>> loader = SynaDataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     prefetch_factor=2
        ... )
        >>> for batch in loader:
        ...     # process batch
    
    Args:
        dataset: A SynaDataset instance.
        batch_size: Number of samples per batch (default: 1).
        shuffle: Whether to shuffle data at each epoch (default: False).
        num_workers: Number of subprocesses for data loading (default: 0).
        prefetch_factor: Number of batches to prefetch per worker (default: 2).
        **kwargs: Additional arguments passed to torch.utils.data.DataLoader.
    """
    
    def __init__(
        self,
        dataset: SynaDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        """
        Initialize the SynaDataLoader.
        
        Args:
            dataset: A SynaDataset instance to load from.
            batch_size: How many samples per batch to load.
            shuffle: Set to True to reshuffle data at every epoch.
            num_workers: How many subprocesses to use for data loading.
            prefetch_factor: Number of batches loaded in advance by each worker.
            **kwargs: Additional arguments for DataLoader.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SynaDataLoader. "
                "Install it with: pip install torch"
            )
        
        # prefetch_factor is only valid when num_workers > 0
        if num_workers > 0:
            kwargs['prefetch_factor'] = prefetch_factor
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )


def create_distributed_loader(
    dataset: SynaDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    **kwargs,
) -> Tuple[SynaDataLoader, "DistributedSampler"]:
    """
    Create a DataLoader configured for distributed training.
    
    This helper function creates a SynaDataLoader with a DistributedSampler
    for multi-GPU training scenarios.
    
    Example:
        >>> dataset = SynaDataset("data.db", pattern="train/*")
        >>> loader, sampler = create_distributed_loader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         # train
    
    Args:
        dataset: A SynaDataset instance.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
        prefetch_factor: Number of batches to prefetch per worker.
        **kwargs: Additional arguments for DataLoader.
        
    Returns:
        Tuple of (SynaDataLoader, DistributedSampler). The sampler's
        set_epoch() method should be called at the start of each epoch.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for distributed training. "
            "Install it with: pip install torch"
        )
    
    sampler = DistributedSampler(dataset)
    
    loader = SynaDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sampler handles shuffling
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
        **kwargs,
    )
    
    return loader, sampler


def get_distributed_sampler(
    dataset: SynaDataset,
    num_replicas: int = None,
    rank: int = None,
    shuffle: bool = True,
) -> "DistributedSampler":
    """
    Get a DistributedSampler for multi-GPU training.
    
    This function creates a DistributedSampler configured for the given
    dataset, enabling data parallelism across multiple GPUs or nodes.
    
    The sampler divides the dataset into non-overlapping chunks, one per
    replica (GPU/process). Each replica only sees its assigned chunk,
    ensuring no data duplication during distributed training.
    
    Example:
        >>> dataset = SynaDataset("data.db", pattern="train/*")
        >>> sampler = get_distributed_sampler(dataset)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)  # Important for shuffling
        ...     for batch in loader:
        ...         # train
    
    Args:
        dataset: A SynaDataset instance to sample from.
        num_replicas: Number of processes participating in distributed
                      training. If None, will be inferred from the
                      distributed environment (torch.distributed).
        rank: Rank of the current process within num_replicas. If None,
              will be inferred from the distributed environment.
        shuffle: If True (default), sampler will shuffle the indices
                 at each epoch. Call sampler.set_epoch(epoch) to enable
                 proper shuffling across epochs.
    
    Returns:
        A DistributedSampler configured for the dataset.
        
    Raises:
        ImportError: If PyTorch is not installed.
        
    Note:
        When using this sampler, remember to:
        1. Set shuffle=False in the DataLoader (sampler handles shuffling)
        2. Call sampler.set_epoch(epoch) at the start of each epoch
        
    Requirements: 13.2
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for DistributedSampler. "
            "Install it with: pip install torch"
        )
    
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
    )


# Convenience function to check if PyTorch is available
def is_torch_available() -> bool:
    """
    Check if PyTorch is available.
    
    Returns:
        True if PyTorch is installed and can be imported.
    """
    return TORCH_AVAILABLE


__all__ = [
    "SynaDataset",
    "SynaDataLoader",
    "create_distributed_loader",
    "get_distributed_sampler",
    "is_torch_available",
    "TORCH_AVAILABLE",
]

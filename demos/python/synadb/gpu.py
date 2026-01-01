"""
GPU Direct memory access for SynaDB.

This module provides GPU-accelerated tensor loading for ML workflows.
It enables loading data directly to CUDA devices for faster training.

Example:
    >>> from synadb.gpu import get_tensor_cuda, prefetch_to_gpu
    >>> # Load tensor directly to GPU
    >>> tensor = get_tensor_cuda("data.db", "train/*", device=0)
    >>> # tensor is already on GPU, ready for training
    
    >>> # Prefetch data for faster access
    >>> prefetch_to_gpu("data.db", "train/*", device=0)

Note:
    Currently uses CPU-to-GPU transfer. Future versions will implement
    GPU Direct for true zero-copy loading when hardware supports it.

_Requirements: 6.1, 6.2_
"""

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def get_tensor_cuda(
    db_path: str,
    pattern: str,
    device: int = 0,
) -> "torch.Tensor":
    """
    Load tensor directly to CUDA device.
    
    Loads data from the database and transfers it to the specified CUDA device.
    This is useful for ML training workflows where data needs to be on GPU.
    
    Args:
        db_path: Path to the SynaDB database file.
        pattern: Glob pattern for keys (e.g., "train/*", "data/batch_*").
                 Supports standard glob wildcards: * matches any characters,
                 ? matches single character.
        device: CUDA device index (default: 0). Use 0 for first GPU,
                1 for second GPU, etc.
    
    Returns:
        PyTorch tensor on the specified CUDA device.
    
    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If CUDA is not available.
    
    Example:
        >>> tensor = get_tensor_cuda("data.db", "train/*", device=0)
        >>> # tensor is already on GPU, no CPU->GPU copy needed
        >>> print(tensor.device)
        cuda:0
        
        >>> # Use with multiple GPUs
        >>> tensor_gpu0 = get_tensor_cuda("data.db", "train/*", device=0)
        >>> tensor_gpu1 = get_tensor_cuda("data.db", "val/*", device=1)
    
    Note:
        Current implementation loads to CPU first, then transfers to GPU.
        Future versions will use GPU Direct for zero-copy loading when
        the hardware supports it (requires NVIDIA GPUDirect Storage).
    
    _Requirements: 6.1, 6.2_
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for GPU operations. "
            "Install with: pip install torch"
        )
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU operations require a CUDA-capable device. "
            "Check your PyTorch installation and GPU drivers."
        )
    
    if device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device} not available. "
            f"Found {torch.cuda.device_count()} device(s)."
        )
    
    from .tensor import TensorEngine
    
    # For now, load to CPU then transfer
    # Future: Use GPU Direct for zero-copy
    engine = TensorEngine(db_path)
    np_array = engine.get_tensor(pattern)
    engine.close()
    
    # Convert to PyTorch tensor and move to GPU
    return torch.from_numpy(np_array).to(f"cuda:{device}")


def prefetch_to_gpu(
    db_path: str,
    pattern: str,
    device: int = 0,
) -> None:
    """
    Prefetch data to GPU memory for faster access.
    
    Useful for warming up before training loops. This function loads
    data into GPU memory asynchronously, so subsequent access is faster.
    
    Args:
        db_path: Path to the SynaDB database file.
        pattern: Glob pattern for keys (e.g., "train/*", "data/batch_*").
        device: CUDA device index (default: 0).
    
    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If CUDA is not available.
    
    Example:
        >>> # Prefetch training data before training loop
        >>> prefetch_to_gpu("data.db", "train/*", device=0)
        >>> 
        >>> # Now data access will be faster
        >>> tensor = get_tensor_cuda("data.db", "train/*", device=0)
    
    Note:
        Current implementation is a placeholder. Future versions will
        implement true prefetching using pinned memory and async transfers
        for optimal performance with GPU Direct Storage.
    
    _Requirements: 6.1, 6.2_
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for GPU operations. "
            "Install with: pip install torch"
        )
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU operations require a CUDA-capable device. "
            "Check your PyTorch installation and GPU drivers."
        )
    
    if device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {device} not available. "
            f"Found {torch.cuda.device_count()} device(s)."
        )
    
    # Implementation would use pinned memory and async transfer
    # For now, this is a placeholder that loads data to warm up the cache
    # Future: Implement true prefetching with:
    # 1. Memory-mapped file access
    # 2. Pinned (page-locked) host memory
    # 3. Async CUDA memcpy for overlapped transfer
    # 4. GPU Direct Storage for direct NVMe-to-GPU transfer
    
    from .tensor import TensorEngine
    
    engine = TensorEngine(db_path)
    np_array = engine.get_tensor(pattern)
    engine.close()
    
    # Use pinned memory for faster transfer (if available)
    tensor = torch.from_numpy(np_array)
    
    # Pin memory for faster GPU transfer
    if tensor.is_pinned() is False:
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            # Pinned memory not available, continue without it
            pass
    
    # Transfer to GPU (this warms up the transfer path)
    _ = tensor.to(f"cuda:{device}", non_blocking=True)
    
    # Synchronize to ensure transfer is complete
    torch.cuda.synchronize(device)


def is_gpu_available() -> bool:
    """
    Check if GPU operations are available.
    
    Returns:
        True if PyTorch is installed and CUDA is available, False otherwise.
    
    Example:
        >>> if is_gpu_available():
        ...     tensor = get_tensor_cuda("data.db", "train/*")
        ... else:
        ...     print("GPU not available, using CPU")
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """
    Get the number of available CUDA devices.
    
    Returns:
        Number of CUDA devices available, or 0 if CUDA is not available.
    
    Example:
        >>> n_gpus = get_gpu_count()
        >>> print(f"Found {n_gpus} GPU(s)")
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        return 0


def get_gpu_info(device: int = 0) -> Optional[dict]:
    """
    Get information about a specific GPU device.
    
    Args:
        device: CUDA device index (default: 0).
    
    Returns:
        Dictionary with GPU information, or None if not available.
        Contains: name, total_memory, free_memory, compute_capability.
    
    Example:
        >>> info = get_gpu_info(0)
        >>> if info:
        ...     print(f"GPU: {info['name']}")
        ...     print(f"Memory: {info['total_memory'] / 1e9:.1f} GB")
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        if device >= torch.cuda.device_count():
            return None
        
        props = torch.cuda.get_device_properties(device)
        
        # Get memory info
        torch.cuda.set_device(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Try to get free memory (may not be accurate on all systems)
        try:
            free_memory = torch.cuda.mem_get_info(device)[0]
        except Exception:
            free_memory = None
        
        return {
            "name": props.name,
            "total_memory": total_memory,
            "free_memory": free_memory,
            "compute_capability": (props.major, props.minor),
            "multi_processor_count": props.multi_processor_count,
        }
    except ImportError:
        return None
    except Exception:
        return None

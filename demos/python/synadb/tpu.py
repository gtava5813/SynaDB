"""TPU support via TensorFlow/JAX framework integration.

TPUs use XLA compilation and managed memory, so we integrate at the
framework level rather than direct memory access.

Example:
    >>> from synadb.tpu import get_tensor_dataset, get_tensor_jax
    >>> 
    >>> # TensorFlow/TPU usage
    >>> dataset = get_tensor_dataset("data.db", "train/*", batch_size=512)
    >>> dataset = dataset.prefetch(tf.data.AUTOTUNE)
    >>> 
    >>> # JAX/TPU usage
    >>> tensor = get_tensor_jax("data.db", "embeddings/*", device="tpu")

Note:
    TPUs don't support direct memory access like CUDA. Instead, this module
    provides framework-level integrations that work with TPU's XLA-based
    execution model.

_Requirements: 6.1 (Accelerator Support)_
"""

from typing import Iterator, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf
    import jax


def get_tensor_dataset(
    db_path: str,
    pattern: str,
    batch_size: int = 1024,
    dtype: str = "float32",
) -> "tf.data.Dataset":
    """
    Create a tf.data.Dataset for TPU-compatible data loading.

    The dataset can be distributed across TPU cores and supports
    prefetching to TPU infeed queues.

    Example:
        >>> dataset = get_tensor_dataset("data.db", "train/*", batch_size=512)
        >>> dataset = dataset.prefetch(tf.data.AUTOTUNE)
        >>> # Use with TPU strategy
        >>> with strategy.scope():
        ...     for batch in dataset:
        ...         train_step(batch)

    Args:
        db_path: Path to SynaDB database
        pattern: Key pattern to match (e.g., "train/*")
        batch_size: Batch size for TPU (should be divisible by 8 for TPU)
        dtype: Data type ("float32", "bfloat16" for TPU efficiency)

    Returns:
        tf.data.Dataset compatible with TPU distribution

    Raises:
        ImportError: If TensorFlow is not installed.

    _Requirements: 6.1 (Accelerator Support)_
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow not installed. Install with: pip install tensorflow")

    from .tensor import TensorEngine

    engine = TensorEngine(db_path)

    def generator():
        """Generator that yields chunks from the database."""
        data = engine.get_tensor_chunked(pattern)
        # Handle both tuple return (data, shape) and direct array return
        if isinstance(data, tuple):
            data = data[0]
        # Yield in batches suitable for TPU
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # Create dataset with proper dtype for TPU
    tf_dtype = tf.bfloat16 if dtype == "bfloat16" else tf.float32

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf_dtype)
    )

    # Batch and optimize for TPU
    return dataset.batch(batch_size, drop_remainder=True)


def get_tensor_jax(
    db_path: str,
    pattern: str,
    device: Optional[str] = None,
    dtype: str = "float32",
) -> "jax.Array":
    """
    Load tensor and place on JAX device (TPU/GPU/CPU).

    For TPU, data is automatically sharded across available cores
    when used with pjit or other JAX parallelism primitives.

    Example:
        >>> # Single device
        >>> tensor = get_tensor_jax("data.db", "embeddings/*")
        >>> 
        >>> # Explicit TPU placement
        >>> tensor = get_tensor_jax("data.db", "weights/*", device="tpu")
        >>> 
        >>> # With sharding for multi-TPU
        >>> from jax.sharding import PositionalSharding
        >>> sharding = PositionalSharding(jax.devices())
        >>> tensor = jax.device_put(tensor, sharding)

    Args:
        db_path: Path to SynaDB database
        pattern: Key pattern to match
        device: Target device ("tpu", "gpu", "cpu", or None for default)
        dtype: Data type ("float32", "bfloat16")

    Returns:
        jax.Array on the specified device

    Raises:
        ImportError: If JAX is not installed.
        RuntimeError: If the specified device is not available.

    _Requirements: 6.1 (Accelerator Support)_
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("jax not installed. Install with: pip install jax jaxlib")

    from .tensor import TensorEngine

    engine = TensorEngine(db_path)
    result = engine.get_tensor_chunked(pattern)
    # Handle both tuple return (data, shape) and direct array return
    if isinstance(result, tuple):
        np_array = result[0]
    else:
        np_array = result
    engine.close()

    # Convert dtype
    jax_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    np_array = np_array.astype(np.float32)  # JAX converts from float32

    # Place on device
    if device == "tpu":
        devices = jax.devices("tpu")
        if not devices:
            raise RuntimeError("No TPU devices available")
        return jax.device_put(jnp.array(np_array, dtype=jax_dtype), devices[0])
    elif device == "gpu":
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("No GPU devices available")
        return jax.device_put(jnp.array(np_array, dtype=jax_dtype), devices[0])
    elif device == "cpu":
        return jax.device_put(jnp.array(np_array, dtype=jax_dtype), jax.devices("cpu")[0])
    else:
        # Default device
        return jnp.array(np_array, dtype=jax_dtype)


def iter_batches_for_tpu(
    db_path: str,
    pattern: str,
    batch_size: int = 1024,
    global_batch_size: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Iterator that yields batches sized for TPU training.

    TPU training typically requires:
    - Batch sizes divisible by 8 (for TPU core count)
    - Consistent batch sizes (drop_remainder=True behavior)

    Example:
        >>> for batch in iter_batches_for_tpu("data.db", "train/*", batch_size=512):
        ...     # batch.shape[0] is always 512
        ...     train_step(batch)

    Args:
        db_path: Path to SynaDB database
        pattern: Key pattern to match
        batch_size: Per-replica batch size
        global_batch_size: Total batch size across all TPU cores (optional)

    Yields:
        numpy arrays of shape (batch_size, ...)

    _Requirements: 6.1 (Accelerator Support)_
    """
    from .tensor import TensorEngine

    engine = TensorEngine(db_path)
    result = engine.get_tensor_chunked(pattern)
    # Handle both tuple return (data, shape) and direct array return
    if isinstance(result, tuple):
        data = result[0]
    else:
        data = result
    engine.close()

    # Ensure batch size is TPU-friendly
    if batch_size % 8 != 0:
        import warnings
        warnings.warn(
            f"batch_size={batch_size} is not divisible by 8. "
            "TPU performance may be suboptimal."
        )

    # Yield complete batches only
    n_batches = len(data) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        yield data[start:end]


def is_tpu_available() -> bool:
    """
    Check if TPU devices are available via JAX.

    Returns:
        True if JAX is installed and TPU devices are available, False otherwise.

    Example:
        >>> if is_tpu_available():
        ...     tensor = get_tensor_jax("data.db", "train/*", device="tpu")
        ... else:
        ...     print("TPU not available, using CPU")
    """
    try:
        import jax
        devices = jax.devices("tpu")
        return len(devices) > 0
    except (ImportError, RuntimeError):
        return False


def get_tpu_count() -> int:
    """
    Get the number of available TPU devices.

    Returns:
        Number of TPU devices available, or 0 if TPU is not available.

    Example:
        >>> n_tpus = get_tpu_count()
        >>> print(f"Found {n_tpus} TPU core(s)")
    """
    try:
        import jax
        devices = jax.devices("tpu")
        return len(devices)
    except (ImportError, RuntimeError):
        return 0


def get_tpu_info() -> Optional[dict]:
    """
    Get information about available TPU devices.

    Returns:
        Dictionary with TPU information, or None if not available.
        Contains: device_count, devices (list of device info).

    Example:
        >>> info = get_tpu_info()
        >>> if info:
        ...     print(f"TPU cores: {info['device_count']}")
        ...     for dev in info['devices']:
        ...         print(f"  {dev['id']}: {dev['platform']}")
    """
    try:
        import jax
        devices = jax.devices("tpu")
        if not devices:
            return None

        return {
            "device_count": len(devices),
            "devices": [
                {
                    "id": str(dev.id),
                    "platform": dev.platform,
                    "device_kind": getattr(dev, "device_kind", "tpu"),
                }
                for dev in devices
            ],
        }
    except (ImportError, RuntimeError):
        return None

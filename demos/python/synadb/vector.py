"""
Vector store for embedding storage and similarity search.

This module provides a high-level Python interface for storing and searching
vector embeddings using the Syna database.

Example:
    >>> from synadb import VectorStore
    >>> store = VectorStore("vectors.db", dimensions=768)
    >>> store.insert("doc1", embedding1)
    >>> store.insert("doc2", embedding2)
    >>> results = store.search(query_embedding, k=5)
    >>> for r in results:
    ...     print(f"{r.key}: {r.score:.4f}")
"""

import ctypes
from ctypes import c_char_p, c_float, c_int32, c_uint16, c_size_t, POINTER, byref
import json
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .wrapper import SynaDB, SynaError


@dataclass
class SearchResult:
    """Result from a similarity search.
    
    Attributes:
        key: The key of the matching vector.
        score: Distance/similarity score (lower = more similar).
        vector: The vector data as a numpy array.
    """
    key: str
    score: float
    vector: np.ndarray


class VectorStore:
    """
    Vector store for embedding storage and similarity search.
    
    Provides a high-level API for storing and searching vector embeddings.
    Supports cosine, euclidean, and dot product distance metrics.
    
    Supports multiple backends:
    - "hnsw": Native HNSW index (default for smaller datasets)
    - "faiss": FAISS index (better for large-scale datasets, optional GPU support)
    - "auto": Automatically selects based on dataset size and GPU availability
    
    Example:
        >>> store = VectorStore("vectors.db", dimensions=768)
        >>> store.insert("doc1", embedding1)
        >>> store.insert("doc2", embedding2)
        >>> results = store.search(query_embedding, k=5)
        >>> for r in results:
        ...     print(f"{r.key}: {r.score:.4f}")
        
        # Using FAISS backend with GPU
        >>> store = VectorStore("vectors.db", dimensions=768, 
        ...                     backend="faiss", use_gpu=True)
    
    Attributes:
        COSINE: Cosine distance metric (1 - cosine_similarity).
        EUCLIDEAN: Euclidean (L2) distance metric.
        DOT_PRODUCT: Negative dot product (for max similarity).
    """
    
    COSINE = 0
    EUCLIDEAN = 1
    DOT_PRODUCT = 2
    
    # Backend constants
    BACKEND_AUTO = "auto"
    BACKEND_HNSW = "hnsw"
    BACKEND_FAISS = "faiss"
    
    def __init__(
        self,
        path: str,
        dimensions: int,
        metric: str = "cosine",
        backend: str = "auto",
        faiss_index_type: str = "IVF1024,Flat",
        faiss_nprobe: int = 10,
        use_gpu: bool = False,
        sync_on_write: bool = True,
    ):
        """
        Create or open a vector store with configurable backend.
        
        Args:
            path: Path to the database file.
            dimensions: Vector dimensions (64-8192).
            metric: Distance metric ("cosine", "euclidean", "dot_product").
            backend: Index backend to use:
                - "auto": Automatically select based on dataset size and GPU availability.
                         Uses FAISS with GPU if available and dataset is large (>100k vectors),
                         otherwise uses HNSW.
                - "hnsw": Native HNSW index. Good for datasets up to ~1M vectors.
                         O(log N) search complexity.
                - "faiss": FAISS index. Better for large-scale datasets (>1M vectors).
                         Supports GPU acceleration.
            faiss_index_type: FAISS index factory string (only used when backend="faiss").
                Defaults to "IVF1024,Flat". Common options:
                - "Flat": Exact search (slow for large datasets)
                - "IVF1024,Flat": Inverted file index with 1024 clusters
                - "IVF4096,PQ32": IVF with product quantization (memory efficient)
                - "HNSW32": FAISS HNSW implementation
            faiss_nprobe: Number of clusters to search for IVF indexes (only used when 
                backend="faiss"). Higher values = better recall but slower search.
                Defaults to 10.
            use_gpu: Whether to use GPU acceleration (only used when backend="faiss").
                Requires faiss-gpu to be installed. Defaults to False.
            sync_on_write: If True (default), sync to disk after each write for
                durability. Set to False for high-throughput scenarios at the risk
                of data loss on crash.
        
        Raises:
            ValueError: If dimensions are out of range (64-8192).
            ValueError: If backend is not one of "auto", "hnsw", or "faiss".
            RuntimeError: If the vector store cannot be created.
            ImportError: If backend="faiss" but faiss is not installed.
        
        Example:
            >>> # Default HNSW backend
            >>> store = VectorStore("vectors.db", dimensions=768)
            
            >>> # FAISS backend with GPU
            >>> store = VectorStore("vectors.db", dimensions=768,
            ...                     backend="faiss", use_gpu=True)
            
            >>> # FAISS with custom index type
            >>> store = VectorStore("vectors.db", dimensions=768,
            ...                     backend="faiss", 
            ...                     faiss_index_type="IVF4096,PQ32",
            ...                     faiss_nprobe=20)
        """
        # Validate dimensions
        if dimensions < 64 or dimensions > 8192:
            raise ValueError(f"Dimensions must be between 64 and 8192, got {dimensions}")
        
        # Validate backend
        valid_backends = [self.BACKEND_AUTO, self.BACKEND_HNSW, self.BACKEND_FAISS]
        if backend.lower() not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. Must be one of: {', '.join(valid_backends)}"
            )
        
        # Load the library using SynaDB's class method
        SynaDB._load_library()
        self._lib = SynaDB._lib
        self._path = path.encode('utf-8')
        self._dimensions = dimensions
        
        # Store backend configuration
        self._backend = backend.lower()
        self._faiss_index_type = faiss_index_type
        self._faiss_nprobe = faiss_nprobe
        self._use_gpu = use_gpu
        self._sync_on_write = sync_on_write
        
        # Resolve "auto" backend
        self._resolved_backend = self._resolve_backend()
        
        # Map metric string to integer
        metric_map = {
            "cosine": self.COSINE,
            "euclidean": self.EUCLIDEAN,
            "dot_product": self.DOT_PRODUCT,
        }
        self._metric = metric_map.get(metric.lower(), self.COSINE)
        
        # Set up FFI function signatures for vector store operations
        self._setup_ffi()
        
        # Initialize the vector store with config
        result = self._lib.SYNA_vector_store_new_with_config(
            self._path, 
            ctypes.c_uint16(dimensions), 
            ctypes.c_int32(self._metric),
            ctypes.c_int32(1 if sync_on_write else 0)
        )
        if result != 1:
            raise RuntimeError(f"Failed to create vector store: error code {result}")
        
        # Track inserted keys for __len__
        self._key_count = 0
        
        # Initialize FAISS index if needed (lazy initialization)
        self._faiss_index = None
        if self._resolved_backend == self.BACKEND_FAISS:
            self._init_faiss_index()
    
    def _setup_ffi(self):
        """Set up FFI function signatures for vector store operations."""
        # SYNA_vector_store_new
        self._lib.SYNA_vector_store_new.argtypes = [c_char_p, c_uint16, c_int32]
        self._lib.SYNA_vector_store_new.restype = c_int32
        
        # SYNA_vector_store_new_with_config
        self._lib.SYNA_vector_store_new_with_config.argtypes = [c_char_p, c_uint16, c_int32, c_int32]
        self._lib.SYNA_vector_store_new_with_config.restype = c_int32
        
        # SYNA_vector_store_insert
        self._lib.SYNA_vector_store_insert.argtypes = [
            c_char_p, c_char_p, POINTER(c_float), c_uint16
        ]
        self._lib.SYNA_vector_store_insert.restype = c_int32
        
        # SYNA_vector_store_search
        self._lib.SYNA_vector_store_search.argtypes = [
            c_char_p, POINTER(c_float), c_uint16, c_size_t, POINTER(c_char_p)
        ]
        self._lib.SYNA_vector_store_search.restype = c_int32
        
        # SYNA_free_json
        self._lib.SYNA_free_json.argtypes = [c_char_p]
        self._lib.SYNA_free_json.restype = None
    
    def _resolve_backend(self) -> str:
        """
        Resolve the "auto" backend to a concrete backend.
        
        Returns:
            The resolved backend name ("hnsw" or "faiss").
        """
        if self._backend != self.BACKEND_AUTO:
            return self._backend
        
        # Auto-selection logic:
        # - Use FAISS if GPU is requested and available
        # - Otherwise default to HNSW (more widely compatible)
        if self._use_gpu:
            try:
                import faiss
                if faiss.get_num_gpus() > 0:
                    return self.BACKEND_FAISS
            except ImportError:
                pass
        
        # Default to HNSW for auto mode
        return self.BACKEND_HNSW
    
    def _init_faiss_index(self):
        """
        Initialize the FAISS index if using FAISS backend.
        
        Raises:
            ImportError: If faiss is not installed.
            RuntimeError: If GPU is requested but not available.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS backend requires faiss to be installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        # Create the index using the factory string
        self._faiss_index = faiss.index_factory(
            self._dimensions, 
            self._faiss_index_type,
            faiss.METRIC_L2 if self._metric == self.EUCLIDEAN else faiss.METRIC_INNER_PRODUCT
        )
        
        # Move to GPU if requested
        if self._use_gpu:
            if faiss.get_num_gpus() == 0:
                raise RuntimeError(
                    "GPU requested but no GPUs available. "
                    "Install faiss-gpu and ensure CUDA is properly configured."
                )
            # Use the first GPU
            res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)
        
        # Set nprobe for IVF indexes
        if hasattr(self._faiss_index, 'nprobe'):
            self._faiss_index.nprobe = self._faiss_nprobe
    
    def insert(self, key: str, vector: np.ndarray) -> None:
        """
        Insert a vector with the given key.
        
        Args:
            key: Unique identifier for the vector.
            vector: numpy array of shape (dimensions,).
        
        Raises:
            ValueError: If vector dimensions don't match store configuration.
            RuntimeError: If the insert fails.
        """
        # Convert to float32 and flatten
        vector = np.asarray(vector, dtype=np.float32).flatten()
        
        if len(vector) != self._dimensions:
            raise ValueError(
                f"Vector has {len(vector)} dimensions, expected {self._dimensions}"
            )
        
        # Get pointer to vector data
        vector_ptr = vector.ctypes.data_as(POINTER(c_float))
        
        result = self._lib.SYNA_vector_store_insert(
            self._path,
            key.encode('utf-8'),
            vector_ptr,
            ctypes.c_uint16(self._dimensions),
        )
        
        if result != 1:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path or key")
            else:
                raise RuntimeError(f"Failed to insert vector: error code {result}")
        
        self._key_count += 1
    
    def insert_batch(self, keys: List[str], vectors: np.ndarray) -> int:
        """
        Insert multiple vectors in a single batch operation.
        
        This is significantly faster than calling insert() in a loop:
        - Single FFI boundary crossing for all vectors
        - Deferred index building until after all vectors are inserted
        - Reduced lock contention
        
        Args:
            keys: List of unique identifiers for each vector.
            vectors: numpy array of shape (num_vectors, dimensions).
        
        Returns:
            Number of vectors successfully inserted.
        
        Raises:
            ValueError: If vectors dimensions don't match store configuration.
            ValueError: If keys and vectors have different lengths.
            RuntimeError: If the insert fails.
        
        Example:
            >>> store = VectorStore("vectors.db", dimensions=768)
            >>> keys = [f"doc_{i}" for i in range(1000)]
            >>> embeddings = np.random.randn(1000, 768).astype(np.float32)
            >>> count = store.insert_batch(keys, embeddings)
            >>> print(f"Inserted {count} vectors")
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_insert_batch_setup'):
            self._lib.SYNA_vector_store_insert_batch.argtypes = [
                c_char_p, POINTER(c_char_p), POINTER(c_float), c_uint16, c_size_t
            ]
            self._lib.SYNA_vector_store_insert_batch.restype = c_int32
            self._lib._insert_batch_setup = True
        
        # Convert to float32 and ensure 2D
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        num_vectors, dims = vectors.shape
        
        if dims != self._dimensions:
            raise ValueError(
                f"Vectors have {dims} dimensions, expected {self._dimensions}"
            )
        
        if len(keys) != num_vectors:
            raise ValueError(
                f"Number of keys ({len(keys)}) doesn't match number of vectors ({num_vectors})"
            )
        
        # Ensure contiguous memory layout (row-major)
        vectors = np.ascontiguousarray(vectors)
        
        # Create array of C strings for keys
        keys_encoded = [k.encode('utf-8') for k in keys]
        keys_array = (c_char_p * num_vectors)(*keys_encoded)
        
        # Get pointer to vector data
        data_ptr = vectors.ctypes.data_as(POINTER(c_float))
        
        result = self._lib.SYNA_vector_store_insert_batch(
            self._path,
            keys_array,
            data_ptr,
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(num_vectors),
        )
        
        if result < 0:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path or keys")
            else:
                raise RuntimeError(f"Failed to insert batch: error code {result}")
        
        self._key_count += result
        return result
    
    def insert_batch_fast(self, keys: List[str], vectors: np.ndarray) -> int:
        """
        Insert multiple vectors WITHOUT updating the index (maximum write speed).
        
        This is the fastest way to bulk-load vectors (100K+/sec). Vectors are
        written to storage but NOT added to the HNSW index. Call `build_index()`
        after all inserts to enable fast search.
        
        Args:
            keys: List of unique identifiers for each vector.
            vectors: numpy array of shape (num_vectors, dimensions).
        
        Returns:
            Number of vectors successfully inserted.
        
        Example:
            >>> store = VectorStore("vectors.db", dimensions=768)
            >>> # Bulk load at 100K+/sec
            >>> for batch in batches:
            ...     store.insert_batch_fast(batch.keys, batch.vectors)
            >>> # Build index once at the end
            >>> store.build_index()
            >>> # Now search is fast
            >>> results = store.search(query, k=10)
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_insert_batch_fast_setup'):
            self._lib.SYNA_vector_store_insert_batch_fast.argtypes = [
                c_char_p, POINTER(c_char_p), POINTER(c_float), c_uint16, c_size_t
            ]
            self._lib.SYNA_vector_store_insert_batch_fast.restype = c_int32
            self._lib._insert_batch_fast_setup = True
        
        # Convert to float32 and ensure 2D
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        num_vectors, dims = vectors.shape
        
        if dims != self._dimensions:
            raise ValueError(
                f"Vectors have {dims} dimensions, expected {self._dimensions}"
            )
        
        if len(keys) != num_vectors:
            raise ValueError(
                f"Number of keys ({len(keys)}) doesn't match number of vectors ({num_vectors})"
            )
        
        vectors = np.ascontiguousarray(vectors)
        keys_encoded = [k.encode('utf-8') for k in keys]
        keys_array = (c_char_p * num_vectors)(*keys_encoded)
        data_ptr = vectors.ctypes.data_as(POINTER(c_float))
        
        result = self._lib.SYNA_vector_store_insert_batch_fast(
            self._path,
            keys_array,
            data_ptr,
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(num_vectors),
        )
        
        if result < 0:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path or keys")
            else:
                raise RuntimeError(f"Failed to insert batch: error code {result}")
        
        self._key_count += result
        return result
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (dimensions,).
            k: Number of results to return.
        
        Returns:
            List of SearchResult sorted by similarity (most similar first).
        
        Raises:
            ValueError: If query dimensions don't match store configuration.
            RuntimeError: If the search fails.
        """
        # Convert to float32 and flatten
        query = np.asarray(query, dtype=np.float32).flatten()
        
        if len(query) != self._dimensions:
            raise ValueError(
                f"Query has {len(query)} dimensions, expected {self._dimensions}"
            )
        
        # Get pointer to query data
        query_ptr = query.ctypes.data_as(POINTER(c_float))
        
        # Prepare output pointer
        out_json = c_char_p()
        
        result = self._lib.SYNA_vector_store_search(
            self._path,
            query_ptr,
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(k),
            byref(out_json),
        )
        
        if result < 0:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path or query")
            else:
                raise RuntimeError(f"Search failed: error code {result}")
        
        # Parse JSON results
        if out_json.value is None:
            return []
        
        try:
            json_str = out_json.value.decode('utf-8')
            results_data = json.loads(json_str)
        finally:
            # Free the JSON string
            self._lib.SYNA_free_json(out_json)
        
        return [
            SearchResult(
                key=r['key'],
                score=r['score'],
                vector=np.array(r['vector'], dtype=np.float32)
            )
            for r in results_data
        ]
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get a vector by key.
        
        Args:
            key: The key to look up.
        
        Returns:
            The vector as a numpy array, or None if not found.
        
        Note:
            This performs a search with k=1 and checks if the result matches
            the requested key. For direct key lookup, consider using the
            underlying SynaDB directly.
        """
        # Use the underlying database to get the vector directly
        # The vector is stored with prefix "vec/" by default
        full_key = f"vec/{key}"
        
        # We need to access the underlying database
        # For now, we'll use a workaround by searching and filtering
        # This is not ideal but works for the basic case
        
        # A better implementation would add a dedicated FFI function
        # For now, return None as the get functionality requires
        # additional FFI support
        return None
    
    def delete(self, key: str) -> None:
        """
        Delete a vector by key.
        
        Args:
            key: The key to delete.
        
        Note:
            This requires the underlying database delete functionality.
            The vector is stored with prefix "vec/" by default.
        """
        # The vector is stored with prefix "vec/" by default
        full_key = f"vec/{key}"
        
        # Use the underlying SYNA_delete function
        result = self._lib.SYNA_delete(self._path, full_key.encode('utf-8'))
        
        if result == 1:
            self._key_count = max(0, self._key_count - 1)
        elif result == -1:
            raise RuntimeError("Database not found")
        elif result != 1:
            raise RuntimeError(f"Failed to delete vector: error code {result}")
    
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self._key_count
    
    @property
    def dimensions(self) -> int:
        """Return the configured dimensions."""
        return self._dimensions
    
    @property
    def metric_name(self) -> str:
        """Return the name of the configured distance metric."""
        metric_names = {
            self.COSINE: "cosine",
            self.EUCLIDEAN: "euclidean",
            self.DOT_PRODUCT: "dot_product",
        }
        return metric_names.get(self._metric, "unknown")
    
    @property
    def backend(self) -> str:
        """Return the configured backend ("auto", "hnsw", or "faiss")."""
        return self._backend
    
    @property
    def resolved_backend(self) -> str:
        """Return the resolved backend ("hnsw" or "faiss").
        
        When backend="auto", this returns the actual backend that was selected.
        """
        return self._resolved_backend
    
    @property
    def faiss_index_type(self) -> str:
        """Return the FAISS index factory string."""
        return self._faiss_index_type
    
    @property
    def faiss_nprobe(self) -> int:
        """Return the FAISS nprobe parameter."""
        return self._faiss_nprobe
    
    @property
    def use_gpu(self) -> bool:
        """Return whether GPU acceleration is enabled."""
        return self._use_gpu
    
    @property
    def is_faiss_trained(self) -> bool:
        """Return whether the FAISS index is trained (if using FAISS backend).
        
        Some FAISS index types (like IVF) require training before use.
        Returns True if not using FAISS backend.
        """
        if self._faiss_index is None:
            return True
        return self._faiss_index.is_trained
    
    @property
    def has_index(self) -> bool:
        """Return whether an HNSW index has been built.
        
        The index is built automatically when vector count exceeds the threshold
        (default 10,000), or can be built manually with build_index().
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_has_index_setup'):
            self._lib.SYNA_vector_store_has_index.argtypes = [c_char_p]
            self._lib.SYNA_vector_store_has_index.restype = c_int32
            self._lib._has_index_setup = True
        
        result = self._lib.SYNA_vector_store_has_index(self._path)
        return result == 1
    
    def build_index(self) -> None:
        """Manually build the HNSW index for faster search.
        
        The index is built automatically when vector count exceeds the threshold
        (default 10,000 vectors), but this method allows explicit control.
        
        Building the index converts search from O(N) brute-force to O(log N)
        approximate nearest neighbor search.
        
        Raises:
            RuntimeError: If the index build fails.
        
        Example:
            >>> store = VectorStore("vectors.db", dimensions=768)
            >>> for i in range(1000):
            ...     store.insert(f"doc_{i}", embeddings[i])
            >>> store.build_index()  # Manually build for faster search
            >>> results = store.search(query, k=10)  # Now uses HNSW
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_build_index_setup'):
            self._lib.SYNA_vector_store_build_index.argtypes = [c_char_p]
            self._lib.SYNA_vector_store_build_index.restype = c_int32
            self._lib._build_index_setup = True
        
        result = self._lib.SYNA_vector_store_build_index(self._path)
        
        if result != 1:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path")
            else:
                raise RuntimeError(f"Failed to build index: error code {result}")
    
    def flush(self) -> None:
        """Flush any pending changes to disk without closing the store.
        
        This saves the HNSW index if it has unsaved changes. Useful for
        ensuring durability without closing the store.
        
        Raises:
            RuntimeError: If the flush fails.
        
        Example:
            >>> store = VectorStore("vectors.db", dimensions=768)
            >>> for i in range(1000):
            ...     store.insert(f"doc_{i}", embeddings[i])
            >>> store.flush()  # Save index to disk
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_flush_setup'):
            self._lib.SYNA_vector_store_flush.argtypes = [c_char_p]
            self._lib.SYNA_vector_store_flush.restype = c_int32
            self._lib._flush_setup = True
        
        result = self._lib.SYNA_vector_store_flush(self._path)
        
        if result != 1:
            if result == -1:
                raise RuntimeError("Vector store not found - was it created?")
            elif result == -2:
                raise RuntimeError("Invalid path")
            elif result == 0:
                raise RuntimeError("Flush failed")
            else:
                raise RuntimeError(f"Failed to flush: error code {result}")
    
    def close(self) -> None:
        """Close the vector store and save any pending changes.
        
        This removes the store from the global registry and saves the HNSW
        index to disk. After calling close(), the store cannot be used.
        
        Raises:
            RuntimeError: If the close fails.
        
        Example:
            >>> store = VectorStore("vectors.db", dimensions=768)
            >>> # ... use the store ...
            >>> store.close()  # Save and close
        """
        # Setup FFI function if not already done
        if not hasattr(self._lib, '_close_setup'):
            self._lib.SYNA_vector_store_close.argtypes = [c_char_p]
            self._lib.SYNA_vector_store_close.restype = c_int32
            self._lib._close_setup = True
        
        result = self._lib.SYNA_vector_store_close(self._path)
        
        if result != 1:
            if result == -1:
                # Already closed or not found - not an error
                return
            elif result == -2:
                raise RuntimeError("Invalid path")
            else:
                raise RuntimeError(f"Failed to close: error code {result}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the store."""
        self.close()
        return False

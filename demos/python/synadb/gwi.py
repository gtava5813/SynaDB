"""
Gravity Well Index (GWI) - Append-Only Vector Indexing.

A novel vector indexing algorithm designed for SynaDB's append-only,
mmap-friendly architecture. Vectors "fall" into gravity wells (attractors)
rather than being connected in a mutable graph like HNSW.

Key advantages over HNSW:
- O(N) index build time vs O(N log N) for HNSW
- Truly append-only (no graph mutations)
- Predictable memory usage
- Faster index construction (10-100x faster than HNSW)

Example:
    >>> from synadb import GravityWellIndex
    >>> gwi = GravityWellIndex("vectors.gwi", dimensions=768)
    >>> 
    >>> # Initialize attractors from sample data
    >>> gwi.initialize(sample_vectors)
    >>> 
    >>> # Insert vectors (O(log M) each)
    >>> gwi.insert("doc1", embedding)
    >>> 
    >>> # Search (O(log M + cluster_size))
    >>> results = gwi.search(query, k=10)
"""

import ctypes
import json
import os
import platform
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class GwiSearchResult:
    """Result from a GWI similarity search."""
    key: str
    score: float
    vector: Optional[np.ndarray] = None


class GravityWellIndex:
    """
    Gravity Well Index for append-only vector indexing.
    
    This implementation uses a hierarchical attractor-based approach where
    vectors "fall" into gravity wells rather than being connected in a
    mutable graph. This enables:
    
    - O(N) index build time (vs O(N log N) for HNSW)
    - Truly append-only storage
    - Predictable memory usage
    - 10-100x faster index construction
    
    Args:
        path: Path to the GWI file
        dimensions: Number of dimensions (64-8192)
        branching_factor: Number of children per attractor (default: 16)
        num_levels: Number of hierarchy levels (default: 3)
        initial_capacity: Pre-allocated capacity in number of vectors
    
    Example:
        >>> gwi = GravityWellIndex("vectors.gwi", dimensions=768)
        >>> 
        >>> # Initialize attractors from sample data (required before insert)
        >>> sample = np.random.randn(1000, 768).astype(np.float32)
        >>> gwi.initialize(sample)
        >>> 
        >>> # Insert vectors
        >>> for i, vec in enumerate(embeddings):
        ...     gwi.insert(f"doc{i}", vec)
        >>> 
        >>> # Search
        >>> results = gwi.search(query, k=10)
    """
    
    def __init__(
        self,
        path: str,
        dimensions: int = 768,
        branching_factor: int = 16,
        num_levels: int = 3,
        initial_capacity: int = 100_000,
    ):
        self._lib = self._load_library()
        self._path = os.path.abspath(path)
        self._dimensions = dimensions
        self._branching_factor = branching_factor
        self._num_levels = num_levels
        self._initial_capacity = initial_capacity
        self._closed = False
        self._initialized = False
        
        # Check if file exists - open existing or create new
        if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
            # Open existing index
            result = self._lib.SYNA_gwi_open(self._path.encode("utf-8"))
            if result != 1:
                raise RuntimeError(f"Failed to open existing GravityWellIndex: error code {result}")
            # Existing index is already initialized
            self._initialized = True
        else:
            # Create new index
            result = self._lib.SYNA_gwi_new(
                self._path.encode("utf-8"),
                ctypes.c_uint16(dimensions),
                ctypes.c_uint16(branching_factor),
                ctypes.c_uint8(num_levels),
                ctypes.c_size_t(initial_capacity),
            )
            if result != 1:
                raise RuntimeError(f"Failed to create GravityWellIndex: error code {result}")

    def _load_library(self):
        """Load the native library."""
        system = platform.system()
        machine = platform.machine().lower()
        
        # Determine library name based on platform
        if system == "Windows":
            lib_name = "synadb.dll"
        elif system == "Darwin":
            if machine in ("arm64", "aarch64"):
                lib_name = "libsynadb-arm64.dylib"
            else:
                lib_name = "libsynadb-x86_64.dylib"
        else:
            lib_name = "libsynadb.so"
        
        # Search paths
        search_paths = [
            os.path.join(os.path.dirname(__file__), lib_name),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "target", "release", lib_name),
            lib_name,
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                lib = ctypes.CDLL(path)
                self._setup_functions(lib)
                return lib
        
        raise RuntimeError(f"Could not find native library: {lib_name}")
    
    def _setup_functions(self, lib):
        """Set up function signatures."""
        # SYNA_gwi_new
        lib.SYNA_gwi_new.argtypes = [
            ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint16, 
            ctypes.c_uint8, ctypes.c_size_t
        ]
        lib.SYNA_gwi_new.restype = ctypes.c_int32
        
        # SYNA_gwi_open
        lib.SYNA_gwi_open.argtypes = [ctypes.c_char_p]
        lib.SYNA_gwi_open.restype = ctypes.c_int32
        
        # SYNA_gwi_initialize
        lib.SYNA_gwi_initialize.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_uint16
        ]
        lib.SYNA_gwi_initialize.restype = ctypes.c_int32
        
        # SYNA_gwi_insert
        lib.SYNA_gwi_insert.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint16
        ]
        lib.SYNA_gwi_insert.restype = ctypes.c_int32
        
        # SYNA_gwi_insert_batch
        lib.SYNA_gwi_insert_batch.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint16, ctypes.c_size_t
        ]
        lib.SYNA_gwi_insert_batch.restype = ctypes.c_int32
        
        # SYNA_gwi_search
        lib.SYNA_gwi_search.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint16, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)
        ]
        lib.SYNA_gwi_search.restype = ctypes.c_int32
        
        # SYNA_gwi_flush
        lib.SYNA_gwi_flush.argtypes = [ctypes.c_char_p]
        lib.SYNA_gwi_flush.restype = ctypes.c_int32
        
        # SYNA_gwi_close
        lib.SYNA_gwi_close.argtypes = [ctypes.c_char_p]
        lib.SYNA_gwi_close.restype = ctypes.c_int32
        
        # SYNA_gwi_len
        lib.SYNA_gwi_len.argtypes = [ctypes.c_char_p]
        lib.SYNA_gwi_len.restype = ctypes.c_int64
        
        # SYNA_free_json (shared with other modules)
        lib.SYNA_free_json.argtypes = [ctypes.c_char_p]
        lib.SYNA_free_json.restype = None

    def initialize(self, sample_vectors: np.ndarray) -> None:
        """
        Initialize attractors from sample vectors.
        
        This must be called before inserting any vectors. The sample vectors
        are used to build the hierarchical attractor structure using K-means.
        
        Args:
            sample_vectors: 2D array of shape (n_samples, dimensions)
                           Recommend 1000-10000 samples for good coverage.
        
        Raises:
            RuntimeError: If initialization fails
            ValueError: If dimensions don't match
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        sample_vectors = np.asarray(sample_vectors, dtype=np.float32)
        if sample_vectors.ndim == 1:
            sample_vectors = sample_vectors.reshape(1, -1)
        
        if sample_vectors.shape[1] != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions} dimensions, got {sample_vectors.shape[1]}"
            )
        
        # Flatten to contiguous array
        flat_vectors = sample_vectors.flatten()
        num_vectors = sample_vectors.shape[0]
        
        result = self._lib.SYNA_gwi_initialize(
            self._path.encode("utf-8"),
            flat_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(num_vectors),
            ctypes.c_uint16(self._dimensions),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to initialize attractors: error code {result}")
        
        self._initialized = True

    def insert(self, key: str, vector: np.ndarray) -> None:
        """
        Insert a vector with the given key.
        
        The vector will "fall" into the nearest gravity well (cluster)
        based on hierarchical descent through the attractor tree.
        
        Args:
            key: Unique identifier for the vector
            vector: The vector data (must match configured dimensions)
        
        Raises:
            RuntimeError: If index is closed or not initialized
            ValueError: If dimensions don't match
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call initialize() first.")
        
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(vector)}")
        
        result = self._lib.SYNA_gwi_insert(
            self._path.encode("utf-8"),
            key.encode("utf-8"),
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to insert vector: error code {result}")
    
    def insert_batch(self, keys: List[str], vectors: np.ndarray) -> int:
        """
        Insert multiple vectors in a batch.
        
        Args:
            keys: List of key strings
            vectors: 2D array of shape (n_vectors, dimensions)
        
        Returns:
            Number of vectors successfully inserted
        
        Raises:
            RuntimeError: If index is closed or not initialized
            ValueError: If dimensions don't match or key/vector count mismatch
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call initialize() first.")
        
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self._dimensions:
            raise ValueError(
                f"Expected {self._dimensions} dimensions, got {vectors.shape[1]}"
            )
        
        if len(keys) != vectors.shape[0]:
            raise ValueError(
                f"Number of keys ({len(keys)}) must match number of vectors ({vectors.shape[0]})"
            )
        
        # Prepare keys as C strings
        key_bytes = [k.encode("utf-8") for k in keys]
        key_array = (ctypes.c_char_p * len(keys))(*key_bytes)
        
        # Flatten vectors to contiguous array
        flat_vectors = vectors.flatten()
        
        result = self._lib.SYNA_gwi_insert_batch(
            self._path.encode("utf-8"),
            key_array,
            flat_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(len(keys)),
        )
        
        if result < 0:
            raise RuntimeError(f"Failed to insert batch: error code {result}")
        
        return result

    def search(self, query: np.ndarray, k: int = 10, nprobe: int = 3) -> List[GwiSearchResult]:
        """
        Search for the k nearest neighbors.
        
        The search descends through the attractor hierarchy to find the
        primary cluster, then probes nearby clusters for candidates.
        
        Args:
            query: Query vector
            k: Number of results to return
            nprobe: Number of clusters to probe (higher = better recall, slower)
                - nprobe=3: Fast, ~5-15% recall (default)
                - nprobe=10: Balanced, ~30-50% recall
                - nprobe=30: High quality, ~70-90% recall
                - nprobe=100: Near-exact, ~95%+ recall
        
        Returns:
            List of GwiSearchResult objects sorted by score (ascending)
        
        Raises:
            RuntimeError: If index is closed or not initialized
            ValueError: If dimensions don't match
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        if not self._initialized:
            raise RuntimeError("Index not initialized. Call initialize() first.")
        
        query = np.asarray(query, dtype=np.float32).flatten()
        if len(query) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(query)}")
        
        out_json = ctypes.c_char_p()
        
        result = self._lib.SYNA_gwi_search_nprobe(
            self._path.encode("utf-8"),
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(k),
            ctypes.c_size_t(nprobe),
            ctypes.byref(out_json),
        )
        
        if result < 0:
            raise RuntimeError(f"Search failed: error code {result}")
        
        # Parse JSON results
        if out_json.value:
            json_str = out_json.value.decode("utf-8")
            self._lib.SYNA_free_json(out_json)
            
            results_data = json.loads(json_str)
            return [
                GwiSearchResult(key=r["key"], score=r["score"])
                for r in results_data
            ]
        
        return []
    
    def flush(self) -> None:
        """Flush any pending changes to disk."""
        if self._closed:
            return
        
        result = self._lib.SYNA_gwi_flush(self._path.encode("utf-8"))
        
        if result != 1:
            raise RuntimeError(f"Failed to flush: error code {result}")
    
    def close(self) -> None:
        """Close the index and release resources."""
        if self._closed:
            return
        
        self._lib.SYNA_gwi_close(self._path.encode("utf-8"))
        self._closed = True
    
    def __len__(self) -> int:
        """Return the number of vectors stored."""
        if self._closed:
            return 0
        
        result = self._lib.SYNA_gwi_len(self._path.encode("utf-8"))
        return max(0, result)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()
    
    @property
    def dimensions(self) -> int:
        """Return the configured dimensions."""
        return self._dimensions
    
    @property
    def branching_factor(self) -> int:
        """Return the configured branching factor."""
        return self._branching_factor
    
    @property
    def num_levels(self) -> int:
        """Return the configured number of levels."""
        return self._num_levels
    
    @property
    def num_clusters(self) -> int:
        """Return the number of leaf clusters."""
        return self._branching_factor ** self._num_levels
    
    @property
    def initialized(self) -> bool:
        """Return whether attractors have been initialized."""
        return self._initialized

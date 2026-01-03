"""Cascade Index: Fast vector index with O(N) build time.

Cascade Index combines Locality-Sensitive Hashing (LSH) with adaptive buckets
and a sparse graph to achieve O(N) build time without requiring initialization
samples (unlike GWI) or quadratic neighbor search (unlike HNSW).

Example:
    >>> from synadb import CascadeIndex
    >>> import numpy as np
    >>> 
    >>> index = CascadeIndex("vectors.cascade", dimensions=768)
    >>> index.insert("doc1", np.random.randn(768).astype(np.float32))
    >>> results = index.search(query, k=10)
"""

import ctypes
import json
import os
import platform
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


def _load_library():
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
            return ctypes.CDLL(path)
    
    raise RuntimeError(f"Could not find native library: {lib_name}")


@dataclass
class SearchResult:
    """Search result from Cascade Index."""
    key: str
    score: float
    vector: Optional[np.ndarray] = None


class CascadeIndex:
    """
    Cascade Index for fast approximate nearest neighbor search.
    
    Combines LSH with sparse graph for O(N) build time and high recall.
    No initialization samples required (unlike GWI).
    
    Args:
        path: Path to the index file
        dimensions: Vector dimensions (64-8192)
        metric: Distance metric ("cosine", "euclidean", "dot_product")
        num_probes: Number of LSH probes for search (default: 3)
        ef_search: Search expansion factor (default: 50)
    
    Example:
        >>> index = CascadeIndex("vectors.cascade", dimensions=768)
        >>> index.insert("doc1", embedding)
        >>> results = index.search(query, k=10)
        >>> for r in results:
        ...     print(f"{r.key}: {r.score:.4f}")
    """
    
    def __init__(
        self,
        path: str,
        dimensions: int = 768,
        metric: str = "cosine",
        num_probes: int = 16,
        ef_search: int = 80,
    ):
        """Initialize Cascade Index."""
        self._lib = _load_library()
        self._path = path
        self._dimensions = dimensions
        self._num_probes = num_probes
        self._ef_search = ef_search
        self._closed = False
        
        # Set up function signatures
        self._setup_functions(self._lib)
        
        # Create the index
        result = self._lib.SYNA_cascade_new(
            self._path.encode("utf-8"),
            ctypes.c_uint16(dimensions),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to create Cascade Index: error code {result}")
    
    def _setup_functions(self, lib):
        """Set up function signatures."""
        # SYNA_cascade_new
        lib.SYNA_cascade_new.argtypes = [ctypes.c_char_p, ctypes.c_uint16]
        lib.SYNA_cascade_new.restype = ctypes.c_int32
        
        # SYNA_cascade_insert
        lib.SYNA_cascade_insert.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint16
        ]
        lib.SYNA_cascade_insert.restype = ctypes.c_int32
        
        # SYNA_cascade_insert_batch
        lib.SYNA_cascade_insert_batch.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint16, ctypes.c_size_t
        ]
        lib.SYNA_cascade_insert_batch.restype = ctypes.c_int32
        
        # SYNA_cascade_search_params
        lib.SYNA_cascade_search_params.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint16, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p)
        ]
        lib.SYNA_cascade_search_params.restype = ctypes.c_int32
        
        # SYNA_cascade_flush
        lib.SYNA_cascade_flush.argtypes = [ctypes.c_char_p]
        lib.SYNA_cascade_flush.restype = ctypes.c_int32
        
        # SYNA_cascade_close
        lib.SYNA_cascade_close.argtypes = [ctypes.c_char_p]
        lib.SYNA_cascade_close.restype = ctypes.c_int32
        
        # SYNA_cascade_len
        lib.SYNA_cascade_len.argtypes = [ctypes.c_char_p]
        lib.SYNA_cascade_len.restype = ctypes.c_int64
        
        # SYNA_free_json (shared)
        lib.SYNA_free_json.argtypes = [ctypes.c_char_p]
        lib.SYNA_free_json.restype = None

    
    def insert(self, key: str, vector: np.ndarray) -> None:
        """Insert a single vector.
        
        Args:
            key: Unique identifier for the vector
            vector: Vector data (must match dimensions)
        
        Raises:
            ValueError: If vector dimensions don't match
            RuntimeError: If insert fails
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(vector)}")
        
        result = self._lib.SYNA_cascade_insert(
            self._path.encode("utf-8"),
            key.encode("utf-8"),
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to insert vector: error code {result}")
    
    def insert_batch(self, keys: List[str], vectors: np.ndarray) -> None:
        """Insert multiple vectors efficiently.
        
        Args:
            keys: List of unique identifiers
            vectors: 2D array of shape (n, dimensions)
        
        Raises:
            ValueError: If dimensions don't match or keys/vectors length mismatch
            RuntimeError: If insert fails
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if len(keys) != vectors.shape[0]:
            raise ValueError(f"Keys ({len(keys)}) and vectors ({vectors.shape[0]}) must have same length")
        
        if vectors.shape[1] != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {vectors.shape[1]}")
        
        # Create array of C strings
        key_array = (ctypes.c_char_p * len(keys))()
        for i, key in enumerate(keys):
            key_array[i] = key.encode("utf-8")
        
        # Flatten vectors
        flat_vectors = vectors.flatten().astype(np.float32)
        
        result = self._lib.SYNA_cascade_insert_batch(
            self._path.encode("utf-8"),
            key_array,
            flat_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(len(keys)),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to insert batch: error code {result}")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        num_probes: Optional[int] = None,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results to return
            num_probes: Number of LSH probes (default: use init value)
            ef_search: Search expansion factor (default: use init value)
        
        Returns:
            List of SearchResult objects sorted by score (ascending)
        
        Raises:
            ValueError: If query dimensions don't match
            RuntimeError: If search fails
        """
        if self._closed:
            raise RuntimeError("Index is closed")
        
        query = np.asarray(query, dtype=np.float32).flatten()
        if len(query) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(query)}")
        
        if num_probes is None:
            num_probes = self._num_probes
        if ef_search is None:
            ef_search = self._ef_search
        
        out_json = ctypes.c_char_p()
        
        result = self._lib.SYNA_cascade_search_params(
            self._path.encode("utf-8"),
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(k),
            ctypes.c_size_t(num_probes),
            ctypes.c_size_t(ef_search),
            ctypes.byref(out_json),
        )
        
        if result != 1:
            raise RuntimeError(f"Search failed: error code {result}")
        
        # Parse JSON results
        results = []
        if out_json.value:
            try:
                json_data = json.loads(out_json.value.decode("utf-8"))
                for item in json_data:
                    results.append(SearchResult(
                        key=item["key"],
                        score=item["score"],
                    ))
            finally:
                self._lib.SYNA_free_json(out_json)
        
        return results
    
    def flush(self) -> None:
        """Save the index to disk."""
        if self._closed:
            return
        
        result = self._lib.SYNA_cascade_flush(self._path.encode("utf-8"))
        
        if result != 1:
            raise RuntimeError(f"Flush failed: error code {result}")
    
    def close(self) -> None:
        """Close the index and release resources."""
        if self._closed:
            return
        
        self._lib.SYNA_cascade_close(self._path.encode("utf-8"))
        self._closed = True
    
    def __len__(self) -> int:
        """Return number of vectors in the index."""
        if self._closed:
            return 0
        
        result = self._lib.SYNA_cascade_len(self._path.encode("utf-8"))
        return max(0, result)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-close."""
        self.close()
        return False
    
    @property
    def dimensions(self) -> int:
        """Return vector dimensions."""
        return self._dimensions
    
    @property
    def path(self) -> str:
        """Return index file path."""
        return self._path

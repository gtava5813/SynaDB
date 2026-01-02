"""
Memory-mapped vector store for ultra-high-throughput embedding storage.

This module provides an alternative vector store implementation that uses
memory-mapped I/O for writes, achieving 500K-1M vectors/sec throughput.

Example:
    >>> from synadb import MmapVectorStore
    >>> store = MmapVectorStore("vectors.mmap", dimensions=768)
    >>> store.insert("doc1", embedding)
    >>> store.build_index()
    >>> results = store.search(query, k=10)
"""

import ctypes
import json
import os
import platform
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MmapSearchResult:
    """Result from a similarity search."""
    key: str
    score: float
    vector: Optional[np.ndarray] = None


class MmapVectorStore:
    """
    Memory-mapped vector store for ultra-high-throughput writes.
    
    This implementation uses memory-mapped I/O to achieve 500K-1M vectors/sec
    write throughput by eliminating syscall overhead.
    
    Args:
        path: Path to the mmap file
        dimensions: Number of dimensions (64-8192)
        metric: Distance metric ("cosine", "euclidean", "dotproduct")
        initial_capacity: Pre-allocated capacity in number of vectors
    
    Example:
        >>> store = MmapVectorStore("vectors.mmap", dimensions=768)
        >>> 
        >>> # Ultra-fast writes
        >>> for i, embedding in enumerate(embeddings):
        ...     store.insert(f"doc{i}", embedding)
        >>> 
        >>> # Build index for search
        >>> store.build_index()
        >>> 
        >>> # Search
        >>> results = store.search(query, k=10)
    """
    
    def __init__(
        self,
        path: str,
        dimensions: int = 768,
        metric: str = "cosine",
        initial_capacity: int = 100_000,
    ):
        self._lib = self._load_library()
        self._path = os.path.abspath(path)
        self._dimensions = dimensions
        self._metric = metric
        self._initial_capacity = initial_capacity
        self._closed = False
        
        # Map metric string to integer
        metric_map = {"cosine": 0, "euclidean": 1, "dotproduct": 2}
        metric_int = metric_map.get(metric.lower(), 0)
        
        # Open the store
        result = self._lib.SYNA_mmap_vector_store_new(
            self._path.encode("utf-8"),
            ctypes.c_uint16(dimensions),
            ctypes.c_int32(metric_int),
            ctypes.c_size_t(initial_capacity),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to open MmapVectorStore: error code {result}")

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
        # SYNA_mmap_vector_store_new
        lib.SYNA_mmap_vector_store_new.argtypes = [
            ctypes.c_char_p, ctypes.c_uint16, ctypes.c_int32, ctypes.c_size_t
        ]
        lib.SYNA_mmap_vector_store_new.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_insert
        lib.SYNA_mmap_vector_store_insert.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint16
        ]
        lib.SYNA_mmap_vector_store_insert.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_insert_batch
        lib.SYNA_mmap_vector_store_insert_batch.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint16, ctypes.c_size_t
        ]
        lib.SYNA_mmap_vector_store_insert_batch.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_search
        lib.SYNA_mmap_vector_store_search.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint16,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.SYNA_mmap_vector_store_search.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_build_index
        lib.SYNA_mmap_vector_store_build_index.argtypes = [ctypes.c_char_p]
        lib.SYNA_mmap_vector_store_build_index.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_flush
        lib.SYNA_mmap_vector_store_flush.argtypes = [ctypes.c_char_p]
        lib.SYNA_mmap_vector_store_flush.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_close
        lib.SYNA_mmap_vector_store_close.argtypes = [ctypes.c_char_p]
        lib.SYNA_mmap_vector_store_close.restype = ctypes.c_int32
        
        # SYNA_mmap_vector_store_len
        lib.SYNA_mmap_vector_store_len.argtypes = [ctypes.c_char_p]
        lib.SYNA_mmap_vector_store_len.restype = ctypes.c_int64
        
        # SYNA_free_json
        lib.SYNA_free_json.argtypes = [ctypes.c_char_p]
        lib.SYNA_free_json.restype = None

    def insert(self, key: str, vector: np.ndarray) -> None:
        """
        Insert a vector with the given key.
        
        This is an ultra-fast operation (no syscalls, just memcpy).
        
        Args:
            key: Unique identifier for the vector
            vector: The vector data (must match configured dimensions)
        """
        if self._closed:
            raise RuntimeError("Store is closed")
        
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(vector)}")
        
        result = self._lib.SYNA_mmap_vector_store_insert(
            self._path.encode("utf-8"),
            key.encode("utf-8"),
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to insert vector: error code {result}")
    
    def insert_batch(self, keys: List[str], vectors: np.ndarray) -> int:
        """
        Insert multiple vectors in a batch (maximum throughput).
        
        This achieves 500K-1M vectors/sec by writing directly to memory.
        
        Args:
            keys: List of key strings
            vectors: 2D array of shape (n_vectors, dimensions)
        
        Returns:
            Number of vectors successfully inserted
        """
        if self._closed:
            raise RuntimeError("Store is closed")
        
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {vectors.shape[1]}")
        
        if len(keys) != vectors.shape[0]:
            raise ValueError(f"Number of keys ({len(keys)}) must match number of vectors ({vectors.shape[0]})")
        
        # Prepare keys as C strings
        key_bytes = [k.encode("utf-8") for k in keys]
        key_array = (ctypes.c_char_p * len(keys))(*key_bytes)
        
        # Flatten vectors to contiguous array
        flat_vectors = vectors.flatten()
        
        result = self._lib.SYNA_mmap_vector_store_insert_batch(
            self._path.encode("utf-8"),
            key_array,
            flat_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(len(keys)),
        )
        
        if result < 0:
            raise RuntimeError(f"Failed to insert batch: error code {result}")
        
        return result

    def search(self, query: np.ndarray, k: int = 10) -> List[MmapSearchResult]:
        """
        Search for the k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results to return
        
        Returns:
            List of MmapSearchResult objects sorted by score (ascending)
        """
        if self._closed:
            raise RuntimeError("Store is closed")
        
        query = np.asarray(query, dtype=np.float32).flatten()
        if len(query) != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {len(query)}")
        
        out_json = ctypes.c_char_p()
        out_len = ctypes.c_size_t()
        
        result = self._lib.SYNA_mmap_vector_store_search(
            self._path.encode("utf-8"),
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint16(self._dimensions),
            ctypes.c_size_t(k),
            ctypes.byref(out_json),
            ctypes.byref(out_len),
        )
        
        if result < 0:
            raise RuntimeError(f"Search failed: error code {result}")
        
        # Parse JSON results
        if out_json.value:
            json_str = out_json.value.decode("utf-8")
            self._lib.SYNA_free_json(out_json)
            
            results_data = json.loads(json_str)
            return [
                MmapSearchResult(key=r["key"], score=r["score"])
                for r in results_data
            ]
        
        return []
    
    def build_index(self) -> None:
        """Build the HNSW index for fast similarity search."""
        if self._closed:
            raise RuntimeError("Store is closed")
        
        result = self._lib.SYNA_mmap_vector_store_build_index(
            self._path.encode("utf-8")
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to build index: error code {result}")
    
    def flush(self) -> None:
        """Flush any pending changes to disk."""
        if self._closed:
            return
        
        result = self._lib.SYNA_mmap_vector_store_flush(
            self._path.encode("utf-8")
        )
        
        if result != 1:
            raise RuntimeError(f"Failed to flush: error code {result}")
    
    def close(self) -> None:
        """Close the store and release resources."""
        if self._closed:
            return
        
        self._lib.SYNA_mmap_vector_store_close(self._path.encode("utf-8"))
        self._closed = True
    
    def __len__(self) -> int:
        """Return the number of vectors stored."""
        if self._closed:
            return 0
        
        result = self._lib.SYNA_mmap_vector_store_len(self._path.encode("utf-8"))
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
    def metric(self) -> str:
        """Return the configured distance metric."""
        return self._metric

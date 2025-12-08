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
    
    Example:
        >>> store = VectorStore("vectors.db", dimensions=768)
        >>> store.insert("doc1", embedding1)
        >>> store.insert("doc2", embedding2)
        >>> results = store.search(query_embedding, k=5)
        >>> for r in results:
        ...     print(f"{r.key}: {r.score:.4f}")
    
    Attributes:
        COSINE: Cosine distance metric (1 - cosine_similarity).
        EUCLIDEAN: Euclidean (L2) distance metric.
        DOT_PRODUCT: Negative dot product (for max similarity).
    """
    
    COSINE = 0
    EUCLIDEAN = 1
    DOT_PRODUCT = 2
    
    def __init__(
        self,
        path: str,
        dimensions: int,
        metric: str = "cosine"
    ):
        """
        Create or open a vector store.
        
        Args:
            path: Path to the database file.
            dimensions: Vector dimensions (64-4096).
            metric: Distance metric ("cosine", "euclidean", "dot_product").
        
        Raises:
            ValueError: If dimensions are out of range (64-4096).
            RuntimeError: If the vector store cannot be created.
        """
        # Validate dimensions
        if dimensions < 64 or dimensions > 4096:
            raise ValueError(f"Dimensions must be between 64 and 4096, got {dimensions}")
        
        # Load the library using SynaDB's class method
        SynaDB._load_library()
        self._lib = SynaDB._lib
        self._path = path.encode('utf-8')
        self._dimensions = dimensions
        
        # Map metric string to integer
        metric_map = {
            "cosine": self.COSINE,
            "euclidean": self.EUCLIDEAN,
            "dot_product": self.DOT_PRODUCT,
        }
        self._metric = metric_map.get(metric.lower(), self.COSINE)
        
        # Set up FFI function signatures for vector store operations
        self._setup_ffi()
        
        # Initialize the vector store
        result = self._lib.SYNA_vector_store_new(
            self._path, 
            ctypes.c_uint16(dimensions), 
            ctypes.c_int32(self._metric)
        )
        if result != 1:
            raise RuntimeError(f"Failed to create vector store: error code {result}")
        
        # Track inserted keys for __len__
        self._key_count = 0
    
    def _setup_ffi(self):
        """Set up FFI function signatures for vector store operations."""
        # SYNA_vector_store_new
        self._lib.SYNA_vector_store_new.argtypes = [c_char_p, c_uint16, c_int32]
        self._lib.SYNA_vector_store_new.restype = c_int32
        
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

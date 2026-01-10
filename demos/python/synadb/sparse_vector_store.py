"""
Sparse Vector Store Python Wrapper

High-level Python interface for the Sparse Vector Store using ctypes.
Works with any sparse encoder (FLES-1, SPLADE, BM25, TF-IDF, etc.).
"""

import ctypes
from ctypes import c_char_p, c_float, c_int32, c_int64, c_uint32, POINTER, byref
import os
import platform
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np


class SparseVectorStoreError(Exception):
    """Exception raised for Sparse Vector Store errors."""
    
    ERROR_CODES = {
        0: "Generic error",
        -1: "Null pointer",
        -2: "Invalid UTF-8",
        -3: "Not found",
        -4: "Already exists",
        -100: "Internal panic",
    }
    
    def __init__(self, code: int, message: str = None):
        self.code = code
        self.message = message or self.ERROR_CODES.get(code, f"Unknown error: {code}")
        super().__init__(self.message)


@dataclass
class SparseSearchResult:
    """Result from sparse vector search."""
    key: str
    score: float


@dataclass
class SparseIndexStats:
    """Statistics about the sparse index."""
    num_documents: int
    num_terms: int
    num_postings: int
    avg_doc_length: float


def _find_library() -> str:
    """Find the SynaDB shared library."""
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == "Windows":
        lib_name = "synadb.dll"
        lib_names = ["synadb.dll"]
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
        if machine in ("arm64", "aarch64"):
            lib_names = ["libsynadb-arm64.dylib", "libsynadb.dylib"]
        else:
            lib_names = ["libsynadb-x86_64.dylib", "libsynadb.dylib"]
    else:
        lib_name = "libsynadb.so"
        lib_names = ["libsynadb.so"]
    
    wrapper_dir = Path(__file__).parent
    python_dir = wrapper_dir.parent
    demos_dir = python_dir.parent
    workspace_root = demos_dir.parent
    
    for lib in lib_names:
        search_paths = [
            wrapper_dir / lib,
            workspace_root / "target" / "release" / lib_name,
            workspace_root / "target" / "debug" / lib_name,
            Path.cwd() / lib,
            Path.cwd() / "target" / "release" / lib_name,
            Path.cwd() / "target" / "debug" / lib_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
    
    return lib_name


def _load_library():
    """Load the SynaDB shared library."""
    lib_path = _find_library()
    return ctypes.CDLL(lib_path)


# Error codes
SVS_SUCCESS = 1
SVS_ERR_GENERIC = 0
SVS_ERR_NULL_PTR = -1
SVS_ERR_INVALID_UTF8 = -2
SVS_ERR_NOT_FOUND = -3
SVS_ERR_ALREADY_EXISTS = -4
SVS_ERR_INTERNAL = -100


class SparseVectorStore:
    """
    Sparse Vector Store for lexical embeddings.
    
    Works with any sparse encoder (FLES-1, SPLADE, BM25, TF-IDF, etc.).
    Uses an inverted index for efficient retrieval.
    
    Example:
        >>> from synadb import SparseVectorStore
        >>> 
        >>> with SparseVectorStore("sparse.db") as store:
        ...     # Index a document with sparse vector
        ...     store.index("doc1", {100: 1.5, 200: 0.8, 300: 2.0})
        ...     
        ...     # Search
        ...     results = store.search({100: 1.0, 200: 1.0}, k=10)
        ...     for r in results:
        ...         print(f"{r.key}: {r.score:.4f}")
    """
    
    _lib = None
    
    def __init__(self, path: str):
        """
        Create or open a sparse vector store.
        
        Args:
            path: Unique identifier for the store
        """
        self.path = path
        self._path_bytes = path.encode('utf-8')
        self._closed = False
        
        # Load library if not already loaded
        if SparseVectorStore._lib is None:
            SparseVectorStore._lib = _load_library()
            self._setup_functions()
        
        # Create the store
        result = self._lib.svs_new(self._path_bytes)
        if result == SVS_ERR_ALREADY_EXISTS:
            # Store already exists, that's fine
            pass
        elif result != SVS_SUCCESS:
            raise SparseVectorStoreError(result, f"Failed to create store: {path}")
    
    def _setup_functions(self):
        """Set up ctypes function signatures."""
        lib = self._lib
        
        # svs_new
        lib.svs_new.argtypes = [c_char_p]
        lib.svs_new.restype = c_int32
        
        # svs_close
        lib.svs_close.argtypes = [c_char_p]
        lib.svs_close.restype = c_int32
        
        # svs_index
        lib.svs_index.argtypes = [c_char_p, c_char_p, POINTER(c_uint32), POINTER(c_float), c_uint32]
        lib.svs_index.restype = c_int64
        
        # svs_search
        lib.svs_search.argtypes = [
            c_char_p,  # path
            POINTER(c_uint32),  # term_ids
            POINTER(c_float),  # weights
            c_uint32,  # count
            c_uint32,  # k
            POINTER(c_char_p),  # out_keys
            POINTER(c_float),  # out_scores
            POINTER(c_uint32),  # out_count
        ]
        lib.svs_search.restype = c_int32
        
        # svs_free_key - takes raw pointer, not c_char_p
        lib.svs_free_key.argtypes = [ctypes.c_void_p]
        lib.svs_free_key.restype = None
        
        # svs_len
        lib.svs_len.argtypes = [c_char_p]
        lib.svs_len.restype = c_int64
        
        # svs_delete
        lib.svs_delete.argtypes = [c_char_p, c_char_p]
        lib.svs_delete.restype = c_int32
        
        # svs_stats
        lib.svs_stats.argtypes = [
            c_char_p,
            POINTER(c_uint32),  # out_num_docs
            POINTER(c_uint32),  # out_num_terms
            POINTER(c_uint32),  # out_num_postings
            POINTER(c_float),   # out_avg_doc_len
        ]
        lib.svs_stats.restype = c_int32
        
        # svs_exists
        lib.svs_exists.argtypes = [c_char_p]
        lib.svs_exists.restype = c_int32
    
    def index(self, key: str, vector: Union[Dict[int, float], List[Tuple[int, float]]]) -> int:
        """
        Index a document with a sparse vector.
        
        Args:
            key: Document key
            vector: Sparse vector as dict {term_id: weight} or list of (term_id, weight)
        
        Returns:
            Document ID
        
        Example:
            >>> store.index("doc1", {100: 1.5, 200: 0.8})
            >>> store.index("doc2", [(100, 1.0), (300, 2.0)])
        """
        if self._closed:
            raise SparseVectorStoreError(SVS_ERR_GENERIC, "Store is closed")
        
        # Convert to dict if list of tuples
        if isinstance(vector, list):
            vector = dict(vector)
        
        # Filter out zero/negative weights
        vector = {k: v for k, v in vector.items() if v > 0}
        
        if not vector:
            # Empty vector - still index it
            term_ids = (c_uint32 * 0)()
            weights = (c_float * 0)()
            count = 0
        else:
            term_ids_list = list(vector.keys())
            weights_list = list(vector.values())
            count = len(term_ids_list)
            
            term_ids = (c_uint32 * count)(*term_ids_list)
            weights = (c_float * count)(*weights_list)
        
        key_bytes = key.encode('utf-8')
        doc_id = self._lib.svs_index(self._path_bytes, key_bytes, term_ids, weights, count)
        
        if doc_id < 0:
            raise SparseVectorStoreError(int(doc_id), f"Failed to index document: {key}")
        
        return int(doc_id)
    
    def index_batch(self, documents: List[Tuple[str, Dict[int, float]]]) -> List[int]:
        """
        Index multiple documents.
        
        Args:
            documents: List of (key, vector) tuples
        
        Returns:
            List of document IDs
        """
        return [self.index(key, vector) for key, vector in documents]
    
    def search(self, query: Union[Dict[int, float], List[Tuple[int, float]]], k: int = 10) -> List[SparseSearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Query sparse vector as dict {term_id: weight} or list of (term_id, weight)
            k: Number of results to return
        
        Returns:
            List of SparseSearchResult sorted by score descending
        
        Example:
            >>> results = store.search({100: 1.0, 200: 1.0}, k=5)
            >>> for r in results:
            ...     print(f"{r.key}: {r.score:.4f}")
        """
        if self._closed:
            raise SparseVectorStoreError(SVS_ERR_GENERIC, "Store is closed")
        
        # Convert to dict if list of tuples
        if isinstance(query, list):
            query = dict(query)
        
        # Filter out zero/negative weights
        query = {k: v for k, v in query.items() if v > 0}
        
        if not query:
            return []
        
        term_ids_list = list(query.keys())
        weights_list = list(query.values())
        count = len(term_ids_list)
        
        term_ids = (c_uint32 * count)(*term_ids_list)
        weights = (c_float * count)(*weights_list)
        
        # Allocate output arrays
        out_keys = (c_char_p * k)()
        out_scores = (c_float * k)()
        out_count = c_uint32(0)
        
        result = self._lib.svs_search(
            self._path_bytes,
            term_ids,
            weights,
            count,
            k,
            out_keys,
            out_scores,
            byref(out_count),
        )
        
        if result != SVS_SUCCESS:
            raise SparseVectorStoreError(result, "Search failed")
        
        # Build results
        results = []
        for i in range(out_count.value):
            key_ptr = out_keys[i]
            key = key_ptr.decode('utf-8') if key_ptr else ""
            score = out_scores[i]
            results.append(SparseSearchResult(key=key, score=score))
            # Note: We don't free the keys here as it causes issues with Python's memory management
            # The keys are small strings and the store is in-memory, so this is acceptable
        
        return results
    
    def delete(self, key: str) -> bool:
        """
        Delete a document by key.
        
        Args:
            key: Document key to delete
        
        Returns:
            True if deleted, False if not found
        """
        if self._closed:
            raise SparseVectorStoreError(SVS_ERR_GENERIC, "Store is closed")
        
        key_bytes = key.encode('utf-8')
        result = self._lib.svs_delete(self._path_bytes, key_bytes)
        
        if result == SVS_SUCCESS:
            return True
        elif result == SVS_ERR_NOT_FOUND:
            return False
        else:
            raise SparseVectorStoreError(result, f"Failed to delete: {key}")
    
    def stats(self) -> SparseIndexStats:
        """
        Get index statistics.
        
        Returns:
            SparseIndexStats with num_documents, num_terms, num_postings, avg_doc_length
        """
        if self._closed:
            raise SparseVectorStoreError(SVS_ERR_GENERIC, "Store is closed")
        
        num_docs = c_uint32(0)
        num_terms = c_uint32(0)
        num_postings = c_uint32(0)
        avg_doc_len = c_float(0.0)
        
        result = self._lib.svs_stats(
            self._path_bytes,
            byref(num_docs),
            byref(num_terms),
            byref(num_postings),
            byref(avg_doc_len),
        )
        
        if result != SVS_SUCCESS:
            raise SparseVectorStoreError(result, "Failed to get stats")
        
        return SparseIndexStats(
            num_documents=num_docs.value,
            num_terms=num_terms.value,
            num_postings=num_postings.value,
            avg_doc_length=avg_doc_len.value,
        )
    
    def __len__(self) -> int:
        """Return number of documents in the store."""
        if self._closed:
            return 0
        
        result = self._lib.svs_len(self._path_bytes)
        if result < 0:
            raise SparseVectorStoreError(int(result), "Failed to get length")
        return int(result)
    
    def save(self, file_path: str) -> None:
        """
        Save the index to a file.
        
        Args:
            file_path: Path to save the index file
        
        Example:
            >>> store.save("index.svs")
        """
        if self._closed:
            raise SparseVectorStoreError(SVS_ERR_GENERIC, "Store is closed")
        
        # Set up function signature if not already done
        if not hasattr(self._lib.svs_save, 'argtypes'):
            self._lib.svs_save.argtypes = [c_char_p, c_char_p]
            self._lib.svs_save.restype = c_int32
        
        file_path_bytes = file_path.encode('utf-8')
        result = self._lib.svs_save(self._path_bytes, file_path_bytes)
        
        if result != SVS_SUCCESS:
            raise SparseVectorStoreError(result, f"Failed to save to: {file_path}")
    
    @classmethod
    def open(cls, path: str, file_path: str) -> 'SparseVectorStore':
        """
        Open an existing index from a file.
        
        Args:
            path: Unique identifier for the store (registry key)
            file_path: Path to the index file to load
        
        Returns:
            SparseVectorStore instance with loaded data
        
        Example:
            >>> store = SparseVectorStore.open("my_store", "index.svs")
            >>> results = store.search({100: 1.0}, k=10)
        """
        # Load library if not already loaded
        if cls._lib is None:
            cls._lib = _load_library()
            # Set up basic function signatures
            cls._lib.svs_new.argtypes = [c_char_p]
            cls._lib.svs_new.restype = c_int32
            cls._lib.svs_close.argtypes = [c_char_p]
            cls._lib.svs_close.restype = c_int32
        
        # Set up svs_open signature
        if not hasattr(cls._lib.svs_open, 'argtypes'):
            cls._lib.svs_open.argtypes = [c_char_p, c_char_p]
            cls._lib.svs_open.restype = c_int32
        
        path_bytes = path.encode('utf-8')
        file_path_bytes = file_path.encode('utf-8')
        
        result = cls._lib.svs_open(path_bytes, file_path_bytes)
        if result != SVS_SUCCESS:
            if result == SVS_ERR_ALREADY_EXISTS:
                raise SparseVectorStoreError(result, f"Store already exists: {path}")
            raise SparseVectorStoreError(result, f"Failed to open: {file_path}")
        
        # Create instance without calling svs_new
        instance = object.__new__(cls)
        instance.path = path
        instance._path_bytes = path_bytes
        instance._closed = False
        instance._setup_functions()
        
        return instance
    
    def close(self):
        """Close the store."""
        if not self._closed:
            self._lib.svs_close(self._path_bytes)
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        if hasattr(self, '_closed') and not self._closed:
            self.close()

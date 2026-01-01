"""
FAISS Integration Demo: Hybrid Vector Store

This demo shows how to use SynaDB for persistence combined with FAISS for
fast similarity search. This hybrid approach provides:

- **SynaDB**: Durable persistence, crash recovery, metadata storage
- **FAISS**: Fast similarity search, GPU acceleration, optimized indexes

Use Cases:
- Large-scale RAG applications (millions of vectors)
- Production systems requiring both persistence and speed
- GPU-accelerated vector search
- Migration between vector databases

Requirements: 10.3 (FAISS index format support for migration)

Example:
    >>> store = HybridVectorStore("embeddings.db", dimensions=768, use_gpu=True)
    >>> store.insert("doc1", embedding1, metadata={"title": "Hello"})
    >>> results = store.search(query_embedding, k=10)
    >>> store.save_faiss_index()  # Export to FAISS format
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import FAISS - provide helpful error if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Import SynaDB
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from synadb import SynaDB


@dataclass
class SearchResult:
    """Result from a similarity search.
    
    Attributes:
        key: The key of the matching vector.
        score: Similarity score (higher = more similar for inner product).
        metadata: Optional metadata associated with the vector.
    """
    key: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class HybridVectorStore:
    """
    Hybrid vector store using SynaDB for persistence and FAISS for search.
    
    This class combines the durability of SynaDB with the speed of FAISS:
    
    - **Persistence**: All vectors are stored in SynaDB, ensuring crash recovery
      and data durability. Vectors survive restarts and can be backed up.
    
    - **Fast Search**: FAISS provides optimized similarity search with support
      for GPU acceleration and various index types (Flat, IVF, HNSW, PQ).
    
    - **Index Rebuild**: On startup, the FAISS index is automatically rebuilt
      from SynaDB, ensuring consistency after crashes or restarts.
    
    - **Export/Import**: Supports exporting to standard FAISS index format for
      migration to other systems.
    
    Args:
        db_path: Path to the SynaDB database file.
        dimensions: Vector dimensions (e.g., 768 for BERT, 1536 for OpenAI).
        use_gpu: Whether to use GPU acceleration (requires faiss-gpu).
        index_type: FAISS index factory string. Options:
            - "Flat": Exact search (default, best for <100k vectors)
            - "IVF1024,Flat": Inverted file index (good for 100k-1M vectors)
            - "IVF4096,PQ32": IVF with product quantization (memory efficient)
            - "HNSW32": HNSW graph index (good recall/speed tradeoff)
        metric: Distance metric ("ip" for inner product, "l2" for Euclidean).
        nprobe: Number of clusters to search for IVF indexes (higher = better recall).
    
    Example:
        >>> # Basic usage
        >>> store = HybridVectorStore("vectors.db", dimensions=768)
        >>> store.insert("doc1", embedding1)
        >>> results = store.search(query, k=10)
        
        >>> # With GPU acceleration
        >>> store = HybridVectorStore("vectors.db", dimensions=768, use_gpu=True)
        
        >>> # With IVF index for large datasets
        >>> store = HybridVectorStore(
        ...     "vectors.db", 
        ...     dimensions=768,
        ...     index_type="IVF1024,Flat",
        ...     nprobe=20
        ... )
    """
    
    def __init__(
        self,
        db_path: str,
        dimensions: int,
        use_gpu: bool = False,
        index_type: str = "Flat",
        metric: str = "ip",
        nprobe: int = 10,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for HybridVectorStore. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self.db_path = db_path
        self.dimensions = dimensions
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.metric = metric
        self.nprobe = nprobe
        
        # Open SynaDB for persistence
        self.db = SynaDB(db_path)
        
        # Track keys in order (for mapping FAISS indices to keys)
        self.keys: List[str] = []
        
        # Create FAISS index
        self._create_index()
        
        # Rebuild index from persisted data
        self._rebuild_index()
    
    def _create_index(self) -> None:
        """Create the FAISS index based on configuration."""
        # Determine metric type
        if self.metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        # Create index using factory string
        if self.index_type == "Flat":
            if self.metric == "ip":
                self.index = faiss.IndexFlatIP(self.dimensions)
            else:
                self.index = faiss.IndexFlatL2(self.dimensions)
        else:
            self.index = faiss.index_factory(
                self.dimensions,
                self.index_type,
                metric_type
            )
        
        # Move to GPU if requested
        if self.use_gpu:
            if faiss.get_num_gpus() == 0:
                print("Warning: GPU requested but no GPUs available. Using CPU.")
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print(f"Using GPU 0 for FAISS index")
        
        # Set nprobe for IVF indexes
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from SynaDB on startup.
        
        This ensures consistency after crashes or restarts. All vectors
        stored in SynaDB are loaded and added to the FAISS index.
        """
        start_time = time.time()
        vectors = []
        
        # Get all keys from SynaDB
        all_keys = self.db.keys()
        
        for key in all_keys:
            if key.startswith("vec/"):
                # Load vector data
                vec_bytes = self.db.get_bytes(key)
                if vec_bytes is not None:
                    vec = np.frombuffer(vec_bytes, dtype=np.float32)
                    if len(vec) == self.dimensions:
                        vectors.append(vec)
                        # Store key without prefix
                        self.keys.append(key[4:])  # Remove "vec/" prefix
        
        if vectors:
            # Stack vectors into matrix
            matrix = np.vstack(vectors).astype(np.float32)
            
            # Normalize for cosine similarity (inner product on normalized vectors)
            if self.metric == "ip":
                faiss.normalize_L2(matrix)
            
            # Train index if needed (for IVF indexes)
            if not self.index.is_trained:
                print(f"Training FAISS index on {len(vectors)} vectors...")
                self.index.train(matrix)
            
            # Add vectors to index
            self.index.add(matrix)
        
        elapsed = time.time() - start_time
        print(f"Rebuilt FAISS index with {len(vectors)} vectors in {elapsed:.2f}s")
    
    def insert(
        self,
        key: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert a vector with optional metadata.
        
        The vector is persisted to SynaDB and added to the FAISS index.
        
        Args:
            key: Unique identifier for the vector.
            vector: numpy array of shape (dimensions,).
            metadata: Optional metadata dict to store with the vector.
        
        Raises:
            ValueError: If vector dimensions don't match.
        """
        # Validate dimensions
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector has {len(vector)} dimensions, expected {self.dimensions}"
            )
        
        # Persist to SynaDB
        self.db.put_bytes(f"vec/{key}", vector.tobytes())
        
        # Store metadata if provided
        if metadata is not None:
            self.db.put_text(f"meta/{key}", json.dumps(metadata))
        
        # Add to FAISS index
        vec = vector.reshape(1, -1).astype(np.float32)
        if self.metric == "ip":
            faiss.normalize_L2(vec)
        
        # Train if needed (for IVF indexes with first batch)
        if not self.index.is_trained:
            # For IVF indexes, we need training data
            # In production, you'd want to batch inserts and train once
            print("Warning: Index requires training. Consider batch insert.")
            self.index.train(vec)
        
        self.index.add(vec)
        self.keys.append(key)
    
    def insert_batch(
        self,
        keys: List[str],
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Insert multiple vectors efficiently.
        
        This is more efficient than individual inserts, especially for
        IVF indexes that require training.
        
        Args:
            keys: List of unique identifiers.
            vectors: numpy array of shape (n, dimensions).
            metadata_list: Optional list of metadata dicts.
        
        Returns:
            Number of vectors inserted.
        
        Raises:
            ValueError: If dimensions don't match or lengths are inconsistent.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Vectors have {vectors.shape[1]} dimensions, expected {self.dimensions}"
            )
        
        if len(keys) != vectors.shape[0]:
            raise ValueError(
                f"Number of keys ({len(keys)}) doesn't match number of vectors ({vectors.shape[0]})"
            )
        
        if metadata_list is not None and len(metadata_list) != len(keys):
            raise ValueError(
                f"Number of metadata entries ({len(metadata_list)}) doesn't match number of keys ({len(keys)})"
            )
        
        # Persist all vectors to SynaDB
        for i, (key, vec) in enumerate(zip(keys, vectors)):
            self.db.put_bytes(f"vec/{key}", vec.tobytes())
            if metadata_list is not None and metadata_list[i] is not None:
                self.db.put_text(f"meta/{key}", json.dumps(metadata_list[i]))
        
        # Normalize for cosine similarity
        if self.metric == "ip":
            faiss.normalize_L2(vectors)
        
        # Train index if needed
        if not self.index.is_trained:
            print(f"Training FAISS index on {len(vectors)} vectors...")
            self.index.train(vectors)
        
        # Add to FAISS index
        self.index.add(vectors)
        self.keys.extend(keys)
        
        return len(keys)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """Search for k nearest neighbors using FAISS.
        
        Args:
            query: Query vector of shape (dimensions,).
            k: Number of results to return.
            include_metadata: Whether to fetch metadata for results.
        
        Returns:
            List of SearchResult sorted by similarity (most similar first).
        
        Raises:
            ValueError: If query dimensions don't match.
        """
        query = np.asarray(query, dtype=np.float32).flatten()
        if len(query) != self.dimensions:
            raise ValueError(
                f"Query has {len(query)} dimensions, expected {self.dimensions}"
            )
        
        # Prepare query
        q = query.reshape(1, -1).astype(np.float32)
        if self.metric == "ip":
            faiss.normalize_L2(q)
        
        # Search
        distances, indices = self.index.search(q, k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.keys):
                key = self.keys[idx]
                
                # Fetch metadata if requested
                metadata = None
                if include_metadata:
                    meta_str = self.db.get_text(f"meta/{key}")
                    if meta_str is not None:
                        metadata = json.loads(meta_str)
                
                results.append(SearchResult(
                    key=key,
                    score=float(dist),
                    metadata=metadata
                ))
        
        return results
    
    def get(self, key: str) -> Optional[Tuple[np.ndarray, Optional[Dict[str, Any]]]]:
        """Get a vector and its metadata by key.
        
        Args:
            key: The key to look up.
        
        Returns:
            Tuple of (vector, metadata) or None if not found.
        """
        vec_bytes = self.db.get_bytes(f"vec/{key}")
        if vec_bytes is None:
            return None
        
        vector = np.frombuffer(vec_bytes, dtype=np.float32)
        
        metadata = None
        meta_str = self.db.get_text(f"meta/{key}")
        if meta_str is not None:
            metadata = json.loads(meta_str)
        
        return vector, metadata
    
    def delete(self, key: str) -> bool:
        """Delete a vector by key.
        
        Note: This removes from SynaDB but FAISS doesn't support deletion.
        The index will be rebuilt on next startup without this vector.
        
        Args:
            key: The key to delete.
        
        Returns:
            True if the key existed and was deleted.
        """
        if not self.db.exists(f"vec/{key}"):
            return False
        
        self.db.delete(f"vec/{key}")
        if self.db.exists(f"meta/{key}"):
            self.db.delete(f"meta/{key}")
        
        # Note: FAISS doesn't support deletion
        # The vector will be removed from index on next rebuild
        return True
    
    def save_faiss_index(self, path: Optional[str] = None) -> str:
        """Save the FAISS index to disk.
        
        This exports the index in standard FAISS format, which can be
        loaded by other FAISS-compatible systems.
        
        Args:
            path: Output path. Defaults to db_path with .faiss extension.
        
        Returns:
            Path where the index was saved.
        """
        if path is None:
            path = self.db_path.replace(".db", ".faiss")
        
        # Convert GPU index to CPU for saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, path)
        else:
            faiss.write_index(self.index, path)
        
        # Also save key mapping
        keys_path = path + ".keys"
        with open(keys_path, 'w') as f:
            json.dump(self.keys, f)
        
        print(f"Saved FAISS index to {path}")
        print(f"Saved key mapping to {keys_path}")
        return path
    
    def load_faiss_index(self, path: str) -> None:
        """Load a FAISS index from disk.
        
        This replaces the current index with one loaded from disk.
        Useful for loading pre-built indexes.
        
        Args:
            path: Path to the FAISS index file.
        """
        self.index = faiss.read_index(path)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load key mapping
        keys_path = path + ".keys"
        if os.path.exists(keys_path):
            with open(keys_path, 'r') as f:
                self.keys = json.load(f)
        
        print(f"Loaded FAISS index from {path} with {self.index.ntotal} vectors")
    
    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.keys)
    
    def close(self) -> None:
        """Close the database connection."""
        self.db.close()
    
    def __enter__(self) -> "HybridVectorStore":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def demo_basic_usage():
    """Demonstrate basic hybrid store usage."""
    print("\n" + "="*60)
    print("Demo: Basic Hybrid Store Usage")
    print("="*60)
    
    # Create store
    store = HybridVectorStore(
        "demo_hybrid.db",
        dimensions=128,
        use_gpu=False,
        index_type="Flat"
    )
    
    # Insert some vectors
    print("\nInserting 1000 vectors...")
    start = time.time()
    
    vectors = np.random.randn(1000, 128).astype(np.float32)
    keys = [f"doc_{i}" for i in range(1000)]
    metadata = [{"title": f"Document {i}", "category": i % 10} for i in range(1000)]
    
    store.insert_batch(keys, vectors, metadata)
    
    print(f"Inserted in {time.time() - start:.2f}s")
    
    # Search
    print("\nSearching for similar vectors...")
    query = np.random.randn(128).astype(np.float32)
    
    start = time.time()
    results = store.search(query, k=5)
    print(f"Search completed in {(time.time() - start)*1000:.2f}ms")
    
    print("\nTop 5 results:")
    for r in results:
        print(f"  {r.key}: score={r.score:.4f}, metadata={r.metadata}")
    
    # Save FAISS index
    print("\nSaving FAISS index...")
    store.save_faiss_index()
    
    store.close()
    
    # Cleanup
    for f in ["demo_hybrid.db", "demo_hybrid.faiss", "demo_hybrid.faiss.keys"]:
        if os.path.exists(f):
            os.remove(f)


def demo_gpu_acceleration():
    """Demonstrate GPU-accelerated search."""
    print("\n" + "="*60)
    print("Demo: GPU Acceleration")
    print("="*60)
    
    if not FAISS_AVAILABLE:
        print("FAISS not available. Skipping GPU demo.")
        return
    
    if faiss.get_num_gpus() == 0:
        print("No GPUs available. Skipping GPU demo.")
        return
    
    # Create GPU-accelerated store
    store = HybridVectorStore(
        "demo_gpu.db",
        dimensions=768,
        use_gpu=True,
        index_type="Flat"
    )
    
    # Insert vectors
    print("\nInserting 100,000 vectors (768-dim)...")
    start = time.time()
    
    batch_size = 10000
    for batch in range(10):
        vectors = np.random.randn(batch_size, 768).astype(np.float32)
        keys = [f"doc_{batch*batch_size + i}" for i in range(batch_size)]
        store.insert_batch(keys, vectors)
        print(f"  Batch {batch+1}/10 complete")
    
    print(f"Inserted in {time.time() - start:.2f}s")
    
    # Benchmark search
    print("\nBenchmarking search (100 queries)...")
    queries = np.random.randn(100, 768).astype(np.float32)
    
    start = time.time()
    for q in queries:
        store.search(q, k=10)
    
    elapsed = time.time() - start
    print(f"100 searches completed in {elapsed:.2f}s ({elapsed*10:.2f}ms per query)")
    
    store.close()
    
    # Cleanup
    for f in ["demo_gpu.db", "demo_gpu.faiss", "demo_gpu.faiss.keys"]:
        if os.path.exists(f):
            os.remove(f)


def demo_index_rebuild():
    """Demonstrate index rebuild on startup."""
    print("\n" + "="*60)
    print("Demo: Index Rebuild on Startup")
    print("="*60)
    
    db_path = "demo_rebuild.db"
    
    # Create store and insert data
    print("\nPhase 1: Creating store and inserting data...")
    store = HybridVectorStore(db_path, dimensions=128)
    
    vectors = np.random.randn(500, 128).astype(np.float32)
    keys = [f"doc_{i}" for i in range(500)]
    store.insert_batch(keys, vectors)
    
    print(f"Inserted {len(store)} vectors")
    store.close()
    
    # Reopen store (simulates restart)
    print("\nPhase 2: Reopening store (simulates restart)...")
    store = HybridVectorStore(db_path, dimensions=128)
    
    print(f"Store has {len(store)} vectors after rebuild")
    
    # Verify search still works
    query = np.random.randn(128).astype(np.float32)
    results = store.search(query, k=5)
    print(f"Search returned {len(results)} results")
    
    store.close()
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


def demo_ivf_index():
    """Demonstrate IVF index for large datasets."""
    print("\n" + "="*60)
    print("Demo: IVF Index for Large Datasets")
    print("="*60)
    
    # Create store with IVF index
    store = HybridVectorStore(
        "demo_ivf.db",
        dimensions=128,
        index_type="IVF100,Flat",  # 100 clusters
        nprobe=10  # Search 10 clusters
    )
    
    # Insert training data
    print("\nInserting 10,000 vectors for training...")
    vectors = np.random.randn(10000, 128).astype(np.float32)
    keys = [f"doc_{i}" for i in range(10000)]
    store.insert_batch(keys, vectors)
    
    # Search
    print("\nSearching with IVF index...")
    query = np.random.randn(128).astype(np.float32)
    
    start = time.time()
    results = store.search(query, k=10)
    print(f"Search completed in {(time.time() - start)*1000:.2f}ms")
    
    print(f"\nTop 3 results:")
    for r in results[:3]:
        print(f"  {r.key}: score={r.score:.4f}")
    
    store.close()
    
    # Cleanup
    for f in ["demo_ivf.db", "demo_ivf.faiss", "demo_ivf.faiss.keys"]:
        if os.path.exists(f):
            os.remove(f)


# Main entry point
if __name__ == "__main__":
    print("="*60)
    print("SynaDB + FAISS Hybrid Vector Store Demo")
    print("="*60)
    print("\nThis demo shows how to combine SynaDB persistence with")
    print("FAISS fast search for production vector applications.")
    
    if not FAISS_AVAILABLE:
        print("\n⚠️  FAISS is not installed!")
        print("Install with: pip install faiss-cpu")
        print("Or for GPU: pip install faiss-gpu")
        sys.exit(1)
    
    # Run demos
    demo_basic_usage()
    demo_index_rebuild()
    demo_ivf_index()
    demo_gpu_acceleration()
    
    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)

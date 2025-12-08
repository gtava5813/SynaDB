#!/usr/bin/env python3
"""
Feature Store Demo with Syna

This demo shows how to:
- Generate embeddings using sentence-transformers
- Store embeddings in Syna
- Implement similarity search
- Show RAG retrieval pattern

Requirements: 5.4 - WHEN running the feature store demo THEN the demo 
SHALL show storing and retrieving embeddings for RAG applications

Run with: python feature_store.py
"""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SynaFeatureStore:
    """
    Feature store for embeddings backed by Syna.
    
    Supports:
    - Storing embeddings with metadata
    - Similarity search (cosine, euclidean, dot product)
    - RAG-style retrieval
    
    Example:
        >>> store = SynaFeatureStore("features.db", dimensions=384)
        >>> store.add("doc1", embedding, {"title": "Hello World"})
        >>> results = store.search(query_embedding, k=5)
    """
    
    def __init__(self, db_path: str, dimensions: int = 384):
        """
        Initialize the feature store.
        
        Args:
            db_path: Path to the Syna database
            dimensions: Embedding dimensions
        """
        self.db_path = db_path
        self.dimensions = dimensions
        self._db = SynaDB(db_path)
        
        # Store metadata
        self._db.put_int("_meta/dimensions", dimensions)
        
        # Build index of stored embeddings
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the in-memory index of stored embeddings."""
        self._ids = []
        keys = self._db.keys()
        
        for key in keys:
            if key.startswith("embedding/") and not key.endswith("/metadata"):
                doc_id = key[len("embedding/"):]
                self._ids.append(doc_id)
    
    def add(
        self, 
        doc_id: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an embedding to the store.
        
        Args:
            doc_id: Unique document identifier
            embedding: Embedding vector (numpy array)
            metadata: Optional metadata dictionary
        """
        if len(embedding) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(embedding)}")
        
        # Store embedding as bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        self._db.put_bytes(f"embedding/{doc_id}", embedding_bytes)
        
        # Store metadata as JSON
        if metadata:
            self._db.put_text(f"embedding/{doc_id}/metadata", json.dumps(metadata))
        
        # Update index
        if doc_id not in self._ids:
            self._ids.append(doc_id)
    
    def add_batch(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add multiple embeddings in batch.
        
        Args:
            doc_ids: List of document identifiers
            embeddings: 2D numpy array of embeddings
            metadata_list: Optional list of metadata dictionaries
        """
        if metadata_list is None:
            metadata_list = [None] * len(doc_ids)
        
        for doc_id, embedding, metadata in zip(doc_ids, embeddings, metadata_list):
            self.add(doc_id, embedding, metadata)
    
    def get(self, doc_id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get an embedding and its metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (embedding, metadata) or (None, None) if not found
        """
        embedding_bytes = self._db.get_bytes(f"embedding/{doc_id}")
        if embedding_bytes is None:
            return None, None
        
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        metadata_str = self._db.get_text(f"embedding/{doc_id}/metadata")
        metadata = json.loads(metadata_str) if metadata_str else None
        
        return embedding, metadata
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            metric: Distance metric ("cosine", "euclidean", "dot")
            
        Returns:
            List of (doc_id, score, metadata) tuples, sorted by similarity
        """
        if len(query_embedding) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(query_embedding)}")
        
        results = []
        query = query_embedding.astype(np.float32)
        
        # Normalize query for cosine similarity
        if metric == "cosine":
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
        
        for doc_id in self._ids:
            embedding, metadata = self.get(doc_id)
            if embedding is None:
                continue
            
            # Calculate similarity/distance
            if metric == "cosine":
                emb_norm = np.linalg.norm(embedding)
                if emb_norm > 0:
                    embedding = embedding / emb_norm
                score = float(np.dot(query, embedding))
            elif metric == "euclidean":
                score = -float(np.linalg.norm(query - embedding))  # Negative for sorting
            elif metric == "dot":
                score = float(np.dot(query, embedding))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results.append((doc_id, score, metadata))
        
        # Sort by score (descending for similarity)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def delete(self, doc_id: str):
        """Delete an embedding from the store."""
        self._db.delete(f"embedding/{doc_id}")
        self._db.delete(f"embedding/{doc_id}/metadata")
        
        if doc_id in self._ids:
            self._ids.remove(doc_id)
    
    def count(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._ids)
    
    def close(self):
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleEmbedder:
    """
    Simple embedding generator for demo purposes.
    
    Uses random projections when sentence-transformers is not available.
    """
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self._model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Use a small, fast model
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimensions = self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            2D numpy array of embeddings
        """
        if self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True)
        
        # Fallback: simple hash-based embeddings
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self.dimensions).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            embeddings.append(emb)
        
        return np.array(embeddings)


def create_sample_documents() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Create sample documents for the demo."""
    return [
        ("doc1", "Machine learning is a subset of artificial intelligence.",
         {"title": "ML Basics", "category": "AI"}),
        ("doc2", "Python is a popular programming language for data science.",
         {"title": "Python for DS", "category": "Programming"}),
        ("doc3", "Neural networks are inspired by biological neurons.",
         {"title": "Neural Networks", "category": "AI"}),
        ("doc4", "Databases store and organize data efficiently.",
         {"title": "Database Intro", "category": "Data"}),
        ("doc5", "Deep learning uses multiple layers of neural networks.",
         {"title": "Deep Learning", "category": "AI"}),
        ("doc6", "SQL is used to query relational databases.",
         {"title": "SQL Basics", "category": "Data"}),
        ("doc7", "Natural language processing enables computers to understand text.",
         {"title": "NLP Overview", "category": "AI"}),
        ("doc8", "Vector databases are optimized for similarity search.",
         {"title": "Vector DBs", "category": "Data"}),
        ("doc9", "Transformers revolutionized natural language processing.",
         {"title": "Transformers", "category": "AI"}),
        ("doc10", "Embeddings represent text as dense vectors.",
         {"title": "Embeddings", "category": "AI"}),
    ]


def rag_demo(store: SynaFeatureStore, embedder: SimpleEmbedder, query: str):
    """
    Demonstrate RAG (Retrieval-Augmented Generation) pattern.
    
    Args:
        store: Feature store with documents
        embedder: Embedding model
        query: User query
    """
    print(f"\n   Query: \"{query}\"")
    
    # 1. Encode query
    query_embedding = embedder.encode([query])[0]
    
    # 2. Retrieve relevant documents
    results = store.search(query_embedding, k=3)
    
    print("   Retrieved documents:")
    for doc_id, score, metadata in results:
        title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
        print(f"      - {title} (score: {score:.3f})")
    
    # 3. In a real RAG system, you would now:
    #    - Concatenate retrieved documents as context
    #    - Send to LLM with the query
    #    - Return generated response
    
    print("   [In production: Send context + query to LLM for generation]")


def main():
    print("=" * 60)
    print("Feature Store Demo with Syna")
    print("Requirement 5.4: Store and retrieve embeddings for RAG")
    print("=" * 60 + "\n")
    
    # Configuration
    DB_PATH = os.path.abspath("feature_store_demo.db")
    
    # Check for sentence-transformers
    if HAS_SENTENCE_TRANSFORMERS:
        print("Using: sentence-transformers (all-MiniLM-L6-v2)")
    else:
        print("Note: sentence-transformers not installed.")
        print("      Using simple hash-based embeddings for demo.")
        print("      Install with: pip install sentence-transformers")
    print()
    
    try:
        # 1. Initialize embedder and feature store
        print("1. Initializing embedder and feature store")
        print("-" * 40)
        
        embedder = SimpleEmbedder()
        print(f"   Embedding dimensions: {embedder.dimensions}")
        
        store = SynaFeatureStore(DB_PATH, dimensions=embedder.dimensions)
        print(f"   Feature store created: {DB_PATH}")
        print()
        
        # 2. Create and store embeddings
        print("2. Creating and storing embeddings")
        print("-" * 40)
        
        documents = create_sample_documents()
        print(f"   Documents to store: {len(documents)}")
        
        start = time.time()
        texts = [doc[1] for doc in documents]
        embeddings = embedder.encode(texts)
        encode_time = time.time() - start
        print(f"   Encoding time: {encode_time * 1000:.2f}ms")
        
        start = time.time()
        for (doc_id, text, metadata), embedding in zip(documents, embeddings):
            store.add(doc_id, embedding, {**metadata, "text": text})
        store_time = time.time() - start
        print(f"   Storage time: {store_time * 1000:.2f}ms")
        print(f"   Stored embeddings: {store.count()}")
        print()
        
        # 3. Verify storage
        print("3. Verifying storage")
        print("-" * 40)
        
        embedding, metadata = store.get("doc1")
        print(f"   Retrieved doc1:")
        print(f"      Embedding shape: {embedding.shape}")
        print(f"      Title: {metadata.get('title')}")
        print(f"      Category: {metadata.get('category')}")
        print()
        
        # 4. Similarity search
        print("4. Similarity search")
        print("-" * 40)
        
        # Search for AI-related documents
        query = "How do neural networks learn?"
        query_embedding = embedder.encode([query])[0]
        
        print(f"   Query: \"{query}\"")
        print("   Top 5 results:")
        
        start = time.time()
        results = store.search(query_embedding, k=5, metric="cosine")
        search_time = time.time() - start
        
        for i, (doc_id, score, metadata) in enumerate(results, 1):
            title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
            category = metadata.get('category', 'Unknown') if metadata else 'Unknown'
            print(f"      {i}. {title} [{category}] (score: {score:.3f})")
        
        print(f"   Search time: {search_time * 1000:.2f}ms")
        print()
        
        # 5. Different similarity metrics
        print("5. Comparing similarity metrics")
        print("-" * 40)
        
        for metric in ["cosine", "euclidean", "dot"]:
            results = store.search(query_embedding, k=3, metric=metric)
            print(f"   {metric.capitalize()} similarity:")
            for doc_id, score, metadata in results:
                title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
                print(f"      - {title}: {score:.3f}")
        print()
        
        # 6. RAG demonstration
        print("6. RAG (Retrieval-Augmented Generation) pattern")
        print("-" * 40)
        
        queries = [
            "What is machine learning?",
            "How do I query a database?",
            "Explain transformers in NLP",
        ]
        
        for query in queries:
            rag_demo(store, embedder, query)
        print()
        
        # 7. Benchmark
        print("7. Performance benchmark")
        print("-" * 40)
        
        # Benchmark search
        num_searches = 100
        query_embeddings = embedder.encode(["test query"] * num_searches)
        
        start = time.time()
        for emb in query_embeddings:
            _ = store.search(emb, k=5)
        total_time = time.time() - start
        
        print(f"   Searches: {num_searches}")
        print(f"   Total time: {total_time * 1000:.2f}ms")
        print(f"   Per search: {total_time / num_searches * 1000:.2f}ms")
        print(f"   Throughput: {num_searches / total_time:.0f} searches/sec")
        print()
        
        # 8. Storage statistics
        print("8. Storage statistics")
        print("-" * 40)
        
        db_size = os.path.getsize(DB_PATH)
        bytes_per_embedding = db_size / store.count()
        
        print(f"   Database size: {db_size / 1024:.1f} KB")
        print(f"   Embeddings stored: {store.count()}")
        print(f"   Bytes per embedding: {bytes_per_embedding:.0f}")
        print(f"   Raw embedding size: {embedder.dimensions * 4} bytes")
        print(f"   Overhead: {bytes_per_embedding - embedder.dimensions * 4:.0f} bytes")
        print()
        
        # Cleanup
        store.close()
        
        print("=" * 60)
        print("Feature Store Demo Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


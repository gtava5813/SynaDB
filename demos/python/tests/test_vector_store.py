"""
Test suite for VectorStore Python wrapper.

Tests the VectorStore class for embedding storage and similarity search.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add the demos/python directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import VectorStore, SearchResult


class TestVectorStoreImport:
    """Test that VectorStore can be imported."""
    
    def test_import(self):
        """VectorStore should be importable."""
        assert VectorStore is not None
        assert SearchResult is not None


class TestVectorStoreCreation:
    """Test creating vector stores."""
    
    def test_create_store(self):
        """Should create a vector store with valid dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=128, metric="cosine")
            assert store.dimensions == 128
            assert store.metric_name == "cosine"
    
    def test_invalid_dimensions_too_small(self):
        """Should reject dimensions < 64."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            with pytest.raises(ValueError, match="Dimensions must be between 64 and 4096"):
                VectorStore(db_path, dimensions=32)
    
    def test_invalid_dimensions_too_large(self):
        """Should reject dimensions > 4096."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            with pytest.raises(ValueError, match="Dimensions must be between 64 and 4096"):
                VectorStore(db_path, dimensions=5000)
    
    def test_different_metrics(self):
        """Should support different distance metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for metric in ["cosine", "euclidean", "dot_product"]:
                db_path = os.path.join(tmpdir, f"test_{metric}.db")
                store = VectorStore(db_path, dimensions=64, metric=metric)
                assert store.metric_name == metric


class TestVectorStoreInsert:
    """Test inserting vectors."""
    
    def test_insert_numpy_array(self):
        """Should insert numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=128, metric="cosine")
            
            embedding = np.random.randn(128).astype(np.float32)
            store.insert("doc1", embedding)
            
            assert len(store) == 1
    
    def test_insert_multiple_vectors(self):
        """Should insert multiple vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=128, metric="cosine")
            
            for i in range(10):
                embedding = np.random.randn(128).astype(np.float32)
                store.insert(f"doc{i}", embedding)
            
            assert len(store) == 10
    
    def test_insert_wrong_dimensions(self):
        """Should reject vectors with wrong dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=768, metric="cosine")
            
            wrong_vec = np.random.randn(512).astype(np.float32)
            with pytest.raises(ValueError, match="Vector has 512 dimensions, expected 768"):
                store.insert("wrong", wrong_vec)


class TestVectorStoreSearch:
    """Test similarity search."""
    
    def test_search_returns_results(self):
        """Should return search results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=64, metric="euclidean")
            
            # Insert some vectors
            for i in range(10):
                vec = np.zeros(64, dtype=np.float32)
                vec[i] = 1.0
                store.insert(f"doc{i}", vec)
            
            # Search
            query = np.zeros(64, dtype=np.float32)
            query[0] = 1.0
            
            results = store.search(query, k=3)
            
            assert len(results) == 3
            assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_finds_exact_match(self):
        """Should find exact match first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=64, metric="euclidean")
            
            # Insert vectors
            for i in range(10):
                vec = np.zeros(64, dtype=np.float32)
                vec[i] = 1.0
                store.insert(f"doc{i}", vec)
            
            # Query with doc0's vector
            query = np.zeros(64, dtype=np.float32)
            query[0] = 1.0
            
            results = store.search(query, k=3)
            
            assert results[0].key == "doc0"
            assert results[0].score < 0.001  # Near-zero distance
    
    def test_search_sorted_by_score(self):
        """Results should be sorted by score (ascending)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=64, metric="euclidean")
            
            # Insert vectors
            for i in range(10):
                vec = np.zeros(64, dtype=np.float32)
                vec[i] = 1.0
                store.insert(f"doc{i}", vec)
            
            query = np.zeros(64, dtype=np.float32)
            query[0] = 1.0
            
            results = store.search(query, k=5)
            
            # Verify sorted by score
            for i in range(1, len(results)):
                assert results[i-1].score <= results[i].score
    
    def test_cosine_similarity(self):
        """Should correctly compute cosine similarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_vectors.db")
            store = VectorStore(db_path, dimensions=64, metric="cosine")
            
            # Insert vectors
            v1 = np.array([1.0] + [0.0] * 63, dtype=np.float32)
            v2 = np.array([0.0, 1.0] + [0.0] * 62, dtype=np.float32)
            v3 = np.array([0.707, 0.707] + [0.0] * 62, dtype=np.float32)
            
            store.insert("v1", v1)
            store.insert("v2", v2)
            store.insert("v3", v3)
            
            # Query with v1
            results = store.search(v1, k=3)
            
            # v1 should be most similar to itself
            assert results[0].key == "v1"
            assert results[0].score < 0.001  # Cosine distance ~0
            
            # v3 should be second (45 degrees)
            assert results[1].key == "v3"
            assert abs(results[1].score - 0.293) < 0.01  # Cosine distance ~0.29
            
            # v2 should be last (90 degrees)
            assert results[2].key == "v2"
            assert abs(results[2].score - 1.0) < 0.01  # Cosine distance ~1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for SparseVectorStore Python wrapper.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synadb import SparseVectorStore, SparseSearchResult, SparseIndexStats, SparseVectorStoreError


class TestSparseVectorStoreBasic:
    """Basic functionality tests."""
    
    def test_create_and_close(self):
        """Test creating and closing a store."""
        store = SparseVectorStore("test_create_close")
        assert len(store) == 0
        store.close()
    
    def test_context_manager(self):
        """Test using store as context manager."""
        with SparseVectorStore("test_context_manager") as store:
            assert len(store) == 0
    
    def test_index_dict(self):
        """Test indexing with dict format."""
        with SparseVectorStore("test_index_dict") as store:
            doc_id = store.index("doc1", {100: 1.5, 200: 0.8})
            assert doc_id >= 0
            assert len(store) == 1
    
    def test_index_list_of_tuples(self):
        """Test indexing with list of tuples format."""
        with SparseVectorStore("test_index_list") as store:
            doc_id = store.index("doc1", [(100, 1.5), (200, 0.8)])
            assert doc_id >= 0
            assert len(store) == 1
    
    def test_index_filters_zero_weights(self):
        """Test that zero and negative weights are filtered."""
        with SparseVectorStore("test_filter_weights") as store:
            store.index("doc1", {100: 1.5, 200: 0.0, 300: -0.5})
            stats = store.stats()
            # Only term 100 should be indexed (positive weight)
            assert stats.num_terms == 1
    
    def test_index_batch(self):
        """Test batch indexing."""
        with SparseVectorStore("test_batch") as store:
            docs = [
                ("doc1", {100: 1.0}),
                ("doc2", {200: 2.0}),
                ("doc3", {300: 3.0}),
            ]
            doc_ids = store.index_batch(docs)
            assert len(doc_ids) == 3
            assert len(store) == 3


class TestSparseVectorStoreSearch:
    """Search functionality tests."""
    
    def test_search_basic(self):
        """Test basic search."""
        with SparseVectorStore("test_search_basic") as store:
            store.index("doc1", {100: 2.0, 200: 1.0})
            store.index("doc2", {100: 1.0, 300: 3.0})
            
            results = store.search({100: 1.0}, k=10)
            
            assert len(results) == 2
            assert results[0].key == "doc1"  # Higher score (2.0 vs 1.0)
            assert results[0].score > results[1].score
    
    def test_search_empty_query(self):
        """Test search with empty query returns empty results."""
        with SparseVectorStore("test_search_empty") as store:
            store.index("doc1", {100: 1.0})
            
            results = store.search({}, k=10)
            assert len(results) == 0
    
    def test_search_no_overlap(self):
        """Test search with no term overlap returns empty results."""
        with SparseVectorStore("test_search_no_overlap") as store:
            store.index("doc1", {100: 1.0})
            
            results = store.search({999: 1.0}, k=10)
            assert len(results) == 0
    
    def test_search_k_limit(self):
        """Test that search respects k limit."""
        with SparseVectorStore("test_search_k") as store:
            for i in range(10):
                store.index(f"doc{i}", {100: float(i + 1)})
            
            results = store.search({100: 1.0}, k=3)
            assert len(results) == 3
    
    def test_search_result_type(self):
        """Test that search returns SparseSearchResult objects."""
        with SparseVectorStore("test_search_result") as store:
            store.index("doc1", {100: 1.0})
            
            results = store.search({100: 1.0}, k=10)
            
            assert len(results) == 1
            assert isinstance(results[0], SparseSearchResult)
            assert results[0].key == "doc1"
            assert isinstance(results[0].score, float)


class TestSparseVectorStoreDelete:
    """Delete functionality tests."""
    
    def test_delete_existing(self):
        """Test deleting an existing document."""
        with SparseVectorStore("test_delete_existing") as store:
            store.index("doc1", {100: 1.0})
            assert len(store) == 1
            
            result = store.delete("doc1")
            assert result is True
            assert len(store) == 0
    
    def test_delete_nonexistent(self):
        """Test deleting a non-existent document."""
        with SparseVectorStore("test_delete_nonexistent") as store:
            result = store.delete("nonexistent")
            assert result is False
    
    def test_delete_removes_from_search(self):
        """Test that deleted documents don't appear in search."""
        with SparseVectorStore("test_delete_search") as store:
            store.index("doc1", {100: 2.0})
            store.index("doc2", {100: 1.0})
            
            store.delete("doc1")
            
            results = store.search({100: 1.0}, k=10)
            assert len(results) == 1
            assert results[0].key == "doc2"


class TestSparseVectorStoreStats:
    """Statistics functionality tests."""
    
    def test_stats_empty(self):
        """Test stats on empty store."""
        with SparseVectorStore("test_stats_empty") as store:
            stats = store.stats()
            
            assert isinstance(stats, SparseIndexStats)
            assert stats.num_documents == 0
            assert stats.num_terms == 0
            assert stats.num_postings == 0
            assert stats.avg_doc_length == 0.0
    
    def test_stats_with_data(self):
        """Test stats with indexed data."""
        with SparseVectorStore("test_stats_data") as store:
            store.index("doc1", {100: 1.0, 200: 2.0})
            store.index("doc2", {100: 1.0, 300: 3.0})
            
            stats = store.stats()
            
            assert stats.num_documents == 2
            assert stats.num_terms == 3  # 100, 200, 300
            assert stats.num_postings == 4  # 2 docs × 2 terms each
            assert abs(stats.avg_doc_length - 2.0) < 0.001


class TestSparseVectorStoreReplace:
    """Document replacement tests."""
    
    def test_replace_document(self):
        """Test that indexing with same key replaces the document."""
        with SparseVectorStore("test_replace") as store:
            store.index("doc1", {100: 1.0})
            store.index("doc1", {200: 2.0})  # Replace
            
            assert len(store) == 1
            
            # Search should find new terms, not old
            results = store.search({100: 1.0}, k=10)
            assert len(results) == 0
            
            results = store.search({200: 1.0}, k=10)
            assert len(results) == 1
            assert results[0].key == "doc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSparseVectorStorePersistence:
    """Persistence functionality tests."""
    
    def test_save_and_open(self, tmp_path):
        """Test saving and opening an index."""
        file_path = str(tmp_path / "test_index.svs")
        
        # Create and populate store
        with SparseVectorStore("test_save_open_1") as store:
            store.index("doc1", {100: 2.0, 200: 1.0})
            store.index("doc2", {100: 1.0, 300: 3.0})
            store.save(file_path)
        
        # Open from file
        store2 = SparseVectorStore.open("test_save_open_2", file_path)
        try:
            assert len(store2) == 2
            
            # Search should work
            results = store2.search({100: 1.0}, k=10)
            assert len(results) == 2
            assert results[0].key == "doc1"  # Higher score
        finally:
            store2.close()
    
    def test_save_empty(self, tmp_path):
        """Test saving an empty index."""
        file_path = str(tmp_path / "empty_index.svs")
        
        with SparseVectorStore("test_save_empty_1") as store:
            store.save(file_path)
        
        store2 = SparseVectorStore.open("test_save_empty_2", file_path)
        try:
            assert len(store2) == 0
        finally:
            store2.close()
    
    def test_save_preserves_stats(self, tmp_path):
        """Test that save/open preserves statistics."""
        file_path = str(tmp_path / "stats_index.svs")
        
        with SparseVectorStore("test_save_stats_1") as store:
            store.index("doc1", {100: 1.0, 200: 2.0})
            store.index("doc2", {100: 1.0, 300: 3.0})
            original_stats = store.stats()
            store.save(file_path)
        
        store2 = SparseVectorStore.open("test_save_stats_2", file_path)
        try:
            loaded_stats = store2.stats()
            assert loaded_stats.num_documents == original_stats.num_documents
            assert loaded_stats.num_terms == original_stats.num_terms
            assert loaded_stats.num_postings == original_stats.num_postings
        finally:
            store2.close()
    
    def test_open_nonexistent_file(self, tmp_path):
        """Test opening a non-existent file raises error."""
        file_path = str(tmp_path / "nonexistent.svs")
        
        with pytest.raises(SparseVectorStoreError):
            SparseVectorStore.open("test_open_nonexistent", file_path)
    
    def test_save_closed_store_raises(self, tmp_path):
        """Test saving a closed store raises error."""
        file_path = str(tmp_path / "closed.svs")
        
        store = SparseVectorStore("test_save_closed")
        store.close()
        
        with pytest.raises(SparseVectorStoreError):
            store.save(file_path)

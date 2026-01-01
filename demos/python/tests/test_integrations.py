"""
Integration tests for LLM frameworks.

Tests the integrations with LangChain, LlamaIndex, and Haystack.
These tests use pytest.importorskip to skip tests when the required
framework is not installed.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add the demos/python directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockEmbeddings:
    """Mock embeddings for testing without API calls."""
    
    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
    
    def embed_documents(self, texts):
        """Generate deterministic embeddings for documents."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embeddings.append(np.random.randn(self.dimensions).tolist())
        return embeddings
    
    def embed_query(self, text):
        """Generate deterministic embedding for query."""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimensions).tolist()


class TestLangChainIntegration:
    """Tests for LangChain integration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        # Create a temporary file and close it immediately
        # This avoids file locking issues on Windows
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # Remove the file so the database can create it fresh
        os.unlink(path)
        yield path
        # Cleanup
        try:
            os.unlink(path)
        except Exception:
            pass
    
    def test_vectorstore_creation(self, temp_db):
        """Should create a LangChain VectorStore."""
        langchain_core = pytest.importorskip("langchain_core")
        from synadb.integrations.langchain import SynaVectorStore
        
        embeddings = MockEmbeddings(dimensions=128)
        store = SynaVectorStore(
            path=temp_db,
            embedding=embeddings,
            dimensions=128,
            metric="cosine"
        )
        
        assert store is not None
        assert store.embeddings == embeddings
    
    def test_vectorstore_add_texts(self, temp_db):
        """Should add texts to the VectorStore."""
        langchain_core = pytest.importorskip("langchain_core")
        from synadb.integrations.langchain import SynaVectorStore
        
        embeddings = MockEmbeddings(dimensions=128)
        store = SynaVectorStore(
            path=temp_db,
            embedding=embeddings,
            dimensions=128,
            metric="cosine"
        )
        
        texts = ["Hello world", "Machine learning is great", "Python is awesome"]
        ids = store.add_texts(texts)
        
        assert len(ids) == 3
        assert all(isinstance(id_, str) for id_ in ids)
    
    def test_vectorstore_similarity_search(self, temp_db):
        """Should perform similarity search."""
        langchain_core = pytest.importorskip("langchain_core")
        from synadb.integrations.langchain import SynaVectorStore
        
        embeddings = MockEmbeddings(dimensions=128)
        store = SynaVectorStore(
            path=temp_db,
            embedding=embeddings,
            dimensions=128,
            metric="cosine"
        )
        
        texts = ["Hello world", "Machine learning is great", "Python is awesome"]
        store.add_texts(texts)
        
        # Search for similar documents
        results = store.similarity_search("Hello world", k=2)
        
        assert len(results) == 2
        # First result should be exact match
        assert results[0].page_content == "Hello world"
    
    def test_vectorstore_similarity_search_with_score(self, temp_db):
        """Should return similarity scores."""
        langchain_core = pytest.importorskip("langchain_core")
        from synadb.integrations.langchain import SynaVectorStore
        
        embeddings = MockEmbeddings(dimensions=128)
        store = SynaVectorStore(
            path=temp_db,
            embedding=embeddings,
            dimensions=128,
            metric="cosine"
        )
        
        texts = ["Hello world", "Machine learning is great"]
        store.add_texts(texts)
        
        results = store.similarity_search_with_score("Hello world", k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # First result should have lowest score (most similar)
        assert results[0][1] <= results[1][1]
    
    def test_vectorstore_from_texts(self, temp_db):
        """Should create VectorStore from texts."""
        langchain_core = pytest.importorskip("langchain_core")
        from synadb.integrations.langchain import SynaVectorStore
        
        embeddings = MockEmbeddings(dimensions=128)
        texts = ["Document 1", "Document 2", "Document 3"]
        
        store = SynaVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db,
            dimensions=128
        )
        
        assert store is not None
        results = store.similarity_search("Document 1", k=1)
        assert len(results) == 1
    
    def test_chat_history_creation(self, temp_db):
        """Should create a chat message history."""
        langchain_core = pytest.importorskip("langchain_core")
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            pytest.skip("langchain_core.messages not available")
        
        from synadb.integrations.langchain import SynaChatMessageHistory
        
        history = SynaChatMessageHistory(path=temp_db, session_id="test_session")
        
        assert history is not None
        assert len(history.messages) == 0
    
    def test_chat_history_add_messages(self, temp_db):
        """Should add and retrieve messages."""
        langchain_core = pytest.importorskip("langchain_core")
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            pytest.skip("langchain_core.messages not available")
        
        from synadb.integrations.langchain import SynaChatMessageHistory
        
        history = SynaChatMessageHistory(path=temp_db, session_id="test_session")
        
        # Add messages
        history.add_message(HumanMessage(content="Hello!"))
        history.add_message(AIMessage(content="Hi there!"))
        
        messages = history.messages
        assert len(messages) == 2
        assert messages[0].content == "Hello!"
        assert messages[1].content == "Hi there!"
    
    def test_chat_history_clear(self, temp_db):
        """Should clear chat history."""
        langchain_core = pytest.importorskip("langchain_core")
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain_core.messages not available")
        
        from synadb.integrations.langchain import SynaChatMessageHistory
        
        history = SynaChatMessageHistory(path=temp_db, session_id="test_session")
        history.add_message(HumanMessage(content="Hello!"))
        
        assert len(history.messages) == 1
        
        history.clear()
        
        assert len(history.messages) == 0
    
    def test_loader_creation(self, temp_db):
        """Should create a document loader."""
        langchain_core = pytest.importorskip("langchain_core")
        try:
            from langchain_core.document_loaders import BaseLoader
        except ImportError:
            pytest.skip("langchain_core.document_loaders not available")
        
        from synadb.integrations.langchain import SynaLoader
        from synadb import SynaDB
        
        # First, add some documents to the database
        db = SynaDB(temp_db)
        db.put_text("documents/doc1", "This is document 1")
        db.put_text("documents/doc2", "This is document 2")
        db.close()
        
        loader = SynaLoader(path=temp_db, pattern="documents/*")
        documents = loader.load()
        
        assert len(documents) == 2
        assert all(doc.page_content for doc in documents)


class TestLlamaIndexIntegration:
    """Tests for LlamaIndex integration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        try:
            os.unlink(f.name)
        except Exception:
            pass
    
    def test_vectorstore_creation(self, temp_db):
        """Should create a LlamaIndex VectorStore."""
        llama_index = pytest.importorskip("llama_index")
        from synadb.integrations.llamaindex import SynaVectorStore
        
        store = SynaVectorStore(
            path=temp_db,
            dimensions=128,
            metric="cosine"
        )
        
        assert store is not None
        assert store._dimensions == 128
    
    def test_vectorstore_add_nodes(self, temp_db):
        """Should add nodes to the VectorStore."""
        llama_index = pytest.importorskip("llama_index")
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            pytest.skip("llama_index.core.schema not available")
        
        from synadb.integrations.llamaindex import SynaVectorStore
        
        store = SynaVectorStore(
            path=temp_db,
            dimensions=128,
            metric="cosine"
        )
        
        # Create nodes with embeddings
        nodes = [
            TextNode(
                text="Hello world",
                id_="node1",
                embedding=np.random.randn(128).tolist()
            ),
            TextNode(
                text="Machine learning",
                id_="node2",
                embedding=np.random.randn(128).tolist()
            )
        ]
        
        ids = store.add(nodes)
        
        assert len(ids) == 2
        assert "node1" in ids
        assert "node2" in ids
    
    def test_vectorstore_query(self, temp_db):
        """Should query the VectorStore."""
        llama_index = pytest.importorskip("llama_index")
        try:
            from llama_index.core.schema import TextNode
            from llama_index.core.vector_stores.types import VectorStoreQuery
        except ImportError:
            pytest.skip("llama_index.core not available")
        
        from synadb.integrations.llamaindex import SynaVectorStore
        
        store = SynaVectorStore(
            path=temp_db,
            dimensions=128,
            metric="cosine"
        )
        
        # Create and add nodes
        embedding1 = np.random.randn(128).tolist()
        nodes = [
            TextNode(
                text="Hello world",
                id_="node1",
                embedding=embedding1
            ),
            TextNode(
                text="Machine learning",
                id_="node2",
                embedding=np.random.randn(128).tolist()
            )
        ]
        store.add(nodes)
        
        # Query with the same embedding
        query = VectorStoreQuery(
            query_embedding=embedding1,
            similarity_top_k=2
        )
        result = store.query(query)
        
        assert result is not None
        assert len(result.nodes) == 2
        # First result should be the exact match
        assert result.ids[0] == "node1"
    
    def test_vectorstore_delete(self, temp_db):
        """Should delete nodes from the VectorStore."""
        llama_index = pytest.importorskip("llama_index")
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            pytest.skip("llama_index.core.schema not available")
        
        from synadb.integrations.llamaindex import SynaVectorStore
        
        store = SynaVectorStore(
            path=temp_db,
            dimensions=128,
            metric="cosine"
        )
        
        nodes = [
            TextNode(
                text="Hello world",
                id_="node1",
                embedding=np.random.randn(128).tolist()
            )
        ]
        store.add(nodes)
        
        # Delete the node
        store.delete("node1")
        
        # Verify deletion (metadata cache should be cleared)
        assert "node1" not in store._metadata_cache
    
    def test_chat_store_creation(self, temp_db):
        """Should create a LlamaIndex ChatStore."""
        llama_index = pytest.importorskip("llama_index")
        try:
            from llama_index.core.storage.chat_store import BaseChatStore
        except ImportError:
            pytest.skip("llama_index.core.storage.chat_store not available")
        
        from synadb.integrations.llamaindex import SynaChatStore
        
        store = SynaChatStore(path=temp_db)
        
        assert store is not None
    
    def test_chat_store_messages(self, temp_db):
        """Should store and retrieve chat messages."""
        llama_index = pytest.importorskip("llama_index")
        try:
            from llama_index.core.llms import ChatMessage
        except ImportError:
            pytest.skip("llama_index.core.llms not available")
        
        from synadb.integrations.llamaindex import SynaChatStore
        
        store = SynaChatStore(path=temp_db)
        
        messages = [
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!")
        ]
        
        store.set_messages("session1", messages)
        retrieved = store.get_messages("session1")
        
        assert len(retrieved) == 2
        assert retrieved[0].content == "Hello!"
        assert retrieved[1].content == "Hi there!"


class TestHaystackIntegration:
    """Tests for Haystack integration."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        try:
            os.unlink(f.name)
        except Exception:
            pass
    
    def test_document_store_creation(self, temp_db):
        """Should create a Haystack DocumentStore."""
        haystack = pytest.importorskip("haystack")
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        assert store is not None
        assert store._embedding_dim == 128
    
    def test_document_store_write_documents(self, temp_db):
        """Should write documents to the store."""
        haystack = pytest.importorskip("haystack")
        try:
            from haystack import Document
        except ImportError:
            pytest.skip("haystack.Document not available")
        
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        documents = [
            Document(content="Hello world", id="doc1"),
            Document(content="Machine learning", id="doc2")
        ]
        
        written = store.write_documents(documents)
        
        assert written == 2
        assert store.count_documents() == 2
    
    def test_document_store_write_with_embeddings(self, temp_db):
        """Should write documents with embeddings."""
        haystack = pytest.importorskip("haystack")
        try:
            from haystack import Document
        except ImportError:
            pytest.skip("haystack.Document not available")
        
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        documents = [
            Document(
                content="Hello world",
                id="doc1",
                embedding=np.random.randn(128).tolist()
            )
        ]
        
        written = store.write_documents(documents)
        
        assert written == 1
    
    def test_document_store_filter_documents(self, temp_db):
        """Should filter documents by metadata."""
        haystack = pytest.importorskip("haystack")
        try:
            from haystack import Document
        except ImportError:
            pytest.skip("haystack.Document not available")
        
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        documents = [
            Document(content="Hello world", id="doc1", meta={"category": "greeting"}),
            Document(content="Machine learning", id="doc2", meta={"category": "tech"})
        ]
        store.write_documents(documents)
        
        # Filter by category
        filtered = store.filter_documents(filters={"category": "tech"})
        
        assert len(filtered) == 1
        assert filtered[0].content == "Machine learning"
    
    def test_document_store_delete_documents(self, temp_db):
        """Should delete documents."""
        haystack = pytest.importorskip("haystack")
        try:
            from haystack import Document
        except ImportError:
            pytest.skip("haystack.Document not available")
        
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        documents = [
            Document(content="Hello world", id="doc1"),
            Document(content="Machine learning", id="doc2")
        ]
        store.write_documents(documents)
        
        assert store.count_documents() == 2
        
        store.delete_documents(["doc1"])
        
        assert store.count_documents() == 1
    
    def test_document_store_serialization(self, temp_db):
        """Should serialize and deserialize the store."""
        haystack = pytest.importorskip("haystack")
        from synadb.integrations.haystack import SynaDocumentStore
        
        store = SynaDocumentStore(
            path=temp_db,
            embedding_dim=128,
            metric="cosine"
        )
        
        # Serialize
        data = store.to_dict()
        
        assert data["type"] == "synadb.integrations.haystack.SynaDocumentStore"
        assert data["init_parameters"]["path"] == temp_db
        assert data["init_parameters"]["embedding_dim"] == 128
        
        # Deserialize
        restored = SynaDocumentStore.from_dict(data)
        
        assert restored._embedding_dim == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

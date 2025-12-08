"""
LangChain integration for Syna.

This module provides LangChain-compatible components backed by Syna:
- SynaVectorStore: VectorStore implementation for RAG applications
- SynaChatMessageHistory: Chat history persistence for conversational AI

Example:
    >>> from langchain_openai import OpenAIEmbeddings
    >>> from synadb.integrations.langchain import SynaVectorStore, SynaChatMessageHistory
    >>> 
    >>> vectorstore = SynaVectorStore.from_documents(
    ...     documents,
    ...     embedding=OpenAIEmbeddings(),
    ...     path="langchain.db"
    ... )
    >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    >>> 
    >>> # Chat history
    >>> history = SynaChatMessageHistory(path="chat.db", session_id="user_123")
    >>> history.add_user_message("Hello!")
    >>> history.add_ai_message("Hi there!")

Requirements:
    pip install langchain langchain-core
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import json
import numpy as np

try:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create placeholder classes for type hints when langchain is not installed
    VectorStore = object
    Document = None
    Embeddings = None

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, messages_from_dict
    LANGCHAIN_CHAT_AVAILABLE = True
except ImportError:
    LANGCHAIN_CHAT_AVAILABLE = False
    BaseChatMessageHistory = object
    BaseMessage = None
    messages_from_dict = None

try:
    from langchain_core.document_loaders import BaseLoader
    LANGCHAIN_LOADER_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOADER_AVAILABLE = False
    BaseLoader = object


class SynaVectorStore(VectorStore):
    """
    LangChain VectorStore backed by Syna.
    
    This class implements the LangChain VectorStore interface, allowing
    Syna to be used as a vector database for RAG (Retrieval-Augmented
    Generation) applications.
    
    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from synadb.integrations.langchain import SynaVectorStore
        >>> 
        >>> # Create from documents
        >>> vectorstore = SynaVectorStore.from_documents(
        ...     documents,
        ...     embedding=OpenAIEmbeddings(),
        ...     path="langchain.db"
        ... )
        >>> 
        >>> # Use as retriever
        >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        >>> 
        >>> # Or search directly
        >>> docs = vectorstore.similarity_search("What is machine learning?", k=4)
    
    Attributes:
        _store: The underlying Syna VectorStore instance.
        _embedding: The embedding model used to convert text to vectors.
    """
    
    def __init__(
        self,
        path: str,
        embedding: "Embeddings",
        dimensions: Optional[int] = None,
        metric: str = "cosine",
    ):
        """
        Create or open a Syna-backed LangChain VectorStore.
        
        Args:
            path: Path to the Syna database file.
            embedding: LangChain Embeddings instance for text-to-vector conversion.
            dimensions: Vector dimensions. If None, auto-detected from embedding.
            metric: Distance metric ("cosine", "euclidean", "dot_product").
        
        Raises:
            ImportError: If langchain is not installed.
            ValueError: If dimensions cannot be determined.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. "
                "Install it with: pip install langchain langchain-core"
            )
        
        from ..vector import VectorStore as SynaStore
        
        self._embedding = embedding
        self._path = path
        self._metric = metric
        
        # Auto-detect dimensions from embedding if not provided
        if dimensions is None:
            test_embed = embedding.embed_query("test")
            dimensions = len(test_embed)
        
        self._dimensions = dimensions
        self._store = SynaStore(path, dimensions=dimensions, metric=metric)
    
    @property
    def embeddings(self) -> Optional["Embeddings"]:
        """Return the embedding model."""
        return self._embedding
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vectorstore.
        
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata dicts associated with texts.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            List of IDs for the added texts.
        """
        texts = list(texts)
        embeddings = self._embedding.embed_documents(texts)
        
        ids = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Generate a unique ID based on text hash
            doc_id = f"doc_{hash(text) % 10**12}"
            
            # Prepare metadata, including the original text
            metadata = metadatas[i] if metadatas else {}
            metadata["text"] = text
            
            # Insert into Syna VectorStore
            # Note: The current VectorStore doesn't support metadata directly,
            # so we store it as part of the key or in a separate entry
            self._store.insert(doc_id, np.array(embedding, dtype=np.float32))
            
            # Store metadata separately using the underlying database
            # This is a workaround until VectorStore supports metadata natively
            self._store_metadata(doc_id, metadata)
            
            ids.append(doc_id)
        
        return ids
    
    def _store_metadata(self, doc_id: str, metadata: dict) -> None:
        """Store metadata for a document in the underlying database."""
        # Import json for serialization
        import json
        
        # Access the underlying library through the store
        try:
            from ..wrapper import SynaDB
            SynaDB._load_library()
            lib = SynaDB._lib
            
            # Store metadata as JSON text
            metadata_key = f"meta/{doc_id}"
            metadata_json = json.dumps(metadata)
            
            lib.SYNA_put_text(
                self._path.encode('utf-8'),
                metadata_key.encode('utf-8'),
                metadata_json.encode('utf-8')
            )
        except Exception:
            # If metadata storage fails, continue without it
            # The vector is still stored
            pass
    
    def _get_metadata(self, doc_id: str) -> dict:
        """Retrieve metadata for a document from the underlying database."""
        import json
        import ctypes
        
        try:
            from ..wrapper import SynaDB
            SynaDB._load_library()
            lib = SynaDB._lib
            
            metadata_key = f"meta/{doc_id}"
            
            # Set up the function signature if not already done
            if not hasattr(lib, '_metadata_setup'):
                lib.SYNA_get_text.argtypes = [
                    ctypes.c_char_p, 
                    ctypes.c_char_p, 
                    ctypes.POINTER(ctypes.c_char_p)
                ]
                lib.SYNA_get_text.restype = ctypes.c_int32
                lib._metadata_setup = True
            
            out_text = ctypes.c_char_p()
            result = lib.SYNA_get_text(
                self._path.encode('utf-8'),
                metadata_key.encode('utf-8'),
                ctypes.byref(out_text)
            )
            
            if result == 1 and out_text.value:
                metadata_json = out_text.value.decode('utf-8')
                # Free the string
                lib.SYNA_free_string(out_text)
                return json.loads(metadata_json)
        except Exception:
            pass
        
        return {}
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List["Document"]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string to search for.
            k: Number of documents to return.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            List of Document objects most similar to the query.
        """
        # Convert query to embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Search in Syna VectorStore
        results = self._store.search(np.array(query_embedding, dtype=np.float32), k=k)
        
        # Convert results to LangChain Documents
        documents = []
        for r in results:
            # Try to get metadata
            metadata = self._get_metadata(r.key)
            
            # Extract text from metadata
            text = metadata.pop("text", "")
            
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple["Document", float]]:
        """
        Search for documents similar to the query, returning scores.
        
        Args:
            query: Query string to search for.
            k: Number of documents to return.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            List of (Document, score) tuples, sorted by similarity.
        """
        # Convert query to embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Search in Syna VectorStore
        results = self._store.search(np.array(query_embedding, dtype=np.float32), k=k)
        
        # Convert results to LangChain Documents with scores
        documents_with_scores = []
        for r in results:
            # Try to get metadata
            metadata = self._get_metadata(r.key)
            
            # Extract text from metadata
            text = metadata.pop("text", "")
            
            doc = Document(
                page_content=text,
                metadata=metadata,
            )
            documents_with_scores.append((doc, r.score))
        
        return documents_with_scores
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List["Document"]:
        """
        Search for documents similar to the given embedding vector.
        
        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            List of Document objects most similar to the embedding.
        """
        # Search in Syna VectorStore
        results = self._store.search(np.array(embedding, dtype=np.float32), k=k)
        
        # Convert results to LangChain Documents
        documents = []
        for r in results:
            # Try to get metadata
            metadata = self._get_metadata(r.key)
            
            # Extract text from metadata
            text = metadata.pop("text", "")
            
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )
        
        return documents
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: "Embeddings",
        metadatas: Optional[List[dict]] = None,
        path: str = "langchain.db",
        **kwargs: Any,
    ) -> "SynaVectorStore":
        """
        Create a SynaVectorStore from a list of texts.
        
        Args:
            texts: List of texts to add to the vectorstore.
            embedding: Embeddings instance for text-to-vector conversion.
            metadatas: Optional list of metadata dicts.
            path: Path to the Syna database file.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            A new SynaVectorStore instance with the texts added.
        """
        store = cls(path=path, embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store
    
    @classmethod
    def from_documents(
        cls,
        documents: List["Document"],
        embedding: "Embeddings",
        path: str = "langchain.db",
        **kwargs: Any,
    ) -> "SynaVectorStore":
        """
        Create a SynaVectorStore from a list of LangChain Documents.
        
        Args:
            documents: List of Document objects to add.
            embedding: Embeddings instance for text-to-vector conversion.
            path: Path to the Syna database file.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            A new SynaVectorStore instance with the documents added.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(texts, embedding, metadatas, path=path, **kwargs)
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            True if deletion was successful, None if not supported.
        """
        if ids is None:
            return None
        
        try:
            for doc_id in ids:
                self._store.delete(doc_id)
            return True
        except Exception:
            return False


class SynaChatMessageHistory(BaseChatMessageHistory):
    """
    LangChain chat history backed by Syna.
    
    This class implements the LangChain BaseChatMessageHistory interface,
    allowing Syna to be used as a persistent storage backend for chat
    conversations in LangChain applications.
    
    Example:
        >>> history = SynaChatMessageHistory(
        ...     path="chat.db",
        ...     session_id="user_123"
        ... )
        >>> history.add_user_message("Hello!")
        >>> history.add_ai_message("Hi there!")
        >>> print(history.messages)
    
    Attributes:
        _db: The underlying Syna database instance.
        _session_id: Unique identifier for the chat session.
        _key_prefix: Prefix used for storing messages in the database.
    """
    
    def __init__(self, path: str, session_id: str):
        """
        Create or open a Syna-backed chat message history.
        
        Args:
            path: Path to the Syna database file.
            session_id: Unique identifier for the chat session.
        
        Raises:
            ImportError: If langchain is not installed.
        """
        if not LANGCHAIN_CHAT_AVAILABLE:
            raise ImportError(
                "langchain chat history is not available. "
                "Install it with: pip install langchain langchain-core"
            )
        
        from ..wrapper import SynaDB
        self._db = SynaDB(path)
        self._session_id = session_id
        self._key_prefix = f"chat/{session_id}/"
    
    @property
    def messages(self) -> List["BaseMessage"]:
        """
        Get all messages in the session.
        
        Returns:
            List of BaseMessage objects in chronological order.
        """
        messages = []
        keys = sorted([
            k for k in self._db.keys()
            if k.startswith(self._key_prefix)
        ])
        
        for key in keys:
            msg_json = self._db.get_text(key)
            if msg_json:
                msg_dict = json.loads(msg_json)
                messages.extend(messages_from_dict([msg_dict]))
        
        return messages
    
    def add_message(self, message: "BaseMessage") -> None:
        """
        Add a message to the session.
        
        Args:
            message: The message to add to the history.
        """
        import time
        key = f"{self._key_prefix}{int(time.time() * 1000000)}"
        self._db.put_text(key, message.json())
    
    def clear(self) -> None:
        """Clear all messages in the session."""
        for key in self._db.keys():
            if key.startswith(self._key_prefix):
                self._db.delete(key)


class SynaLoader(BaseLoader):
    """
    Load documents from Syna database.
    
    This class implements the LangChain BaseLoader interface, allowing
    documents stored in a Syna database to be loaded as LangChain Document
    objects for use in RAG pipelines and other LangChain applications.
    
    Example:
        >>> loader = SynaLoader(path="docs.db", pattern="documents/*")
        >>> documents = loader.load()
        >>> for doc in documents:
        ...     print(f"Source: {doc.metadata['source']}")
        ...     print(f"Content: {doc.page_content[:100]}...")
    
    Attributes:
        _db: The underlying Syna database instance.
        _pattern: Pattern for filtering keys to load.
    """
    
    def __init__(self, path: str, pattern: str = "*"):
        """
        Create a document loader for a Syna database.
        
        Args:
            path: Path to the Syna database file.
            pattern: Pattern for filtering keys to load.
                     Use "*" to load all keys.
                     Use "prefix/*" to load keys starting with "prefix/".
                     Use exact key name to load a single document.
        
        Raises:
            ImportError: If langchain is not installed.
        """
        if not LANGCHAIN_LOADER_AVAILABLE:
            raise ImportError(
                "langchain document loader is not available. "
                "Install it with: pip install langchain langchain-core"
            )
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. "
                "Install it with: pip install langchain langchain-core"
            )
        
        from ..wrapper import SynaDB
        self._db = SynaDB(path)
        self._pattern = pattern
    
    def load(self) -> List["Document"]:
        """
        Load documents matching the pattern.
        
        Returns:
            List of Document objects with page_content set to the text
            value and metadata containing the source key.
        """
        documents = []
        
        for key in self._db.keys():
            if self._matches_pattern(key, self._pattern):
                text = self._db.get_text(key)
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": key}
                    ))
        
        return documents
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """
        Check if a key matches the given pattern.
        
        Args:
            key: The key to check.
            pattern: The pattern to match against.
                     "*" matches all keys.
                     "prefix/*" matches keys starting with "prefix/".
                     Exact string matches the key exactly.
        
        Returns:
            True if the key matches the pattern, False otherwise.
        """
        if pattern == "*":
            return True
        if pattern.endswith("/*"):
            return key.startswith(pattern[:-2])
        return key == pattern

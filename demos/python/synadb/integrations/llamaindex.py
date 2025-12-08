"""
LlamaIndex integration for Syna.

This module provides LlamaIndex-compatible components backed by Syna:
- SynaVectorStore: VectorStore implementation for RAG applications

Example:
    >>> from llama_index.core import VectorStoreIndex, StorageContext
    >>> from synadb.integrations.llamaindex import SynaVectorStore
    >>> 
    >>> vector_store = SynaVectorStore(path="index.db", dimensions=1536)
    >>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
    >>> index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

Requirements:
    pip install llama-index llama-index-core

**Validates: Requirements 11.4, 11.5**
"""

from typing import Any, Dict, List, Optional
import json
import ctypes
import numpy as np

try:
    from llama_index.core.vector_stores.types import (
        VectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.schema import TextNode
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Create placeholder classes for type hints when llama-index is not installed
    VectorStore = object
    VectorStoreQuery = None
    VectorStoreQueryResult = None
    TextNode = None

try:
    from llama_index.core.storage.chat_store import BaseChatStore
    from llama_index.core.llms import ChatMessage
    LLAMAINDEX_CHAT_AVAILABLE = True
except ImportError:
    LLAMAINDEX_CHAT_AVAILABLE = False
    # Create placeholder classes for type hints when llama-index is not installed
    BaseChatStore = object
    ChatMessage = None


class SynaVectorStore(VectorStore):
    """
    LlamaIndex VectorStore backed by Syna.
    
    This class implements the LlamaIndex VectorStore interface, allowing
    Syna to be used as a vector database for RAG (Retrieval-Augmented
    Generation) applications with LlamaIndex.
    
    Example:
        >>> from llama_index.core import VectorStoreIndex, StorageContext
        >>> from synadb.integrations.llamaindex import SynaVectorStore
        >>> 
        >>> vector_store = SynaVectorStore(path="index.db", dimensions=1536)
        >>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
        >>> index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    
    Attributes:
        stores_text: Whether the store persists text (True for Syna).
        flat_metadata: Whether metadata is stored flat (True for Syna).
    
    **Validates: Requirements 11.4, 11.5**
    """
    
    stores_text: bool = True
    flat_metadata: bool = True
    
    def __init__(self, path: str, dimensions: int, metric: str = "cosine"):
        """
        Create or open a Syna-backed LlamaIndex VectorStore.
        
        Args:
            path: Path to the Syna database file.
            dimensions: Vector dimensions (64-4096).
            metric: Distance metric ("cosine", "euclidean", "dot_product").
        
        Raises:
            ImportError: If llama-index is not installed.
            ValueError: If dimensions are out of range.
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "llama-index is not installed. "
                "Install it with: pip install llama-index llama-index-core"
            )
        
        from ..vector import VectorStore as SynaStore
        self._store = SynaStore(path, dimensions=dimensions, metric=metric)
        self._dimensions = dimensions
        self._path = path
        # Cache for metadata storage
        self._metadata_cache: Dict[str, dict] = {}
    
    def add(self, nodes: List["TextNode"], **kwargs: Any) -> List[str]:
        """
        Add nodes to the vector store.
        
        Args:
            nodes: List of TextNode objects with embeddings.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            List of node IDs that were added.
        
        Raises:
            ValueError: If a node has no embedding.
        """
        ids = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node {node.node_id} has no embedding")
            
            # Prepare metadata including text
            metadata = {
                "text": node.text,
                **node.metadata,
            }
            
            # Insert vector into the store
            self._store.insert(
                node.node_id,
                np.array(node.embedding, dtype=np.float32),
            )
            
            # Store metadata separately in the database
            self._store_metadata(node.node_id, metadata)
            
            ids.append(node.node_id)
        return ids
    
    def _store_metadata(self, node_id: str, metadata: dict) -> None:
        """Store metadata for a node in the underlying database."""
        # Cache metadata locally for fast retrieval
        self._metadata_cache[node_id] = metadata
        
        try:
            from ..wrapper import SynaDB
            SynaDB._load_library()
            lib = SynaDB._lib
            
            # Store metadata as JSON text
            metadata_key = f"llamaindex_meta/{node_id}"
            metadata_json = json.dumps(metadata)
            
            lib.SYNA_put_text(
                self._path.encode('utf-8'),
                metadata_key.encode('utf-8'),
                metadata_json.encode('utf-8')
            )
        except Exception:
            # If metadata storage fails, continue without it
            # The vector is still stored and metadata is in cache
            pass
    
    def _get_metadata(self, node_id: str) -> dict:
        """Retrieve metadata for a node from the underlying database."""
        # Check cache first
        if node_id in self._metadata_cache:
            return self._metadata_cache[node_id].copy()
        
        try:
            from ..wrapper import SynaDB
            SynaDB._load_library()
            lib = SynaDB._lib
            
            metadata_key = f"llamaindex_meta/{node_id}"
            
            # Set up the function signature if not already done
            if not hasattr(lib, '_llamaindex_metadata_setup'):
                lib.SYNA_get_text.argtypes = [
                    ctypes.c_char_p, 
                    ctypes.c_char_p, 
                    ctypes.POINTER(ctypes.c_char_p)
                ]
                lib.SYNA_get_text.restype = ctypes.c_int32
                lib._llamaindex_metadata_setup = True
            
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
                metadata = json.loads(metadata_json)
                # Cache for future use
                self._metadata_cache[node_id] = metadata
                return metadata
        except Exception:
            pass
        
        return {}
    
    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """
        Delete nodes by ref_doc_id.
        
        Args:
            ref_doc_id: The document ID to delete.
            **kwargs: Additional arguments (ignored).
        """
        # Remove from cache
        self._metadata_cache.pop(ref_doc_id, None)
        
        try:
            self._store.delete(ref_doc_id)
            
            # Also delete metadata
            from ..wrapper import SynaDB
            SynaDB._load_library()
            lib = SynaDB._lib
            
            metadata_key = f"llamaindex_meta/{ref_doc_id}"
            lib.SYNA_delete(
                self._path.encode('utf-8'),
                metadata_key.encode('utf-8')
            )
        except Exception:
            pass
    
    def query(
        self, 
        query: "VectorStoreQuery", 
        **kwargs: Any
    ) -> "VectorStoreQueryResult":
        """
        Query the vector store for similar nodes.
        
        Args:
            query: VectorStoreQuery containing the query embedding.
            **kwargs: Additional arguments (ignored).
        
        Returns:
            VectorStoreQueryResult with matching nodes, similarities, and IDs.
        
        Raises:
            ValueError: If query has no embedding.
        """
        if query.query_embedding is None:
            raise ValueError("Query must have embedding")
        
        # Perform similarity search
        results = self._store.search(
            np.array(query.query_embedding, dtype=np.float32),
            k=query.similarity_top_k or 10
        )
        
        nodes = []
        similarities = []
        ids = []
        
        for r in results:
            # Get metadata for this node
            metadata = self._get_metadata(r.key)
            
            # Extract text from metadata
            text = metadata.pop("text", "")
            
            # Create TextNode
            node = TextNode(
                text=text,
                id_=r.key,
                metadata=metadata,
                embedding=r.vector.tolist(),
            )
            nodes.append(node)
            
            # Convert distance to similarity (1 - distance for cosine)
            similarities.append(1.0 - r.score)
            ids.append(r.key)
        
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
    
    @property
    def client(self) -> Any:
        """Return the underlying Syna VectorStore client."""
        return self._store


class SynaChatStore(BaseChatStore):
    """
    LlamaIndex ChatStore backed by Syna.
    
    This class implements the LlamaIndex BaseChatStore interface, allowing
    Syna to be used as a persistent chat message store for LlamaIndex
    applications.
    
    Example:
        >>> from synadb.integrations.llamaindex import SynaChatStore
        >>> chat_store = SynaChatStore(path="chats.db")
        >>> chat_store.set_messages("session_1", messages)
        >>> retrieved = chat_store.get_messages("session_1")
    
    **Validates: Requirements 11.6**
    """
    
    def __init__(self, path: str):
        """
        Create or open a Syna-backed LlamaIndex ChatStore.
        
        Args:
            path: Path to the Syna database file.
        
        Raises:
            ImportError: If llama-index is not installed.
        """
        if not LLAMAINDEX_CHAT_AVAILABLE:
            raise ImportError(
                "llama-index chat store is not available. "
                "Install it with: pip install llama-index llama-index-core"
            )
        
        from ..wrapper import SynaDB
        self._db = SynaDB(path)
        self._path = path
    
    def set_messages(self, key: str, messages: List["ChatMessage"]) -> None:
        """
        Set messages for a session.
        
        Args:
            key: Session identifier.
            messages: List of ChatMessage objects to store.
        """
        data = [{"role": m.role.value, "content": m.content} for m in messages]
        self._db.put_text(f"chat/{key}", json.dumps(data))
    
    def get_messages(self, key: str) -> List["ChatMessage"]:
        """
        Get messages for a session.
        
        Args:
            key: Session identifier.
        
        Returns:
            List of ChatMessage objects for the session, or empty list if not found.
        """
        data = self._db.get_text(f"chat/{key}")
        if not data:
            return []
        messages = json.loads(data)
        return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
    
    def add_message(self, key: str, message: "ChatMessage") -> None:
        """
        Add a message to a session.
        
        Args:
            key: Session identifier.
            message: ChatMessage to add.
        """
        messages = self.get_messages(key)
        messages.append(message)
        self.set_messages(key, messages)
    
    def delete_messages(self, key: str) -> Optional[List["ChatMessage"]]:
        """
        Delete messages for a session.
        
        Args:
            key: Session identifier.
        
        Returns:
            List of deleted ChatMessage objects, or None if session not found.
        """
        messages = self.get_messages(key)
        if not messages:
            return None
        self._db.delete(f"chat/{key}")
        return messages
    
    def get_keys(self) -> List[str]:
        """
        Get all session keys.
        
        Returns:
            List of session key strings (without the "chat/" prefix).
        """
        return [k[5:] for k in self._db.keys() if k.startswith("chat/")]

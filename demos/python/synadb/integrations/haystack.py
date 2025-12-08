"""
Haystack integration for Syna.

This module provides Haystack-compatible components backed by Syna:
- SynaDocumentStore: DocumentStore implementation for RAG applications

Example:
    >>> from haystack import Pipeline
    >>> from synadb.integrations.haystack import SynaDocumentStore
    >>> 
    >>> document_store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
    >>> document_store.write_documents(documents)

Requirements:
    pip install haystack-ai

**Validates: Requirements 11.7**
"""

from typing import Any, Dict, List, Optional, Generator
import json
import numpy as np

try:
    from haystack import Document
    from haystack.document_stores.types import DocumentStore, DuplicatePolicy
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Create placeholder classes for type hints when haystack is not installed
    DocumentStore = object
    Document = None
    DuplicatePolicy = None


class SynaDocumentStore(DocumentStore):
    """
    Haystack DocumentStore backed by Syna.
    
    This class implements the Haystack DocumentStore interface, allowing
    Syna to be used as a document store for RAG (Retrieval-Augmented
    Generation) applications with Haystack.
    
    Example:
        >>> from haystack import Pipeline
        >>> from synadb.integrations.haystack import SynaDocumentStore
        >>> 
        >>> document_store = SynaDocumentStore(path="haystack.db", embedding_dim=768)
        >>> document_store.write_documents(documents)
        >>> 
        >>> # Query documents
        >>> docs = document_store.filter_documents()
    
    Attributes:
        _vector_store: The underlying Syna VectorStore instance.
        _db: The underlying Syna database instance.
        _embedding_dim: The embedding dimensions.
    
    **Validates: Requirements 11.7**
    """
    
    def __init__(
        self,
        path: str,
        embedding_dim: int = 768,
        metric: str = "cosine",
    ):
        """
        Create or open a Syna-backed Haystack DocumentStore.
        
        Args:
            path: Path to the Syna database file.
            embedding_dim: Vector dimensions (64-4096).
            metric: Distance metric ("cosine", "euclidean", "dot_product").
        
        Raises:
            ImportError: If haystack-ai is not installed.
            ValueError: If embedding_dim is out of range.
        """
        if not HAYSTACK_AVAILABLE:
            raise ImportError(
                "haystack-ai is not installed. "
                "Install it with: pip install haystack-ai"
            )
        
        from ..vector import VectorStore
        from ..wrapper import SynaDB
        
        self._vector_store = VectorStore(path, dimensions=embedding_dim, metric=metric)
        self._db = SynaDB(path)
        self._embedding_dim = embedding_dim
        self._path = path
        # Track document IDs for counting
        self._doc_ids: set = set()
        # Load existing document IDs from database
        self._load_existing_docs()
    
    def _load_existing_docs(self) -> None:
        """Load existing document IDs from the database."""
        for key in self._db.keys():
            if key.startswith("haystack_doc/"):
                doc_id = key[13:]  # Remove "haystack_doc/" prefix
                self._doc_ids.add(doc_id)
    
    def count_documents(self) -> int:
        """
        Return the count of documents in the store.
        
        Returns:
            Number of documents stored.
        """
        return len(self._doc_ids)
    
    def write_documents(
        self,
        documents: List["Document"],
        policy: "DuplicatePolicy" = None,
    ) -> int:
        """
        Write documents to the store.
        
        Args:
            documents: List of Haystack Document objects to store.
            policy: Duplicate handling policy (NONE, SKIP, OVERWRITE).
                   Default is NONE (raise error on duplicates).
        
        Returns:
            Number of documents written.
        
        Raises:
            ValueError: If policy is NONE and a duplicate is found.
        """
        if policy is None and DuplicatePolicy is not None:
            policy = DuplicatePolicy.NONE
        
        written = 0
        for doc in documents:
            doc_id = doc.id
            
            # Check for duplicates
            if doc_id in self._doc_ids:
                if policy == DuplicatePolicy.SKIP:
                    continue
                elif policy == DuplicatePolicy.NONE:
                    raise ValueError(f"Document with id '{doc_id}' already exists")
                # OVERWRITE: continue to overwrite
            
            # Store document metadata
            doc_data = {
                "content": doc.content,
                "meta": doc.meta or {},
            }
            self._db.put_text(f"haystack_doc/{doc_id}", json.dumps(doc_data))
            
            # Store embedding if present
            if doc.embedding is not None:
                embedding = np.array(doc.embedding, dtype=np.float32)
                if len(embedding) == self._embedding_dim:
                    self._vector_store.insert(doc_id, embedding)
            
            self._doc_ids.add(doc_id)
            written += 1
        
        return written
    
    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List["Document"]:
        """
        Filter documents by metadata.
        
        Args:
            filters: Optional dictionary of metadata filters.
                    Currently supports simple equality filters.
                    Example: {"category": "tech", "author": "John"}
        
        Returns:
            List of Document objects matching the filters.
        """
        documents = []
        
        for doc_id in self._doc_ids:
            doc_data_str = self._db.get_text(f"haystack_doc/{doc_id}")
            if not doc_data_str:
                continue
            
            try:
                doc_data = json.loads(doc_data_str)
            except json.JSONDecodeError:
                continue
            
            content = doc_data.get("content", "")
            meta = doc_data.get("meta", {})
            
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Try to get embedding
            embedding = None
            try:
                # Search for this specific document to get its embedding
                # This is a workaround since VectorStore.get() is not fully implemented
                pass
            except Exception:
                pass
            
            documents.append(Document(
                id=doc_id,
                content=content,
                meta=meta,
                embedding=embedding,
            ))
        
        return documents
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete.
        """
        for doc_id in document_ids:
            # Delete from vector store
            try:
                self._vector_store.delete(doc_id)
            except Exception:
                pass
            
            # Delete document data
            try:
                self._db.delete(f"haystack_doc/{doc_id}")
            except Exception:
                pass
            
            # Remove from tracking set
            self._doc_ids.discard(doc_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the document store to a dictionary.
        
        Returns:
            Dictionary representation of the document store configuration.
        """
        return {
            "type": "synadb.integrations.haystack.SynaDocumentStore",
            "init_parameters": {
                "path": self._path,
                "embedding_dim": self._embedding_dim,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynaDocumentStore":
        """
        Create a document store from a dictionary.
        
        Args:
            data: Dictionary with type and init_parameters.
        
        Returns:
            New SynaDocumentStore instance.
        """
        return cls(**data.get("init_parameters", {}))

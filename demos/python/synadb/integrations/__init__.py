"""
Syna integrations with popular ML/AI frameworks.

This module provides integrations with:
- LangChain: VectorStore, ChatMessageHistory, DocumentLoader
- LlamaIndex: VectorStore for RAG applications
- Haystack: DocumentStore for RAG applications

Example (LangChain):
    >>> from synadb.integrations.langchain import SynaVectorStore, SynaChatMessageHistory
    >>> from langchain_openai import OpenAIEmbeddings
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

Example (LlamaIndex):
    >>> from llama_index.core import VectorStoreIndex, StorageContext
    >>> from synadb.integrations.llamaindex import SynaVectorStore as LlamaIndexSynaVectorStore
    >>> 
    >>> vector_store = LlamaIndexSynaVectorStore(path="index.db", dimensions=1536)
    >>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
    >>> index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
"""

# Import LangChain integrations if available
try:
    from .langchain import SynaVectorStore, SynaChatMessageHistory, SynaLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    SynaVectorStore = None
    SynaChatMessageHistory = None
    SynaLoader = None

# Import LlamaIndex integrations if available
try:
    from .llamaindex import SynaVectorStore as LlamaIndexSynaVectorStore
    from .llamaindex import SynaChatStore as LlamaIndexSynaChatStore
    from .llamaindex import LLAMAINDEX_AVAILABLE, LLAMAINDEX_CHAT_AVAILABLE
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LLAMAINDEX_CHAT_AVAILABLE = False
    LlamaIndexSynaVectorStore = None
    LlamaIndexSynaChatStore = None

# Import Haystack integrations if available
try:
    from .haystack import SynaDocumentStore as HaystackSynaDocumentStore
    from .haystack import HAYSTACK_AVAILABLE
except ImportError:
    HAYSTACK_AVAILABLE = False
    HaystackSynaDocumentStore = None

__all__ = []

if LANGCHAIN_AVAILABLE:
    __all__.extend(["SynaVectorStore", "SynaChatMessageHistory", "SynaLoader"])

if LLAMAINDEX_AVAILABLE:
    __all__.extend(["LlamaIndexSynaVectorStore"])

if LLAMAINDEX_CHAT_AVAILABLE:
    __all__.extend(["LlamaIndexSynaChatStore"])

if HAYSTACK_AVAILABLE:
    __all__.extend(["HaystackSynaDocumentStore"])

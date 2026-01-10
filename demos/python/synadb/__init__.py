"""
Syna Python Wrapper

A high-level Python interface for the Syna embedded database.

Example:
    >>> from synadb import SynaDB
    >>> with SynaDB("my.db") as db:
    ...     db.put_float("temperature", 23.5)
    ...     print(db.get_float("temperature"))
    23.5

For RL experience collection:
    >>> from synadb import ExperienceCollector
    >>> collector = ExperienceCollector("exp.db", machine_id="gpu_server_1")
    >>> collector.log_transition(state, action, reward, next_state)

For vector storage and similarity search:
    >>> from synadb import VectorStore
    >>> store = VectorStore("vectors.db", dimensions=768)
    >>> store.insert("doc1", embedding)
    >>> results = store.search(query_embedding, k=5)

For batch tensor operations:
    >>> from synadb import TensorEngine
    >>> engine = TensorEngine("data.db")
    >>> engine.put_tensor("train/", X_train)
    >>> X = engine.get_tensor("train/*", dtype=np.float32)

For model versioning and registry:
    >>> from synadb import ModelRegistry
    >>> registry = ModelRegistry("models.db")
    >>> version = registry.save("classifier", model, {"accuracy": "0.95"})
    >>> loaded = registry.load("classifier")

For experiment tracking:
    >>> from synadb import Experiment
    >>> exp = Experiment("mnist", "experiments.db")
    >>> with exp.start_run(tags=["baseline"]) as run:
    ...     run.log_params({"lr": 0.001, "batch_size": 32})
    ...     for epoch in range(100):
    ...         run.log_metric("loss", loss, step=epoch)
    ...     run.log_artifact("model.pt", model.state_dict())

For LLM framework integrations:
    >>> # LangChain integration (lazy loaded)
    >>> langchain = synadb.get_langchain()
    >>> vectorstore = langchain.SynaVectorStore(path="vectors.db", dimensions=768)
    >>> 
    >>> # LlamaIndex integration (lazy loaded)
    >>> llamaindex = synadb.get_llamaindex()
    >>> vector_store = llamaindex.SynaVectorStore(path="index.db", dimensions=1536)
    >>>
    >>> # Haystack integration (lazy loaded)
    >>> haystack = synadb.get_haystack()
    >>> doc_store = haystack.SynaDocumentStore(path="docs.db")

For PyTorch integration:
    >>> from synadb.torch import SynaDataset, SynaDataLoader
    >>> dataset = SynaDataset("mnist.db", pattern="train/*")
    >>> loader = SynaDataLoader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     # train with batch

For TensorFlow integration:
    >>> from synadb.tensorflow import syna_dataset
    >>> import tensorflow as tf
    >>> dataset = syna_dataset(
    ...     path="data.db",
    ...     pattern="train/*",
    ...     batch_size=32
    ... ).prefetch(tf.data.AUTOTUNE)
    >>> for batch in dataset:
    ...     # train with batch

For GPU Direct memory access:
    >>> gpu = synadb.get_gpu()
    >>> if gpu.is_gpu_available():
    ...     # Load tensor directly to GPU
    ...     tensor = gpu.get_tensor_cuda("data.db", "train/*", device=0)
    ...     print(f"Tensor on {tensor.device}")
    ...     
    ...     # Prefetch for faster access
    ...     gpu.prefetch_to_gpu("data.db", "val/*", device=0)

For TPU support via TensorFlow/JAX:
    >>> tpu = synadb.get_tpu()
    >>> # TensorFlow dataset for TPU training
    >>> dataset = tpu.get_tensor_dataset("data.db", "train/*", batch_size=512)
    >>> dataset = dataset.prefetch(tf.data.AUTOTUNE)
    >>> 
    >>> # JAX tensor on TPU
    >>> if tpu.is_tpu_available():
    ...     tensor = tpu.get_tensor_jax("data.db", "weights/*", device="tpu")
    >>> 
    >>> # TPU-optimized batch iterator
    >>> for batch in tpu.iter_batches_for_tpu("data.db", "train/*", batch_size=512):
    ...     train_step(batch)  # batch.shape[0] is always 512

For Syna Studio (Web UI):
    >>> from synadb import launch_studio
    >>> launch_studio("mydb.db", port=8501)
    # Opens browser to http://localhost:8501
    # Requires: pip install flask

For data export/import:
    >>> with SynaDB("data.db") as db:
    ...     # Export to various formats
    ...     db.export_json("data.json")
    ...     db.export_jsonl("data.jsonl")
    ...     db.export_csv("data.csv")
    ...     db.export_pickle("data.pkl")
    ...     db.export_parquet("data.parquet")  # requires pyarrow
    ...     db.export_arrow("data.arrow")       # requires pyarrow
    ...     db.export_msgpack("data.msgpack")   # requires msgpack
    ...     
    ...     # Filter by key pattern
    ...     db.export_parquet("sensors.parquet", key_pattern="sensor/*")
    ...     
    ...     # Import from files
    ...     db.import_json("config.json", key_prefix="config/")
"""

from .wrapper import SynaDB, SynaError
from .experience import ExperienceCollector, Transition, SessionContext
from .vector import VectorStore, SearchResult
from .mmap_vector import MmapVectorStore, MmapSearchResult
from .gwi import GravityWellIndex, GwiSearchResult
from .cascade import CascadeIndex, SearchResult as CascadeSearchResult
from .sparse_vector_store import SparseVectorStore, SparseSearchResult, SparseIndexStats, SparseVectorStoreError
from .tensor import TensorEngine
from .models import ModelRegistry, ModelVersion, ModelStage
from .experiment import Experiment, Run, RunStatus
from .studio import launch as launch_studio, FLASK_AVAILABLE

# Import integrations submodule
from . import integrations

__version__ = "1.1.0"


# Lazy imports for integrations
def get_langchain():
    """
    Lazy load LangChain integration module.
    
    Returns:
        The langchain integration module with SynaVectorStore, 
        SynaChatMessageHistory, and SynaLoader classes.
        
    Raises:
        ImportError: If langchain is not installed.
        
    Example:
        >>> langchain = synadb.get_langchain()
        >>> vectorstore = langchain.SynaVectorStore(path="vectors.db", dimensions=768)
    """
    from .integrations import langchain
    return langchain


def get_llamaindex():
    """
    Lazy load LlamaIndex integration module.
    
    Returns:
        The llamaindex integration module with SynaVectorStore 
        and SynaChatStore classes.
        
    Raises:
        ImportError: If llama-index is not installed.
        
    Example:
        >>> llamaindex = synadb.get_llamaindex()
        >>> vector_store = llamaindex.SynaVectorStore(path="index.db", dimensions=1536)
    """
    from .integrations import llamaindex
    return llamaindex


def get_haystack():
    """
    Lazy load Haystack integration module.
    
    Returns:
        The haystack integration module with SynaDocumentStore class.
        
    Raises:
        ImportError: If haystack-ai is not installed.
        
    Example:
        >>> haystack = synadb.get_haystack()
        >>> doc_store = haystack.SynaDocumentStore(path="docs.db")
    """
    from .integrations import haystack
    return haystack


def get_mlflow():
    """
    Lazy load MLflow integration module.
    
    Returns:
        The mlflow integration module with SynaTrackingStore class
        and register_syna_tracking_store function.
        
    Raises:
        ImportError: If mlflow is not installed.
        
    Example:
        >>> mlflow_integration = synadb.get_mlflow()
        >>> mlflow_integration.register_syna_tracking_store()
        >>> import mlflow
        >>> mlflow.set_tracking_uri("synadb:///experiments.db")
    """
    from .integrations import mlflow
    return mlflow


def get_torch():
    """
    Lazy load PyTorch integration module.
    
    Returns:
        The torch integration module with SynaDataset and SynaDataLoader classes.
        
    Raises:
        ImportError: If torch is not installed.
        
    Example:
        >>> torch_integration = synadb.get_torch()
        >>> dataset = torch_integration.SynaDataset("data.db", pattern="train/*")
        >>> loader = torch_integration.SynaDataLoader(dataset, batch_size=32)
    """
    from . import torch as torch_module
    return torch_module


def get_tensorflow():
    """
    Lazy load TensorFlow integration module.
    
    Returns:
        The tensorflow integration module with syna_dataset function
        and SynaDataset class.
        
    Raises:
        ImportError: If tensorflow is not installed.
        
    Example:
        >>> tf_integration = synadb.get_tensorflow()
        >>> dataset = tf_integration.syna_dataset("data.db", pattern="train/*", batch_size=32)
        >>> for batch in dataset.prefetch(tf.data.AUTOTUNE):
        ...     # train with batch
    """
    from . import tensorflow as tf_module
    return tf_module


def get_studio():
    """
    Lazy load Studio web UI module.
    
    Returns:
        The studio module with launch function.
        
    Example:
        >>> studio = synadb.get_studio()
        >>> studio.launch("mydb.db", port=8501)
    """
    from . import studio
    return studio


def get_jupyter():
    """
    Lazy load Jupyter magic commands module.
    
    Returns:
        The jupyter module with SynaMagics class and load_ipython_extension function.
        
    Example:
        >>> # In Jupyter notebook:
        >>> %load_ext synadb
        >>> %syna_info mydb.db
    
    Note:
        The preferred way to use Jupyter magic commands is via %load_ext synadb.
        This function is provided for programmatic access to the module.
    """
    from . import jupyter
    return jupyter


def get_gpu():
    """
    Lazy load GPU integration module.
    
    Returns:
        The gpu module with get_tensor_cuda, prefetch_to_gpu, and utility functions.
        
    Raises:
        ImportError: If torch is not installed.
        
    Example:
        >>> gpu = synadb.get_gpu()
        >>> if gpu.is_gpu_available():
        ...     tensor = gpu.get_tensor_cuda("data.db", "train/*", device=0)
        ...     print(f"Loaded tensor on {tensor.device}")
    """
    from . import gpu
    return gpu


def get_tpu():
    """
    Lazy load TPU integration module.
    
    Returns:
        The tpu module with get_tensor_dataset, get_tensor_jax, iter_batches_for_tpu,
        and utility functions for TPU support via TensorFlow/JAX framework integration.
        
    Raises:
        ImportError: If tensorflow or jax is not installed (depending on function used).
        
    Example:
        >>> tpu = synadb.get_tpu()
        >>> if tpu.is_tpu_available():
        ...     tensor = tpu.get_tensor_jax("data.db", "train/*", device="tpu")
        ...     print(f"Loaded tensor on TPU")
        >>> 
        >>> # TensorFlow dataset for TPU
        >>> dataset = tpu.get_tensor_dataset("data.db", "train/*", batch_size=512)
    """
    from . import tpu
    return tpu


__all__ = [
    "SynaDB",
    "SynaError",
    "ExperienceCollector",
    "Transition",
    "SessionContext",
    "VectorStore",
    "SearchResult",
    "MmapVectorStore",
    "MmapSearchResult",
    "GravityWellIndex",
    "GwiSearchResult",
    "CascadeIndex",
    "CascadeSearchResult",
    "SparseVectorStore",
    "SparseSearchResult",
    "SparseIndexStats",
    "SparseVectorStoreError",
    "TensorEngine",
    "ModelRegistry",
    "ModelVersion",
    "ModelStage",
    "Experiment",
    "Run",
    "RunStatus",
    "integrations",
    "get_langchain",
    "get_llamaindex",
    "get_haystack",
    "get_mlflow",
    "get_torch",
    "get_tensorflow",
    "get_studio",
    "get_jupyter",
    "get_gpu",
    "get_tpu",
    "launch_studio",
    "FLASK_AVAILABLE",
]


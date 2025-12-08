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
"""

from .wrapper import SynaDB, SynaError
from .experience import ExperienceCollector, Transition, SessionContext
from .vector import VectorStore, SearchResult
from .tensor import TensorEngine
from .models import ModelRegistry, ModelVersion, ModelStage
from .experiment import Experiment, Run, RunStatus

# Import integrations submodule
from . import integrations

__version__ = "0.2.0"
__all__ = [
    "SynaDB",
    "SynaError",
    "ExperienceCollector",
    "Transition",
    "SessionContext",
    "VectorStore",
    "SearchResult",
    "TensorEngine",
    "ModelRegistry",
    "ModelVersion",
    "ModelStage",
    "Experiment",
    "Run",
    "RunStatus",
    "integrations",
]


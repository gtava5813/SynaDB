"""
Experiment tracking for ML workflows.

This module provides a high-level Python interface for tracking ML experiments
using the Syna database. It supports logging parameters, metrics, and artifacts
with automatic run management.

Example:
    >>> from synadb import Experiment
    >>> exp = Experiment("mnist", "experiments.db")
    >>> with exp.start_run(tags=["baseline"]) as run:
    ...     run.log_params({"lr": 0.001, "batch_size": 32})
    ...     for epoch in range(100):
    ...         run.log_metric("loss", loss, step=epoch)
    ...     run.log_artifact("model.pt", model.state_dict())
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .wrapper import SynaDB, SynaError


class RunStatus(Enum):
    """Status of an experiment run.
    
    Attributes:
        RUNNING: Run is currently in progress.
        COMPLETED: Run finished successfully.
        FAILED: Run failed with an error.
        KILLED: Run was manually terminated.
    """
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class Run:
    """A single experiment run.
    
    Represents a single execution of an experiment with its parameters,
    metrics, and artifacts. Supports context manager protocol for automatic
    run completion.
    
    Attributes:
        id: Unique identifier for the run.
        experiment_name: Name of the parent experiment.
        started_at: Unix timestamp when the run started.
        ended_at: Unix timestamp when the run ended (None if still running).
        status: Current status of the run.
        params: Dictionary of hyperparameters logged for this run.
        tags: List of tags associated with this run.
    
    Example:
        >>> with exp.start_run() as run:
        ...     run.log_param("lr", 0.001)
        ...     run.log_metric("loss", 0.5, step=1)
    
    _Requirements: 5.1, 5.2, 5.3_
    """
    id: str
    experiment_name: str
    started_at: int
    ended_at: Optional[int] = None
    status: RunStatus = RunStatus.RUNNING
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    _tracker: "ExperimentTracker" = field(repr=False, default=None)

    def log_param(self, key: str, value: Any) -> None:
        """Log a hyperparameter.
        
        Args:
            key: Parameter name (e.g., "learning_rate", "batch_size").
            value: Parameter value (will be converted to string for storage).
        
        Example:
            >>> run.log_param("lr", 0.001)
            >>> run.log_param("optimizer", "adam")
        
        _Requirements: 5.2_
        """
        self.params[key] = value
        self._tracker._log_param(self.id, key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters.
        
        Args:
            params: Dictionary of parameter names to values.
        
        Example:
            >>> run.log_params({"lr": 0.001, "batch_size": 32, "epochs": 100})
        
        _Requirements: 5.2_
        """
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        """Log a metric value.
        
        Args:
            key: Metric name (e.g., "loss", "accuracy").
            value: Metric value (must be numeric).
            step: Optional step number for time-series metrics (e.g., epoch).
        
        Example:
            >>> run.log_metric("loss", 0.5)
            >>> run.log_metric("accuracy", 0.95, step=10)
        
        _Requirements: 5.3_
        """
        self._tracker._log_metric(self.id, key, value, step)

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number for all metrics.
        
        Example:
            >>> run.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=10)
        
        _Requirements: 5.3_
        """
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (file, plot, model).
        
        Artifacts are stored as bytes. If data is not bytes, it will be
        serialized using pickle.
        
        Args:
            name: Artifact name (e.g., "model.pt", "confusion_matrix.png").
            data: Artifact data (bytes, or any picklable object).
        
        Example:
            >>> run.log_artifact("model.pt", model.state_dict())
            >>> run.log_artifact("plot.png", open("plot.png", "rb").read())
        
        _Requirements: 5.4_
        """
        self._tracker._log_artifact(self.id, name, data)

    def end(self, status: str = "completed") -> None:
        """End the run.
        
        Args:
            status: Final status ("completed", "failed", "killed").
        
        Example:
            >>> run.end("completed")
            >>> run.end("failed")
        
        _Requirements: 5.1_
        """
        self.status = RunStatus(status)
        self.ended_at = int(time.time())
        self._tracker._end_run(self.id, status)

    def __enter__(self) -> "Run":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic status handling.
        
        If an exception occurred, marks the run as failed.
        Otherwise, marks it as completed if still running.
        """
        if exc_type is not None:
            self.end("failed")
        elif self.status == RunStatus.RUNNING:
            self.end("completed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary for serialization.
        
        Returns:
            Dictionary representation of the run.
        """
        return {
            "id": self.id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status.value,
            "params": self.params,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], tracker: "ExperimentTracker" = None) -> "Run":
        """Create a Run from a dictionary.
        
        Args:
            data: Dictionary with run data.
            tracker: Optional ExperimentTracker instance.
        
        Returns:
            Run instance.
        """
        # Support both 'experiment' and 'experiment_name' keys for compatibility
        experiment_name = data.get("experiment_name") or data.get("experiment", "")
        return cls(
            id=data["id"],
            experiment_name=experiment_name,
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            status=RunStatus(data.get("status", "running")),
            params=data.get("params", {}),
            tags=data.get("tags", []),
            _tracker=tracker,
        )


class ExperimentTracker:
    """
    Internal tracker for experiment operations.
    
    Handles the low-level database operations for experiment tracking.
    Users should interact with the Experiment class instead.
    """
    
    def __init__(self, db: SynaDB, experiment_name: str):
        """Initialize the tracker.
        
        Args:
            db: SynaDB instance for storage.
            experiment_name: Name of the experiment.
        """
        self._db = db
        self._experiment_name = experiment_name
        self._metric_steps: Dict[str, int] = {}  # Track auto-increment steps
    
    def _log_param(self, run_id: str, key: str, value: Any) -> None:
        """Store a parameter in the database."""
        param_key = f"exp/{self._experiment_name}/run/{run_id}/param/{key}"
        self._db.put_text(param_key, str(value))
    
    def _log_metric(self, run_id: str, key: str, value: float, step: int = None) -> None:
        """Store a metric in the database."""
        # Auto-increment step if not provided
        metric_id = f"{run_id}/{key}"
        if step is None:
            step = self._metric_steps.get(metric_id, 0)
            self._metric_steps[metric_id] = step + 1
        else:
            self._metric_steps[metric_id] = step + 1
        
        # Store metric value with step
        metric_key = f"exp/{self._experiment_name}/run/{run_id}/metric/{key}/{step}"
        self._db.put_float(metric_key, value)
        
        # Also store in a time-series key for tensor extraction
        ts_key = f"exp/{self._experiment_name}/run/{run_id}/metric_ts/{key}"
        self._db.put_float(ts_key, value)
    
    def _log_artifact(self, run_id: str, name: str, data: Any) -> None:
        """Store an artifact in the database."""
        import pickle
        
        artifact_key = f"exp/{self._experiment_name}/run/{run_id}/artifact/{name}"
        
        if isinstance(data, bytes):
            self._db.put_bytes(artifact_key, data)
        else:
            # Serialize with pickle
            self._db.put_bytes(artifact_key, pickle.dumps(data))
    
    def _end_run(self, run_id: str, status: str) -> None:
        """Update run metadata with end status."""
        meta_key = f"exp/{self._experiment_name}/run/{run_id}/meta"
        meta_json = self._db.get_text(meta_key)
        
        if meta_json:
            meta = json.loads(meta_json)
            meta["status"] = status
            meta["ended_at"] = int(time.time())
            self._db.put_text(meta_key, json.dumps(meta))
    
    def _update_meta(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Update run metadata."""
        meta_key = f"exp/{self._experiment_name}/run/{run_id}/meta"
        meta_json = self._db.get_text(meta_key)
        
        if meta_json:
            meta = json.loads(meta_json)
            meta.update(updates)
            self._db.put_text(meta_key, json.dumps(meta))


class Experiment:
    """
    Experiment tracking for ML workflows.
    
    Provides a high-level API for tracking ML experiments, including
    parameters, metrics, and artifacts. Supports querying and comparing
    runs across experiments.
    
    Example:
        >>> exp = Experiment("mnist", "experiments.db")
        >>> with exp.start_run(tags=["baseline"]) as run:
        ...     run.log_params({"lr": 0.001, "batch_size": 32})
        ...     for epoch in range(100):
        ...         run.log_metric("loss", loss, step=epoch)
        ...     run.log_artifact("model.pt", model.state_dict())
    
    Attributes:
        name: Name of the experiment.
        path: Path to the database file.
    
    _Requirements: 5.1, 5.2, 5.3, 5.4, 8.1_
    """

    def __init__(self, name: str, db_path: str = "experiments.db"):
        """Create or connect to an experiment.
        
        Args:
            name: Name of the experiment (e.g., "mnist_classifier").
            db_path: Path to the database file. Will be created if it doesn't exist.
        
        Raises:
            SynaError: If the database cannot be opened.
        
        Example:
            >>> exp = Experiment("mnist", "experiments.db")
        """
        self.name = name
        self._path = db_path
        self._db = SynaDB(db_path)
        self._tracker = ExperimentTracker(self._db, name)

    def start_run(self, tags: List[str] = None) -> Run:
        """Start a new run.
        
        Creates a new run with a unique ID and stores initial metadata.
        
        Args:
            tags: Optional list of tags for the run (e.g., ["baseline", "v1"]).
        
        Returns:
            Run object for logging parameters, metrics, and artifacts.
        
        Example:
            >>> run = exp.start_run(tags=["baseline"])
            >>> # Or use as context manager
            >>> with exp.start_run() as run:
            ...     run.log_param("lr", 0.001)
        
        _Requirements: 5.1_
        """
        run = Run(
            id=str(uuid.uuid4()),
            experiment_name=self.name,
            started_at=int(time.time()),
            tags=tags or [],
            _tracker=self._tracker,
        )
        
        # Store run metadata
        meta = {
            "id": run.id,
            "experiment": self.name,
            "started_at": run.started_at,
            "tags": run.tags,
            "status": "running",
            "params": {},
        }
        self._db.put_text(
            f"exp/{self.name}/run/{run.id}/meta",
            json.dumps(meta)
        )
        
        return run

    def query(
        self,
        filter: Dict[str, Any] = None,
        sort_by: str = None,
        ascending: bool = True,
    ) -> List[Run]:
        """Query runs with optional filtering and sorting.
        
        Args:
            filter: Optional filter dictionary. Supports:
                - "status": Filter by run status (e.g., "completed").
                - "tags": Filter by tags (run must have all specified tags).
                - Parameter filters: e.g., {"lr": 0.001} to filter by param value.
            sort_by: Optional metric or field to sort by (e.g., "accuracy", "started_at").
            ascending: Sort order (True for ascending, False for descending).
        
        Returns:
            List of Run objects matching the filter, sorted as specified.
        
        Example:
            >>> # Get all completed runs
            >>> runs = exp.query(filter={"status": "completed"})
            >>> # Get runs sorted by accuracy
            >>> runs = exp.query(sort_by="accuracy", ascending=False)
            >>> # Filter by parameter
            >>> runs = exp.query(filter={"lr": 0.001})
        
        _Requirements: 5.5, 5.6_
        """
        runs = []
        all_keys = self._db.keys()
        
        # Find all run metadata keys
        prefix = f"exp/{self.name}/run/"
        meta_suffix = "/meta"
        
        run_ids = set()
        for key in all_keys:
            if key.startswith(prefix) and key.endswith(meta_suffix):
                # Extract run ID: exp/{name}/run/{run_id}/meta
                parts = key.split("/")
                if len(parts) >= 5:
                    run_ids.add(parts[3])
        
        # Load each run
        for run_id in run_ids:
            meta_key = f"exp/{self.name}/run/{run_id}/meta"
            meta_json = self._db.get_text(meta_key)
            
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                    
                    # Load params from individual keys
                    params = {}
                    param_prefix = f"exp/{self.name}/run/{run_id}/param/"
                    for key in all_keys:
                        if key.startswith(param_prefix):
                            param_name = key[len(param_prefix):]
                            param_value = self._db.get_text(key)
                            if param_value:
                                params[param_name] = param_value
                    
                    meta["params"] = params
                    
                    run = Run.from_dict(meta, self._tracker)
                    runs.append(run)
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Apply filters
        if filter:
            filtered_runs = []
            for run in runs:
                match = True
                
                # Status filter
                if "status" in filter:
                    if run.status.value != filter["status"]:
                        match = False
                
                # Tags filter
                if "tags" in filter and match:
                    required_tags = filter["tags"]
                    if isinstance(required_tags, str):
                        required_tags = [required_tags]
                    if not all(tag in run.tags for tag in required_tags):
                        match = False
                
                # Parameter filters
                for key, value in filter.items():
                    if key not in ("status", "tags") and match:
                        if key not in run.params or str(run.params[key]) != str(value):
                            match = False
                
                if match:
                    filtered_runs.append(run)
            
            runs = filtered_runs
        
        # Apply sorting
        if sort_by:
            if sort_by == "started_at":
                runs.sort(key=lambda r: r.started_at, reverse=not ascending)
            elif sort_by == "ended_at":
                runs.sort(key=lambda r: r.ended_at or 0, reverse=not ascending)
            else:
                # Sort by metric - need to load metric values
                def get_metric_value(run: Run) -> float:
                    # Get the latest value for this metric
                    ts_key = f"exp/{self.name}/run/{run.id}/metric_ts/{sort_by}"
                    tensor = self._db.get_history_tensor(ts_key)
                    if len(tensor) > 0:
                        return tensor[-1]  # Latest value
                    return float('-inf') if ascending else float('inf')
                
                runs.sort(key=get_metric_value, reverse=not ascending)
        
        return runs

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a specific run by ID.
        
        Args:
            run_id: The unique run identifier.
        
        Returns:
            Run object if found, None otherwise.
        
        Example:
            >>> run = exp.get_run("abc123")
        """
        meta_key = f"exp/{self.name}/run/{run_id}/meta"
        meta_json = self._db.get_text(meta_key)
        
        if meta_json is None:
            return None
        
        try:
            meta = json.loads(meta_json)
            
            # Load params
            all_keys = self._db.keys()
            params = {}
            param_prefix = f"exp/{self.name}/run/{run_id}/param/"
            for key in all_keys:
                if key.startswith(param_prefix):
                    param_name = key[len(param_prefix):]
                    param_value = self._db.get_text(key)
                    if param_value:
                        params[param_name] = param_value
            
            meta["params"] = params
            return Run.from_dict(meta, self._tracker)
        except (json.JSONDecodeError, KeyError):
            return None

    def get_metrics(self, run_id: str, metric_name: str) -> List[tuple]:
        """Get all values for a metric as (step, value) pairs.
        
        Args:
            run_id: The run identifier.
            metric_name: Name of the metric.
        
        Returns:
            List of (step, value) tuples in step order.
        
        Example:
            >>> metrics = exp.get_metrics(run.id, "loss")
            >>> for step, value in metrics:
            ...     print(f"Step {step}: {value}")
        
        _Requirements: 5.3_
        """
        all_keys = self._db.keys()
        prefix = f"exp/{self.name}/run/{run_id}/metric/{metric_name}/"
        
        metrics = []
        for key in all_keys:
            if key.startswith(prefix):
                step_str = key[len(prefix):]
                try:
                    step = int(step_str)
                    value = self._db.get_float(key)
                    if value is not None:
                        metrics.append((step, value))
                except ValueError:
                    pass
        
        # Sort by step
        metrics.sort(key=lambda x: x[0])
        return metrics

    def get_metric_tensor(self, run_id: str, metric_name: str) -> "np.ndarray":
        """Get all values for a metric as a numpy array.
        
        Optimized for ML workflows - returns contiguous float64 array.
        
        Args:
            run_id: The run identifier.
            metric_name: Name of the metric.
        
        Returns:
            numpy array of metric values in chronological order.
        
        Example:
            >>> loss = exp.get_metric_tensor(run.id, "loss")
            >>> plt.plot(loss)
        
        _Requirements: 5.3_
        """
        ts_key = f"exp/{self.name}/run/{run_id}/metric_ts/{metric_name}"
        return self._db.get_history_tensor(ts_key)

    def get_artifact(self, run_id: str, name: str) -> Optional[Any]:
        """Get an artifact by name.
        
        Args:
            run_id: The run identifier.
            name: Artifact name.
        
        Returns:
            Artifact data (deserialized if it was pickled), or None if not found.
        
        Example:
            >>> model_state = exp.get_artifact(run.id, "model.pt")
        
        _Requirements: 5.4_
        """
        import pickle
        
        artifact_key = f"exp/{self.name}/run/{run_id}/artifact/{name}"
        data = self._db.get_bytes(artifact_key)
        
        if data is None:
            return None
        
        # Try to unpickle
        try:
            return pickle.loads(data)
        except (pickle.UnpicklingError, Exception):
            # Return raw bytes if unpickling fails
            return data

    def list_artifacts(self, run_id: str) -> List[str]:
        """List all artifact names for a run.
        
        Args:
            run_id: The run identifier.
        
        Returns:
            List of artifact names.
        
        Example:
            >>> artifacts = exp.list_artifacts(run.id)
            >>> print(artifacts)  # ["model.pt", "confusion_matrix.png"]
        
        _Requirements: 5.4_
        """
        all_keys = self._db.keys()
        prefix = f"exp/{self.name}/run/{run_id}/artifact/"
        
        artifacts = []
        for key in all_keys:
            if key.startswith(prefix):
                artifact_name = key[len(prefix):]
                artifacts.append(artifact_name)
        
        return artifacts

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare.
        
        Returns:
            Dictionary with comparison data including params and metrics.
        
        Example:
            >>> comparison = exp.compare_runs([run1.id, run2.id])
            >>> print(comparison)
        
        _Requirements: 5.7_
        """
        comparison = {
            "runs": [],
            "params": {},
            "metrics": {},
        }
        
        # Collect all param and metric names
        all_params = set()
        all_metrics = set()
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                comparison["runs"].append(run.to_dict())
                all_params.update(run.params.keys())
                
                # Find metric names
                all_keys = self._db.keys()
                metric_prefix = f"exp/{self.name}/run/{run_id}/metric_ts/"
                for key in all_keys:
                    if key.startswith(metric_prefix):
                        metric_name = key[len(metric_prefix):]
                        all_metrics.add(metric_name)
        
        # Build param comparison
        for param in all_params:
            comparison["params"][param] = {}
            for run_id in run_ids:
                run = self.get_run(run_id)
                if run and param in run.params:
                    comparison["params"][param][run_id] = run.params[param]
        
        # Build metric comparison (latest values)
        for metric in all_metrics:
            comparison["metrics"][metric] = {}
            for run_id in run_ids:
                tensor = self.get_metric_tensor(run_id, metric)
                if len(tensor) > 0:
                    comparison["metrics"][metric][run_id] = {
                        "latest": float(tensor[-1]),
                        "min": float(tensor.min()),
                        "max": float(tensor.max()),
                        "mean": float(tensor.mean()),
                    }
        
        return comparison

    def delete_run(self, run_id: str) -> int:
        """Delete a run and all its data.
        
        Args:
            run_id: The run identifier.
        
        Returns:
            Number of keys deleted.
        
        Example:
            >>> exp.delete_run(run.id)
        """
        all_keys = self._db.keys()
        prefix = f"exp/{self.name}/run/{run_id}/"
        
        count = 0
        for key in all_keys:
            if key.startswith(prefix):
                try:
                    self._db.delete(key)
                    count += 1
                except SynaError:
                    pass
        
        return count

    def list_runs(self) -> List[str]:
        """List all run IDs for this experiment.
        
        Returns:
            List of run IDs.
        
        Example:
            >>> run_ids = exp.list_runs()
        """
        all_keys = self._db.keys()
        prefix = f"exp/{self.name}/run/"
        meta_suffix = "/meta"
        
        run_ids = []
        for key in all_keys:
            if key.startswith(prefix) and key.endswith(meta_suffix):
                parts = key.split("/")
                if len(parts) >= 5:
                    run_ids.append(parts[3])
        
        return run_ids

    def close(self) -> None:
        """Close the database."""
        self._db.close()

    def __enter__(self) -> "Experiment":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass

    @property
    def path(self) -> str:
        """Return the database path."""
        return self._path

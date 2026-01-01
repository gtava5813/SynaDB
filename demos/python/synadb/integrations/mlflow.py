"""
MLflow integration for Syna.

This module provides MLflow tracking backend backed by Syna database,
allowing you to use Syna as the storage layer for MLflow experiments.

Example:
    >>> import mlflow
    >>> from synadb.integrations.mlflow import register_syna_tracking_store
    >>> 
    >>> # Register Syna as a tracking store
    >>> register_syna_tracking_store()
    >>> 
    >>> # Use Syna as tracking URI
    >>> mlflow.set_tracking_uri("synadb:///experiments.db")
    >>> 
    >>> # Use MLflow as normal
    >>> with mlflow.start_run():
    ...     mlflow.log_param("lr", 0.001)
    ...     mlflow.log_metric("accuracy", 0.95)

Requirements:
    pip install mlflow
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pathlib import Path

if TYPE_CHECKING:
    from mlflow.entities import (
        Experiment, Run, RunInfo, RunData, Metric, Param, RunTag,
        ViewType, LifecycleStage, RunStatus as MLflowRunStatus
    )

try:
    from mlflow.tracking import MlflowClient
    from mlflow.store.tracking.abstract_store import AbstractStore
    from mlflow.entities import (
        Experiment, Run, RunInfo, RunData, Metric, Param, RunTag,
        ViewType, LifecycleStage, RunStatus as MLflowRunStatus
    )
    from mlflow.entities.experiment import Experiment as MLflowExperiment
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    AbstractStore = object
    MLflowExperiment = None


class SynaTrackingStore(AbstractStore):
    """
    MLflow tracking store backed by Syna.
    
    This class implements the MLflow AbstractStore interface, allowing
    Syna to be used as a backend for MLflow experiment tracking.
    
    Usage:
        Register the store and set tracking URI:
        >>> from synadb.integrations.mlflow import register_syna_tracking_store
        >>> register_syna_tracking_store()
        >>> mlflow.set_tracking_uri("synadb:///experiments.db")
    
    _Requirements: 12.1_
    """
    
    def __init__(self, store_uri: str, artifact_uri: Optional[str] = None):
        """Initialize the Syna tracking store.
        
        Args:
            store_uri: URI in format "synadb:///path/to/db" or "synadb://path/to/db"
            artifact_uri: Optional artifact storage location
        
        Raises:
            ImportError: If mlflow is not installed
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")
        
        # Parse URI: synadb:///path/to/db or synadb://path/to/db
        path = store_uri.replace("synadb://", "")
        if path.startswith("/"):
            path = path[1:]
        
        # Import here to avoid circular imports
        from ..wrapper import SynaDB
        from ..experiment import Experiment as SynaExperiment
        
        self._db = SynaDB(path)
        self._path = path
        self._artifact_uri = artifact_uri or str(Path(path).parent / f"{Path(path).stem}_artifacts")
        
        # Cache for experiments
        self._experiments: Dict[str, Dict[str, Any]] = {}
    
    def _get_experiment_meta_key(self, experiment_id: str) -> str:
        """Get the database key for experiment metadata."""
        return f"mlflow/exp/{experiment_id}/meta"
    
    def _get_run_meta_key(self, run_id: str) -> str:
        """Get the database key for run metadata."""
        return f"mlflow/run/{run_id}/meta"
    
    def _get_param_key(self, run_id: str, key: str) -> str:
        """Get the database key for a parameter."""
        return f"mlflow/run/{run_id}/param/{key}"
    
    def _get_metric_key(self, run_id: str, key: str, timestamp: int, step: int) -> str:
        """Get the database key for a metric."""
        return f"mlflow/run/{run_id}/metric/{key}/{timestamp}/{step}"
    
    def _get_tag_key(self, run_id: str, key: str) -> str:
        """Get the database key for a tag."""
        return f"mlflow/run/{run_id}/tag/{key}"
    
    def create_experiment(self, name: str, artifact_location: Optional[str] = None, tags: Optional[List["RunTag"]] = None) -> str:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            artifact_location: Optional artifact storage location
            tags: Optional list of tags
        
        Returns:
            Experiment ID
        """
        # Generate experiment ID from name hash
        exp_id = str(abs(hash(name)) % 10**12)
        
        # Store experiment metadata
        meta = {
            "experiment_id": exp_id,
            "name": name,
            "artifact_location": artifact_location or f"{self._artifact_uri}/{exp_id}",
            "lifecycle_stage": "active",
            "creation_time": int(time.time() * 1000),  # milliseconds
            "last_update_time": int(time.time() * 1000),
            "tags": {tag.key: tag.value for tag in (tags or [])}
        }
        
        self._db.put_text(
            self._get_experiment_meta_key(exp_id),
            json.dumps(meta)
        )
        
        self._experiments[exp_id] = meta
        return exp_id
    
    def get_experiment(self, experiment_id: str) -> Optional["MLflowExperiment"]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Experiment object or None if not found
        """
        # Check cache first
        if experiment_id in self._experiments:
            meta = self._experiments[experiment_id]
        else:
            meta_json = self._db.get_text(self._get_experiment_meta_key(experiment_id))
            if not meta_json:
                return None
            meta = json.loads(meta_json)
            self._experiments[experiment_id] = meta
        
        return Experiment(
            experiment_id=meta["experiment_id"],
            name=meta["name"],
            artifact_location=meta.get("artifact_location"),
            lifecycle_stage=meta.get("lifecycle_stage", "active"),
            tags=meta.get("tags", {}),
        )
    
    def get_experiment_by_name(self, experiment_name: str) -> Optional["MLflowExperiment"]:
        """Get experiment by name.
        
        Args:
            experiment_name: Experiment name
        
        Returns:
            Experiment object or None if not found
        """
        # Search through all experiments
        all_keys = self._db.keys()
        for key in all_keys:
            if key.startswith("mlflow/exp/") and key.endswith("/meta"):
                meta_json = self._db.get_text(key)
                if meta_json:
                    meta = json.loads(meta_json)
                    if meta.get("name") == experiment_name:
                        exp_id = meta["experiment_id"]
                        self._experiments[exp_id] = meta
                        return self.get_experiment(exp_id)
        return None
    
    def list_experiments(self, view_type: "ViewType" = None, max_results: int = 1000, page_token: Optional[str] = None) -> List["MLflowExperiment"]:
        """List experiments.
        
        Args:
            view_type: Filter by lifecycle stage (defaults to ACTIVE_ONLY)
            max_results: Maximum number of results
            page_token: Pagination token (not implemented)
        
        Returns:
            List of Experiment objects
        """
        # Set default view_type if not provided
        if view_type is None and MLFLOW_AVAILABLE:
            view_type = ViewType.ACTIVE_ONLY
        
        experiments = []
        all_keys = self._db.keys()
        
        for key in all_keys:
            if key.startswith("mlflow/exp/") and key.endswith("/meta"):
                meta_json = self._db.get_text(key)
                if meta_json:
                    meta = json.loads(meta_json)
                    
                    # Filter by lifecycle stage
                    stage = meta.get("lifecycle_stage", "active")
                    if MLFLOW_AVAILABLE and view_type:
                        if view_type == ViewType.ACTIVE_ONLY and stage != "active":
                            continue
                        elif view_type == ViewType.DELETED_ONLY and stage != "deleted":
                            continue
                    
                    exp_id = meta["experiment_id"]
                    self._experiments[exp_id] = meta
                    experiments.append(self.get_experiment(exp_id))
                    
                    if len(experiments) >= max_results:
                        break
        
        return experiments
    
    def create_run(self, experiment_id: str, user_id: str, start_time: int, tags: List["RunTag"], run_name: Optional[str] = None) -> "Run":
        """Create a new run.
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            start_time: Start time in milliseconds
            tags: List of tags
            run_name: Optional run name
        
        Returns:
            Run object
        """
        import uuid
        
        run_id = str(uuid.uuid4()).replace("-", "")
        
        # Store run metadata
        meta = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": "RUNNING",
            "start_time": start_time,
            "end_time": None,
            "lifecycle_stage": "active",
            "artifact_uri": f"{self._artifact_uri}/{experiment_id}/{run_id}/artifacts",
            "run_name": run_name,
        }
        
        self._db.put_text(
            self._get_run_meta_key(run_id),
            json.dumps(meta)
        )
        
        # Store tags
        for tag in (tags or []):
            self._db.put_text(
                self._get_tag_key(run_id, tag.key),
                tag.value
            )
        
        return Run(
            run_info=RunInfo(
                run_uuid=run_id,
                run_id=run_id,
                experiment_id=experiment_id,
                user_id=user_id,
                status=MLflowRunStatus.to_string(MLflowRunStatus.RUNNING),
                start_time=start_time,
                end_time=None,
                lifecycle_stage=LifecycleStage.ACTIVE,
                artifact_uri=meta["artifact_uri"],
                run_name=run_name,
            ),
            run_data=RunData(
                metrics=[],
                params=[],
                tags=[tag for tag in (tags or [])],
            ),
        )
    
    def update_run_info(self, run_id: str, run_status: "MLflowRunStatus", end_time: int, run_name: Optional[str] = None) -> "RunInfo":
        """Update run information.
        
        Args:
            run_id: Run ID
            run_status: New run status
            end_time: End time in milliseconds
            run_name: Optional run name
        
        Returns:
            Updated RunInfo
        """
        meta_json = self._db.get_text(self._get_run_meta_key(run_id))
        if not meta_json:
            raise Exception(f"Run {run_id} not found")
        
        meta = json.loads(meta_json)
        meta["status"] = MLflowRunStatus.to_string(run_status)
        meta["end_time"] = end_time
        if run_name:
            meta["run_name"] = run_name
        
        self._db.put_text(
            self._get_run_meta_key(run_id),
            json.dumps(meta)
        )
        
        return RunInfo(
            run_uuid=run_id,
            run_id=run_id,
            experiment_id=meta["experiment_id"],
            user_id=meta["user_id"],
            status=meta["status"],
            start_time=meta["start_time"],
            end_time=meta["end_time"],
            lifecycle_stage=meta.get("lifecycle_stage", "active"),
            artifact_uri=meta.get("artifact_uri"),
            run_name=meta.get("run_name"),
        )
    
    def get_run(self, run_id: str) -> "Run":
        """Get run by ID.
        
        Args:
            run_id: Run ID
        
        Returns:
            Run object
        
        Raises:
            Exception: If run not found
        """
        meta_json = self._db.get_text(self._get_run_meta_key(run_id))
        if not meta_json:
            raise Exception(f"Run {run_id} not found")
        
        meta = json.loads(meta_json)
        
        # Load params
        params = []
        all_keys = self._db.keys()
        param_prefix = f"mlflow/run/{run_id}/param/"
        for key in all_keys:
            if key.startswith(param_prefix):
                param_name = key[len(param_prefix):]
                param_value = self._db.get_text(key)
                if param_value:
                    params.append(Param(param_name, param_value))
        
        # Load metrics (latest values only for run data)
        metrics = []
        metric_prefix = f"mlflow/run/{run_id}/metric/"
        metric_names = set()
        for key in all_keys:
            if key.startswith(metric_prefix):
                parts = key[len(metric_prefix):].split("/")
                if len(parts) >= 1:
                    metric_names.add(parts[0])
        
        for metric_name in metric_names:
            # Get latest value
            latest_value = None
            latest_timestamp = 0
            latest_step = 0
            
            for key in all_keys:
                if key.startswith(f"{metric_prefix}{metric_name}/"):
                    parts = key[len(metric_prefix):].split("/")
                    if len(parts) == 3:
                        try:
                            timestamp = int(parts[1])
                            step = int(parts[2])
                            value = self._db.get_float(key)
                            if value is not None and timestamp >= latest_timestamp:
                                latest_value = value
                                latest_timestamp = timestamp
                                latest_step = step
                        except (ValueError, TypeError):
                            pass
            
            if latest_value is not None:
                metrics.append(Metric(
                    key=metric_name,
                    value=latest_value,
                    timestamp=latest_timestamp,
                    step=latest_step
                ))
        
        # Load tags
        tags = []
        tag_prefix = f"mlflow/run/{run_id}/tag/"
        for key in all_keys:
            if key.startswith(tag_prefix):
                tag_name = key[len(tag_prefix):]
                tag_value = self._db.get_text(key)
                if tag_value:
                    tags.append(RunTag(tag_name, tag_value))
        
        return Run(
            run_info=RunInfo(
                run_uuid=run_id,
                run_id=run_id,
                experiment_id=meta["experiment_id"],
                user_id=meta["user_id"],
                status=meta["status"],
                start_time=meta["start_time"],
                end_time=meta.get("end_time"),
                lifecycle_stage=meta.get("lifecycle_stage", "active"),
                artifact_uri=meta.get("artifact_uri"),
                run_name=meta.get("run_name"),
            ),
            run_data=RunData(
                metrics=metrics,
                params=params,
                tags=tags,
            ),
        )
    
    def log_metric(self, run_id: str, metric: "Metric") -> None:
        """Log a metric.
        
        Args:
            run_id: Run ID
            metric: Metric object
        """
        key = self._get_metric_key(
            run_id,
            metric.key,
            metric.timestamp,
            metric.step
        )
        self._db.put_float(key, metric.value)
    
    def log_param(self, run_id: str, param: "Param") -> None:
        """Log a parameter.
        
        Args:
            run_id: Run ID
            param: Param object
        """
        key = self._get_param_key(run_id, param.key)
        self._db.put_text(key, str(param.value))
    
    def set_tag(self, run_id: str, tag: "RunTag") -> None:
        """Set a tag.
        
        Args:
            run_id: Run ID
            tag: RunTag object
        """
        key = self._get_tag_key(run_id, tag.key)
        self._db.put_text(key, tag.value)
    
    def delete_tag(self, run_id: str, key: str) -> None:
        """Delete a tag.
        
        Args:
            run_id: Run ID
            key: Tag key
        """
        tag_key = self._get_tag_key(run_id, key)
        self._db.delete(tag_key)
    
    def log_batch(self, run_id: str, metrics: List["Metric"], params: List["Param"], tags: List["RunTag"]) -> None:
        """Log a batch of metrics, params, and tags.
        
        Args:
            run_id: Run ID
            metrics: List of metrics
            params: List of params
            tags: List of tags
        """
        for metric in metrics:
            self.log_metric(run_id, metric)
        for param in params:
            self.log_param(run_id, param)
        for tag in tags:
            self.set_tag(run_id, tag)
    
    def delete_run(self, run_id: str) -> None:
        """Delete a run.
        
        Args:
            run_id: Run ID
        """
        # Mark as deleted
        meta_json = self._db.get_text(self._get_run_meta_key(run_id))
        if meta_json:
            meta = json.loads(meta_json)
            meta["lifecycle_stage"] = "deleted"
            self._db.put_text(
                self._get_run_meta_key(run_id),
                json.dumps(meta)
            )
    
    def restore_run(self, run_id: str) -> None:
        """Restore a deleted run.
        
        Args:
            run_id: Run ID
        """
        meta_json = self._db.get_text(self._get_run_meta_key(run_id))
        if meta_json:
            meta = json.loads(meta_json)
            meta["lifecycle_stage"] = "active"
            self._db.put_text(
                self._get_run_meta_key(run_id),
                json.dumps(meta)
            )
    
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment.
        
        Args:
            experiment_id: Experiment ID
        """
        meta_json = self._db.get_text(self._get_experiment_meta_key(experiment_id))
        if meta_json:
            meta = json.loads(meta_json)
            meta["lifecycle_stage"] = "deleted"
            self._db.put_text(
                self._get_experiment_meta_key(experiment_id),
                json.dumps(meta)
            )
            if experiment_id in self._experiments:
                del self._experiments[experiment_id]
    
    def restore_experiment(self, experiment_id: str) -> None:
        """Restore a deleted experiment.
        
        Args:
            experiment_id: Experiment ID
        """
        meta_json = self._db.get_text(self._get_experiment_meta_key(experiment_id))
        if meta_json:
            meta = json.loads(meta_json)
            meta["lifecycle_stage"] = "active"
            self._db.put_text(
                self._get_experiment_meta_key(experiment_id),
                json.dumps(meta)
            )
            self._experiments[experiment_id] = meta


def register_syna_tracking_store():
    """
    Register Syna as an MLflow tracking store.
    
    After calling this function, you can use "synadb:///" URIs with MLflow.
    
    Example:
        >>> from synadb.integrations.mlflow import register_syna_tracking_store
        >>> import mlflow
        >>> 
        >>> register_syna_tracking_store()
        >>> mlflow.set_tracking_uri("synadb:///experiments.db")
        >>> 
        >>> with mlflow.start_run():
        ...     mlflow.log_param("lr", 0.001)
        ...     mlflow.log_metric("accuracy", 0.95)
    
    _Requirements: 12.1_
    """
    if not MLFLOW_AVAILABLE:
        raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
    
    # Register the synadb:// scheme
    TrackingStoreRegistry.register("synadb", SynaTrackingStore)
    TrackingStoreRegistry.register("synadb", SynaTrackingStore)  # Also register without ://


# Export public API
__all__ = [
    "SynaTrackingStore",
    "register_syna_tracking_store",
    "MLFLOW_AVAILABLE",
]

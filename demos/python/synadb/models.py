"""
Model registry for versioned artifact storage.

This module provides a high-level Python interface for storing and versioning
ML models using the Syna database. It supports automatic versioning, checksum
verification, and stage management (development, staging, production).

Example:
    >>> from synadb import ModelRegistry
    >>> registry = ModelRegistry("models.db")
    >>> # Save a PyTorch model
    >>> version = registry.save("classifier", model, {"accuracy": "0.95"})
    >>> print(f"Saved as v{version.version}")
    >>> # Load latest
    >>> loaded = registry.load("classifier")
    >>> # Promote to production
    >>> registry.promote("classifier", version.version, "production")
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from .wrapper import SynaDB, SynaError


class ModelStage(Enum):
    """Stage of a model in the deployment lifecycle.
    
    Attributes:
        DEVELOPMENT: Model is in development/experimentation.
        STAGING: Model is being tested for production readiness.
        PRODUCTION: Model is deployed in production.
        ARCHIVED: Model is archived and no longer active.
    """
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Metadata for a model version.
    
    Attributes:
        name: Name of the model.
        version: Version number (auto-incremented).
        created_at: Unix timestamp when the model was saved.
        checksum: SHA-256 checksum of the model data.
        size_bytes: Size of the serialized model in bytes.
        metadata: User-provided metadata (e.g., accuracy, hyperparameters).
        stage: Current deployment stage of the model.
    """
    name: str
    version: int
    created_at: int
    checksum: str
    size_bytes: int
    metadata: Dict[str, str]
    stage: ModelStage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "stage": self.stage.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            created_at=data["created_at"],
            checksum=data["checksum"],
            size_bytes=data["size_bytes"],
            metadata=data.get("metadata", {}),
            stage=ModelStage(data.get("stage", "development")),
        )


class ModelRegistry:
    """
    Model registry for storing and versioning ML models.
    
    Provides a high-level API for saving, loading, and managing ML model
    artifacts with automatic versioning, checksum verification, and
    deployment stage tracking.
    
    Supports: PyTorch, TensorFlow/Keras, scikit-learn, ONNX, or raw bytes.
    
    Example:
        >>> registry = ModelRegistry("models.db")
        >>> # Save a PyTorch model
        >>> version = registry.save("classifier", model, {"accuracy": "0.95"})
        >>> print(f"Saved as v{version.version}")
        >>> # Load latest
        >>> loaded = registry.load("classifier")
        >>> # Promote to production
        >>> registry.promote("classifier", version.version, "production")
    
    Attributes:
        path: Path to the database file.
    
    _Requirements: 8.1, 8.2_
    """
    
    def __init__(self, path: str):
        """
        Create or open a model registry.
        
        Args:
            path: Path to the database file. Will be created if it doesn't exist.
        
        Raises:
            SynaError: If the database cannot be opened.
        """
        self._db = SynaDB(path)
        self._path = path
    
    def _serialize_model(self, model: Any) -> tuple:
        """
        Serialize a model to bytes with format marker.
        
        Supports PyTorch, TensorFlow/Keras, scikit-learn, ONNX, or raw bytes.
        
        Args:
            model: The model to serialize.
        
        Returns:
            Tuple of (format_marker, serialized_bytes).
        """
        # If already bytes, return as-is with marker
        if isinstance(model, bytes):
            return ("raw", model)
        
        # Try PyTorch
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                import io
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                return ("pytorch", buffer.getvalue())
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try TensorFlow/Keras
        try:
            import tensorflow as tf
            if isinstance(model, (tf.keras.Model, tf.Module)):
                import io
                import tempfile
                import os
                # Save to temp file and read bytes
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, "model")
                    model.save(path)
                    # Read all files from saved model directory
                    import zipfile
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, path)
                                zf.write(file_path, arcname)
                    return ("tensorflow", buffer.getvalue())
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try ONNX
        try:
            import onnx
            if isinstance(model, onnx.ModelProto):
                return ("onnx", model.SerializeToString())
        except ImportError:
            pass
        except Exception:
            pass
        
        # Default: use pickle
        return ("pickle", pickle.dumps(model))
    
    def _deserialize_model(self, data: bytes, format_marker: str = None) -> Any:
        """
        Deserialize a model from bytes.
        
        Args:
            data: Serialized model bytes.
            format_marker: Format marker from serialization ("raw", "pytorch", 
                           "tensorflow", "onnx", "pickle"). If None, tries to auto-detect.
        
        Returns:
            Deserialized model.
        """
        # Raw bytes - return as-is
        if format_marker == "raw":
            return data
        
        # PyTorch
        if format_marker == "pytorch":
            try:
                import torch
                import io
                buffer = io.BytesIO(data)
                state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
                return state_dict
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch model: {e}")
        
        # TensorFlow/Keras
        if format_marker == "tensorflow":
            try:
                import tensorflow as tf
                import io
                import tempfile
                import zipfile
                
                buffer = io.BytesIO(data)
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(buffer, 'r') as zf:
                        zf.extractall(tmpdir)
                    model = tf.keras.models.load_model(tmpdir)
                    return model
            except Exception as e:
                raise RuntimeError(f"Failed to load TensorFlow model: {e}")
        
        # ONNX
        if format_marker == "onnx":
            try:
                import onnx
                model = onnx.load_from_string(data)
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load ONNX model: {e}")
        
        # Pickle (default)
        if format_marker in ("pickle", None):
            return pickle.loads(data)
        
        raise ValueError(f"Unknown format marker: {format_marker}")
    
    def save(
        self,
        name: str,
        model: Any,
        metadata: Dict[str, str] = None,
    ) -> ModelVersion:
        """
        Save a model with automatic versioning.
        
        Serializes the model, computes a checksum, assigns a version number,
        and stores it in the database.
        
        Supports: PyTorch, TensorFlow/Keras, scikit-learn, ONNX, or raw bytes.
        
        Args:
            name: Name of the model (e.g., "classifier", "recommender").
            model: The model to save. Can be a PyTorch nn.Module, TensorFlow
                   Model, scikit-learn estimator, ONNX model, or raw bytes.
            metadata: Optional metadata dictionary (e.g., {"accuracy": "0.95"}).
                      Values should be strings for consistent serialization.
        
        Returns:
            ModelVersion with version info and checksum.
        
        Raises:
            SynaError: If the save operation fails.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> version = registry.save("classifier", model, {"accuracy": "0.95"})
            >>> print(f"Saved {version.name} v{version.version}")
        
        _Requirements: 8.1, 8.2_
        """
        # Serialize model (returns format_marker and data)
        format_marker, data = self._serialize_model(model)
        
        # Compute checksum
        checksum = hashlib.sha256(data).hexdigest()
        
        # Get next version number
        versions = self.list(name)
        version = max([v.version for v in versions], default=0) + 1
        
        # Store model data
        data_key = f"model/{name}/v{version}/data"
        self._db.put_bytes(data_key, data)
        
        # Store format marker
        format_key = f"model/{name}/v{version}/format"
        self._db.put_text(format_key, format_marker)
        
        # Create version metadata
        model_version = ModelVersion(
            name=name,
            version=version,
            created_at=int(time.time()),
            checksum=checksum,
            size_bytes=len(data),
            metadata=metadata or {},
            stage=ModelStage.DEVELOPMENT,
        )
        
        # Store metadata
        meta_key = f"model/{name}/v{version}/meta"
        self._db.put_text(meta_key, json.dumps(model_version.to_dict()))
        
        return model_version
    
    def load(
        self,
        name: str,
        version: int = None,
        verify_checksum: bool = True,
    ) -> Any:
        """
        Load a model (auto-detects framework).
        
        Retrieves the model from the database, optionally verifies the checksum,
        and deserializes it.
        
        Args:
            name: Name of the model to load.
            version: Specific version to load. If None, loads the latest version.
            verify_checksum: Whether to verify the checksum (default: True).
        
        Returns:
            The deserialized model.
        
        Raises:
            ValueError: If the model or version is not found.
            RuntimeError: If checksum verification fails.
            SynaError: If the load operation fails.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> model = registry.load("classifier")  # Load latest
            >>> model_v1 = registry.load("classifier", version=1)  # Load specific
        
        _Requirements: 8.1, 8.2_
        """
        # Get version info
        if version is None:
            versions = self.list(name)
            if not versions:
                raise ValueError(f"Model '{name}' not found")
            version = max(v.version for v in versions)
        
        # Load metadata
        meta_key = f"model/{name}/v{version}/meta"
        meta_json = self._db.get_text(meta_key)
        if meta_json is None:
            raise ValueError(f"Model '{name}' version {version} not found")
        
        model_version = ModelVersion.from_dict(json.loads(meta_json))
        
        # Load model data
        data_key = f"model/{name}/v{version}/data"
        data = self._db.get_bytes(data_key)
        if data is None:
            raise ValueError(f"Model data for '{name}' version {version} not found")
        
        # Verify checksum
        if verify_checksum:
            actual_checksum = hashlib.sha256(data).hexdigest()
            if actual_checksum != model_version.checksum:
                raise RuntimeError(
                    f"Checksum mismatch for '{name}' v{version}: "
                    f"expected {model_version.checksum}, got {actual_checksum}"
                )
        
        # Load format marker
        format_key = f"model/{name}/v{version}/format"
        format_marker = self._db.get_text(format_key)
        
        # Deserialize and return
        return self._deserialize_model(data, format_marker)
    
    def list(self, name: str = None) -> List[ModelVersion]:
        """
        List all models or versions of a specific model.
        
        Args:
            name: Optional model name. If provided, lists versions of that model.
                  If None, lists all models (latest version of each).
        
        Returns:
            List of ModelVersion objects, sorted by version number.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> # List all versions of a model
            >>> versions = registry.list("classifier")
            >>> for v in versions:
            ...     print(f"v{v.version}: {v.stage.value}")
            >>> # List all models
            >>> all_models = registry.list()
        
        _Requirements: 8.1, 8.2_
        """
        all_keys = self._db.keys()
        versions = []
        
        if name is not None:
            # List versions of a specific model
            prefix = f"model/{name}/v"
            meta_suffix = "/meta"
            
            for key in all_keys:
                if key.startswith(prefix) and key.endswith(meta_suffix):
                    meta_json = self._db.get_text(key)
                    if meta_json:
                        try:
                            versions.append(ModelVersion.from_dict(json.loads(meta_json)))
                        except (json.JSONDecodeError, KeyError):
                            pass
        else:
            # List all models (latest version of each)
            model_names = set()
            for key in all_keys:
                if key.startswith("model/") and "/meta" in key:
                    # Extract model name: model/{name}/v{version}/meta
                    parts = key.split("/")
                    if len(parts) >= 4:
                        model_names.add(parts[1])
            
            for model_name in model_names:
                model_versions = self.list(model_name)
                if model_versions:
                    # Get latest version
                    latest = max(model_versions, key=lambda v: v.version)
                    versions.append(latest)
        
        # Sort by version number
        return sorted(versions, key=lambda v: (v.name, v.version))
    
    def promote(self, name: str, version: int, stage: str) -> None:
        """
        Promote a model to a deployment stage.
        
        Updates the stage of a model version (development, staging, production,
        archived).
        
        Args:
            name: Name of the model.
            version: Version number to promote.
            stage: Target stage ("development", "staging", "production", "archived").
        
        Raises:
            ValueError: If the model/version is not found or stage is invalid.
            SynaError: If the update operation fails.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> registry.promote("classifier", 3, "production")
        
        _Requirements: 8.1, 8.2_
        """
        # Validate stage
        try:
            new_stage = ModelStage(stage.lower())
        except ValueError:
            valid_stages = [s.value for s in ModelStage]
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")
        
        # Load current metadata
        meta_key = f"model/{name}/v{version}/meta"
        meta_json = self._db.get_text(meta_key)
        if meta_json is None:
            raise ValueError(f"Model '{name}' version {version} not found")
        
        # Update stage
        model_version = ModelVersion.from_dict(json.loads(meta_json))
        model_version.stage = new_stage
        
        # Save updated metadata
        self._db.put_text(meta_key, json.dumps(model_version.to_dict()))
    
    def delete(self, name: str, version: int = None) -> int:
        """
        Delete a model or specific version.
        
        Args:
            name: Name of the model.
            version: Specific version to delete. If None, deletes all versions.
        
        Returns:
            Number of versions deleted.
        
        Raises:
            SynaError: If the delete operation fails.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> registry.delete("classifier", version=1)  # Delete v1
            >>> registry.delete("old_model")  # Delete all versions
        """
        count = 0
        
        if version is not None:
            # Delete specific version
            data_key = f"model/{name}/v{version}/data"
            meta_key = f"model/{name}/v{version}/meta"
            
            try:
                self._db.delete(data_key)
                count += 1
            except SynaError:
                pass
            
            try:
                self._db.delete(meta_key)
            except SynaError:
                pass
        else:
            # Delete all versions
            versions = self.list(name)
            for v in versions:
                count += self.delete(name, v.version)
        
        return count
    
    def get_version(self, name: str, version: int) -> Optional[ModelVersion]:
        """
        Get metadata for a specific model version.
        
        Args:
            name: Name of the model.
            version: Version number.
        
        Returns:
            ModelVersion if found, None otherwise.
        """
        meta_key = f"model/{name}/v{version}/meta"
        meta_json = self._db.get_text(meta_key)
        if meta_json is None:
            return None
        
        try:
            return ModelVersion.from_dict(json.loads(meta_json))
        except (json.JSONDecodeError, KeyError):
            return None
    
    def get_production_model(self, name: str) -> Optional[Any]:
        """
        Load the production version of a model.
        
        Convenience method to load the model version that is currently
        in the PRODUCTION stage.
        
        Args:
            name: Name of the model.
        
        Returns:
            The production model, or None if no production version exists.
        """
        versions = self.list(name)
        production_versions = [v for v in versions if v.stage == ModelStage.PRODUCTION]
        
        if not production_versions:
            return None
        
        # Get the latest production version
        latest = max(production_versions, key=lambda v: v.version)
        return self.load(name, latest.version)
    
    def compare(self, name: str, version1: int, version2: int) -> Dict[str, Any]:
        """
        Compare metadata between two model versions.
        
        Args:
            name: Name of the model.
            version1: First version to compare.
            version2: Second version to compare.
        
        Returns:
            Dictionary with comparison results.
        
        Example:
            >>> registry = ModelRegistry("models.db")
            >>> diff = registry.compare("classifier", 1, 2)
            >>> print(diff)
        """
        v1 = self.get_version(name, version1)
        v2 = self.get_version(name, version2)
        
        if v1 is None:
            raise ValueError(f"Model '{name}' version {version1} not found")
        if v2 is None:
            raise ValueError(f"Model '{name}' version {version2} not found")
        
        return {
            "version1": version1,
            "version2": version2,
            "size_diff_bytes": v2.size_bytes - v1.size_bytes,
            "time_diff_seconds": v2.created_at - v1.created_at,
            "stage_v1": v1.stage.value,
            "stage_v2": v2.stage.value,
            "metadata_v1": v1.metadata,
            "metadata_v2": v2.metadata,
            "checksum_v1": v1.checksum,
            "checksum_v2": v2.checksum,
        }
    
    def close(self) -> None:
        """Close the database."""
        self._db.close()
    
    def __enter__(self) -> "ModelRegistry":
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

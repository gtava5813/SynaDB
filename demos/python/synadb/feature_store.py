"""
SynaDB Feature Store — Python wrapper.

A fully embedded, zero-server feature management system for ML engineers.

Usage:
    from synadb.feature_store import FeatureStore

    with FeatureStore("features.db") as fs:
        # Ingest features
        fs.ingest("user_features", "user_123", event_ts=1000000,
                  values={"purchase_count": 5, "avg_spend": 42.50})

        # Serve latest values
        vector = fs.serve("user_features", "user_123", ["purchase_count", "avg_spend"])

        # Point-in-time query
        vector = fs.get_as_of("user_features", "user_123", cutoff_ts=500000,
                              features=["purchase_count"])

        # Version-based query
        value = fs.get_at_version("user_features", "user_123", "purchase_count", version=-2)
"""

import ctypes
import os
import platform
from ctypes import c_char_p, c_double, c_int32, c_int64, c_uint64, POINTER, byref
from typing import Any, Dict, List, Optional


def _load_library():
    """Load the SynaDB shared library."""
    system = platform.system()

    # Search paths
    search_paths = []

    # Check environment variable
    lib_path = os.environ.get("SYNADB_LIB_PATH")
    if lib_path:
        search_paths.append(lib_path)

    # Check relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(this_dir)
    search_paths.append(os.path.join(this_dir, "..", "..", "..", "target", "release"))
    search_paths.append(os.path.join(this_dir, "..", "..", "..", "target", "debug"))

    if system == "Linux":
        lib_name = "libsynadb.so"
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
    elif system == "Windows":
        lib_name = "synadb.dll"
    else:
        lib_name = "libsynadb.so"

    for path in search_paths:
        full_path = os.path.join(path, lib_name)
        if os.path.exists(full_path):
            return ctypes.CDLL(full_path)

    # Try system path
    try:
        return ctypes.CDLL(lib_name)
    except OSError:
        raise RuntimeError(
            f"Could not find SynaDB library ({lib_name}). "
            f"Set SYNADB_LIB_PATH or build with: cargo build --release"
        )


# Load library
_lib = _load_library()

# Define function signatures
_lib.SYNA_fs_new.argtypes = [c_char_p]
_lib.SYNA_fs_new.restype = c_int32

_lib.SYNA_fs_close.argtypes = [c_char_p]
_lib.SYNA_fs_close.restype = c_int32

_lib.SYNA_fs_ingest_float.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_double, c_uint64]
_lib.SYNA_fs_ingest_float.restype = c_int32

_lib.SYNA_fs_ingest_int.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_int64, c_uint64]
_lib.SYNA_fs_ingest_int.restype = c_int32

_lib.SYNA_fs_serve_float.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, POINTER(c_double)]
_lib.SYNA_fs_serve_float.restype = c_int32

_lib.SYNA_fs_get_at_version.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_int64, POINTER(c_double)]
_lib.SYNA_fs_get_at_version.restype = c_int32

_lib.SYNA_fs_get_at_timestamp.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_uint64, POINTER(c_double)]
_lib.SYNA_fs_get_at_timestamp.restype = c_int32

_lib.SYNA_fs_flush.argtypes = [c_char_p]
_lib.SYNA_fs_flush.restype = c_int32


class FeatureStore:
    """
    Embedded feature store for ML engineers.

    Provides typed feature schemas, point-in-time queries, sub-millisecond
    online serving, and training dataset generation — all in a single file.

    Args:
        path: Path to the feature store database file.

    Example:
        >>> with FeatureStore("features.db") as fs:
        ...     fs.ingest("users", "u1", 1000, {"score": 0.95})
        ...     result = fs.serve("users", "u1", ["score"])
        ...     print(result)  # {"score": 0.95}
    """

    def __init__(self, path: str):
        self._path = path.encode("utf-8") if isinstance(path, str) else path
        self._closed = False

        result = _lib.SYNA_fs_new(self._path)
        if result != 1:
            raise RuntimeError(f"Failed to open feature store at '{path}': error code {result}")

    def close(self):
        """Close the feature store, flushing all buffers."""
        if not self._closed:
            _lib.SYNA_fs_close(self._path)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    def ingest(
        self,
        group: str,
        entity_key: str,
        event_ts: int,
        values: Dict[str, Any],
    ) -> None:
        """
        Ingest feature values for an entity.

        Args:
            group: Feature group name.
            entity_key: Entity identifier.
            event_ts: Event timestamp in microseconds.
            values: Dictionary of feature_name → value.

        Raises:
            RuntimeError: If ingestion fails (schema validation, etc.)
        """
        group_b = group.encode("utf-8")
        entity_b = entity_key.encode("utf-8")

        for feature, value in values.items():
            feature_b = feature.encode("utf-8")

            if isinstance(value, float):
                result = _lib.SYNA_fs_ingest_float(
                    self._path, group_b, entity_b, feature_b,
                    c_double(value), c_uint64(event_ts),
                )
            elif isinstance(value, int):
                result = _lib.SYNA_fs_ingest_int(
                    self._path, group_b, entity_b, feature_b,
                    c_int64(value), c_uint64(event_ts),
                )
            else:
                raise TypeError(f"Unsupported value type for feature '{feature}': {type(value)}")

            if result != 1:
                raise RuntimeError(
                    f"Failed to ingest feature '{feature}' for entity '{entity_key}': "
                    f"error code {result}"
                )

    def serve(
        self,
        group: str,
        entity_key: str,
        features: List[str],
    ) -> Dict[str, Optional[float]]:
        """
        Serve latest feature values for an entity.

        Returns values from the in-memory cache with O(1) lookup.

        Args:
            group: Feature group name.
            entity_key: Entity identifier.
            features: List of feature names to retrieve.

        Returns:
            Dictionary of feature_name → value (None if not found).
        """
        group_b = group.encode("utf-8")
        entity_b = entity_key.encode("utf-8")
        result = {}

        for feature in features:
            feature_b = feature.encode("utf-8")
            out = c_double(0.0)
            ret = _lib.SYNA_fs_serve_float(
                self._path, group_b, entity_b, feature_b, byref(out),
            )
            if ret == 1:
                result[feature] = out.value
            else:
                result[feature] = None

        return result

    def get_at_version(
        self,
        group: str,
        entity_key: str,
        feature: str,
        version: int = 0,
    ) -> Optional[float]:
        """
        Get the Nth-most-recent value for a feature.

        Args:
            group: Feature group name.
            entity_key: Entity identifier.
            feature: Feature name.
            version: 0 or -1 for latest, -N for Nth-most-recent.

        Returns:
            The feature value, or None if the version doesn't exist.
        """
        out = c_double(0.0)
        ret = _lib.SYNA_fs_get_at_version(
            self._path,
            group.encode("utf-8"),
            entity_key.encode("utf-8"),
            feature.encode("utf-8"),
            c_int64(version),
            byref(out),
        )
        return out.value if ret == 1 else None

    def get_at_timestamp(
        self,
        group: str,
        entity_key: str,
        feature: str,
        timestamp: int,
    ) -> Optional[float]:
        """
        Get the value as it existed at a specific timestamp.

        Point-in-time query: returns the latest value with event_ts <= timestamp.

        Args:
            group: Feature group name.
            entity_key: Entity identifier.
            feature: Feature name.
            timestamp: Cutoff timestamp in microseconds.

        Returns:
            The feature value, or None if no value exists before the timestamp.
        """
        out = c_double(0.0)
        ret = _lib.SYNA_fs_get_at_timestamp(
            self._path,
            group.encode("utf-8"),
            entity_key.encode("utf-8"),
            feature.encode("utf-8"),
            c_uint64(timestamp),
            byref(out),
        )
        return out.value if ret == 1 else None

    def flush(self) -> None:
        """Flush the write buffer to disk."""
        result = _lib.SYNA_fs_flush(self._path)
        if result != 1:
            raise RuntimeError(f"Failed to flush feature store: error code {result}")

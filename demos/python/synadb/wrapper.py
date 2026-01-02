"""
Syna Database Python Wrapper

High-level Python interface using ctypes to call the Syna C-ABI.
"""

import ctypes
from ctypes import c_char_p, c_double, c_int64, c_int32, c_size_t, c_uint8, POINTER, byref
import os
import platform
from pathlib import Path
from typing import Optional, List, Union
import numpy as np


class SynaError(Exception):
    """Exception raised for Syna database errors."""
    
    ERROR_CODES = {
        0: "Generic error",
        -1: "Database not found in registry",
        -2: "Invalid path or UTF-8",
        -3: "I/O error",
        -4: "Serialization error",
        -5: "Key not found",
        -6: "Type mismatch",
        -7: "Empty key not allowed",
        -8: "Key too long",
        -100: "Internal panic",
    }
    
    def __init__(self, code: int, message: str = None):
        self.code = code
        self.message = message or self.ERROR_CODES.get(code, f"Unknown error: {code}")
        super().__init__(self.message)


def _find_library() -> str:
    """Find the Syna shared library."""
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == "Windows":
        lib_name = "synadb.dll"
        lib_names = ["synadb.dll"]
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
        # macOS: try architecture-specific first, then generic
        if machine in ("arm64", "aarch64"):
            lib_names = ["libsynadb-arm64.dylib", "libsynadb.dylib"]
        else:
            lib_names = ["libsynadb-x86_64.dylib", "libsynadb.dylib"]
    else:
        lib_name = "libsynadb.so"
        lib_names = ["libsynadb.so"]
    
    # Get the workspace root (demos/python/synadb/wrapper.py -> demos/python/synadb -> demos/python -> demos -> root)
    wrapper_dir = Path(__file__).parent  # synadb/
    python_dir = wrapper_dir.parent       # demos/python/
    demos_dir = python_dir.parent         # demos/
    workspace_root = demos_dir.parent     # root
    
    # Search paths for each library name variant
    for lib in lib_names:
        search_paths = [
            # Inside installed package (pip install synadb)
            wrapper_dir / lib,
            # Workspace root target directories (development)
            workspace_root / "target" / "release" / lib_name,
            workspace_root / "target" / "debug" / lib_name,
            # Current directory
            Path.cwd() / lib,
            Path.cwd() / "target" / "release" / lib_name,
            Path.cwd() / "target" / "debug" / lib_name,
            # Walk up from cwd looking for target/release
            Path.cwd().parent / "target" / "release" / lib_name,
            Path.cwd().parent.parent / "target" / "release" / lib_name,
            Path.cwd().parent.parent.parent / "target" / "release" / lib_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
    
    # Try system library path as fallback
    return lib_name


class SynaDB:
    """
    High-level Python wrapper for Syna database.
    
    Example:
        >>> with SynaDB("my.db") as db:
        ...     db.put_float("key", 3.14)
        ...     print(db.get_float("key"))
        3.14
    """
    
    _lib = None
    _lib_path = None
    
    @classmethod
    def _load_library(cls):
        """Load the shared library if not already loaded."""
        if cls._lib is not None:
            return
        
        lib_path = _find_library()
        cls._lib_path = lib_path
        cls._lib = ctypes.CDLL(lib_path)
        
        # Define function signatures (using SYNA_ prefix for FFI functions)
        cls._lib.SYNA_open.argtypes = [c_char_p]
        cls._lib.SYNA_open.restype = c_int32
        
        cls._lib.SYNA_close.argtypes = [c_char_p]
        cls._lib.SYNA_close.restype = c_int32
        
        cls._lib.SYNA_put_float.argtypes = [c_char_p, c_char_p, c_double]
        cls._lib.SYNA_put_float.restype = c_int64
        
        cls._lib.SYNA_put_int.argtypes = [c_char_p, c_char_p, c_int64]
        cls._lib.SYNA_put_int.restype = c_int64
        
        cls._lib.SYNA_put_text.argtypes = [c_char_p, c_char_p, c_char_p]
        cls._lib.SYNA_put_text.restype = c_int64
        
        cls._lib.SYNA_put_bytes.argtypes = [c_char_p, c_char_p, POINTER(c_uint8), c_size_t]
        cls._lib.SYNA_put_bytes.restype = c_int64
        
        cls._lib.SYNA_get_float.argtypes = [c_char_p, c_char_p, POINTER(c_double)]
        cls._lib.SYNA_get_float.restype = c_int32
        
        cls._lib.SYNA_get_int.argtypes = [c_char_p, c_char_p, POINTER(c_int64)]
        cls._lib.SYNA_get_int.restype = c_int32
        
        cls._lib.SYNA_get_history_tensor.argtypes = [c_char_p, c_char_p, POINTER(c_size_t)]
        cls._lib.SYNA_get_history_tensor.restype = POINTER(c_double)
        
        cls._lib.SYNA_free_tensor.argtypes = [POINTER(c_double), c_size_t]
        cls._lib.SYNA_free_tensor.restype = None
        
        cls._lib.SYNA_delete.argtypes = [c_char_p, c_char_p]
        cls._lib.SYNA_delete.restype = c_int32
        
        cls._lib.SYNA_exists.argtypes = [c_char_p, c_char_p]
        cls._lib.SYNA_exists.restype = c_int32
        
        cls._lib.SYNA_compact.argtypes = [c_char_p]
        cls._lib.SYNA_compact.restype = c_int32
        
        cls._lib.SYNA_keys.argtypes = [c_char_p, POINTER(c_size_t)]
        cls._lib.SYNA_keys.restype = POINTER(c_char_p)
        
        cls._lib.SYNA_free_keys.argtypes = [POINTER(c_char_p), c_size_t]
        cls._lib.SYNA_free_keys.restype = None
        
        cls._lib.SYNA_get_text.argtypes = [c_char_p, c_char_p, POINTER(c_size_t)]
        cls._lib.SYNA_get_text.restype = POINTER(ctypes.c_char)
        
        cls._lib.SYNA_free_text.argtypes = [POINTER(ctypes.c_char), c_size_t]
        cls._lib.SYNA_free_text.restype = None
        
        cls._lib.SYNA_get_bytes.argtypes = [c_char_p, c_char_p, POINTER(c_size_t)]
        cls._lib.SYNA_get_bytes.restype = POINTER(c_uint8)
        
        cls._lib.SYNA_free_bytes.argtypes = [POINTER(c_uint8), c_size_t]
        cls._lib.SYNA_free_bytes.restype = None
        
        # New: open with config for sync_on_write control
        cls._lib.SYNA_open_with_config.argtypes = [c_char_p, c_int32]
        cls._lib.SYNA_open_with_config.restype = c_int32
        
        # New: batch write for high-throughput ingestion
        cls._lib.SYNA_put_floats_batch.argtypes = [c_char_p, c_char_p, POINTER(c_double), c_size_t]
        cls._lib.SYNA_put_floats_batch.restype = c_int64
    
    def __init__(self, path: str, sync_on_write: bool = True):
        """
        Open or create a database at the given path.
        
        Args:
            path: Path to the database file
            sync_on_write: If True (default), sync to disk after each write for
                durability. Set to False for high-throughput scenarios (100K+ ops/sec)
                at the risk of data loss on crash.
            
        Raises:
            SynaError: If the database cannot be opened
        """
        self._load_library()
        self._path = path.encode('utf-8')
        self._closed = False
        
        # Use config-based open if sync_on_write is False
        if sync_on_write:
            result = self._lib.SYNA_open(self._path)
        else:
            result = self._lib.SYNA_open_with_config(self._path, 0)  # 0 = no sync
        
        if result != 1:
            raise SynaError(result, f"Failed to open database: {path}")
    
    def close(self) -> None:
        """Close the database."""
        if not self._closed:
            self._lib.SYNA_close(self._path)
            self._closed = True
    
    def __enter__(self) -> 'SynaDB':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
    
    def _check_open(self):
        """Raise error if database is closed."""
        if self._closed:
            raise SynaError(-1, "Database is closed")

    
    def put_float(self, key: str, value: float) -> int:
        """
        Write a float value to the database.
        
        Args:
            key: The key (non-empty string, max 65535 bytes)
            value: The float value to store
            
        Returns:
            Byte offset where the entry was written
            
        Raises:
            SynaError: If the write fails
        """
        self._check_open()
        result = self._lib.SYNA_put_float(self._path, key.encode('utf-8'), value)
        if result < 0:
            raise SynaError(int(result))
        return result
    
    def put_floats_batch(self, key: str, values: np.ndarray) -> int:
        """
        Write multiple float values in a single batch operation.
        
        This is optimized for high-throughput ingestion scenarios like sensor data.
        All values are written under the same key, building up a history that can
        be extracted as a tensor with `get_history_tensor()`.
        
        Args:
            key: The key (non-empty string, max 65535 bytes)
            values: numpy array of float values to store
            
        Returns:
            Number of values written
            
        Raises:
            SynaError: If the write fails
            
        Performance:
            This method is significantly faster than calling `put_float()` in a loop:
            - Single FFI boundary crossing
            - Single mutex lock for all writes
            - Single fsync at the end (if sync_on_write is enabled)
            
        Example:
            >>> db = SynaDB("sensors.db", sync_on_write=False)
            >>> readings = np.array([23.5, 23.6, 23.7, 23.8])
            >>> count = db.put_floats_batch("sensor/temp", readings)
            >>> print(count)  # 4
        """
        self._check_open()
        # Ensure contiguous float64 array
        arr = np.ascontiguousarray(values, dtype=np.float64)
        ptr = arr.ctypes.data_as(POINTER(c_double))
        result = self._lib.SYNA_put_floats_batch(
            self._path,
            key.encode('utf-8'),
            ptr,
            len(arr)
        )
        if result < 0:
            raise SynaError(int(result))
        return int(result)
    
    def put_int(self, key: str, value: int) -> int:
        """
        Write an integer value to the database.
        
        Args:
            key: The key (non-empty string, max 65535 bytes)
            value: The integer value to store
            
        Returns:
            Byte offset where the entry was written
            
        Raises:
            SynaError: If the write fails
        """
        self._check_open()
        result = self._lib.SYNA_put_int(self._path, key.encode('utf-8'), value)
        if result < 0:
            raise SynaError(int(result))
        return result
    
    def put_text(self, key: str, value: str) -> int:
        """
        Write a text value to the database.
        
        Args:
            key: The key (non-empty string, max 65535 bytes)
            value: The text value to store
            
        Returns:
            Byte offset where the entry was written
            
        Raises:
            SynaError: If the write fails
        """
        self._check_open()
        result = self._lib.SYNA_put_text(
            self._path, 
            key.encode('utf-8'), 
            value.encode('utf-8')
        )
        if result < 0:
            raise SynaError(int(result))
        return result
    
    def put_bytes(self, key: str, value: bytes) -> int:
        """
        Write a bytes value to the database.
        
        Args:
            key: The key (non-empty string, max 65535 bytes)
            value: The bytes value to store
            
        Returns:
            Byte offset where the entry was written
            
        Raises:
            SynaError: If the write fails
        """
        self._check_open()
        data_ptr = (c_uint8 * len(value)).from_buffer_copy(value)
        result = self._lib.SYNA_put_bytes(
            self._path,
            key.encode('utf-8'),
            ctypes.cast(data_ptr, POINTER(c_uint8)),
            len(value)
        )
        if result < 0:
            raise SynaError(int(result))
        return result
    
    def get_float(self, key: str) -> Optional[float]:
        """
        Read a float value from the database.
        
        Args:
            key: The key to read
            
        Returns:
            The float value, or None if key not found
            
        Raises:
            SynaError: If the read fails (except key not found)
        """
        self._check_open()
        out = c_double()
        result = self._lib.SYNA_get_float(self._path, key.encode('utf-8'), byref(out))
        if result == 1:
            return out.value
        elif result == -5:  # Key not found
            return None
        else:
            raise SynaError(result)
    
    def get_int(self, key: str) -> Optional[int]:
        """
        Read an integer value from the database.
        
        Args:
            key: The key to read
            
        Returns:
            The integer value, or None if key not found
            
        Raises:
            SynaError: If the read fails (except key not found)
        """
        self._check_open()
        out = c_int64()
        result = self._lib.SYNA_get_int(self._path, key.encode('utf-8'), byref(out))
        if result == 1:
            return out.value
        elif result == -5:  # Key not found
            return None
        else:
            raise SynaError(result)
    
    def get_history_tensor(self, key: str) -> np.ndarray:
        """
        Get the complete history of float values for a key as a numpy array.
        
        This is optimized for ML workloads - the returned array can be
        used directly with PyTorch or TensorFlow.
        
        Args:
            key: The key to read history for
            
        Returns:
            numpy array of float64 values in chronological order
            
        Raises:
            SynaError: If the read fails
        """
        self._check_open()
        length = c_size_t()
        ptr = self._lib.SYNA_get_history_tensor(
            self._path, 
            key.encode('utf-8'), 
            byref(length)
        )
        
        if not ptr:
            return np.array([], dtype=np.float64)
        
        try:
            # Create numpy array from pointer (copies data)
            arr = np.ctypeslib.as_array(ptr, shape=(length.value,)).copy()
            return arr
        finally:
            # Free the tensor memory
            self._lib.SYNA_free_tensor(ptr, length)
    
    def delete(self, key: str) -> None:
        """
        Delete a key from the database.
        
        Args:
            key: The key to delete
            
        Raises:
            SynaError: If the delete fails
        """
        self._check_open()
        result = self._lib.SYNA_delete(self._path, key.encode('utf-8'))
        if result != 1:
            raise SynaError(result)
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not deleted.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        self._check_open()
        result = self._lib.SYNA_exists(self._path, key.encode('utf-8'))
        if result < 0:
            raise SynaError(result)
        return result == 1
    
    def keys(self) -> List[str]:
        """
        List all non-deleted keys in the database.
        
        Returns:
            List of key strings
        """
        self._check_open()
        length = c_size_t()
        ptr = self._lib.SYNA_keys(self._path, byref(length))
        
        if not ptr or length.value == 0:
            return []
        
        try:
            keys = []
            for i in range(length.value):
                key_bytes = ctypes.string_at(ptr[i])
                keys.append(key_bytes.decode('utf-8'))
            return keys
        finally:
            self._lib.SYNA_free_keys(ptr, length)
    
    def compact(self) -> None:
        """
        Compact the database to reclaim disk space.
        
        This removes deleted entries and old versions of keys.
        After compaction, get_history_tensor() will only return
        the latest value for each key.
        """
        self._check_open()
        result = self._lib.SYNA_compact(self._path)
        if result != 1:
            raise SynaError(result)
    
    # Convenience methods for pandas integration
    
    def put_numpy(self, key: str, arr: np.ndarray) -> int:
        """
        Store a numpy array as bytes.
        
        Args:
            key: The key
            arr: numpy array to store
            
        Returns:
            Byte offset where the entry was written
        """
        return self.put_bytes(key, arr.tobytes())
    
    def get_text(self, key: str) -> Optional[str]:
        """
        Read a text value from the database.
        
        Args:
            key: The key to read
            
        Returns:
            The text value, or None if key not found
            
        Raises:
            SynaError: If the read fails (except key not found)
        """
        self._check_open()
        length = c_size_t()
        ptr = self._lib.SYNA_get_text(
            self._path,
            key.encode('utf-8'),
            byref(length)
        )
        
        if not ptr:
            return None
        
        try:
            # Read the string (ptr is already null-terminated)
            result = ctypes.string_at(ptr, length.value).decode('utf-8')
            return result
        finally:
            self._lib.SYNA_free_text(ptr, length.value)
    
    def get_bytes(self, key: str) -> Optional[bytes]:
        """
        Read a bytes value from the database.
        
        Args:
            key: The key to read
            
        Returns:
            The bytes value, or None if key not found
            
        Raises:
            SynaError: If the read fails (except key not found)
        """
        self._check_open()
        length = c_size_t()
        ptr = self._lib.SYNA_get_bytes(
            self._path,
            key.encode('utf-8'),
            byref(length)
        )
        
        if not ptr or length.value == 0:
            return None
        
        try:
            # Copy bytes from pointer
            result = bytes(ctypes.cast(ptr, POINTER(c_uint8 * length.value)).contents)
            return result
        finally:
            self._lib.SYNA_free_bytes(ptr, length.value)
    
    def get_numpy(self, key: str, dtype=np.float64, shape=None) -> Optional[np.ndarray]:
        """
        Read a numpy array stored as bytes.
        
        Args:
            key: The key
            dtype: numpy dtype of the array
            shape: Optional shape to reshape to
            
        Returns:
            numpy array, or None if key not found
        """
        self._check_open()
        data = self.get_bytes(key)
        if data is None:
            return None
        
        arr = np.frombuffer(data, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr


    def to_dataframe(self, key_pattern: str = None) -> 'pd.DataFrame':
        """
        Load data into a pandas DataFrame.
        
        Args:
            key_pattern: Optional glob pattern to filter keys (e.g., "sensor/*")
            
        Returns:
            DataFrame with keys as index and values as columns
        """
        import pandas as pd
        
        self._check_open()
        all_keys = self.keys()
        
        # Filter keys if pattern provided
        if key_pattern:
            import fnmatch
            all_keys = [k for k in all_keys if fnmatch.fnmatch(k, key_pattern)]
        
        # Build DataFrame from float histories
        data = {}
        max_len = 0
        
        for key in all_keys:
            try:
                history = self.get_history_tensor(key)
                if len(history) > 0:
                    data[key] = history
                    max_len = max(max_len, len(history))
            except:
                pass  # Skip non-float keys
        
        if not data:
            return pd.DataFrame()
        
        # Pad shorter series with NaN
        for key in data:
            if len(data[key]) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(data[key])] = data[key]
                data[key] = padded
        
        return pd.DataFrame(data)
    
    def from_dataframe(self, df: 'pd.DataFrame', key_prefix: str = "") -> int:
        """
        Store a pandas DataFrame into the database.
        
        Each column becomes a key with the column name (prefixed if specified).
        Each row value is appended to that key's history.
        
        Args:
            df: DataFrame to store
            key_prefix: Optional prefix for keys (e.g., "data/")
            
        Returns:
            Number of entries written
        """
        self._check_open()
        count = 0
        
        for col in df.columns:
            key = f"{key_prefix}{col}" if key_prefix else str(col)
            
            for value in df[col].dropna():
                if isinstance(value, (int, np.integer)):
                    self.put_int(key, int(value))
                elif isinstance(value, (float, np.floating)):
                    self.put_float(key, float(value))
                elif isinstance(value, str):
                    self.put_text(key, value)
                else:
                    # Try to convert to float
                    try:
                        self.put_float(key, float(value))
                    except:
                        self.put_text(key, str(value))
                count += 1
        
        return count
    
    def to_timeseries_dataframe(self, key: str) -> 'pd.DataFrame':
        """
        Load a single key's history as a time-indexed DataFrame.
        
        Args:
            key: The key to load
            
        Returns:
            DataFrame with timestamp index and value column
        """
        import pandas as pd
        
        self._check_open()
        history = self.get_history_tensor(key)
        
        # Create a simple integer index (we don't have timestamps in the wrapper yet)
        return pd.DataFrame({
            'value': history
        })

    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def _collect_data(self, key_pattern: str = None) -> List[dict]:
        """
        Collect all data as a list of dicts for export.
        
        Args:
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            List of dicts with 'key', 'type', and 'value' fields
        """
        all_keys = self.keys()
        
        # Filter keys if pattern provided
        if key_pattern:
            import fnmatch
            all_keys = [k for k in all_keys if fnmatch.fnmatch(k, key_pattern)]
        
        records = []
        for key in all_keys:
            # Try each type in order
            value = self.get_float(key)
            if value is not None:
                records.append({'key': key, 'type': 'float', 'value': value})
                continue
            
            value = self.get_int(key)
            if value is not None:
                records.append({'key': key, 'type': 'int', 'value': value})
                continue
            
            value = self.get_text(key)
            if value is not None:
                records.append({'key': key, 'type': 'text', 'value': value})
                continue
            
            value = self.get_bytes(key)
            if value is not None:
                records.append({'key': key, 'type': 'bytes', 'value': value.hex()})
                continue
        
        return records
    
    def export_json(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to JSON file.
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
        """
        import json
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        with open(path, 'w') as f:
            json.dump({r['key']: r['value'] for r in records}, f, indent=2)
        
        return len(records)
    
    def export_jsonl(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to JSON Lines file (one JSON object per line).
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
        """
        import json
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        with open(path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        return len(records)
    
    def export_csv(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to CSV file.
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
        """
        import csv
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'type', 'value'])
            writer.writeheader()
            writer.writerows(records)
        
        return len(records)
    
    def export_pickle(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to Python pickle file.
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
            
        Note:
            Pickle files can only be read by Python. For cross-language
            compatibility, use Parquet or Arrow.
        """
        import pickle
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        with open(path, 'wb') as f:
            pickle.dump(records, f)
        
        return len(records)
    
    def export_parquet(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to Apache Parquet file.
        
        Requires: pyarrow or fastparquet
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
            
        Note:
            Parquet is a columnar format ideal for analytics and ML.
            It's readable by pandas, Spark, DuckDB, and many other tools.
        """
        import pandas as pd
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)
        
        return len(records)
    
    def export_arrow(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to Apache Arrow IPC file (.arrow or .feather).
        
        Requires: pyarrow
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
            
        Note:
            Arrow is a columnar in-memory format with zero-copy reads.
            It's ideal for high-performance data exchange between systems.
        """
        import pyarrow as pa
        import pyarrow.feather as feather
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        # Convert to Arrow table
        table = pa.Table.from_pylist(records)
        
        # Write as Feather (Arrow IPC format)
        feather.write_feather(table, path)
        
        return len(records)
    
    def export_msgpack(self, path: str, key_pattern: str = None) -> int:
        """
        Export database to MessagePack file.
        
        Requires: msgpack
        
        Args:
            path: Output file path
            key_pattern: Optional glob pattern to filter keys
            
        Returns:
            Number of records exported
            
        Note:
            MessagePack is a compact binary format, smaller than JSON.
        """
        import msgpack
        
        self._check_open()
        records = self._collect_data(key_pattern)
        
        with open(path, 'wb') as f:
            msgpack.pack(records, f)
        
        return len(records)
    
    # =========================================================================
    # Import Methods
    # =========================================================================
    
    def import_json(self, path: str, key_prefix: str = "") -> int:
        """
        Import data from JSON file.
        
        Args:
            path: Input file path
            key_prefix: Optional prefix for imported keys
            
        Returns:
            Number of records imported
        """
        import json
        
        self._check_open()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        count = 0
        for key, value in data.items():
            full_key = f"{key_prefix}{key}" if key_prefix else key
            
            if isinstance(value, float):
                self.put_float(full_key, value)
            elif isinstance(value, int):
                self.put_int(full_key, value)
            elif isinstance(value, str):
                self.put_text(full_key, value)
            else:
                self.put_text(full_key, str(value))
            count += 1
        
        return count
    
    def import_jsonl(self, path: str, key_prefix: str = "") -> int:
        """
        Import data from JSON Lines file.
        
        Args:
            path: Input file path
            key_prefix: Optional prefix for imported keys
            
        Returns:
            Number of records imported
        """
        import json
        
        self._check_open()
        count = 0
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                record = json.loads(line)
                key = record.get('key', '')
                value = record.get('value')
                
                if not key:
                    continue
                
                full_key = f"{key_prefix}{key}" if key_prefix else key
                
                if isinstance(value, float):
                    self.put_float(full_key, value)
                elif isinstance(value, int):
                    self.put_int(full_key, value)
                elif isinstance(value, str):
                    self.put_text(full_key, value)
                else:
                    self.put_text(full_key, str(value))
                count += 1
        
        return count
    
    def import_csv(self, path: str, key_prefix: str = "") -> int:
        """
        Import data from CSV file.
        
        Expects columns: key, type, value
        
        Args:
            path: Input file path
            key_prefix: Optional prefix for imported keys
            
        Returns:
            Number of records imported
        """
        import csv
        
        self._check_open()
        count = 0
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get('key', '')
                value_type = row.get('type', 'text')
                value = row.get('value', '')
                
                if not key:
                    continue
                
                full_key = f"{key_prefix}{key}" if key_prefix else key
                
                if value_type == 'float':
                    self.put_float(full_key, float(value))
                elif value_type == 'int':
                    self.put_int(full_key, int(value))
                else:
                    self.put_text(full_key, value)
                count += 1
        
        return count
    
    def import_pickle(self, path: str, key_prefix: str = "") -> int:
        """
        Import data from Python pickle file.
        
        Args:
            path: Input file path
            key_prefix: Optional prefix for imported keys
            
        Returns:
            Number of records imported
        """
        import pickle
        
        self._check_open()
        
        with open(path, 'rb') as f:
            records = pickle.load(f)
        
        count = 0
        for record in records:
            key = record.get('key', '')
            value_type = record.get('type', 'text')
            value = record.get('value')
            
            if not key:
                continue
            
            full_key = f"{key_prefix}{key}" if key_prefix else key
            
            if value_type == 'float':
                self.put_float(full_key, float(value))
            elif value_type == 'int':
                self.put_int(full_key, int(value))
            else:
                self.put_text(full_key, str(value))
            count += 1
        
        return count
    
    def import_parquet(self, path: str, key_prefix: str = "") -> int:
        """
        Import data from Apache Parquet file.
        
        Requires: pyarrow or fastparquet
        
        Expects columns: key, type, value
        
        Args:
            path: Input file path
            key_prefix: Optional prefix for imported keys
            
        Returns:
            Number of records imported
        """
        import pandas as pd
        
        self._check_open()
        
        df = pd.read_parquet(path)
        count = 0
        
        for _, row in df.iterrows():
            key = row.get('key', '')
            value_type = row.get('type', 'text')
            value = row.get('value')
            
            if not key:
                continue
            
            full_key = f"{key_prefix}{key}" if key_prefix else key
            
            if value_type == 'float':
                self.put_float(full_key, float(value))
            elif value_type == 'int':
                self.put_int(full_key, int(value))
            else:
                self.put_text(full_key, str(value))
            count += 1
        
        return count


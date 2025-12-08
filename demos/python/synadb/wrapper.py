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
    
    if system == "Windows":
        lib_name = "synadb.dll"
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
    else:
        lib_name = "libsynadb.so"
    
    # Search paths
    search_paths = [
        # Relative to this file
        Path(__file__).parent.parent.parent.parent / "target" / "release" / lib_name,
        Path(__file__).parent.parent.parent.parent / "target" / "debug" / lib_name,
        # Current directory
        Path.cwd() / lib_name,
        Path.cwd() / "target" / "release" / lib_name,
        Path.cwd() / "target" / "debug" / lib_name,
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    
    # Try system library path
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
    
    def __init__(self, path: str):
        """
        Open or create a database at the given path.
        
        Args:
            path: Path to the database file
            
        Raises:
            SynaError: If the database cannot be opened
        """
        self._load_library()
        self._path = path.encode('utf-8')
        self._closed = False
        
        result = self._lib.SYNA_open(self._path)
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


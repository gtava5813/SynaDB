"""
FreshnessIndex: Scalable staleness queries via Forward Decay.

Python wrapper around the ``SYNA_davo_freshness_index_*`` FFI functions.
Uses a deadline-based BTreeMap secondary index for O(k + log N) staleness
scans instead of O(N) full scans.
"""

import ctypes
from ctypes import c_char_p, c_double, c_int32, c_size_t, POINTER, byref
import platform
from pathlib import Path
from typing import List, Optional


# ── Error codes (must match src/ffi_davo.rs) ──────────────────────────

DAVO_SUCCESS = 1
DAVO_ERR_NULL_PTR = -1
DAVO_ERR_INVALID_UTF8 = -2
DAVO_ERR_NOT_FOUND = -3
DAVO_ERR_ALREADY_EXISTS = -4
DAVO_ERR_INTERNAL = -100


class DavoError(Exception):
    """Exception raised for DAVO FFI errors."""

    ERROR_CODES = {
        0: "Generic error",
        -1: "Null pointer",
        -2: "Invalid UTF-8",
        -3: "Not found",
        -4: "Already exists",
        -100: "Internal panic",
    }

    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or self.ERROR_CODES.get(code, f"Unknown error: {code}")
        super().__init__(self.message)


# ── Library loading ───────────────────────────────────────────────────

def _find_library() -> str:
    """Find the SynaDB shared library."""
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        lib_name = "synadb.dll"
        lib_names = ["synadb.dll"]
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
        if machine in ("arm64", "aarch64"):
            lib_names = ["libsynadb-arm64.dylib", "libsynadb.dylib"]
        else:
            lib_names = ["libsynadb-x86_64.dylib", "libsynadb.dylib"]
    else:
        lib_name = "libsynadb.so"
        lib_names = ["libsynadb.so"]

    wrapper_dir = Path(__file__).parent
    workspace_root = wrapper_dir.parent.parent.parent.parent

    for lib in lib_names:
        search_paths = [
            wrapper_dir / lib,
            workspace_root / "target" / "release" / lib_name,
            workspace_root / "target" / "debug" / lib_name,
            Path.cwd() / lib,
            Path.cwd() / "target" / "release" / lib_name,
            Path.cwd() / "target" / "debug" / lib_name,
        ]
        for path in search_paths:
            if path.exists():
                return str(path)

    return lib_name


def _load_library():
    """Load the SynaDB shared library and set up DAVO FFI signatures."""
    lib = ctypes.CDLL(_find_library())

    # FreshnessIndex
    lib.SYNA_davo_freshness_index_new.argtypes = [c_char_p, c_double]
    lib.SYNA_davo_freshness_index_new.restype = c_int32

    lib.SYNA_davo_freshness_index_insert.argtypes = [c_char_p, c_char_p, c_double]
    lib.SYNA_davo_freshness_index_insert.restype = c_int32

    lib.SYNA_davo_freshness_index_get_freshness.argtypes = [
        c_char_p, c_char_p, POINTER(c_double),
    ]
    lib.SYNA_davo_freshness_index_get_freshness.restype = c_int32

    lib.SYNA_davo_freshness_index_query_stale.argtypes = [
        c_char_p, POINTER(POINTER(c_char_p)), POINTER(c_size_t),
    ]
    lib.SYNA_davo_freshness_index_query_stale.restype = c_int32

    lib.SYNA_davo_freshness_index_evict_stale.argtypes = [c_char_p, POINTER(c_size_t)]
    lib.SYNA_davo_freshness_index_evict_stale.restype = c_int32

    lib.SYNA_davo_freshness_index_len.argtypes = [c_char_p]
    lib.SYNA_davo_freshness_index_len.restype = ctypes.c_int64

    lib.SYNA_davo_freshness_index_close.argtypes = [c_char_p]
    lib.SYNA_davo_freshness_index_close.restype = c_int32

    lib.SYNA_davo_freshness_index_save.argtypes = [c_char_p, c_char_p]
    lib.SYNA_davo_freshness_index_save.restype = c_int32

    lib.SYNA_davo_freshness_index_load.argtypes = [c_char_p, c_char_p]
    lib.SYNA_davo_freshness_index_load.restype = c_int32

    lib.SYNA_davo_free_keys.argtypes = [POINTER(c_char_p), c_size_t]
    lib.SYNA_davo_free_keys.restype = None

    return lib


# Module-level library singleton
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ── FreshnessIndex class ─────────────────────────────────────────────

class FreshnessIndex:
    """Scalable freshness index with deadline-based staleness queries.

    Tracks keys with decay rates and provides O(k + log N) staleness
    scans via the Forward Decay technique.

    Args:
        path: Unique identifier for this index instance.
        threshold: Staleness threshold in (0, 1). Default 0.5.

    Example::

        with FreshnessIndex("my_index") as idx:
            idx.insert("sensor/temp", decay_rate=0.001)
            print(idx.get_freshness("sensor/temp"))  # ~1.0
    """

    def __init__(self, path: str, threshold: float = 0.5) -> None:
        self._lib = _get_lib()
        self._path = path.encode("utf-8")
        self._closed = False

        rc = self._lib.SYNA_davo_freshness_index_new(self._path, threshold)
        if rc != DAVO_SUCCESS:
            raise DavoError(rc, f"Failed to create FreshnessIndex at '{path}'")

    def insert(self, key: str, decay_rate: float) -> None:
        """Insert or update a key with the given decay rate λ.

        Args:
            key: Data key to track.
            decay_rate: Decay rate λ (per second). 0.0 = static (never stale).
        """
        rc = self._lib.SYNA_davo_freshness_index_insert(
            self._path, key.encode("utf-8"), decay_rate,
        )
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)

    def get_freshness(self, key: str) -> Optional[float]:
        """Get the freshness of a key (0.0 – 1.0).

        Args:
            key: Data key to query.

        Returns:
            Freshness score, or None if the key is not tracked.
        """
        out = c_double(0.0)
        rc = self._lib.SYNA_davo_freshness_index_get_freshness(
            self._path, key.encode("utf-8"), byref(out),
        )
        if rc == DAVO_ERR_NOT_FOUND:
            return None
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)
        return out.value

    def query_stale(self) -> List[str]:
        """Return all keys whose freshness is below the threshold.

        Returns:
            List of stale key strings.
        """
        out_keys = POINTER(c_char_p)()
        out_count = c_size_t(0)
        rc = self._lib.SYNA_davo_freshness_index_query_stale(
            self._path, byref(out_keys), byref(out_count),
        )
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)

        count = out_count.value
        keys = []
        if count > 0 and out_keys:
            for i in range(count):
                raw = out_keys[i]
                if raw:
                    keys.append(raw.decode("utf-8"))
            self._lib.SYNA_davo_free_keys(out_keys, count)
        return keys

    def evict_stale(self) -> int:
        """Remove all stale entries from the index.

        Returns:
            Number of entries evicted.
        """
        out = c_size_t(0)
        rc = self._lib.SYNA_davo_freshness_index_evict_stale(self._path, byref(out))
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)
        return out.value

    def __len__(self) -> int:
        """Return the number of tracked keys."""
        n = self._lib.SYNA_davo_freshness_index_len(self._path)
        return max(0, n)

    def close(self) -> None:
        """Close the index and release resources."""
        if not self._closed:
            self._lib.SYNA_davo_freshness_index_close(self._path)
            self._closed = True

    def save(self, file_path: str) -> None:
        """Save the index to disk.

        Args:
            file_path: Path to write the persisted index to.
        """
        rc = self._lib.SYNA_davo_freshness_index_save(
            self._path, file_path.encode("utf-8")
        )
        if rc != DAVO_SUCCESS:
            raise DavoError(rc, f"Failed to save index to '{file_path}'")

    @classmethod
    def load(cls, path: str, file_path: str) -> "FreshnessIndex":
        """Load a persisted index from disk.

        Args:
            path: Unique identifier to register the loaded index under.
            file_path: Path to read the persisted index from.

        Returns:
            A FreshnessIndex bound to the loaded data.
        """
        lib = _get_lib()
        rc = lib.SYNA_davo_freshness_index_load(
            path.encode("utf-8"), file_path.encode("utf-8")
        )
        if rc != DAVO_SUCCESS:
            raise DavoError(rc, f"Failed to load index from '{file_path}'")

        # Construct without calling __init__ (index is already in registry)
        instance = cls.__new__(cls)
        instance._lib = lib
        instance._path = path.encode("utf-8")
        instance._closed = False
        return instance

    def __enter__(self) -> "FreshnessIndex":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

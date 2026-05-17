"""
Syna Query — Python wrapper for the EQL/EMQ query language.

Execute SQL-like queries against SynaDB databases from Python.

Example::

    from synadb import SynaDB
    from synadb.query import query_eql, query_emq

    with SynaDB("mydata.db") as db:
        # SQL-like query
        result = query_eql(db.path, "SELECT * FROM 'sensor/*' WHERE value > 30")
        for row in result["rows"]:
            print(f"{row['key']}: {row['value']}")

        # MongoDB-like query
        result = query_emq(db.path, {
            "from": "sensor/*",
            "filter": {"value": {"$gt": 30}},
            "sort": {"value": -1},
            "limit": 10
        })
"""

import ctypes
import json
import platform
from ctypes import c_char_p, c_int32, c_size_t, POINTER, byref
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Error codes ───────────────────────────────────────────────────────

QUERY_SUCCESS = 1
QUERY_ERR_PARSE = -10
QUERY_ERR_INTERNAL = -100


class QueryError(Exception):
    """Exception raised for query execution errors."""

    ERROR_CODES = {
        -10: "Parse error",
        -11: "Type error",
        -12: "Unknown function",
        -13: "Query timeout",
        -14: "Invalid regex",
        -15: "Non-numeric aggregation",
        -16: "Insufficient data",
        -17: "Database error",
        -100: "Internal error",
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
    elif system == "Darwin":
        lib_name = "libsynadb.dylib"
    else:
        lib_name = "libsynadb.so"

    wrapper_dir = Path(__file__).parent
    workspace_root = wrapper_dir.parent.parent.parent

    search_paths = [
        wrapper_dir / lib_name,
        workspace_root / "target" / "release" / lib_name,
        workspace_root / "target" / "debug" / lib_name,
        Path.cwd() / lib_name,
        Path.cwd() / "target" / "release" / lib_name,
        Path.cwd() / "target" / "debug" / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    return lib_name


def _load_library():
    """Load the SynaDB shared library and set up query FFI signatures."""
    lib = ctypes.CDLL(_find_library())

    # Query FFI
    lib.SYNA_query_eql.argtypes = [c_char_p, c_char_p, POINTER(c_char_p), POINTER(c_size_t)]
    lib.SYNA_query_eql.restype = c_int32

    lib.SYNA_query_emq.argtypes = [c_char_p, c_char_p, POINTER(c_char_p), POINTER(c_size_t)]
    lib.SYNA_query_emq.restype = c_int32

    lib.SYNA_query_free_result.argtypes = [c_char_p]
    lib.SYNA_query_free_result.restype = None

    # We also need SYNA_open/SYNA_close to ensure the DB is in the registry
    lib.SYNA_open.argtypes = [c_char_p]
    lib.SYNA_open.restype = c_int32

    lib.SYNA_close.argtypes = [c_char_p]
    lib.SYNA_close.restype = c_int32

    return lib


# Module-level library singleton
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ── Public API ────────────────────────────────────────────────────────

def query_eql(db_path: str, query: str) -> Dict[str, Any]:
    """Execute an EQL (SQL-like) query against a database.

    Args:
        db_path: Path to the SynaDB database file.
        query: EQL query string (e.g., "SELECT * FROM 'sensor/*' WHERE value > 30").

    Returns:
        Dict with "rows" (list of {key, value, timestamp}) and "metadata"
        (execution_time_us, rows_scanned, rows_returned, index_used).

    Raises:
        QueryError: If parsing or execution fails.

    Example::

        result = query_eql("mydata.db", "SELECT * FROM 'sensor/*' LIMIT 10")
        for row in result["rows"]:
            print(f"{row['key']}: {row['value']}")
    """
    lib = _get_lib()
    path_bytes = db_path.encode("utf-8")
    query_bytes = query.encode("utf-8")

    # Ensure DB is open in the global registry
    lib.SYNA_open(path_bytes)

    out_json = c_char_p()
    out_len = c_size_t(0)

    rc = lib.SYNA_query_eql(path_bytes, query_bytes, byref(out_json), byref(out_len))

    if rc != QUERY_SUCCESS:
        raise QueryError(rc)

    try:
        json_str = out_json.value.decode("utf-8") if out_json.value else "{}"
        return json.loads(json_str)
    finally:
        if out_json.value:
            lib.SYNA_query_free_result(out_json)


def query_emq(db_path: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an EMQ (MongoDB-like) query against a database.

    Args:
        db_path: Path to the SynaDB database file.
        document: Query document dict (e.g., {"from": "sensor/*", "filter": {"value": {"$gt": 30}}}).

    Returns:
        Dict with "rows" and "metadata" (same as query_eql).

    Raises:
        QueryError: If parsing or execution fails.

    Example::

        result = query_emq("mydata.db", {
            "from": "sensor/*",
            "filter": {"value": {"$gte": 20}},
            "sort": {"timestamp": -1},
            "limit": 50
        })
    """
    lib = _get_lib()
    path_bytes = db_path.encode("utf-8")
    doc_json = json.dumps(document).encode("utf-8")

    # Ensure DB is open
    lib.SYNA_open(path_bytes)

    out_json = c_char_p()
    out_len = c_size_t(0)

    rc = lib.SYNA_query_emq(path_bytes, doc_json, byref(out_json), byref(out_len))

    if rc != QUERY_SUCCESS:
        raise QueryError(rc)

    try:
        json_str = out_json.value.decode("utf-8") if out_json.value else "{}"
        return json.loads(json_str)
    finally:
        if out_json.value:
            lib.SYNA_query_free_result(out_json)


# ── Convenience class ─────────────────────────────────────────────────

class QueryEngine:
    """High-level query interface bound to a database path.

    Example::

        engine = QueryEngine("mydata.db")
        rows = engine.select("sensor/*", where="value > 30", limit=10)
        count = engine.count("sensor/*")
        avg = engine.avg("sensor/*")
        engine.close()
    """

    def __init__(self, db_path: str):
        self._path = db_path
        self._lib = _get_lib()
        self._lib.SYNA_open(db_path.encode("utf-8"))

    def eql(self, query: str) -> Dict[str, Any]:
        """Execute a raw EQL query."""
        return query_eql(self._path, query)

    def emq(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a raw EMQ query."""
        return query_emq(self._path, document)

    def select(
        self,
        pattern: str,
        where: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Build and execute a SELECT query.

        Args:
            pattern: Key pattern (e.g., "sensor/*").
            where: Optional WHERE clause (e.g., "value > 30").
            order_by: Optional ORDER BY (e.g., "value DESC").
            limit: Optional LIMIT.
            offset: Optional OFFSET.

        Returns:
            List of row dicts with key, value, timestamp.
        """
        q = f"SELECT * FROM '{pattern}'"
        if where:
            q += f" WHERE {where}"
        if order_by:
            q += f" ORDER BY {order_by}"
        if limit is not None:
            q += f" LIMIT {limit}"
        if offset is not None:
            q += f" OFFSET {offset}"
        result = self.eql(q)
        return result.get("rows", [])

    def count(self, pattern: str, where: Optional[str] = None) -> int:
        """Count matching entries."""
        q = f"SELECT COUNT(*) FROM '{pattern}'"
        if where:
            q += f" WHERE {where}"
        result = self.eql(q)
        rows = result.get("rows", [])
        if rows and "value" in rows[0]:
            v = rows[0]["value"]
            if isinstance(v, dict) and "Int" in v:
                return v["Int"]
            return int(v) if v else 0
        return 0

    def avg(self, pattern: str, where: Optional[str] = None) -> Optional[float]:
        """Compute average of matching float values."""
        q = f"SELECT AVG(value) FROM '{pattern}'"
        if where:
            q += f" WHERE {where}"
        result = self.eql(q)
        rows = result.get("rows", [])
        if rows and "value" in rows[0]:
            v = rows[0]["value"]
            if isinstance(v, dict) and "Float" in v:
                return v["Float"]
            if v is not None:
                return float(v)
        return None

    def close(self):
        """Close the database."""
        self._lib.SYNA_close(self._path.encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

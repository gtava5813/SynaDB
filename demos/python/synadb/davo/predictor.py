"""
DecayPredictor: Bayesian learning of decay rates.

Python wrapper around the ``SYNA_davo_predictor_*`` FFI functions.
Maintains a Gamma posterior over the decay rate λ and supports
Thompson Sampling for exploration/exploitation trade-offs.
"""

import ctypes
from ctypes import c_char_p, c_double, c_int32, POINTER, byref
import platform
from pathlib import Path
from typing import Optional


# ── Error codes (must match src/ffi_davo.rs) ──────────────────────────

DAVO_SUCCESS = 1
DAVO_ERR_NULL_PTR = -1
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
    """Load the SynaDB shared library and set up predictor FFI signatures."""
    lib = ctypes.CDLL(_find_library())

    lib.SYNA_davo_predictor_new.argtypes = [c_char_p]
    lib.SYNA_davo_predictor_new.restype = c_int32

    lib.SYNA_davo_predictor_observe.argtypes = [c_char_p, c_double]
    lib.SYNA_davo_predictor_observe.restype = c_int32

    lib.SYNA_davo_predictor_predict.argtypes = [c_char_p, POINTER(c_double)]
    lib.SYNA_davo_predictor_predict.restype = c_int32

    lib.SYNA_davo_predictor_sample.argtypes = [c_char_p, POINTER(c_double)]
    lib.SYNA_davo_predictor_sample.restype = c_int32

    lib.SYNA_davo_predictor_uncertainty.argtypes = [c_char_p, POINTER(c_double)]
    lib.SYNA_davo_predictor_uncertainty.restype = c_int32

    lib.SYNA_davo_predictor_close.argtypes = [c_char_p]
    lib.SYNA_davo_predictor_close.restype = c_int32

    return lib


# Module-level library singleton
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ── DecayPredictor class ─────────────────────────────────────────────

class DecayPredictor:
    """Bayesian decay-rate predictor using a Gamma conjugate prior.

    Learns the optimal decay rate λ from observed staleness outcomes.
    Supports Thompson Sampling for exploration/exploitation.

    Args:
        path: Unique identifier for this predictor instance.

    Example::

        with DecayPredictor("my_pred") as pred:
            for _ in range(100):
                pred.observe(0.05)
            print(pred.predict())      # ~0.05
            print(pred.uncertainty())   # decreasing
    """

    def __init__(self, path: str) -> None:
        self._lib = _get_lib()
        self._path = path.encode("utf-8")
        self._closed = False

        rc = self._lib.SYNA_davo_predictor_new(self._path)
        if rc != DAVO_SUCCESS:
            raise DavoError(rc, f"Failed to create DecayPredictor at '{path}'")

    def observe(self, actual_decay: float) -> None:
        """Feed an observed decay rate to update the posterior.

        Args:
            actual_decay: Observed decay rate (must be > 0 to update).
        """
        rc = self._lib.SYNA_davo_predictor_observe(self._path, actual_decay)
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)

    def predict(self) -> float:
        """Return the point-estimate prediction (posterior mean α/β).

        Returns:
            Predicted decay rate λ.
        """
        out = c_double(0.0)
        rc = self._lib.SYNA_davo_predictor_predict(self._path, byref(out))
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)
        return out.value

    def sample(self) -> float:
        """Sample a decay rate from the posterior (Thompson Sampling).

        Returns:
            Sampled decay rate λ (always > 0).
        """
        out = c_double(0.0)
        rc = self._lib.SYNA_davo_predictor_sample(self._path, byref(out))
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)
        return out.value

    def uncertainty(self) -> float:
        """Return the posterior uncertainty (variance α/β²).

        Returns:
            Posterior variance. Decreases as more observations are added.
        """
        out = c_double(0.0)
        rc = self._lib.SYNA_davo_predictor_uncertainty(self._path, byref(out))
        if rc != DAVO_SUCCESS:
            raise DavoError(rc)
        return out.value

    def close(self) -> None:
        """Close the predictor and release resources."""
        if not self._closed:
            self._lib.SYNA_davo_predictor_close(self._path)
            self._closed = True

    def __enter__(self) -> "DecayPredictor":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

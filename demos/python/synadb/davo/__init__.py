"""
DAVO: Decay-Aware Value Optimization (Experimental).

A self-improving database layer that treats data as decaying assets.
Every value can carry a decay rate λ so that freshness degrades over
time according to ``e^(-λ × age_seconds)``.

Status: **Experimental** — API may change between minor versions.

Quick start::

    from synadb.davo import FreshnessIndex, DecayPredictor

    # Track freshness of keys
    with FreshnessIndex("my_index") as idx:
        idx.insert("sensor/temp", decay_rate=0.001)
        print(idx.get_freshness("sensor/temp"))  # ~1.0

    # Learn decay rates from observations
    with DecayPredictor("my_pred") as pred:
        for _ in range(100):
            pred.observe(0.05)
        print(pred.predict())  # ~0.05
"""

__version__ = "0.1.0-experimental"

from .freshness import FreshnessIndex
from .predictor import DecayPredictor

__all__ = [
    "__version__",
    "FreshnessIndex",
    "DecayPredictor",
]

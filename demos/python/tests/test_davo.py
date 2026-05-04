"""
Tests for the DAVO Python wrapper.

Requires the SynaDB shared library built with --features davo.
"""

import time
import pytest
import sys
from pathlib import Path

# Add synadb package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synadb.davo import FreshnessIndex, DecayPredictor
from synadb.davo.freshness import DavoError


# ── FreshnessIndex tests ─────────────────────────────────────────────

class TestFreshnessIndex:
    """Tests for the FreshnessIndex Python wrapper."""

    def test_create_and_close(self):
        """Test basic lifecycle: create → close."""
        idx = FreshnessIndex("py_test_create", threshold=0.5)
        assert len(idx) == 0
        idx.close()

    def test_context_manager(self):
        """Test context manager auto-close."""
        with FreshnessIndex("py_test_ctx") as idx:
            idx.insert("key1", decay_rate=0.001)
            assert len(idx) == 1

    def test_insert_and_get_freshness(self):
        """Test insert and immediate freshness query."""
        with FreshnessIndex("py_test_insert") as idx:
            idx.insert("sensor/temp", decay_rate=0.001)
            f = idx.get_freshness("sensor/temp")
            assert f is not None
            assert f > 0.99, f"Expected ~1.0, got {f}"

    def test_get_freshness_nonexistent(self):
        """Test get_freshness returns None for unknown key."""
        with FreshnessIndex("py_test_none") as idx:
            assert idx.get_freshness("nonexistent") is None

    def test_static_key_never_stale(self):
        """Test that decay_rate=0.0 means never stale."""
        with FreshnessIndex("py_test_static") as idx:
            idx.insert("config", decay_rate=0.0)
            f = idx.get_freshness("config")
            assert f is not None
            assert abs(f - 1.0) < 0.001

    def test_evict_stale(self):
        """Test that fast-decaying keys get evicted."""
        with FreshnessIndex("py_test_evict") as idx:
            idx.insert("fast", decay_rate=50000.0)
            idx.insert("slow", decay_rate=0.0)
            time.sleep(0.005)
            evicted = idx.evict_stale()
            assert evicted == 1
            assert len(idx) == 1

    def test_query_stale(self):
        """Test query_stale returns correct keys."""
        with FreshnessIndex("py_test_query_stale") as idx:
            idx.insert("stale1", decay_rate=50000.0)
            idx.insert("stale2", decay_rate=50000.0)
            idx.insert("fresh", decay_rate=0.0)
            time.sleep(0.005)
            stale = idx.query_stale()
            assert len(stale) == 2
            assert "fresh" not in stale

    def test_len(self):
        """Test __len__ returns correct count."""
        with FreshnessIndex("py_test_len") as idx:
            assert len(idx) == 0
            idx.insert("a", 0.001)
            idx.insert("b", 0.001)
            assert len(idx) == 2


# ── DecayPredictor tests ─────────────────────────────────────────────

class TestDecayPredictor:
    """Tests for the DecayPredictor Python wrapper."""

    def test_create_and_close(self):
        """Test basic lifecycle: create → close."""
        pred = DecayPredictor("py_test_pred_create")
        pred.close()

    def test_context_manager(self):
        """Test context manager auto-close."""
        with DecayPredictor("py_test_pred_ctx") as pred:
            pred.observe(0.01)

    def test_convergence(self):
        """Test that prediction converges to observed value."""
        with DecayPredictor("py_test_pred_conv") as pred:
            for _ in range(100):
                pred.observe(0.05)
            prediction = pred.predict()
            assert abs(prediction - 0.05) < 0.02, f"Expected ~0.05, got {prediction}"

    def test_sample_positive(self):
        """Test that sample always returns a positive value."""
        with DecayPredictor("py_test_pred_sample") as pred:
            for _ in range(10):
                pred.observe(0.01)
            s = pred.sample()
            assert s > 0.0, f"Sample must be positive, got {s}"

    def test_uncertainty_decreases(self):
        """Test that uncertainty decreases with more observations."""
        with DecayPredictor("py_test_pred_unc") as pred:
            u0 = pred.uncertainty()
            for _ in range(50):
                pred.observe(0.01)
            u1 = pred.uncertainty()
            assert u1 < u0, f"Uncertainty should decrease: {u0} -> {u1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

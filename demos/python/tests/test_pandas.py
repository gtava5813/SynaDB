"""
Unit tests for Syna Python wrapper pandas integration.

Tests cover:
- to_dataframe() method
- from_dataframe() method
- Timestamp indexing for time-series

Requirements: 2.3
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Syna import SynaDB

# Import pandas - skip tests if not available
pd = pytest.importorskip("pandas")


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


class TestPandasIntegration:
    """Test pandas DataFrame integration - Requirements 2.3"""
    
    def test_to_dataframe_basic(self, temp_db):
        """Test converting database to DataFrame."""
        with SynaDB(temp_db) as db:
            # Write some float values
            for i in range(5):
                db.put_float("sensor_a", float(i))
                db.put_float("sensor_b", float(i * 2))
            
            df = db.to_dataframe()
            
            assert isinstance(df, pd.DataFrame)
            assert "sensor_a" in df.columns
            assert "sensor_b" in df.columns
            assert len(df) == 5
    
    def test_to_dataframe_with_pattern(self, temp_db):
        """Test filtering keys with pattern."""
        with SynaDB(temp_db) as db:
            db.put_float("sensor/temp", 23.5)
            db.put_float("sensor/humidity", 45.0)
            db.put_float("config/value", 1.0)
            
            df = db.to_dataframe("sensor/*")
            
            assert len(df.columns) == 2
            assert "sensor/temp" in df.columns
            assert "sensor/humidity" in df.columns
            assert "config/value" not in df.columns
    
    def test_to_dataframe_empty(self, temp_db):
        """Test to_dataframe with empty database."""
        with SynaDB(temp_db) as db:
            df = db.to_dataframe()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
    
    def test_from_dataframe_basic(self, temp_db):
        """Test storing DataFrame to database."""
        with SynaDB(temp_db) as db:
            # Create a simple DataFrame
            df = pd.DataFrame({
                "col_a": [1.0, 2.0, 3.0],
                "col_b": [4.0, 5.0, 6.0]
            })
            
            count = db.from_dataframe(df)
            
            assert count == 6  # 3 values * 2 columns
            
            # Verify data was stored
            tensor_a = db.get_history_tensor("col_a")
            np.testing.assert_array_equal(tensor_a, [1.0, 2.0, 3.0])
    
    def test_from_dataframe_with_prefix(self, temp_db):
        """Test storing DataFrame with key prefix."""
        with SynaDB(temp_db) as db:
            df = pd.DataFrame({
                "temp": [23.5, 24.0],
                "humidity": [45.0, 46.0]
            })
            
            db.from_dataframe(df, key_prefix="sensor/")
            
            keys = db.keys()
            assert "sensor/temp" in keys
            assert "sensor/humidity" in keys
    
    def test_from_dataframe_mixed_types(self, temp_db):
        """Test storing DataFrame with mixed types."""
        with SynaDB(temp_db) as db:
            df = pd.DataFrame({
                "floats": [1.5, 2.5],
                "ints": [1, 2],
                "strings": ["a", "b"]
            })
            
            count = db.from_dataframe(df)
            assert count == 6
    
    def test_to_timeseries_dataframe(self, temp_db):
        """Test loading single key as time-series DataFrame."""
        with SynaDB(temp_db) as db:
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            for v in values:
                db.put_float("sensor", v)
            
            df = db.to_timeseries_dataframe("sensor")
            
            assert isinstance(df, pd.DataFrame)
            assert "value" in df.columns
            assert len(df) == 5
            np.testing.assert_array_equal(df["value"].values, values)
    
    def test_roundtrip_dataframe(self, temp_db):
        """Test storing and retrieving DataFrame preserves data."""
        with SynaDB(temp_db) as db:
            # Create original DataFrame
            original = pd.DataFrame({
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0]
            })
            
            # Store it
            db.from_dataframe(original)
            
            # Retrieve it
            retrieved = db.to_dataframe()
            
            # Compare (order may differ)
            for col in original.columns:
                np.testing.assert_array_equal(
                    retrieved[col].values,
                    original[col].values
                )


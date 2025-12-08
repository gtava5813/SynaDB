#!/usr/bin/env python3
"""
Syna Pandas Integration Demo

This demo shows how to use Syna with Pandas for data analysis:
- Loading time-series into DataFrame with timestamp index
- Storing DataFrame back to Syna
- Query patterns using pandas

Requirements: 2.3 - WHEN a developer views the Python pandas demo THEN the demo 
SHALL show loading time-series data into DataFrames

Run with: python pandas_integration.py
"""

import os
import sys
import time
import tempfile
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from Syna import SynaDB


def demo_load_timeseries_to_dataframe():
    """Demonstrate loading time-series into DataFrame."""
    print("=" * 60)
    print("1. LOADING TIME-SERIES INTO DATAFRAME")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "pandas_demo.db")
        
        with SynaDB(db_path) as db:
            # Simulate sensor data collection
            print("\nSimulating Sensor Data Collection")
            print("-" * 40)
            
            np.random.seed(42)
            n_samples = 100
            
            # Generate correlated sensor data
            base_temp = 22.0
            for i in range(n_samples):
                temp = base_temp + np.sin(i * 0.1) * 3 + np.random.randn() * 0.5
                humidity = 50 + np.cos(i * 0.1) * 10 + np.random.randn() * 2
                pressure = 1013 + np.random.randn() * 2
                
                db.put_float("sensor/temperature", temp)
                db.put_float("sensor/humidity", humidity)
                db.put_float("sensor/pressure", pressure)
            
            print(f"✓ Stored {n_samples} readings for 3 sensors")
            
            # Method 1: Using to_dataframe()
            print("\nMethod 1: Using to_dataframe()")
            print("-" * 40)
            print("""
df = db.to_dataframe("sensor/*")
""")
            
            df = db.to_dataframe("sensor/*")
            print(f"✓ DataFrame shape: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            # Method 2: Manual extraction with custom index
            print("\nMethod 2: Manual Extraction with Timestamp Index")
            print("-" * 40)
            
            # Extract each sensor
            temp = db.get_history_tensor("sensor/temperature")
            humidity = db.get_history_tensor("sensor/humidity")
            pressure = db.get_history_tensor("sensor/pressure")
            
            # Create timestamp index
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            timestamps = [start_time + timedelta(minutes=i) for i in range(len(temp))]
            
            df_manual = pd.DataFrame({
                'temperature': temp,
                'humidity': humidity,
                'pressure': pressure
            }, index=pd.DatetimeIndex(timestamps, name='timestamp'))
            
            print(f"✓ DataFrame with DatetimeIndex:")
            print(df_manual.head())
            print(f"\nIndex type: {type(df_manual.index)}")
            
            # Method 3: Single key history
            print("\nMethod 3: Single Key to Series")
            print("-" * 40)
            print("""
df = db.to_timeseries_dataframe("sensor/temperature")
""")
            
            df_single = db.to_timeseries_dataframe("sensor/temperature")
            print(f"✓ Single key DataFrame:")
            print(df_single.head())
    
    print()


def demo_store_dataframe_to_Syna():
    """Demonstrate storing DataFrame back to Syna."""
    print("=" * 60)
    print("2. STORING DATAFRAME TO Syna")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "pandas_demo.db")
        
        with SynaDB(db_path) as db:
            # Create sample DataFrame
            print("\nCreating Sample DataFrame")
            print("-" * 40)
            
            np.random.seed(42)
            n_rows = 50
            
            df = pd.DataFrame({
                'temperature': np.random.randn(n_rows) * 5 + 25,
                'humidity': np.random.randn(n_rows) * 10 + 50,
                'status': np.random.choice(['ok', 'warning', 'error'], n_rows),
                'count': np.random.randint(0, 100, n_rows)
            })
            
            print(f"Original DataFrame:")
            print(df.head())
            print(f"\nShape: {df.shape}")
            print(f"dtypes:\n{df.dtypes}")
            
            # Store using from_dataframe()
            print("\nStoring with from_dataframe()")
            print("-" * 40)
            print("""
count = db.from_dataframe(df, key_prefix="data/")
""")
            
            count = db.from_dataframe(df, key_prefix="data/")
            print(f"✓ Stored {count} entries")
            
            # Verify storage
            print("\nVerifying Storage")
            print("-" * 40)
            
            keys = db.keys()
            print(f"Keys created: {sorted(keys)}")
            
            # Check numeric columns
            temp_history = db.get_history_tensor("data/temperature")
            print(f"\ndata/temperature history length: {len(temp_history)}")
            print(f"First 5 values: {temp_history[:5]}")
            
            # Reload as DataFrame
            print("\nReloading as DataFrame")
            print("-" * 40)
            
            df_reloaded = db.to_dataframe("data/*")
            print(f"Reloaded DataFrame shape: {df_reloaded.shape}")
            print(df_reloaded.head())
            
            # Note: Text columns are not included in to_dataframe()
            print("\nNote: to_dataframe() only loads float columns.")
            print("Text columns like 'status' need separate handling.")
    
    print()


def demo_pandas_query_patterns():
    """Demonstrate query patterns using pandas."""
    print("=" * 60)
    print("3. QUERY PATTERNS WITH PANDAS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "pandas_demo.db")
        
        with SynaDB(db_path) as db:
            # Setup: Create time-series data
            print("\nSetup: Creating Time-Series Data")
            print("-" * 40)
            
            np.random.seed(42)
            n_samples = 200
            
            # Simulate 24 hours of data at 5-minute intervals
            for i in range(n_samples):
                hour = (i * 5 / 60) % 24
                # Temperature varies with time of day
                temp = 20 + 5 * np.sin((hour - 6) * np.pi / 12) + np.random.randn() * 0.5
                humidity = 60 - 10 * np.sin((hour - 6) * np.pi / 12) + np.random.randn() * 2
                
                db.put_float("sensor/temp", temp)
                db.put_float("sensor/humidity", humidity)
            
            print(f"✓ Created {n_samples} samples (simulated 24h at 5min intervals)")
            
            # Load into DataFrame with timestamp index
            temp = db.get_history_tensor("sensor/temp")
            humidity = db.get_history_tensor("sensor/humidity")
            
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            timestamps = [start_time + timedelta(minutes=i*5) for i in range(len(temp))]
            
            df = pd.DataFrame({
                'temperature': temp,
                'humidity': humidity
            }, index=pd.DatetimeIndex(timestamps, name='timestamp'))
            
            print(f"\nDataFrame loaded: {df.shape}")
            
            # Query 1: Basic statistics
            print("\nQuery 1: Basic Statistics")
            print("-" * 40)
            print("""
df.describe()
""")
            print(df.describe())
            
            # Query 2: Time-based filtering
            print("\nQuery 2: Time-Based Filtering")
            print("-" * 40)
            print("""
# Morning hours (6 AM - 12 PM)
morning = df.between_time('06:00', '12:00')
""")
            
            morning = df.between_time('06:00', '12:00')
            print(f"Morning samples: {len(morning)}")
            print(f"Morning avg temp: {morning['temperature'].mean():.2f}°C")
            
            # Query 3: Rolling statistics
            print("\nQuery 3: Rolling Statistics")
            print("-" * 40)
            print("""
# 1-hour rolling average (12 samples at 5min intervals)
df['temp_rolling'] = df['temperature'].rolling(window=12).mean()
""")
            
            df['temp_rolling'] = df['temperature'].rolling(window=12).mean()
            print(df[['temperature', 'temp_rolling']].tail(10))
            
            # Query 4: Resampling
            print("\nQuery 4: Resampling (Hourly Aggregation)")
            print("-" * 40)
            print("""
# Resample to hourly averages
hourly = df.resample('h').agg({
    'temperature': ['mean', 'min', 'max'],
    'humidity': 'mean'
})
""")
            
            hourly = df[['temperature', 'humidity']].resample('h').agg({
                'temperature': ['mean', 'min', 'max'],
                'humidity': 'mean'
            })
            print(hourly.head(10))
            
            # Query 5: Conditional filtering
            print("\nQuery 5: Conditional Filtering")
            print("-" * 40)
            print("""
# Find high temperature events
high_temp = df[df['temperature'] > 24]
""")
            
            high_temp = df[df['temperature'] > 24]
            print(f"High temperature events (>24°C): {len(high_temp)}")
            print(f"Percentage: {len(high_temp)/len(df)*100:.1f}%")
            
            # Query 6: Correlation analysis
            print("\nQuery 6: Correlation Analysis")
            print("-" * 40)
            print("""
df[['temperature', 'humidity']].corr()
""")
            
            corr = df[['temperature', 'humidity']].corr()
            print(corr)
            print(f"\nTemp-Humidity correlation: {corr.loc['temperature', 'humidity']:.3f}")
            
            # Query 7: Anomaly detection
            print("\nQuery 7: Simple Anomaly Detection")
            print("-" * 40)
            print("""
# Flag values outside 2 standard deviations
mean = df['temperature'].mean()
std = df['temperature'].std()
df['anomaly'] = (df['temperature'] - mean).abs() > 2 * std
""")
            
            mean = df['temperature'].mean()
            std = df['temperature'].std()
            df['anomaly'] = (df['temperature'] - mean).abs() > 2 * std
            
            anomalies = df[df['anomaly']]
            print(f"Anomalies detected: {len(anomalies)}")
            if len(anomalies) > 0:
                print(f"Anomaly timestamps:\n{anomalies.index.tolist()[:5]}")
    
    print()


def demo_multi_sensor_analysis():
    """Demonstrate multi-sensor analysis workflow."""
    print("=" * 60)
    print("4. MULTI-SENSOR ANALYSIS WORKFLOW")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "pandas_demo.db")
        
        with SynaDB(db_path) as db:
            # Setup: Multiple sensors in different locations
            print("\nSetup: Multiple Sensors in Different Locations")
            print("-" * 40)
            
            np.random.seed(42)
            n_samples = 100
            locations = ['room1', 'room2', 'room3', 'outdoor']
            
            # Base temperatures for each location
            base_temps = {'room1': 22, 'room2': 21, 'room3': 23, 'outdoor': 15}
            
            for i in range(n_samples):
                for loc in locations:
                    temp = base_temps[loc] + np.random.randn() * 1.5
                    db.put_float(f"sensor/{loc}/temp", temp)
            
            print(f"✓ Created {n_samples} samples for {len(locations)} locations")
            
            # Load all sensors into single DataFrame
            print("\nLoading All Sensors")
            print("-" * 40)
            
            data = {}
            for loc in locations:
                data[loc] = db.get_history_tensor(f"sensor/{loc}/temp")
            
            df = pd.DataFrame(data)
            print(f"Combined DataFrame shape: {df.shape}")
            print(df.head())
            
            # Compare locations
            print("\nLocation Comparison")
            print("-" * 40)
            print(df.describe())
            
            # Find warmest/coldest
            print("\nWarmest/Coldest Analysis")
            print("-" * 40)
            
            means = df.mean()
            print(f"Average temperatures:")
            for loc in means.sort_values(ascending=False).index:
                print(f"  {loc}: {means[loc]:.2f}°C")
            
            print(f"\nWarmest location: {means.idxmax()} ({means.max():.2f}°C)")
            print(f"Coldest location: {means.idxmin()} ({means.min():.2f}°C)")
            
            # Cross-location correlation
            print("\nCross-Location Correlation")
            print("-" * 40)
            print(df.corr().round(3))
            
            # Store analysis results back
            print("\nStoring Analysis Results")
            print("-" * 40)
            
            for loc in locations:
                db.put_float(f"analysis/{loc}/mean", float(df[loc].mean()))
                db.put_float(f"analysis/{loc}/std", float(df[loc].std()))
                db.put_float(f"analysis/{loc}/min", float(df[loc].min()))
                db.put_float(f"analysis/{loc}/max", float(df[loc].max()))
            
            print("✓ Stored mean, std, min, max for each location")
            
            # Verify
            analysis_keys = [k for k in db.keys() if k.startswith("analysis/")]
            print(f"✓ Created {len(analysis_keys)} analysis keys")
    
    print()


def demo_export_import():
    """Demonstrate DataFrame export/import patterns."""
    print("=" * 60)
    print("5. DATAFRAME EXPORT/IMPORT PATTERNS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "pandas_demo.db")
        csv_path = os.path.join(tmpdir, "export.csv")
        
        with SynaDB(db_path) as db:
            # Create data
            print("\nCreating Sample Data")
            print("-" * 40)
            
            np.random.seed(42)
            for i in range(50):
                db.put_float("data/x", float(i))
                db.put_float("data/y", float(i**2 + np.random.randn() * 10))
            
            # Export to CSV
            print("\nExport: Syna → DataFrame → CSV")
            print("-" * 40)
            print("""
df = db.to_dataframe("data/*")
df.to_csv("export.csv")
""")
            
            df = db.to_dataframe("data/*")
            df.to_csv(csv_path)
            print(f"✓ Exported {len(df)} rows to CSV")
            print(f"✓ File size: {os.path.getsize(csv_path)} bytes")
            
            # Import from CSV
            print("\nImport: CSV → DataFrame → Syna")
            print("-" * 40)
            print("""
df_imported = pd.read_csv("export.csv", index_col=0)
db.from_dataframe(df_imported, key_prefix="imported/")
""")
            
            df_imported = pd.read_csv(csv_path, index_col=0)
            count = db.from_dataframe(df_imported, key_prefix="imported/")
            print(f"✓ Imported {count} entries from CSV")
            
            # Verify round-trip
            print("\nVerifying Round-Trip")
            print("-" * 40)
            
            original_x = db.get_history_tensor("data/x")
            imported_x = db.get_history_tensor("imported/data/x")
            
            print(f"Original data/x length: {len(original_x)}")
            print(f"Imported data/x length: {len(imported_x)}")
            print(f"✓ Values match: {np.allclose(original_x, imported_x)}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("   Syna PANDAS INTEGRATION DEMO")
    print("   Requirements: 2.3")
    print("=" * 60 + "\n")
    
    try:
        demo_load_timeseries_to_dataframe()
        demo_store_dataframe_to_Syna()
        demo_pandas_query_patterns()
        demo_multi_sensor_analysis()
        demo_export_import()
        
        print("=" * 60)
        print("   ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


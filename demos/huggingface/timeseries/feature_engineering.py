#!/usr/bin/env python3
"""
Feature Engineering Demo

This demo shows how to:
- Compute rolling statistics (mean, std, min, max)
- Use Syna history for window functions
- Store computed features back to DB
- Show latency metrics for real-time suitability

Run with: python feature_engineering.py

Requirements: numpy
"""

import os
import sys
import time
import math
import random
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


class SensorSimulator:
    """Simulates realistic IoT sensor data."""
    
    def __init__(self, sensor_id: str, sensor_type: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.time_offset = random.random() * 1000
        
        if sensor_type == "temperature":
            self.base = 22.0
            self.amplitude = 3.0
            self.noise = 0.2
        elif sensor_type == "humidity":
            self.base = 50.0
            self.amplitude = 15.0
            self.noise = 2.0
        elif sensor_type == "pressure":
            self.base = 1013.25
            self.amplitude = 5.0
            self.noise = 0.5
        else:
            self.base = 0.0
            self.amplitude = 1.0
            self.noise = 0.1
    
    def read(self, timestamp: float) -> float:
        daily_cycle = math.sin((timestamp + self.time_offset) * 2 * math.pi / 86400)
        drift = math.sin(timestamp * 2 * math.pi / 604800) * 0.5
        value = self.base + self.amplitude * daily_cycle + drift * self.amplitude
        value += random.gauss(0, self.noise)
        return value


class FeatureComputer:
    """Computes rolling features from time-series data."""
    
    def __init__(self, window_sizes: list = None):
        self.window_sizes = window_sizes or [10, 20]
    
    def compute_rolling_features(self, data: np.ndarray, window_size: int) -> dict:
        """Compute rolling statistics for a given window size."""
        n = len(data)
        if n < window_size:
            return None
        
        # Pre-allocate arrays
        rolling_mean = np.zeros(n - window_size + 1)
        rolling_std = np.zeros(n - window_size + 1)
        rolling_min = np.zeros(n - window_size + 1)
        rolling_max = np.zeros(n - window_size + 1)
        rolling_range = np.zeros(n - window_size + 1)
        rolling_median = np.zeros(n - window_size + 1)
        
        # Compute rolling statistics
        for i in range(n - window_size + 1):
            window = data[i:i + window_size]
            rolling_mean[i] = np.mean(window)
            rolling_std[i] = np.std(window)
            rolling_min[i] = np.min(window)
            rolling_max[i] = np.max(window)
            rolling_range[i] = rolling_max[i] - rolling_min[i]
            rolling_median[i] = np.median(window)
        
        return {
            'mean': rolling_mean,
            'std': rolling_std,
            'min': rolling_min,
            'max': rolling_max,
            'range': rolling_range,
            'median': rolling_median
        }
    
    def compute_all_features(self, data: np.ndarray) -> dict:
        """Compute features for all window sizes."""
        all_features = {}
        
        for window_size in self.window_sizes:
            features = self.compute_rolling_features(data, window_size)
            if features:
                for name, values in features.items():
                    key = f"w{window_size}_{name}"
                    all_features[key] = values
        
        return all_features
    
    def compute_derivative_features(self, data: np.ndarray) -> dict:
        """Compute derivative-based features."""
        # First derivative (rate of change)
        diff1 = np.diff(data)
        
        # Second derivative (acceleration)
        diff2 = np.diff(diff1)
        
        return {
            'diff1': diff1,
            'diff2': diff2,
            'abs_diff1': np.abs(diff1),
            'abs_diff2': np.abs(diff2)
        }
    
    def compute_lag_features(self, data: np.ndarray, lags: list = None) -> dict:
        """Compute lag features."""
        lags = lags or [1, 5, 10, 20]
        features = {}
        
        for lag in lags:
            if len(data) > lag:
                features[f'lag_{lag}'] = data[:-lag]
                features[f'diff_lag_{lag}'] = data[lag:] - data[:-lag]
        
        return features


def main():
    print("=== Feature Engineering Demo ===\n")
    
    import tempfile
    import shutil
    
    # Use temporary directory for database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "feature_engineering.db")
    
    try:
        # 1. Generate raw sensor data
        print("1. Generating raw sensor data...")
        sensors = [
            SensorSimulator("sensor-001", "temperature"),
            SensorSimulator("sensor-002", "humidity"),
            SensorSimulator("sensor-003", "pressure"),
        ]
        
        num_readings = 500
        
        with SynaDB(db_path) as db:
            for i in range(num_readings):
                timestamp = i * 60  # 1 minute intervals
                for sensor in sensors:
                    key = f"raw/{sensor.sensor_type}/{sensor.sensor_id}"
                    value = sensor.read(timestamp)
                    db.put_float(key, value)
        
        print(f"   ✓ Generated {num_readings * len(sensors)} readings\n")
        
        # 2. Load raw data from Syna
        print("2. Loading raw data from Syna...")
        raw_data = {}
        
        with SynaDB(db_path) as db:
            for sensor in sensors:
                key = f"raw/{sensor.sensor_type}/{sensor.sensor_id}"
                data = db.get_history_tensor(key)
                raw_data[f"{sensor.sensor_type}/{sensor.sensor_id}"] = data
                print(f"   {key}: {len(data)} values")
        
        print()
        
        # 3. Compute rolling features
        print("3. Computing rolling features...")
        feature_computer = FeatureComputer(window_sizes=[10, 20])
        
        all_computed_features = {}
        start_time = time.time()
        
        for sensor_key, data in raw_data.items():
            print(f"   Processing {sensor_key}...")
            
            # Rolling features
            rolling_features = feature_computer.compute_all_features(data)
            for feat_name, feat_values in rolling_features.items():
                full_key = f"{sensor_key}/{feat_name}"
                all_computed_features[full_key] = feat_values
            
            # Derivative features
            deriv_features = feature_computer.compute_derivative_features(data)
            for feat_name, feat_values in deriv_features.items():
                full_key = f"{sensor_key}/{feat_name}"
                all_computed_features[full_key] = feat_values
            
            # Lag features
            lag_features = feature_computer.compute_lag_features(data, lags=[1, 5, 10])
            for feat_name, feat_values in lag_features.items():
                full_key = f"{sensor_key}/{feat_name}"
                all_computed_features[full_key] = feat_values
        
        compute_time = time.time() - start_time
        print(f"   ✓ Computed {len(all_computed_features)} feature series in {compute_time:.3f}s\n")
        
        # 4. Store computed features back to Syna
        print("4. Storing computed features to Syna...")
        
        start_time = time.time()
        total_values = 0
        
        with SynaDB(db_path) as db:
            for feat_key, feat_values in all_computed_features.items():
                key = f"features/{feat_key}"
                for value in feat_values:
                    db.put_float(key, float(value))
                    total_values += 1
        
        store_time = time.time() - start_time
        print(f"   ✓ Stored {total_values} feature values in {store_time:.2f}s")
        print(f"   Throughput: {total_values / store_time:.0f} writes/sec\n")
        
        # 5. Verify stored features
        print("5. Verifying stored features...")
        
        with SynaDB(db_path) as db:
            # Sample verification
            sample_key = f"features/temperature/sensor-001/w10_mean"
            stored = db.get_history_tensor(sample_key)
            original = all_computed_features["temperature/sensor-001/w10_mean"]
            
            if len(stored) == len(original):
                match = np.allclose(stored, original)
                print(f"   {sample_key}: {len(stored)} values, match={match}")
            else:
                print(f"   {sample_key}: length mismatch ({len(stored)} vs {len(original)})")
        
        print()
        
        # 6. Show feature statistics
        print("6. Feature statistics (temperature/sensor-001):")
        
        with SynaDB(db_path) as db:
            # Raw data stats
            raw = db.get_history_tensor("raw/temperature/sensor-001")
            print(f"   Raw data:")
            print(f"      Mean: {np.mean(raw):.2f}°C")
            print(f"      Std:  {np.std(raw):.2f}°C")
            print(f"      Range: [{np.min(raw):.2f}, {np.max(raw):.2f}]°C")
            
            # Rolling mean stats
            rolling_mean = db.get_history_tensor("features/temperature/sensor-001/w20_mean")
            print(f"\n   Rolling mean (window=20):")
            print(f"      Mean: {np.mean(rolling_mean):.2f}°C")
            print(f"      Std:  {np.std(rolling_mean):.4f}°C (smoothed)")
            
            # Rolling std stats
            rolling_std = db.get_history_tensor("features/temperature/sensor-001/w20_std")
            print(f"\n   Rolling std (window=20):")
            print(f"      Mean: {np.mean(rolling_std):.4f}°C")
            print(f"      Range: [{np.min(rolling_std):.4f}, {np.max(rolling_std):.4f}]°C")
        
        print()
        
        # 7. Benchmark feature extraction
        print("7. Benchmarking feature extraction latency...")
        
        with SynaDB(db_path) as db:
            # Measure extraction time
            latencies = []
            for _ in range(100):
                start = time.time()
                _ = db.get_history_tensor("features/temperature/sensor-001/w20_mean")
                latencies.append((time.time() - start) * 1000)
            
            latencies = np.array(latencies)
            print(f"   Feature extraction (100 iterations):")
            print(f"      Mean: {np.mean(latencies):.3f}ms")
            print(f"      P50:  {np.percentile(latencies, 50):.3f}ms")
            print(f"      P95:  {np.percentile(latencies, 95):.3f}ms")
            print(f"      P99:  {np.percentile(latencies, 99):.3f}ms")
        
        print()
        
        # 8. Show storage statistics
        print("8. Storage statistics:")
        file_size = os.path.getsize(db_path)
        print(f"   Database size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Total values stored: {total_values + num_readings * len(sensors)}")
        print(f"   Bytes per value: {file_size / (total_values + num_readings * len(sensors)):.1f}")
        
        print()
        
        # 9. Demonstrate real-time feature computation
        print("9. Real-time feature computation demo...")
        
        # Simulate streaming with incremental feature updates
        window_size = 20
        buffer = list(raw_data["temperature/sensor-001"][-window_size:])
        
        latencies = []
        for i in range(100):
            # Simulate new reading
            new_value = sensors[0].read((num_readings + i) * 60)
            
            start = time.time()
            
            # Update buffer
            buffer.pop(0)
            buffer.append(new_value)
            
            # Compute features for current window
            window = np.array(buffer)
            features = {
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window),
                'median': np.median(window)
            }
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        latencies = np.array(latencies)
        print(f"   Real-time feature computation (100 iterations):")
        print(f"      Mean: {np.mean(latencies):.4f}ms")
        print(f"      P99:  {np.percentile(latencies, 99):.4f}ms")
        print(f"      Suitable for real-time: {'Yes' if np.percentile(latencies, 99) < 1.0 else 'No'}")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    
    print("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())


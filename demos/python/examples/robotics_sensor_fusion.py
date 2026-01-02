#!/usr/bin/env python3
"""
Robotics Sensor Fusion Demo

Simulates a warehouse robot with multiple sensors:
- LiDAR: Distance readings
- IMU: Accelerometer (3 axes)
- Temperature: Thermal readings

Demonstrates:
1. High-frequency data ingestion (100K+ ops/sec with sync_on_write=False)
2. Zero-copy tensor extraction for ML inference
3. Delta compression for storage efficiency
4. Real-time anomaly detection

Usage:
    python robotics_sensor_fusion.py              # Default (420 writes)
    python robotics_sensor_fusion.py --50k        # 50,000 writes
    python robotics_sensor_fusion.py --100k       # 100,000 writes
    python robotics_sensor_fusion.py --1m         # 1,000,000 writes
    python robotics_sensor_fusion.py --1m --batch # 1M writes with batch API (~20x faster)

Requirements:
    pip install synadb numpy
"""

import argparse
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from synadb import SynaDB


@dataclass
class SensorConfig:
    """Configuration for sensor simulation."""
    lidar_samples: int = 100
    imu_samples: int = 100
    temp_samples: int = 20
    
    @classmethod
    def default(cls) -> 'SensorConfig':
        """Default config: ~420 writes."""
        return cls(lidar_samples=100, imu_samples=100, temp_samples=20)
    
    @classmethod
    def scale_50k(cls) -> 'SensorConfig':
        """50K config: ~50,000 writes."""
        return cls(lidar_samples=10000, imu_samples=10000, temp_samples=10000)
    
    @classmethod
    def scale_100k(cls) -> 'SensorConfig':
        """100K config: ~100,000 writes."""
        return cls(lidar_samples=20000, imu_samples=20000, temp_samples=20000)
    
    @classmethod
    def scale_1m(cls) -> 'SensorConfig':
        """1M config: ~1,000,000 writes."""
        return cls(lidar_samples=200000, imu_samples=200000, temp_samples=200000)
    
    @property
    def total_writes(self) -> int:
        return self.lidar_samples + self.imu_samples * 3 + self.temp_samples


class SimulatedSensors:
    """Simulates robot sensors with realistic noise patterns."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def generate_lidar(self, n: int) -> np.ndarray:
        """Generate LiDAR distance readings (meters)."""
        t = np.linspace(0, 5, n)
        base = 2.0 + 0.5 * np.sin(t * 0.5)
        noise = self.rng.normal(0, 0.02, n)
        return base + noise
    
    def generate_imu(self, n: int) -> np.ndarray:
        """Generate IMU accelerometer readings (m/s², Nx3)."""
        t = np.linspace(0, 5, n)
        ax = 0.1 * np.sin(t * 2) + self.rng.normal(0, 0.05, n)
        ay = 0.05 * np.cos(t * 1.5) + self.rng.normal(0, 0.05, n)
        az = 9.81 + self.rng.normal(0, 0.02, n)
        return np.stack([ax, ay, az], axis=1)
    
    def generate_temperature(self, n: int) -> np.ndarray:
        """Generate temperature readings (Celsius)."""
        t = np.linspace(0, 5, n)
        base = 25.0 + 0.5 * t
        noise = self.rng.normal(0, 0.1, n)
        return base + noise


class MLInference:
    """Simulated ML model that consumes sensor tensors."""
    
    def __init__(self, db: SynaDB):
        self.db = db
        
    def get_lidar_tensor(self) -> np.ndarray:
        return self.db.get_history_tensor("lidar/distance")
    
    def get_imu_tensor(self) -> np.ndarray:
        ax = self.db.get_history_tensor("imu/accel_x")
        ay = self.db.get_history_tensor("imu/accel_y")
        az = self.db.get_history_tensor("imu/accel_z")
        
        min_len = min(len(ax), len(ay), len(az))
        if min_len == 0:
            return np.zeros((0, 3))
        return np.stack([ax[:min_len], ay[:min_len], az[:min_len]], axis=1)
    
    def get_temperature_tensor(self) -> np.ndarray:
        return self.db.get_history_tensor("thermal/motor")
    
    def detect_anomaly(self, window_size: int = 50) -> Optional[str]:
        lidar = self.get_lidar_tensor()
        
        if len(lidar) < window_size:
            return None
        
        recent = lidar[-window_size:]
        diff = np.abs(np.diff(recent))
        
        if np.max(diff) > 0.5:
            return f"OBSTACLE DETECTED: {np.max(diff):.2f}m sudden change"
        
        mean_first = np.mean(recent[:window_size//2])
        mean_last = np.mean(recent[window_size//2:])
        
        if abs(mean_last - mean_first) > 0.3:
            return f"DRIFT WARNING: {abs(mean_last - mean_first):.2f}m drift"
        
        return None
    
    def compute_statistics(self) -> dict:
        lidar = self.get_lidar_tensor()
        imu = self.get_imu_tensor()
        temp = self.get_temperature_tensor()
        
        stats = {}
        
        if len(lidar) > 0:
            stats['lidar_mean'] = np.mean(lidar)
            stats['lidar_std'] = np.std(lidar)
            stats['lidar_min'] = np.min(lidar)
            stats['lidar_max'] = np.max(lidar)
        
        if len(imu) > 0:
            stats['imu_magnitude_mean'] = np.mean(np.linalg.norm(imu, axis=1))
        
        if len(temp) > 0:
            stats['temp_mean'] = np.mean(temp)
            stats['temp_trend'] = temp[-1] - temp[0] if len(temp) > 1 else 0
        
        return stats


def run_demo(config: SensorConfig, scale_name: str, use_batch: bool = False):
    """Run the robotics sensor fusion demo."""
    print("=" * 60)
    print("SynaDB Robotics Sensor Fusion Demo")
    print("=" * 60)
    print()
    
    db_path = os.path.abspath("robot_demo.db")
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"Scale: {scale_name}")
    print(f"Mode: {'Batch API' if use_batch else 'Individual writes'}")
    print(f"Configuration:")
    print(f"  LiDAR samples: {config.lidar_samples:,}")
    print(f"  IMU samples: {config.imu_samples:,} (x3 axes)")
    print(f"  Temperature samples: {config.temp_samples:,}")
    print(f"  Total writes: {config.total_writes:,}")
    print()
    
    # Generate sensor data
    print("Phase 1: Generating Sensor Data")
    print("-" * 40)
    
    sensors = SimulatedSensors()
    
    gen_start = time.time()
    lidar_data = sensors.generate_lidar(config.lidar_samples)
    imu_data = sensors.generate_imu(config.imu_samples)
    temp_data = sensors.generate_temperature(config.temp_samples)
    gen_time = time.time() - gen_start
    
    print(f"  LiDAR: {lidar_data.shape}")
    print(f"  IMU: {imu_data.shape}")
    print(f"  Temperature: {temp_data.shape}")
    print(f"  Generation time: {gen_time:.2f}s")
    print()
    
    # Ingest data
    print("Phase 2: Data Ingestion")
    print("-" * 40)
    
    # Use sync_on_write=False for high-throughput ingestion
    db = SynaDB(db_path, sync_on_write=False)
    
    start = time.time()
    
    if use_batch:
        # Batch API: single FFI call per sensor type
        db.put_floats_batch("lidar/distance", lidar_data)
        db.put_floats_batch("imu/accel_x", imu_data[:, 0])
        db.put_floats_batch("imu/accel_y", imu_data[:, 1])
        db.put_floats_batch("imu/accel_z", imu_data[:, 2])
        db.put_floats_batch("thermal/motor", temp_data)
        write_count = config.total_writes
    else:
        # Individual writes
        write_count = 0
        
        # Progress reporting for large datasets
        report_interval = max(1, config.total_writes // 10)
        
        # Write LiDAR
        for i, val in enumerate(lidar_data):
            db.put_float("lidar/distance", float(val))
            write_count += 1
            if write_count % report_interval == 0:
                elapsed = time.time() - start
                rate = write_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {write_count:,}/{config.total_writes:,} ({rate:.0f} ops/sec)")
        
        # Write IMU (3 axes)
        for ax, ay, az in imu_data:
            db.put_float("imu/accel_x", float(ax))
            db.put_float("imu/accel_y", float(ay))
            db.put_float("imu/accel_z", float(az))
            write_count += 3
            if write_count % report_interval == 0:
                elapsed = time.time() - start
                rate = write_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {write_count:,}/{config.total_writes:,} ({rate:.0f} ops/sec)")
        
        # Write temperature
        for val in temp_data:
            db.put_float("thermal/motor", float(val))
            write_count += 1
    
    ingestion_time = time.time() - start
    
    print(f"  Ingestion complete!")
    print(f"  Total writes: {config.total_writes:,}")
    print(f"  Time: {ingestion_time:.2f}s")
    print(f"  Throughput: {config.total_writes / ingestion_time:,.0f} ops/sec")
    print()
    if use_batch:
        print("  Note: Using batch API (put_floats_batch) for maximum throughput.")
    else:
        print("  Note: Using sync_on_write=False for high throughput.")
    print()
    
    # Tensor extraction
    print("Phase 3: ML Tensor Extraction")
    print("-" * 40)
    
    ml = MLInference(db)
    
    t0 = time.time()
    lidar_tensor = ml.get_lidar_tensor()
    lidar_time = (time.time() - t0) * 1000
    
    t0 = time.time()
    imu_tensor = ml.get_imu_tensor()
    imu_time = (time.time() - t0) * 1000
    
    t0 = time.time()
    temp_tensor = ml.get_temperature_tensor()
    temp_time = (time.time() - t0) * 1000
    
    total_extraction_time = lidar_time + imu_time + temp_time
    total_values = len(lidar_tensor) + len(imu_tensor) * 3 + len(temp_tensor)
    
    print(f"  LiDAR tensor: {lidar_tensor.shape} in {lidar_time:.2f}ms")
    print(f"  IMU tensor: {imu_tensor.shape} in {imu_time:.2f}ms")
    print(f"  Temperature tensor: {temp_tensor.shape} in {temp_time:.2f}ms")
    print(f"  Total: {total_values:,} values in {total_extraction_time:.2f}ms")
    print()
    
    # ML features
    print("Phase 4: ML Feature Computation")
    print("-" * 40)
    
    stats = ml.compute_statistics()
    print("  Computed features:")
    for key, value in stats.items():
        print(f"    {key}: {value:.4f}")
    print()
    
    # Anomaly detection
    print("Phase 5: Anomaly Detection")
    print("-" * 40)
    
    anomaly = ml.detect_anomaly()
    if anomaly:
        print(f"  ⚠️  {anomaly}")
    else:
        print("  ✅ No anomalies detected")
    print()
    
    # Storage
    print("Phase 6: Storage Analysis")
    print("-" * 40)
    
    db.close()
    
    db_size = os.path.getsize(db_path)
    bytes_per_write = db_size / config.total_writes if config.total_writes > 0 else 0
    
    print(f"  Database size: {db_size:,} bytes ({db_size/1024/1024:.2f} MB)")
    print(f"  Bytes per write: {bytes_per_write:.1f}")
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"  Scale: {scale_name}")
    print(f"  Writes: {config.total_writes:,}")
    print(f"  Ingestion: {ingestion_time:.2f}s ({config.total_writes / ingestion_time:,.0f} ops/sec)")
    print(f"  Extraction: {total_extraction_time:.2f}ms for {total_values:,} values")
    print(f"  Storage: {db_size/1024/1024:.2f} MB ({bytes_per_write:.1f} bytes/write)")
    print()
    print(f"Database saved to: {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SynaDB Robotics Sensor Fusion Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robotics_sensor_fusion.py              # Default (~420 writes)
  python robotics_sensor_fusion.py --50k        # 50,000 writes
  python robotics_sensor_fusion.py --100k       # 100,000 writes
  python robotics_sensor_fusion.py --1m         # 1,000,000 writes
  python robotics_sensor_fusion.py --1m --batch # 1M writes with batch API (~20x faster)
        """
    )
    
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument('--50k', dest='scale', action='store_const', 
                             const='50k', help='50,000 writes')
    scale_group.add_argument('--100k', dest='scale', action='store_const',
                             const='100k', help='100,000 writes')
    scale_group.add_argument('--1m', '--1M', dest='scale', action='store_const',
                             const='1m', help='1,000,000 writes')
    
    parser.add_argument('--batch', action='store_true',
                        help='Use batch API (put_floats_batch) for ~20x faster ingestion')
    
    args = parser.parse_args()
    
    if args.scale == '50k':
        config = SensorConfig.scale_50k()
        scale_name = "50K writes"
    elif args.scale == '100k':
        config = SensorConfig.scale_100k()
        scale_name = "100K writes"
    elif args.scale == '1m':
        config = SensorConfig.scale_1m()
        scale_name = "1M writes"
    else:
        config = SensorConfig.default()
        scale_name = "Default (~420 writes)"
    
    run_demo(config, scale_name, use_batch=args.batch)


if __name__ == "__main__":
    main()

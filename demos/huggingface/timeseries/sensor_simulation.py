#!/usr/bin/env python3
"""
IoT Sensor Simulation Demo

This demo shows how to:
- Simulate real-time IoT sensor data streams
- Store data in Syna with high throughput
- Extract data for analysis and ML

Run with: python sensor_simulation.py
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
        self.time_offset = random.random() * 1000  # Random phase
        
        # Sensor-specific parameters
        if sensor_type == "temperature":
            self.base = 22.0  # Base temperature in Celsius
            self.amplitude = 3.0  # Daily variation
            self.noise = 0.2  # Random noise
        elif sensor_type == "humidity":
            self.base = 50.0  # Base humidity %
            self.amplitude = 15.0
            self.noise = 2.0
        elif sensor_type == "pressure":
            self.base = 1013.25  # Base pressure in hPa
            self.amplitude = 5.0
            self.noise = 0.5
        elif sensor_type == "light":
            self.base = 500.0  # Base light in lux
            self.amplitude = 400.0
            self.noise = 50.0
        else:
            self.base = 0.0
            self.amplitude = 1.0
            self.noise = 0.1
    
    def read(self, timestamp: float) -> float:
        """Generate a sensor reading for the given timestamp."""
        # Simulate daily cycle
        daily_cycle = math.sin((timestamp + self.time_offset) * 2 * math.pi / 86400)
        
        # Add some slower drift
        drift = math.sin(timestamp * 2 * math.pi / 604800) * 0.5  # Weekly
        
        # Calculate value
        value = self.base + self.amplitude * daily_cycle + drift * self.amplitude
        
        # Add noise
        value += random.gauss(0, self.noise)
        
        return value


def main():
    print("=== IoT Sensor Simulation Demo ===\n")
    
    import tempfile
    
    # Use temporary directory for database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "sensor_simulation.db")
    
    try:
        # 1. Create sensors (10 IoT sensors with realistic patterns)
        print("1. Creating simulated sensors...")
        sensors = [
            SensorSimulator("sensor-001", "temperature"),
            SensorSimulator("sensor-002", "temperature"),
            SensorSimulator("sensor-003", "humidity"),
            SensorSimulator("sensor-004", "humidity"),
            SensorSimulator("sensor-005", "pressure"),
            SensorSimulator("sensor-006", "pressure"),
            SensorSimulator("sensor-007", "light"),
            SensorSimulator("sensor-008", "light"),
            SensorSimulator("sensor-009", "temperature"),
            SensorSimulator("sensor-010", "humidity"),
        ]
        print(f"   ✓ Created {len(sensors)} sensors\n")
        
        # 2. Simulate data collection
        print("2. Simulating data collection (1,000 readings per sensor)...")
        
        num_readings = 1000
        interval = 1.0  # 1 second between readings
        
        start_time = time.time()
        
        with SynaDB(db_path) as db:
            for i in range(num_readings):
                timestamp = i * interval
                
                for sensor in sensors:
                    key = f"sensors/{sensor.sensor_type}/{sensor.sensor_id}"
                    value = sensor.read(timestamp)
                    db.put_float(key, value)
                
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) * len(sensors) / elapsed
                    print(f"   {i + 1}/{num_readings} readings ({rate:.0f} writes/sec)")
        
        total_time = time.time() - start_time
        total_writes = num_readings * len(sensors)
        print(f"   ✓ Wrote {total_writes} readings in {total_time:.2f}s")
        print(f"   ✓ Throughput: {total_writes / total_time:.0f} writes/sec\n")
        
        # 3. Check storage
        file_size = os.path.getsize(db_path)
        print("3. Storage statistics:")
        print(f"   Database size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Bytes per reading: {file_size / total_writes:.1f}\n")
        
        # 4. Extract and analyze data
        print("4. Extracting data for analysis...")
        
        with SynaDB(db_path) as db:
            # Get temperature sensor data
            temp_data = db.get_history_tensor("sensors/temperature/sensor-001")
            print(f"   Temperature sensor-001: {len(temp_data)} readings")
            
            if len(temp_data) > 0:
                print(f"      Mean: {np.mean(temp_data):.2f}°C")
                print(f"      Std:  {np.std(temp_data):.2f}°C")
                print(f"      Min:  {np.min(temp_data):.2f}°C")
                print(f"      Max:  {np.max(temp_data):.2f}°C")
            
            # Get humidity data
            humidity_data = db.get_history_tensor("sensors/humidity/sensor-003")
            print(f"\n   Humidity sensor-003: {len(humidity_data)} readings")
            
            if len(humidity_data) > 0:
                print(f"      Mean: {np.mean(humidity_data):.1f}%")
                print(f"      Std:  {np.std(humidity_data):.1f}%")
        
        print()
        
        # 5. Benchmark tensor extraction
        print("5. Benchmarking tensor extraction...")
        
        with SynaDB(db_path) as db:
            start = time.time()
            for _ in range(100):
                _ = db.get_history_tensor("sensors/temperature/sensor-001")
            extract_time = time.time() - start
            
            print(f"   100 tensor extractions: {extract_time * 1000:.2f}ms")
            print(f"   Throughput: {100 / extract_time:.0f} extractions/sec")
            print(f"   Data rate: {100 * num_readings / extract_time / 1e6:.2f} M values/sec\n")
        
        # 6. Demonstrate real-time pattern
        print("6. Demonstrating real-time ingestion pattern...")
        
        with SynaDB(db_path) as db:
            # Simulate 1 second of real-time data at 100Hz
            start = time.time()
            for i in range(100):
                timestamp = time.time()
                for sensor in sensors[:2]:  # Just 2 sensors for speed
                    key = f"realtime/{sensor.sensor_type}/{sensor.sensor_id}"
                    value = sensor.read(timestamp)
                    db.put_float(key, value)
            
            realtime_duration = time.time() - start
            print(f"   200 real-time writes in {realtime_duration * 1000:.2f}ms")
            print(f"   Latency per write: {realtime_duration / 200 * 1000:.3f}ms\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    
    print("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
Anomaly Detection Demo

This demo shows how to:
- Train an autoencoder on normal sensor data
- Stream new data and detect anomalies
- Show real-time alerting pattern
- Display latency metrics for real-time suitability

Run with: python anomaly_detection.py

Requirements: torch, numpy
"""

import os
import sys
import time
import math
import random
import numpy as np
from collections import deque

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Install with: pip install torch")


class SensorSimulator:
    """Simulates realistic IoT sensor data."""
    
    def __init__(self, sensor_id: str, anomaly_prob: float = 0.0):
        self.sensor_id = sensor_id
        self.anomaly_prob = anomaly_prob
        self.time_offset = random.random() * 1000
        self.base = 22.0
        self.amplitude = 3.0
        self.noise = 0.2
    
    def read(self, timestamp: float) -> tuple:
        """Returns (value, is_anomaly)."""
        daily_cycle = math.sin((timestamp + self.time_offset) * 2 * math.pi / 86400)
        drift = math.sin(timestamp * 2 * math.pi / 604800) * 0.5
        value = self.base + self.amplitude * daily_cycle + drift * self.amplitude
        value += random.gauss(0, self.noise)
        
        is_anomaly = False
        if random.random() < self.anomaly_prob:
            # Inject anomaly: spike or drop
            if random.random() < 0.5:
                value += random.uniform(5, 10)  # Spike
            else:
                value -= random.uniform(5, 10)  # Drop
            is_anomaly = True
        
        return value, is_anomaly


class TimeSeriesDataset(Dataset):
    """Dataset for autoencoder training."""
    
    def __init__(self, data: np.ndarray, window_size: int = 20):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return window, window  # Input = target for autoencoder


class Autoencoder(nn.Module):
    """Simple autoencoder for anomaly detection."""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 8):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly scoring."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=-1)
        return error


class AnomalyDetector:
    """Real-time anomaly detector using autoencoder."""
    
    def __init__(self, model: Autoencoder, window_size: int, threshold: float, 
                 mean: float, std: float, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.window_size = window_size
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.device = device
        self.buffer = deque(maxlen=window_size)
    
    def add_reading(self, value: float) -> dict:
        """Add a reading and check for anomaly."""
        # Normalize
        normalized = (value - self.mean) / (self.std + 1e-8)
        self.buffer.append(normalized)
        
        if len(self.buffer) < self.window_size:
            return {'ready': False, 'anomaly': False, 'score': 0.0}
        
        # Create tensor from buffer
        window = torch.FloatTensor(list(self.buffer)).unsqueeze(0).to(self.device)
        
        # Calculate reconstruction error
        error = self.model.reconstruction_error(window).item()
        
        is_anomaly = error > self.threshold
        
        return {
            'ready': True,
            'anomaly': is_anomaly,
            'score': error,
            'threshold': self.threshold
        }


def normalize_data(data: np.ndarray):
    """Normalize data to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8), mean, std


def train_autoencoder(model, train_loader, epochs=30, lr=0.001, device='cpu'):
    """Train the autoencoder on normal data."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.6f}")
    
    return losses


def calculate_threshold(model, data_loader, percentile=99, device='cpu'):
    """Calculate anomaly threshold from normal data."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            error = model.reconstruction_error(x)
            errors.extend(error.cpu().numpy())
    
    threshold = np.percentile(errors, percentile)
    return threshold, np.array(errors)


def main():
    print("=== Anomaly Detection Demo ===\n")
    
    if not HAS_TORCH:
        print("Error: PyTorch is required for this demo.")
        print("Install with: pip install torch")
        return 1
    
    import tempfile
    import shutil
    
    # Use temporary directory for database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "anomaly_detection.db")
    
    try:
        # 1. Generate normal training data
        print("1. Generating normal training data...")
        sensor_normal = SensorSimulator("sensor-001", anomaly_prob=0.0)
        num_train = 3000
        
        with SynaDB(db_path) as db:
            for i in range(num_train):
                timestamp = i * 60
                value, _ = sensor_normal.read(timestamp)
                db.put_float("train/temperature", value)
        
        print(f"   ✓ Generated {num_train} normal readings\n")
        
        # 2. Load and prepare training data
        print("2. Loading training data from Syna...")
        with SynaDB(db_path) as db:
            train_data = db.get_history_tensor("train/temperature")
        
        train_norm, mean, std = normalize_data(train_data)
        print(f"   ✓ Loaded {len(train_data)} values")
        print(f"   Mean: {mean:.2f}°C, Std: {std:.2f}°C\n")
        
        # 3. Create and train autoencoder
        print("3. Training autoencoder on normal data...")
        window_size = 20
        batch_size = 32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        train_dataset = TimeSeriesDataset(train_norm, window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = Autoencoder(input_size=window_size, hidden_size=8)
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        train_autoencoder(model, train_loader, epochs=30, lr=0.001, device=device)
        train_time = time.time() - start_time
        print(f"   ✓ Training completed in {train_time:.2f}s\n")
        
        # 4. Calculate anomaly threshold
        print("4. Calculating anomaly threshold...")
        threshold, train_errors = calculate_threshold(model, train_loader, percentile=99, device=device)
        print(f"   Normal error range: [{train_errors.min():.6f}, {train_errors.max():.6f}]")
        print(f"   Threshold (99th percentile): {threshold:.6f}\n")
        
        # 5. Create anomaly detector
        print("5. Creating real-time anomaly detector...")
        detector = AnomalyDetector(model, window_size, threshold, mean, std, device)
        print(f"   ✓ Detector ready\n")
        
        # 6. Stream test data with anomalies
        print("6. Streaming test data with injected anomalies...")
        sensor_test = SensorSimulator("sensor-002", anomaly_prob=0.05)  # 5% anomaly rate
        
        num_test = 500
        true_anomalies = 0
        detected_anomalies = 0
        true_positives = 0
        false_positives = 0
        
        alerts = []
        latencies = []
        
        with SynaDB(db_path) as db:
            for i in range(num_test):
                timestamp = (num_train + i) * 60
                value, is_true_anomaly = sensor_test.read(timestamp)
                
                # Store in Syna
                db.put_float("test/temperature", value)
                
                # Detect anomaly
                start = time.time()
                result = detector.add_reading(value)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                if is_true_anomaly:
                    true_anomalies += 1
                
                if result['ready'] and result['anomaly']:
                    detected_anomalies += 1
                    if is_true_anomaly:
                        true_positives += 1
                    else:
                        false_positives += 1
                    
                    alerts.append({
                        'index': i,
                        'value': value,
                        'score': result['score'],
                        'true_anomaly': is_true_anomaly
                    })
        
        print(f"   Processed {num_test} readings")
        print(f"   True anomalies injected: {true_anomalies}")
        print(f"   Anomalies detected: {detected_anomalies}")
        print(f"   True positives: {true_positives}")
        print(f"   False positives: {false_positives}\n")
        
        # 7. Show detection metrics
        print("7. Detection metrics:")
        if true_anomalies > 0:
            recall = true_positives / true_anomalies
            print(f"   Recall (sensitivity): {recall:.2%}")
        if detected_anomalies > 0:
            precision = true_positives / detected_anomalies
            print(f"   Precision: {precision:.2%}")
        print()
        
        # 8. Show latency metrics
        print("8. Real-time latency metrics:")
        latencies = np.array(latencies)
        print(f"   Mean latency: {np.mean(latencies):.3f}ms")
        print(f"   P50 latency:  {np.percentile(latencies, 50):.3f}ms")
        print(f"   P95 latency:  {np.percentile(latencies, 95):.3f}ms")
        print(f"   P99 latency:  {np.percentile(latencies, 99):.3f}ms")
        print(f"   Max latency:  {np.max(latencies):.3f}ms\n")
        
        # 9. Show sample alerts
        print("9. Sample alerts (first 5):")
        for alert in alerts[:5]:
            status = "TRUE ANOMALY" if alert['true_anomaly'] else "FALSE POSITIVE"
            print(f"   Index {alert['index']}: value={alert['value']:.2f}°C, "
                  f"score={alert['score']:.4f}, {status}")
        
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


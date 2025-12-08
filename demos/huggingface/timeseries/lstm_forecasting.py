#!/usr/bin/env python3
"""
LSTM Forecasting Demo

This demo shows how to:
- Load sensor data from Syna
- Train an LSTM for next-value prediction
- Evaluate on held-out data
- Show latency metrics for real-time suitability

Run with: python lstm_forecasting.py

Requirements: torch, numpy
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
    """Simulates realistic IoT sensor data for training."""
    
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


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series sequences from Syna."""
    
    def __init__(self, data: np.ndarray, seq_length: int = 20):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x.unsqueeze(-1), y  # Add feature dimension


class LSTMForecaster(nn.Module):
    """Simple LSTM for next-value prediction."""
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction.squeeze(-1)


def normalize_data(data: np.ndarray):
    """Normalize data to zero mean and unit variance."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-8), mean, std


def denormalize_data(data: np.ndarray, mean: float, std: float):
    """Reverse normalization."""
    return data * std + mean


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the LSTM model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    
    return train_losses, val_losses


def evaluate_model(model, test_loader, mean, std, device='cpu'):
    """Evaluate model on test data."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Denormalize
    predictions = denormalize_data(predictions, mean, std)
    actuals = denormalize_data(actuals, mean, std)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae}


def main():
    print("=== LSTM Forecasting Demo ===\n")
    
    if not HAS_TORCH:
        print("Error: PyTorch is required for this demo.")
        print("Install with: pip install torch")
        return 1
    
    import tempfile
    import shutil
    
    # Use temporary directory for database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "lstm_forecasting.db")
    
    try:
        # 1. Generate and store training data
        print("1. Generating sensor data...")
        sensor = SensorSimulator("sensor-001", "temperature")
        num_readings = 5000
        
        with SynaDB(db_path) as db:
            for i in range(num_readings):
                timestamp = i * 60  # 1 minute intervals
                value = sensor.read(timestamp)
                db.put_float("sensor/temperature", value)
        
        print(f"   ✓ Generated {num_readings} readings\n")
        
        # 2. Load data from Syna
        print("2. Loading data from Syna...")
        with SynaDB(db_path) as db:
            data = db.get_history_tensor("sensor/temperature")
        
        print(f"   ✓ Loaded {len(data)} values")
        print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]\n")
        
        # 3. Prepare datasets
        print("3. Preparing train/val/test splits...")
        
        # Normalize data
        data_norm, mean, std = normalize_data(data)
        
        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(data_norm))
        val_size = int(0.15 * len(data_norm))
        
        train_data = data_norm[:train_size]
        val_data = data_norm[train_size:train_size + val_size]
        test_data = data_norm[train_size + val_size:]
        
        seq_length = 20
        batch_size = 32
        
        train_dataset = TimeSeriesDataset(train_data, seq_length)
        val_dataset = TimeSeriesDataset(val_data, seq_length)
        test_dataset = TimeSeriesDataset(test_data, seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Sequence length: {seq_length}\n")
        
        # 4. Create and train model
        print("4. Training LSTM model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        model = LSTMForecaster(input_size=1, hidden_size=32, num_layers=2)
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            epochs=50, lr=0.001, device=device
        )
        train_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {train_time:.2f}s\n")
        
        # 5. Evaluate on test data
        print("5. Evaluating on held-out test data...")
        predictions, actuals, metrics = evaluate_model(model, test_loader, mean, std, device)
        
        print(f"   MSE:  {metrics['mse']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.4f}°C")
        print(f"   MAE:  {metrics['mae']:.4f}°C\n")
        
        # 6. Measure inference latency
        print("6. Measuring inference latency...")
        model.eval()
        
        # Single sample inference
        sample_x = torch.FloatTensor(data_norm[:seq_length]).unsqueeze(0).unsqueeze(-1).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_x)
        
        # Measure
        latencies = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(sample_x)
            latencies.append((time.time() - start) * 1000)
        
        latencies = np.array(latencies)
        print(f"   Single inference latency:")
        print(f"      Mean: {np.mean(latencies):.3f}ms")
        print(f"      P50:  {np.percentile(latencies, 50):.3f}ms")
        print(f"      P95:  {np.percentile(latencies, 95):.3f}ms")
        print(f"      P99:  {np.percentile(latencies, 99):.3f}ms\n")
        
        # 7. Show sample predictions
        print("7. Sample predictions vs actuals:")
        for i in range(min(5, len(predictions))):
            print(f"   Predicted: {predictions[i]:.2f}°C, Actual: {actuals[i]:.2f}°C, Error: {abs(predictions[i] - actuals[i]):.3f}°C")
        
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


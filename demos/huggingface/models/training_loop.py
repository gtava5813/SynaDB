#!/usr/bin/env python3
"""
Training Loop Demo with Syna

This demo shows how to:
- Train a simple CNN on MNIST data stored in Syna
- Show complete epoch iteration
- Log training metrics

Requirements: 5.2 - WHEN running the training loop demo THEN the demo 
SHALL show a complete training iteration reading from Syna

Run with: python training_loop.py
"""

import os
import sys
import time
import numpy as np
from typing import Optional

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.
    
    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool
    - Flatten -> Linear(1600, 128) -> ReLU -> Linear(128, 10)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SynaDataset(Dataset):
    """PyTorch Dataset backed by Syna database."""
    
    def __init__(self, db_path: str, key_prefix: str = "train"):
        self.db_path = db_path
        self.key_prefix = key_prefix
        self._db = SynaDB(db_path)
        
        # Build index
        all_keys = self._db.keys()
        label_prefix = f"{key_prefix}/label/"
        self._indices = sorted([
            int(k[len(label_prefix):]) 
            for k in all_keys 
            if k.startswith(label_prefix)
        ])
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, idx):
        actual_idx = self._indices[idx]
        
        # Load image
        image_bytes = self._db.get_bytes(f"{self.key_prefix}/image/{actual_idx}")
        if image_bytes:
            image = np.frombuffer(image_bytes, dtype=np.uint8).astype(np.float32)
            image = image.reshape(1, 28, 28) / 255.0
        else:
            image = np.zeros((1, 28, 28), dtype=np.float32)
        
        # Load label
        label = self._db.get_int(f"{self.key_prefix}/label/{actual_idx}") or 0
        
        return torch.from_numpy(image), label
    
    def close(self):
        if self._db:
            self._db.close()
            self._db = None


class TrainingMetrics:
    """Track and log training metrics."""
    
    def __init__(self):
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.batch_losses = []
        self.training_time = 0
    
    def log_batch(self, loss: float):
        self.batch_losses.append(loss)
    
    def log_epoch(self, loss: float, accuracy: float):
        self.epoch_losses.append(loss)
        self.epoch_accuracies.append(accuracy)
    
    def print_summary(self):
        print("\n   Training Summary:")
        print(f"   - Total epochs: {len(self.epoch_losses)}")
        print(f"   - Final loss: {self.epoch_losses[-1]:.4f}")
        print(f"   - Final accuracy: {self.epoch_accuracies[-1]:.2%}")
        print(f"   - Training time: {self.training_time:.2f}s")
        if len(self.epoch_losses) > 1:
            improvement = self.epoch_losses[0] - self.epoch_losses[-1]
            print(f"   - Loss improvement: {improvement:.4f}")


def create_mnist_database(db_path: str, num_train: int = 1000, num_test: int = 200):
    """Create a synthetic MNIST-like database for training."""
    print(f"   Creating database with {num_train} train, {num_test} test samples...")
    
    with SynaDB(db_path) as db:
        # Create training data
        for i in range(num_train):
            # Create synthetic digit-like patterns
            label = i % 10
            image = create_synthetic_digit(label)
            db.put_bytes(f"train/image/{i}", image.tobytes())
            db.put_int(f"train/label/{i}", label)
        
        # Create test data
        for i in range(num_test):
            label = i % 10
            image = create_synthetic_digit(label)
            db.put_bytes(f"test/image/{i}", image.tobytes())
            db.put_int(f"test/label/{i}", label)
    
    print(f"   ✓ Database created")


def create_synthetic_digit(label: int) -> np.ndarray:
    """
    Create a synthetic digit-like pattern.
    
    Each digit has a distinct pattern that the CNN can learn.
    """
    image = np.zeros((28, 28), dtype=np.uint8)
    
    # Create different patterns for each digit
    if label == 0:
        # Circle
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - 14)**2 + (j - 14)**2)
                if 8 < dist < 12:
                    image[i, j] = 255
    elif label == 1:
        # Vertical line
        image[4:24, 12:16] = 255
    elif label == 2:
        # Horizontal lines
        image[6:10, 6:22] = 255
        image[12:16, 6:22] = 255
        image[20:24, 6:22] = 255
    elif label == 3:
        # Three horizontal bars
        image[4:8, 8:20] = 255
        image[12:16, 8:20] = 255
        image[20:24, 8:20] = 255
    elif label == 4:
        # L shape
        image[4:20, 6:10] = 255
        image[12:16, 6:22] = 255
    elif label == 5:
        # S shape
        image[4:8, 6:22] = 255
        image[12:16, 6:22] = 255
        image[20:24, 6:22] = 255
        image[4:16, 6:10] = 255
        image[12:24, 18:22] = 255
    elif label == 6:
        # 6-like shape
        image[4:24, 6:10] = 255
        image[4:8, 6:22] = 255
        image[12:16, 6:22] = 255
        image[20:24, 6:22] = 255
    elif label == 7:
        # 7 shape
        image[4:8, 6:22] = 255
        image[4:24, 18:22] = 255
    elif label == 8:
        # 8-like shape (two circles)
        image[4:8, 8:20] = 255
        image[12:16, 8:20] = 255
        image[20:24, 8:20] = 255
        image[4:16, 8:12] = 255
        image[4:16, 16:20] = 255
        image[12:24, 8:12] = 255
        image[12:24, 16:20] = 255
    else:  # 9
        # 9-like shape
        image[4:16, 6:10] = 255
        image[4:16, 18:22] = 255
        image[4:8, 6:22] = 255
        image[12:16, 6:22] = 255
        image[12:24, 18:22] = 255
    
    # Add some noise
    noise = np.random.randint(0, 30, (28, 28), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise - 15, 0, 255).astype(np.uint8)
    
    return image.flatten()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics: TrainingMetrics
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        metrics.log_batch(loss.item())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("Training Loop Demo with Syna")
    print("Requirement 5.2: Complete training iteration from Syna")
    print("=" * 60 + "\n")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        return 1
    
    # Configuration
    DB_PATH = os.path.abspath("training_demo.db")
    NUM_TRAIN = 500
    NUM_TEST = 100
    BATCH_SIZE = 32
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    try:
        # 1. Create database
        print("1. Creating MNIST-like database")
        print("-" * 40)
        create_mnist_database(DB_PATH, NUM_TRAIN, NUM_TEST)
        print()
        
        # 2. Create datasets and dataloaders
        print("2. Creating datasets and dataloaders")
        print("-" * 40)
        train_dataset = SynaDataset(DB_PATH, key_prefix="train")
        test_dataset = SynaDataset(DB_PATH, key_prefix="test")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Train batches: {len(train_loader)}")
        print()
        
        # 3. Create model, optimizer, criterion
        print("3. Creating model and optimizer")
        print("-" * 40)
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model: SimpleCNN")
        print(f"   Parameters: {num_params:,}")
        print(f"   Optimizer: Adam (lr={LEARNING_RATE})")
        print(f"   Loss: CrossEntropyLoss")
        print()
        
        # 4. Training loop
        print("4. Training")
        print("-" * 40)
        metrics = TrainingMetrics()
        
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, metrics
            )
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            epoch_time = time.time() - epoch_start
            metrics.log_epoch(train_loss, train_acc)
            
            print(f"   Epoch {epoch + 1}/{NUM_EPOCHS}:")
            print(f"      Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"      Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2%}")
            print(f"      Time: {epoch_time:.2f}s")
        
        metrics.training_time = time.time() - start_time
        print()
        
        # 5. Final evaluation
        print("5. Final Evaluation")
        print("-" * 40)
        final_loss, final_acc = evaluate(model, test_loader, criterion, device)
        print(f"   Final Test Loss: {final_loss:.4f}")
        print(f"   Final Test Accuracy: {final_acc:.2%}")
        
        metrics.print_summary()
        print()
        
        # 6. Demonstrate inference
        print("6. Sample Inference")
        print("-" * 40)
        model.eval()
        with torch.no_grad():
            # Get a batch
            images, labels = next(iter(test_loader))
            images = images.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            print("   First 10 predictions:")
            for i in range(min(10, len(labels))):
                correct = "✓" if predictions[i].item() == labels[i].item() else "✗"
                print(f"      Sample {i}: Predicted={predictions[i].item()}, "
                      f"Actual={labels[i].item()} {correct}")
        print()
        
        # Cleanup
        train_dataset.close()
        test_dataset.close()
        
        print("=" * 60)
        print("Training Demo Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


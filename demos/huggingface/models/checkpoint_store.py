#!/usr/bin/env python3
"""
Checkpoint Store Demo with Syna

This demo shows how to:
- Save model state_dict to Syna
- Implement versioned checkpoints
- Show checkpoint loading and resumption

Requirements: 5.5 - WHEN running the checkpoint demo THEN the demo 
SHALL show storing model checkpoints with versioning

Run with: python checkpoint_store.py
"""

import os
import sys
import time
import io
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SynaCheckpointStore:
    """
    Checkpoint store for ML models backed by Syna.
    
    Features:
    - Versioned checkpoints with automatic numbering
    - Metadata storage (epoch, loss, metrics)
    - Optimizer state preservation
    - Easy checkpoint listing and loading
    - Checkpoint comparison and rollback
    
    Example:
        >>> store = SynaCheckpointStore("checkpoints.db", "my_model")
        >>> store.save(model, optimizer, epoch=5, metrics={"loss": 0.1})
        >>> model, optimizer, metadata = store.load_latest()
    """
    
    def __init__(self, db_path: str, model_name: str):
        """
        Initialize the checkpoint store.
        
        Args:
            db_path: Path to the Syna database
            model_name: Name of the model (used as key prefix)
        """
        self.db_path = db_path
        self.model_name = model_name
        self._db = SynaDB(db_path)
        
        # Initialize version counter if not exists
        current_version = self._db.get_int(f"{model_name}/_meta/latest_version")
        if current_version is None:
            self._db.put_int(f"{model_name}/_meta/latest_version", 0)
    
    def _get_next_version(self) -> int:
        """Get the next version number."""
        current = self._db.get_int(f"{self.model_name}/_meta/latest_version") or 0
        return current + 1
    
    def _serialize_state_dict(self, state_dict: Dict[str, Any]) -> bytes:
        """Serialize a state dict to bytes."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def _deserialize_state_dict(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to a state dict."""
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=False)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> int:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer to save
            epoch: Current epoch number
            metrics: Optional metrics dictionary
            notes: Optional notes about this checkpoint
            
        Returns:
            Version number of the saved checkpoint
        """
        version = self._get_next_version()
        prefix = f"{self.model_name}/v{version}"
        
        # Save model state
        model_bytes = self._serialize_state_dict(model.state_dict())
        self._db.put_bytes(f"{prefix}/model", model_bytes)
        
        # Save optimizer state if provided
        if optimizer is not None:
            opt_bytes = self._serialize_state_dict(optimizer.state_dict())
            self._db.put_bytes(f"{prefix}/optimizer", opt_bytes)
        
        # Save metadata
        metadata = {
            "version": version,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "notes": notes,
            "has_optimizer": optimizer is not None
        }
        self._db.put_text(f"{prefix}/metadata", json.dumps(metadata))
        
        # Update latest version
        self._db.put_int(f"{self.model_name}/_meta/latest_version", version)
        
        return version
    
    def load(
        self,
        version: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint version.
        
        Args:
            version: Version number to load
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Metadata dictionary
        """
        prefix = f"{self.model_name}/v{version}"
        
        # Load model state
        model_bytes = self._db.get_bytes(f"{prefix}/model")
        if model_bytes is None:
            raise ValueError(f"Checkpoint version {version} not found")
        
        model_state = self._deserialize_state_dict(model_bytes)
        model.load_state_dict(model_state)
        
        # Load optimizer state if requested and available
        if optimizer is not None:
            opt_bytes = self._db.get_bytes(f"{prefix}/optimizer")
            if opt_bytes is not None:
                opt_state = self._deserialize_state_dict(opt_bytes)
                optimizer.load_state_dict(opt_state)
        
        # Load metadata
        metadata_str = self._db.get_text(f"{prefix}/metadata")
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        return metadata
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Metadata dictionary
        """
        latest_version = self._db.get_int(f"{self.model_name}/_meta/latest_version")
        if latest_version is None or latest_version == 0:
            raise ValueError("No checkpoints found")
        
        return self.load(latest_version, model, optimizer)
    
    def load_best(
        self,
        model: nn.Module,
        metric_name: str = "loss",
        minimize: bool = True,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load the checkpoint with the best metric value.
        
        Args:
            model: Model to load weights into
            metric_name: Name of the metric to optimize
            minimize: If True, find minimum; if False, find maximum
            optimizer: Optional optimizer to load state into
            
        Returns:
            Metadata dictionary
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ValueError("No checkpoints found")
        
        # Find best checkpoint
        best_version = None
        best_value = float('inf') if minimize else float('-inf')
        
        for cp in checkpoints:
            value = cp.get('metrics', {}).get(metric_name)
            if value is not None:
                if (minimize and value < best_value) or (not minimize and value > best_value):
                    best_value = value
                    best_version = cp['version']
        
        if best_version is None:
            raise ValueError(f"No checkpoints with metric '{metric_name}' found")
        
        return self.load(best_version, model, optimizer)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of metadata dictionaries
        """
        latest_version = self._db.get_int(f"{self.model_name}/_meta/latest_version") or 0
        checkpoints = []
        
        for version in range(1, latest_version + 1):
            metadata_str = self._db.get_text(f"{self.model_name}/v{version}/metadata")
            if metadata_str:
                checkpoints.append(json.loads(metadata_str))
        
        return checkpoints
    
    def delete(self, version: int):
        """Delete a specific checkpoint version."""
        prefix = f"{self.model_name}/v{version}"
        
        self._db.delete(f"{prefix}/model")
        self._db.delete(f"{prefix}/optimizer")
        self._db.delete(f"{prefix}/metadata")
    
    def get_latest_version(self) -> int:
        """Get the latest version number."""
        return self._db.get_int(f"{self.model_name}/_meta/latest_version") or 0
    
    def close(self):
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def simulate_training_epoch(model: nn.Module, epoch: int) -> Dict[str, float]:
    """Simulate a training epoch and return metrics."""
    # Simulate decreasing loss over epochs
    base_loss = 2.0 * (0.8 ** epoch)
    noise = np.random.uniform(-0.1, 0.1)
    loss = base_loss + noise
    
    # Simulate increasing accuracy
    accuracy = min(0.95, 0.5 + 0.05 * epoch + np.random.uniform(-0.02, 0.02))
    
    # Slightly modify model weights to simulate training
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    return {"loss": loss, "accuracy": accuracy}


def main():
    print("=" * 60)
    print("Checkpoint Store Demo with Syna")
    print("Requirement 5.5: Store model checkpoints with versioning")
    print("=" * 60 + "\n")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install torch")
        return 1
    
    # Configuration
    DB_PATH = os.path.abspath("checkpoint_demo.db")
    NUM_EPOCHS = 5
    
    try:
        # 1. Initialize model and checkpoint store
        print("1. Initializing model and checkpoint store")
        print("-" * 40)
        
        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model: SimpleCNN ({num_params:,} parameters)")
        print(f"   Optimizer: Adam")
        
        store = SynaCheckpointStore(DB_PATH, "cnn_experiment")
        print(f"   Checkpoint store: {DB_PATH}")
        print()
        
        # 2. Simulate training with checkpoints
        print("2. Simulating training with checkpoints")
        print("-" * 40)
        
        for epoch in range(NUM_EPOCHS):
            # Simulate training
            metrics = simulate_training_epoch(model, epoch)
            
            # Save checkpoint
            start = time.time()
            version = store.save(
                model,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                notes=f"Training epoch {epoch}"
            )
            save_time = time.time() - start
            
            print(f"   Epoch {epoch}: loss={metrics['loss']:.4f}, "
                  f"acc={metrics['accuracy']:.2%} → v{version} ({save_time * 1000:.1f}ms)")
        print()
        
        # 3. List all checkpoints
        print("3. Listing all checkpoints")
        print("-" * 40)
        
        checkpoints = store.list_checkpoints()
        print(f"   Total checkpoints: {len(checkpoints)}")
        print()
        print("   Version | Epoch | Loss   | Accuracy | Timestamp")
        print("   " + "-" * 55)
        
        for cp in checkpoints:
            print(f"   v{cp['version']:6} | {cp['epoch']:5} | "
                  f"{cp['metrics'].get('loss', 0):.4f} | "
                  f"{cp['metrics'].get('accuracy', 0):.2%}   | "
                  f"{cp['timestamp'][:19]}")
        print()
        
        # 4. Load latest checkpoint
        print("4. Loading latest checkpoint")
        print("-" * 40)
        
        # Create fresh model
        fresh_model = SimpleCNN()
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.001)
        
        start = time.time()
        metadata = store.load_latest(fresh_model, fresh_optimizer)
        load_time = time.time() - start
        
        print(f"   Loaded version: v{metadata['version']}")
        print(f"   Epoch: {metadata['epoch']}")
        print(f"   Metrics: loss={metadata['metrics']['loss']:.4f}, "
              f"acc={metadata['metrics']['accuracy']:.2%}")
        print(f"   Load time: {load_time * 1000:.1f}ms")
        print()
        
        # 5. Load best checkpoint
        print("5. Loading best checkpoint (lowest loss)")
        print("-" * 40)
        
        best_model = SimpleCNN()
        metadata = store.load_best(best_model, metric_name="loss", minimize=True)
        
        print(f"   Best version: v{metadata['version']}")
        print(f"   Epoch: {metadata['epoch']}")
        print(f"   Loss: {metadata['metrics']['loss']:.4f}")
        print()
        
        # 6. Load specific version
        print("6. Loading specific version (v2)")
        print("-" * 40)
        
        v2_model = SimpleCNN()
        metadata = store.load(2, v2_model)
        
        print(f"   Loaded version: v{metadata['version']}")
        print(f"   Epoch: {metadata['epoch']}")
        print(f"   Notes: {metadata['notes']}")
        print()
        
        # 7. Demonstrate training resumption
        print("7. Demonstrating training resumption")
        print("-" * 40)
        
        # Load from checkpoint
        resume_model = SimpleCNN()
        resume_optimizer = optim.Adam(resume_model.parameters(), lr=0.001)
        
        metadata = store.load(3, resume_model, resume_optimizer)
        start_epoch = metadata['epoch'] + 1
        
        print(f"   Resuming from epoch {start_epoch}")
        
        # Continue training
        for epoch in range(start_epoch, start_epoch + 2):
            metrics = simulate_training_epoch(resume_model, epoch)
            version = store.save(
                resume_model,
                resume_optimizer,
                epoch=epoch,
                metrics=metrics,
                notes=f"Resumed training epoch {epoch}"
            )
            print(f"   Epoch {epoch}: loss={metrics['loss']:.4f} → v{version}")
        print()
        
        # 8. Storage statistics
        print("8. Storage statistics")
        print("-" * 40)
        
        db_size = os.path.getsize(DB_PATH)
        num_checkpoints = store.get_latest_version()
        
        print(f"   Database size: {db_size / 1024:.1f} KB")
        print(f"   Total checkpoints: {num_checkpoints}")
        print(f"   Avg size per checkpoint: {db_size / num_checkpoints / 1024:.1f} KB")
        
        # Model size estimate
        model_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
        print(f"   Model parameters size: {model_size / 1024:.1f} KB")
        print()
        
        # 9. Checkpoint comparison
        print("9. Checkpoint comparison")
        print("-" * 40)
        
        checkpoints = store.list_checkpoints()
        if len(checkpoints) >= 2:
            first = checkpoints[0]
            last = checkpoints[-1]
            
            loss_improvement = first['metrics']['loss'] - last['metrics']['loss']
            acc_improvement = last['metrics']['accuracy'] - first['metrics']['accuracy']
            
            print(f"   First checkpoint (v{first['version']}):")
            print(f"      Loss: {first['metrics']['loss']:.4f}")
            print(f"      Accuracy: {first['metrics']['accuracy']:.2%}")
            print(f"   Last checkpoint (v{last['version']}):")
            print(f"      Loss: {last['metrics']['loss']:.4f}")
            print(f"      Accuracy: {last['metrics']['accuracy']:.2%}")
            print(f"   Improvement:")
            print(f"      Loss: -{loss_improvement:.4f}")
            print(f"      Accuracy: +{acc_improvement:.2%}")
        print()
        
        # Cleanup
        store.close()
        
        print("=" * 60)
        print("Checkpoint Store Demo Complete!")
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


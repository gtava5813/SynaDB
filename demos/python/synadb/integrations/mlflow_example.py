"""
Example usage of Syna MLflow integration.

This example demonstrates how to use Syna as a backend for MLflow tracking.

Requirements:
    pip install mlflow synadb

Usage:
    python mlflow_example.py
"""

import mlflow
from synadb.integrations.mlflow import register_syna_tracking_store

# Register Syna as a tracking store
register_syna_tracking_store()

# Set Syna as the tracking URI
mlflow.set_tracking_uri("synadb:///experiments.db")

# Create an experiment
experiment_name = "mnist_classifier"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except Exception:
    # Experiment might already exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

# Start a run
with mlflow.start_run(experiment_id=experiment_id):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("optimizer", "adam")
    
    # Simulate training loop
    for epoch in range(10):
        # Log metrics
        loss = 2.3 - (epoch * 0.1)  # Simulated decreasing loss
        accuracy = 0.5 + (epoch * 0.04)  # Simulated increasing accuracy
        
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    # Log final metrics
    mlflow.log_metric("final_accuracy", 0.95)
    
    print("\nRun completed successfully!")
    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# Query runs
print("\n--- Listing all runs ---")
runs = mlflow.search_runs(experiment_ids=[experiment_id])
print(runs[["run_id", "params.learning_rate", "metrics.final_accuracy", "status"]])

print("\nMLflow tracking with Syna backend completed!")

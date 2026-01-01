#!/usr/bin/env python3
"""
Syna Studio Launcher

Launch Syna Studio web UI for database exploration and visualization.

Usage:
    # Launch with existing database
    python run_ui.py path/to/database.db
    
    # Launch with test data (creates ui_test.db)
    python run_ui.py --test
    
    # Launch with HuggingFace embeddings
    python run_ui.py --test --use-hf --samples 200
    
    # Custom port
    python run_ui.py --test --port 8080

Requirements:
    - Flask (pip install flask)
    - numpy
    - Optional for --use-hf: datasets, sentence-transformers
"""

import sys
import os
import time
import json
import random
import argparse

# Add the PARENT directory to sys.path for package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from synadb import studio
    from synadb.wrapper import SynaDB
except ImportError as e:
    print(f"Import Error: {e}")
    sys.path.append(current_dir)
    import wrapper as SynaDB_module
    import studio
    SynaDB = SynaDB_module.SynaDB


# =============================================================================
# Test Data Seeding Functions
# =============================================================================

def seed_basic_data(db: SynaDB) -> None:
    """Seed basic key-value data for Keys Explorer."""
    print("  → Seeding basic data types...")
    
    # System metrics (floats)
    db.put_float("system/cpu_usage", 45.2)
    db.put_float("system/memory_usage_gb", 8.4)
    db.put_float("system/disk_usage_percent", 67.3)
    db.put_float("system/network_latency_ms", 12.5)
    
    # Counters (integers)
    db.put_int("system/uptime_seconds", 86400)
    db.put_int("stats/total_requests", 1_234_567)
    db.put_int("stats/active_users", 42)
    db.put_int("stats/errors_today", 3)
    
    # Configuration (text)
    db.put_text("config/theme", "dark")
    db.put_text("config/language", "en-US")
    db.put_text("config/timezone", "America/New_York")
    db.put_text("config/api_version", "v2.1.0")
    
    # Binary data (bytes)
    db.put_bytes("cache/session_token", os.urandom(32))
    db.put_bytes("cache/thumbnail", b"\x89PNG\r\n\x1a\n" + os.urandom(100))


def seed_models(db: SynaDB) -> None:
    """Seed model registry data for AI Models tab."""
    print("  → Seeding model registry...")
    
    models_data = [
        ("text-classifier", 1, "archived", {"accuracy": "0.82", "f1": "0.79"}),
        ("text-classifier", 2, "staging", {"accuracy": "0.89", "f1": "0.87"}),
        ("text-classifier", 3, "production", {"accuracy": "0.94", "f1": "0.93"}),
        ("sentiment-bert", 1, "production", {"accuracy": "0.91", "auc": "0.95"}),
        ("recommender-v2", 1, "development", {"precision@10": "0.42", "ndcg": "0.38"}),
        ("recommender-v2", 2, "staging", {"precision@10": "0.51", "ndcg": "0.47"}),
        ("image-encoder", 1, "production", {"mAP": "0.78", "params": "86M"}),
        ("gpt-finetuned", 1, "archived", {"perplexity": "15.2", "bleu": "32.1"}),
        ("gpt-finetuned", 2, "production", {"perplexity": "11.8", "bleu": "38.4"}),
    ]
    
    for name, ver, stage, meta in models_data:
        full_meta = {
            "name": name,
            "version": ver,
            "stage": stage,
            "created_at": time.time() - random.randint(0, 2_000_000),
            "size_bytes": random.randint(10_000, 500_000_000),
            "metadata": meta
        }
        db.put_text(f"model/{name}/v{ver}/meta", json.dumps(full_meta))
        db.put_bytes(f"model/{name}/v{ver}/data", b"MODEL_WEIGHTS_PLACEHOLDER")


def seed_synthetic_vectors(db: SynaDB, count: int = 100) -> None:
    """Seed synthetic clustered vectors for 3D Clusters visualization."""
    print(f"  → Seeding {count} synthetic vectors (3 clusters)...")
    
    # Create 3 distinct clusters in 8D space
    clusters = [
        {"center": [5, 5, 5, 0, 0, 0, 0, 0], "label": "cluster_a"},
        {"center": [-5, 0, 0, 5, 5, 0, 0, 0], "label": "cluster_b"},
        {"center": [0, -5, 2, 0, 0, 5, 5, 0], "label": "cluster_c"},
    ]
    
    for i in range(count):
        cluster = clusters[i % len(clusters)]
        # Add Gaussian noise around cluster center
        vec = [c + random.gauss(0, 1.5) for c in cluster["center"]]
        # Store as JSON text (studio.py detects JSON arrays as vectors)
        db.put_text(f"vector/{cluster['label']}/emb_{i}", json.dumps(vec))


def seed_hf_embeddings(db: SynaDB, num_samples: int = 100) -> bool:
    """
    Seed real embeddings from HuggingFace datasets.
    
    Uses sentence-transformers to encode text from IMDB reviews.
    Returns True if successful, False if dependencies missing.
    """
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠ HuggingFace mode requires: pip install datasets sentence-transformers")
        return False
    
    print(f"  → Loading IMDB dataset ({num_samples} samples)...")
    dataset = load_dataset("imdb", split=f"train[:{num_samples}]")
    
    print("  → Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast
    
    print("  → Encoding text to embeddings...")
    texts = [sample['text'][:512] for sample in dataset]  # Truncate for speed
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("  → Storing embeddings in database...")
    for i, (sample, emb) in enumerate(zip(dataset, embeddings)):
        label = "positive" if sample['label'] == 1 else "negative"
        # Store embedding as JSON (studio.py detects as vector)
        db.put_text(f"vector/imdb/{label}/emb_{i}", json.dumps(emb.tolist()))
        # Also store the review text
        db.put_text(f"text/imdb/{label}/review_{i}", sample['text'][:500])
    
    return True


def seed_time_series(db: SynaDB, points: int = 50) -> None:
    """Seed time series data for statistics visualization."""
    print(f"  → Seeding {points} time series points...")
    
    # Simulate sensor readings over time
    for i in range(points):
        timestamp = int(time.time()) - (points - i) * 60  # 1 minute intervals
        
        # Temperature with daily pattern + noise
        temp = 20 + 5 * (0.5 + 0.5 * (i % 24) / 24) + random.gauss(0, 0.5)
        db.put_float(f"sensor/temperature/{timestamp}", temp)
        
        # Humidity inversely correlated with temperature
        humidity = 60 - 10 * (temp - 20) / 5 + random.gauss(0, 2)
        db.put_float(f"sensor/humidity/{timestamp}", max(30, min(90, humidity)))


def create_test_database(db_path: str, use_hf: bool, samples: int) -> None:
    """Create and populate a test database."""
    # Clean up existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    print("\nPopulating test database...")
    with SynaDB(db_path) as db:
        seed_basic_data(db)
        seed_models(db)
        seed_time_series(db)
        
        if use_hf:
            success = seed_hf_embeddings(db, samples)
            if not success:
                print("  → Falling back to synthetic vectors...")
                seed_synthetic_vectors(db, samples)
        else:
            seed_synthetic_vectors(db, samples)
        
        key_count = len(db.keys())
        print(f"\n✓ Database ready with {key_count} keys")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Syna Studio - Web UI for SynaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ui.py mydata.db              # Open existing database
  python run_ui.py --test                 # Create test database and launch
  python run_ui.py --test --use-hf        # Use HuggingFace embeddings
  python run_ui.py --test --samples 500   # More test samples
  python run_ui.py --test --port 8080     # Custom port
        """
    )
    
    parser.add_argument("database", nargs="?", default=None,
                        help="Path to database file (optional if --test)")
    parser.add_argument("--test", action="store_true",
                        help="Create a test database with sample data")
    parser.add_argument("--use-hf", action="store_true",
                        help="Use HuggingFace datasets for real embeddings (requires datasets, sentence-transformers)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of vector samples for test mode (default: 100)")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for Studio web UI (default: 8501)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode")
    
    args = parser.parse_args()
    
    # Determine database path
    if args.test:
        db_path = os.path.join(current_dir, "ui_test.db")
    elif args.database:
        db_path = args.database
    else:
        parser.error("Please provide a database path or use --test flag")
    
    print(f"\n{'='*50}")
    print("Syna Studio")
    print(f"{'='*50}\n")
    
    try:
        # Create test database if requested
        if args.test:
            print(f"Mode: Test ({'HuggingFace' if args.use_hf else 'Synthetic'})")
            print(f"Database: {db_path}")
            create_test_database(db_path, args.use_hf, args.samples)
        else:
            print(f"Mode: Production")
            print(f"Database: {db_path}")
            if not os.path.exists(db_path):
                print(f"\n✗ Error: Database not found: {db_path}")
                return 1
        
        # Launch Studio
        print(f"\nStarting Syna Studio on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser\n")
        studio.launch(db_path, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

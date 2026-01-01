"""Demo script to launch Syna Studio with sample data."""
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the synadb package to path
sys.path.insert(0, script_dir)

from synadb import SynaDB, launch_studio

# Create a test database with sample data (in script directory)
db_path = os.path.join(script_dir, "demo_studio.db")

# Clean up any existing demo database
if os.path.exists(db_path):
    os.remove(db_path)

print("Creating demo database with sample data...")

with SynaDB(db_path) as db:
    # Add some float values (sensor data)
    for i in range(10):
        db.put_float(f"sensor/temperature/{i}", 20.0 + i * 0.5)
        db.put_float(f"sensor/humidity/{i}", 45.0 + i * 2.0)
    
    # Add some integer values
    db.put_int("config/batch_size", 32)
    db.put_int("config/epochs", 100)
    db.put_int("config/hidden_dim", 768)
    
    # Add some text values
    db.put_text("model/name", "bert-base-uncased")
    db.put_text("model/version", "1.0.0")
    db.put_text("experiment/description", "Testing the new transformer architecture")
    
    # Add some user data
    db.put_text("user/name", "Alice")
    db.put_float("user/score", 95.5)
    
    print(f"Created {len(db.keys())} keys in {db_path}")

print("\nLaunching Syna Studio...")
print("Open your browser to http://localhost:8501")
print("Press Ctrl+C to stop the server\n")

# Launch the studio
launch_studio(db_path, port=8501, debug=False)

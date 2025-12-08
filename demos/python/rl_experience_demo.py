#!/usr/bin/env python3
"""
RL Experience Collection Demo

Demonstrates using Syna for multi-machine reinforcement learning
experience collection, sync, and merge.

Use Case: Model exploration with Q-learning across multiple machines,
preserving raw (state, action, reward, next_state) tuples for later
retraining with DQN, PPO, or other algorithms.
"""

import random
import tempfile
from pathlib import Path

# Add parent to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent))

from Syna import ExperienceCollector


def simulate_exploration_step():
    """Simulate a single exploration step (replace with your actual logic)."""
    # State: (layer, component, depth, coverage)
    state = (
        random.randint(0, 10),      # layer
        random.randint(0, 5),       # component
        random.randint(0, 20),      # depth
        random.random(),            # coverage
    )
    
    # Action: one of the exploration actions
    actions = ["analyze_weights", "analyze_activations", "analyze_gradients", 
               "compare_layers", "drill_down", "step_back"]
    action = random.choice(actions)
    
    # Reward: based on insight quality
    reward = random.gauss(0.5, 0.3)  # Mean 0.5, std 0.3
    
    # Next state
    next_state = (
        state[0] + random.choice([-1, 0, 1]),
        state[1],
        state[2] + 1,
        min(1.0, state[3] + random.random() * 0.1),
    )
    
    # Metadata
    metadata = {
        "model": "Qwen/Qwen3-4B-Thinking-2507",
        "model_family": "qwen",
        "architecture": "dense",
        "insight_count": random.randint(0, 5),
        "insight_quality_avg": random.random(),
    }
    
    return state, action, reward, next_state, metadata


def demo_single_machine():
    """Demo: Collecting experiences on a single machine."""
    print("=" * 60)
    print("Demo 1: Single Machine Experience Collection")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/experiences.db"
        
        # Create collector with machine ID
        collector = ExperienceCollector(db_path, machine_id="mac_mini_m4")
        
        # Use session context for an exploration episode
        with collector.session(
            model="Qwen/Qwen3-4B",
            episode=1,
            exploration_strategy="epsilon_greedy",
            epsilon=0.3,
        ) as session:
            
            # Simulate 100 exploration steps
            for step in range(100):
                state, action, reward, next_state, metadata = simulate_exploration_step()
                session.log(state, action, reward, next_state, step=step, **metadata)
            
            print(f"Session: {session.session_id}")
            print(f"Transitions logged: {session.transition_count}")
            print(f"Total reward: {session.total_reward:.2f}")
            print(f"Mean reward: {session.mean_reward:.3f}")
        
        # Get stats
        stats = collector.stats()
        print(f"\nDatabase stats:")
        print(f"  Machine ID: {stats['machine_id']}")
        print(f"  Total transitions: {stats['total_transitions']}")
        print(f"  Sessions: {stats['sessions']}")
        
        # Extract rewards as tensor for analysis
        rewards = collector.get_rewards_tensor(session.session_id)
        print(f"\nRewards tensor shape: {rewards.shape}")
        print(f"Rewards mean: {rewards.mean():.3f}, std: {rewards.std():.3f}")
        
        collector.close()


def demo_multi_machine_simulation():
    """Demo: Simulating multi-machine collection and merge."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Machine Collection & Merge")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate 3 machines collecting experiences
        machine_dbs = []
        
        for machine_num in range(3):
            machine_id = f"gpu_server_{machine_num}"
            db_path = f"{tmpdir}/machine_{machine_num}.db"
            machine_dbs.append(db_path)
            
            print(f"\nMachine {machine_num} ({machine_id}):")
            
            with ExperienceCollector(db_path, machine_id=machine_id) as collector:
                with collector.session(
                    model="Qwen/Qwen3-4B",
                    machine=machine_num,
                ) as session:
                    # Each machine collects 50 transitions
                    for step in range(50):
                        state, action, reward, next_state, metadata = simulate_exploration_step()
                        session.log(state, action, reward, next_state, **metadata)
                    
                    print(f"  Collected {session.transition_count} transitions")
                    print(f"  Mean reward: {session.mean_reward:.3f}")
        
        # Merge all databases
        print("\n" + "-" * 40)
        print("Merging databases...")
        
        master_path = f"{tmpdir}/master.db"
        merged_count = ExperienceCollector.merge(machine_dbs, master_path)
        
        print(f"Merged {merged_count} transitions into master.db")
        
        # Analyze merged data
        with ExperienceCollector(master_path) as master:
            stats = master.stats()
            print(f"\nMaster database stats:")
            print(f"  Total transitions: {stats['total_transitions']}")
            print(f"  Sessions: {len(stats['sessions'])}")
            
            # Could now train DQN from this merged data
            print("\nReady for DQN training from merged experiences!")


def demo_data_structure():
    """Demo: Show the data structure being stored."""
    print("\n" + "=" * 60)
    print("Demo 3: Data Structure")
    print("=" * 60)
    
    from Syna.experience import Transition
    import json
    
    # Create a sample transition
    transition = Transition(
        state=(0, 1, 2, 0.5),
        action="analyze_weights",
        reward=0.75,
        next_state=(0, 1, 3, 0.6),
        timestamp=1764791962000000,
        session_id="abc123",
        machine_id="mac_mini_m4",
        metadata={
            "model": "Qwen/Qwen3-4B-Thinking-2507",
            "model_family": "qwen",
            "architecture": "dense",
            "insight_count": 3,
            "insight_quality_avg": 0.8,
        }
    )
    
    # Show the JSON representation
    data = transition.to_dict()
    json_str = json.dumps(data, indent=2)
    
    print("Transition structure:")
    print(json_str)
    
    # Show size
    compact = json.dumps(data, separators=(',', ':'))
    print(f"\nCompact JSON size: {len(compact)} bytes")
    print(f"Content hash (for dedup): {transition.content_hash()}")
    
    # Show key format
    key = f"exp/{transition.session_id}/{transition.timestamp}_{transition.content_hash()}"
    print(f"Storage key: {key}")


def demo_sync_workflow():
    """Demo: Show the sync workflow for distributed collection."""
    print("\n" + "=" * 60)
    print("Demo 4: Sync Workflow")
    print("=" * 60)
    
    print("""
Recommended sync workflow for multi-machine RL:

1. COLLECT (on each machine):
   ```python
   collector = ExperienceCollector("local_exp.db", machine_id="machine_1")
   with collector.session(model="Qwen/Qwen3-4B") as s:
       for step in exploration_loop:
           s.log(state, action, reward, next_state, **metadata)
   ```

2. SYNC (periodic, e.g., hourly):
   ```bash
   # Option A: rsync to central server
   rsync -avz local_exp.db user@server:/data/machine_1/
   
   # Option B: Cloud storage
   aws s3 cp local_exp.db s3://bucket/experiences/machine_1/
   
   # Option C: Simple copy (for local network)
   cp local_exp.db /shared/drive/machine_1/
   ```

3. MERGE (on training server):
   ```python
   sources = glob.glob("/data/*/exp.db")
   ExperienceCollector.merge(sources, "master.db", deduplicate=True)
   ```

4. TRAIN (from merged data):
   ```python
   with ExperienceCollector("master.db") as exp:
       for session in exp.list_sessions():
           rewards = exp.get_rewards_tensor(session)
           # Load transitions into replay buffer
           # Train DQN/PPO/SAC
   ```

Key benefits:
- Append-only: No merge conflicts
- Hash-based keys: Automatic deduplication
- Timestamps: Preserve temporal ordering
- Machine IDs: Track data provenance
- ~200 bytes/transition: 100K transitions = 20MB
""")


if __name__ == "__main__":
    demo_single_machine()
    demo_multi_machine_simulation()
    demo_data_structure()
    demo_sync_workflow()


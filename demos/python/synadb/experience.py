"""
Experience Collector for Reinforcement Learning

A high-level wrapper around SynaDB optimized for collecting, storing,
and merging RL experience tuples across multiple machines.

Example:
    >>> from Syna import ExperienceCollector
    >>> collector = ExperienceCollector("experiences.db", machine_id="mac_mini_m4")
    >>> collector.log_transition(
    ...     state=(0, 1, 2, 0.5),
    ...     action="analyze_weights",
    ...     reward=0.75,
    ...     next_state=(0, 1, 3, 0.6),
    ...     metadata={"model": "Qwen/Qwen3-4B", "insight_count": 3}
    ... )
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .wrapper import SynaDB, SynaError


@dataclass
class Transition:
    """A single RL experience transition."""
    state: Tuple[Any, ...]
    action: Union[str, int]
    reward: float
    next_state: Tuple[Any, ...]
    timestamp: int  # Unix microseconds
    session_id: str
    machine_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": list(self.state),
            "action": self.action,
            "reward": self.reward,
            "next_state": list(self.next_state),
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "machine_id": self.machine_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transition":
        """Create from dictionary."""
        return cls(
            state=tuple(d["state"]),
            action=d["action"],
            reward=d["reward"],
            next_state=tuple(d["next_state"]),
            timestamp=d["timestamp"],
            session_id=d["session_id"],
            machine_id=d["machine_id"],
            metadata=d.get("metadata", {}),
        )
    
    def content_hash(self) -> str:
        """Generate a hash for deduplication (excludes timestamp)."""
        content = json.dumps({
            "state": list(self.state),
            "action": self.action,
            "reward": self.reward,
            "next_state": list(self.next_state),
            "session_id": self.session_id,
            "machine_id": self.machine_id,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ExperienceCollector:
    """
    Collects and stores RL experience tuples using Syna.
    
    Designed for multi-machine data collection with easy sync and merge.
    
    Key Schema:
        exp/{session_id}/{timestamp}_{hash} -> JSON transition
        meta/session/{session_id} -> session metadata
        meta/machine -> machine_id
        idx/reward/{timestamp} -> reward value (for tensor extraction)
    
    Example:
        >>> collector = ExperienceCollector("exp.db", machine_id="gpu_server_1")
        >>> with collector.session(model="Qwen/Qwen3-4B") as session:
        ...     for step in episode:
        ...         session.log(state, action, reward, next_state)
        >>> 
        >>> # On another machine, merge experiences
        >>> ExperienceCollector.merge(["exp1.db", "exp2.db"], "master.db")
    """
    
    def __init__(
        self,
        path: str,
        machine_id: Optional[str] = None,
        auto_flush: bool = True,
    ):
        """
        Open or create an experience database.
        
        Args:
            path: Path to the database file
            machine_id: Unique identifier for this machine (auto-generated if None)
            auto_flush: Whether to flush after each transition (safer but slower)
        """
        self.path = path
        self.db = SynaDB(path)
        self.auto_flush = auto_flush
        self._current_session: Optional[str] = None
        
        # Set or retrieve machine ID
        if machine_id:
            self.machine_id = machine_id
            self.db.put_text("meta/machine", machine_id)
        else:
            # Try to read existing, or generate new
            try:
                existing = self._get_text("meta/machine")
                self.machine_id = existing if existing else self._generate_machine_id()
            except:
                self.machine_id = self._generate_machine_id()
            self.db.put_text("meta/machine", self.machine_id)
    
    def _generate_machine_id(self) -> str:
        """Generate a unique machine identifier."""
        import platform
        import socket
        try:
            hostname = socket.gethostname()[:10]
        except:
            hostname = "unknown"
        return f"{hostname}_{uuid.uuid4().hex[:8]}"
    
    def _get_text(self, key: str) -> Optional[str]:
        """Helper to get text value (wrapper doesn't have get_text yet)."""
        # For now, we store text and retrieve via keys check
        # This is a limitation we'll work around
        if self.db.exists(key):
            # We need to add get_text to the wrapper, for now use a workaround
            # Store as bytes and decode
            return None  # Placeholder until get_text is added
        return None
    
    def close(self) -> None:
        """Close the database."""
        self.db.close()
    
    def __enter__(self) -> "ExperienceCollector":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    
    def log_transition(
        self,
        state: Tuple[Any, ...],
        action: Union[str, int],
        reward: float,
        next_state: Tuple[Any, ...],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a single experience transition.
        
        Args:
            state: Current state tuple (e.g., (layer, component, depth, coverage))
            action: Action taken (string name or integer ID)
            reward: Reward received
            next_state: Resulting state tuple
            session_id: Session identifier (uses current session if None)
            metadata: Additional metadata (model, insight_count, etc.)
            
        Returns:
            Key where the transition was stored
            
        Example:
            >>> collector.log_transition(
            ...     state=(0, 1, 2, 0.5),
            ...     action="analyze_weights",
            ...     reward=0.75,
            ...     next_state=(0, 1, 3, 0.6),
            ...     metadata={"model": "Qwen/Qwen3-4B", "insight_count": 3}
            ... )
        """
        timestamp = int(time.time() * 1_000_000)  # Microseconds
        session = session_id or self._current_session or "default"
        
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            timestamp=timestamp,
            session_id=session,
            machine_id=self.machine_id,
            metadata=metadata or {},
        )
        
        # Generate key with hash for deduplication
        content_hash = transition.content_hash()
        key = f"exp/{session}/{timestamp}_{content_hash}"
        
        # Store as JSON bytes (compact)
        data = json.dumps(transition.to_dict(), separators=(',', ':')).encode('utf-8')
        self.db.put_bytes(key, data)
        
        # Also store reward in a separate index for tensor extraction
        self.db.put_float(f"idx/reward/{session}", reward)
        
        return key
    
    def session(
        self,
        session_id: Optional[str] = None,
        **metadata
    ) -> "SessionContext":
        """
        Create a session context for logging multiple transitions.
        
        Args:
            session_id: Optional session ID (auto-generated if None)
            **metadata: Session-level metadata (model, hyperparams, etc.)
            
        Returns:
            SessionContext for use with `with` statement
            
        Example:
            >>> with collector.session(model="Qwen/Qwen3-4B", lr=0.001) as s:
            ...     s.log(state, action, reward, next_state)
            ...     s.log(state2, action2, reward2, next_state2)
        """
        return SessionContext(self, session_id, metadata)
    
    def get_transition(self, key: str) -> Optional[Transition]:
        """
        Retrieve a single transition by key.
        
        Args:
            key: The transition key
            
        Returns:
            Transition object, or None if not found
        """
        if not self.db.exists(key):
            return None
        
        data = self.db.get_bytes(key)
        if data is None:
            return None
        
        try:
            d = json.loads(data.decode('utf-8'))
            return Transition.from_dict(d)
        except (json.JSONDecodeError, KeyError):
            return None
    
    def iterate_transitions(
        self,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Iterate over all transition keys.
        
        Args:
            session_id: Filter to specific session (None for all)
            
        Yields:
            Transition keys
        """
        prefix = f"exp/{session_id}/" if session_id else "exp/"
        for key in self.db.keys():
            if key.startswith(prefix):
                yield key
    
    def get_rewards_tensor(self, session_id: str = "default") -> np.ndarray:
        """
        Get all rewards for a session as a numpy array.
        
        Optimized for training - returns contiguous float64 array.
        
        Args:
            session_id: Session to get rewards for
            
        Returns:
            numpy array of rewards in chronological order
        """
        return self.db.get_history_tensor(f"idx/reward/{session_id}")
    
    def count_transitions(self, session_id: Optional[str] = None) -> int:
        """Count total transitions, optionally filtered by session."""
        return sum(1 for _ in self.iterate_transitions(session_id))
    
    def list_sessions(self) -> List[str]:
        """List all session IDs in the database."""
        sessions = set()
        for key in self.db.keys():
            if key.startswith("exp/"):
                parts = key.split("/")
                if len(parts) >= 2:
                    sessions.add(parts[1])
        return sorted(sessions)
    
    def export_jsonl(self, output_path: str, session_id: Optional[str] = None) -> int:
        """
        Export transitions to JSON Lines format for sync/merge.
        
        Args:
            output_path: Path to output .jsonl file
            session_id: Filter to specific session (None for all)
            
        Returns:
            Number of transitions exported
        """
        count = 0
        with open(output_path, 'w') as f:
            for key in self.iterate_transitions(session_id):
                transition = self.get_transition(key)
                if transition:
                    f.write(json.dumps(transition.to_dict()) + "\n")
                    count += 1
        return count
    
    def import_jsonl(self, input_path: str, deduplicate: bool = True) -> int:
        """
        Import transitions from JSON Lines format.
        
        Args:
            input_path: Path to input .jsonl file
            deduplicate: Skip transitions that already exist (by hash)
            
        Returns:
            Number of transitions imported
        """
        seen_hashes = set()
        if deduplicate:
            for key in self.iterate_transitions():
                parts = key.split("_")
                if len(parts) >= 2:
                    seen_hashes.add(parts[-1])
        
        count = 0
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    transition = Transition.from_dict(d)
                    
                    if deduplicate:
                        content_hash = transition.content_hash()
                        if content_hash in seen_hashes:
                            continue
                        seen_hashes.add(content_hash)
                    
                    # Store the transition
                    key = f"exp/{transition.session_id}/{transition.timestamp}_{transition.content_hash()}"
                    data = json.dumps(transition.to_dict(), separators=(',', ':')).encode('utf-8')
                    self.db.put_bytes(key, data)
                    self.db.put_float(f"idx/reward/{transition.session_id}", transition.reward)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return count
    
    @staticmethod
    def merge(
        sources: List[str],
        dest: str,
        deduplicate: bool = True,
    ) -> int:
        """
        Merge multiple experience databases into one.
        
        Args:
            sources: List of source database paths
            dest: Destination database path
            deduplicate: Whether to skip duplicate transitions (by hash)
            
        Returns:
            Number of transitions merged
            
        Example:
            >>> ExperienceCollector.merge(
            ...     ["machine1/exp.db", "machine2/exp.db"],
            ...     "master/exp.db"
            ... )
        """
        seen_hashes = set()
        count = 0
        
        with SynaDB(dest) as master:
            # First, collect existing hashes if deduplicating
            if deduplicate:
                for key in master.keys():
                    if key.startswith("exp/"):
                        # Extract hash from key: exp/{session}/{timestamp}_{hash}
                        parts = key.split("_")
                        if len(parts) >= 2:
                            seen_hashes.add(parts[-1])
            
            # Merge each source
            for src_path in sources:
                if not Path(src_path).exists():
                    continue
                    
                with SynaDB(src_path) as src:
                    for key in src.keys():
                        # Skip non-experience keys
                        if not key.startswith("exp/"):
                            # But copy metadata and reward indexes
                            if key.startswith("meta/") or key.startswith("idx/"):
                                try:
                                    if key.startswith("idx/reward/"):
                                        # Copy reward history
                                        session = key.replace("idx/reward/", "")
                                        rewards = src.get_history_tensor(session)
                                        for r in rewards:
                                            master.put_float(key, r)
                                    else:
                                        # Copy metadata as text
                                        text = src.get_text(key)
                                        if text:
                                            master.put_text(key, text)
                                except:
                                    pass
                            continue
                        
                        # Check for duplicate
                        if deduplicate:
                            parts = key.split("_")
                            if len(parts) >= 2:
                                content_hash = parts[-1]
                                if content_hash in seen_hashes:
                                    continue
                                seen_hashes.add(content_hash)
                        
                        # Copy transition data
                        try:
                            data = src.get_bytes(key)
                            if data:
                                master.put_bytes(key, data)
                                count += 1
                        except:
                            pass
        
        return count
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the experience database.
        
        Returns:
            Dictionary with counts, sessions, machine info
        """
        sessions = self.list_sessions()
        total = self.count_transitions()
        
        session_counts = {}
        for session in sessions:
            session_counts[session] = self.count_transitions(session)
        
        return {
            "machine_id": self.machine_id,
            "total_transitions": total,
            "sessions": sessions,
            "session_counts": session_counts,
            "db_path": self.path,
        }


class SessionContext:
    """Context manager for logging transitions within a session."""
    
    def __init__(
        self,
        collector: ExperienceCollector,
        session_id: Optional[str],
        metadata: Dict[str, Any],
    ):
        self.collector = collector
        self.session_id = session_id or f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata
        self._transition_count = 0
        self._total_reward = 0.0
    
    def __enter__(self) -> "SessionContext":
        self.collector._current_session = self.session_id
        # Store session metadata
        meta_key = f"meta/session/{self.session_id}"
        meta_data = {
            "session_id": self.session_id,
            "machine_id": self.collector.machine_id,
            "start_time": int(time.time() * 1_000_000),
            **self.metadata,
        }
        self.collector.db.put_text(meta_key, json.dumps(meta_data))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.collector._current_session = None
    
    def log(
        self,
        state: Tuple[Any, ...],
        action: Union[str, int],
        reward: float,
        next_state: Tuple[Any, ...],
        **extra_metadata,
    ) -> str:
        """
        Log a transition within this session.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            **extra_metadata: Additional per-transition metadata
            
        Returns:
            Key where transition was stored
        """
        merged_metadata = {**self.metadata, **extra_metadata}
        key = self.collector.log_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            session_id=self.session_id,
            metadata=merged_metadata,
        )
        self._transition_count += 1
        self._total_reward += reward
        return key
    
    @property
    def transition_count(self) -> int:
        """Number of transitions logged in this session."""
        return self._transition_count
    
    @property
    def total_reward(self) -> float:
        """Sum of rewards in this session."""
        return self._total_reward
    
    @property
    def mean_reward(self) -> float:
        """Mean reward per transition."""
        if self._transition_count == 0:
            return 0.0
        return self._total_reward / self._transition_count


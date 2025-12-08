#!/usr/bin/env python3
"""
Syna Async Operations Demo

This demo shows how to use Syna with asyncio for non-blocking operations:
- Using asyncio with thread pool executor
- Non-blocking writes
- Concurrent async reads

Requirements: 2.4 - WHEN a developer views the Python async demo THEN the demo 
SHALL show non-blocking database operations

Run with: python async_operations.py
"""

import os
import sys
import time
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from Syna import SynaDB


# Global thread pool for database operations
_db_executor = ThreadPoolExecutor(max_workers=4)


class AsyncSynaDB:
    """
    Async wrapper for SynaDB using thread pool executor.
    
    Since the underlying FFI calls are blocking, we run them in a
    thread pool to avoid blocking the event loop.
    
    Example:
        async with AsyncSynaDB("my.db") as db:
            await db.put_float("key", 3.14)
            value = await db.get_float("key")
    """
    
    def __init__(self, path: str, executor: ThreadPoolExecutor = None):
        """
        Initialize async database wrapper.
        
        Args:
            path: Path to database file
            executor: Optional thread pool executor (uses global if not provided)
        """
        self._path = path
        self._executor = executor or _db_executor
        self._db = None
        self._loop = None
    
    async def __aenter__(self) -> 'AsyncSynaDB':
        """Async context manager entry."""
        self._loop = asyncio.get_event_loop()
        self._db = await self._loop.run_in_executor(
            self._executor,
            lambda: SynaDB(self._path)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._db:
            await self._loop.run_in_executor(
                self._executor,
                self._db.close
            )
    
    async def put_float(self, key: str, value: float) -> int:
        """Async write float value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.put_float(key, value)
        )
    
    async def put_int(self, key: str, value: int) -> int:
        """Async write int value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.put_int(key, value)
        )
    
    async def put_text(self, key: str, value: str) -> int:
        """Async write text value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.put_text(key, value)
        )
    
    async def put_bytes(self, key: str, value: bytes) -> int:
        """Async write bytes value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.put_bytes(key, value)
        )
    
    async def get_float(self, key: str) -> float:
        """Async read float value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.get_float(key)
        )
    
    async def get_int(self, key: str) -> int:
        """Async read int value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.get_int(key)
        )
    
    async def get_text(self, key: str) -> str:
        """Async read text value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.get_text(key)
        )
    
    async def get_bytes(self, key: str) -> bytes:
        """Async read bytes value."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.get_bytes(key)
        )
    
    async def get_history_tensor(self, key: str) -> np.ndarray:
        """Async get history as numpy array."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.get_history_tensor(key)
        )
    
    async def delete(self, key: str) -> None:
        """Async delete key."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.delete(key)
        )
    
    async def exists(self, key: str) -> bool:
        """Async check key existence."""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: self._db.exists(key)
        )
    
    async def keys(self) -> List[str]:
        """Async list all keys."""
        return await self._loop.run_in_executor(
            self._executor,
            self._db.keys
        )


async def demo_basic_async_operations():
    """Demonstrate basic async operations."""
    print("=" * 60)
    print("1. BASIC ASYNC OPERATIONS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        print("\nUsing AsyncSynaDB")
        print("-" * 40)
        print("""
async with AsyncSynaDB("my.db") as db:
    await db.put_float("key", 3.14)
    value = await db.get_float("key")
""")
        
        async with AsyncSynaDB(db_path) as db:
            # Write operations
            await db.put_float("sensor/temp", 23.5)
            await db.put_int("counter", 42)
            await db.put_text("message", "Hello, Async!")
            
            print("✓ Async writes completed")
            
            # Read operations
            temp = await db.get_float("sensor/temp")
            count = await db.get_int("counter")
            msg = await db.get_text("message")
            
            print(f"✓ sensor/temp = {temp}")
            print(f"✓ counter = {count}")
            print(f"✓ message = '{msg}'")
            
            # Check existence
            exists = await db.exists("sensor/temp")
            print(f"✓ exists('sensor/temp') = {exists}")
            
            # List keys
            keys = await db.keys()
            print(f"✓ keys = {keys}")
    
    print()


async def demo_non_blocking_writes():
    """Demonstrate non-blocking writes."""
    print("=" * 60)
    print("2. NON-BLOCKING WRITES")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        async with AsyncSynaDB(db_path) as db:
            # Simulate sensor data stream
            print("\nSimulating Non-Blocking Sensor Writes")
            print("-" * 40)
            
            n_writes = 100
            
            async def write_sensor_data(sensor_id: int, n_values: int):
                """Write sensor data without blocking."""
                for i in range(n_values):
                    value = 20 + np.random.randn() * 2
                    await db.put_float(f"sensor/{sensor_id}/temp", value)
                return sensor_id, n_values
            
            # Sequential writes (for comparison)
            print("\nSequential writes (blocking):")
            start = time.time()
            for sensor_id in range(4):
                await write_sensor_data(sensor_id, n_writes)
            seq_time = time.time() - start
            print(f"✓ {4 * n_writes} writes in {seq_time:.3f}s")
            
            # Clear for next test
            # (Note: We can't actually clear, so we use different keys)
            
            # Concurrent writes (non-blocking)
            print("\nConcurrent writes (non-blocking):")
            start = time.time()
            tasks = [
                write_sensor_data(sensor_id + 10, n_writes)
                for sensor_id in range(4)
            ]
            results = await asyncio.gather(*tasks)
            conc_time = time.time() - start
            print(f"✓ {4 * n_writes} writes in {conc_time:.3f}s")
            
            # Note: Due to the GIL and single-file writes, concurrent writes
            # may not be faster, but they don't block the event loop
            print(f"\nNote: Concurrent writes don't block the event loop,")
            print(f"allowing other async tasks to run during I/O.")
    
    print()


async def demo_concurrent_async_reads():
    """Demonstrate concurrent async reads."""
    print("=" * 60)
    print("3. CONCURRENT ASYNC READS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        async with AsyncSynaDB(db_path) as db:
            # Setup: Write test data
            print("\nSetup: Writing test data")
            print("-" * 40)
            
            n_keys = 20
            n_values = 50
            
            for key_id in range(n_keys):
                for _ in range(n_values):
                    await db.put_float(f"data/{key_id}", np.random.randn())
            
            print(f"✓ Created {n_keys} keys with {n_values} values each")
            
            # Sequential reads
            print("\nSequential Reads:")
            print("-" * 40)
            
            start = time.time()
            results_seq = []
            for key_id in range(n_keys):
                tensor = await db.get_history_tensor(f"data/{key_id}")
                results_seq.append((key_id, tensor.mean()))
            seq_time = time.time() - start
            print(f"✓ Read {n_keys} tensors in {seq_time*1000:.1f}ms")
            
            # Concurrent reads
            print("\nConcurrent Reads:")
            print("-" * 40)
            
            async def read_and_compute(key_id: int) -> Tuple[int, float]:
                """Read tensor and compute mean."""
                tensor = await db.get_history_tensor(f"data/{key_id}")
                return key_id, tensor.mean()
            
            start = time.time()
            tasks = [read_and_compute(key_id) for key_id in range(n_keys)]
            results_conc = await asyncio.gather(*tasks)
            conc_time = time.time() - start
            print(f"✓ Read {n_keys} tensors in {conc_time*1000:.1f}ms")
            
            # Verify results match
            results_seq_sorted = sorted(results_seq)
            results_conc_sorted = sorted(results_conc)
            match = all(
                abs(s[1] - c[1]) < 1e-10 
                for s, c in zip(results_seq_sorted, results_conc_sorted)
            )
            print(f"✓ Results match: {match}")
    
    print()


async def demo_mixed_async_workload():
    """Demonstrate mixed async workload."""
    print("=" * 60)
    print("4. MIXED ASYNC WORKLOAD")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        async with AsyncSynaDB(db_path) as db:
            print("\nSimulating Real-Time Data Pipeline")
            print("-" * 40)
            print("""
This simulates a real-time pipeline where:
- Sensors write data continuously
- Analyzers read and process data
- All operations are non-blocking
""")
            
            # Shared state
            write_count = 0
            read_count = 0
            running = True
            
            async def sensor_writer(sensor_id: int, interval: float):
                """Simulate sensor writing data at intervals."""
                nonlocal write_count
                while running:
                    value = 20 + np.random.randn() * 2
                    await db.put_float(f"sensor/{sensor_id}", value)
                    write_count += 1
                    await asyncio.sleep(interval)
            
            async def data_analyzer(sensor_id: int, interval: float):
                """Analyze sensor data periodically."""
                nonlocal read_count
                while running:
                    await asyncio.sleep(interval)
                    tensor = await db.get_history_tensor(f"sensor/{sensor_id}")
                    if len(tensor) > 0:
                        mean = tensor.mean()
                        read_count += 1
            
            async def status_reporter(interval: float):
                """Report status periodically."""
                while running:
                    await asyncio.sleep(interval)
                    print(f"  Status: {write_count} writes, {read_count} reads")
            
            # Start tasks
            print("\nStarting pipeline (running for 2 seconds)...")
            
            tasks = []
            # 4 sensors writing at different rates
            for i in range(4):
                tasks.append(asyncio.create_task(sensor_writer(i, 0.01 + i * 0.005)))
            # 4 analyzers reading
            for i in range(4):
                tasks.append(asyncio.create_task(data_analyzer(i, 0.05)))
            # Status reporter
            tasks.append(asyncio.create_task(status_reporter(0.5)))
            
            # Run for 2 seconds
            await asyncio.sleep(2.0)
            running = False
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            
            print(f"\nPipeline stopped:")
            print(f"  Total writes: {write_count}")
            print(f"  Total reads: {read_count}")
            print(f"  Write rate: {write_count/2:.0f} writes/sec")
            print(f"  Read rate: {read_count/2:.0f} reads/sec")
    
    print()


async def demo_async_batch_operations():
    """Demonstrate async batch operations."""
    print("=" * 60)
    print("5. ASYNC BATCH OPERATIONS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        async with AsyncSynaDB(db_path) as db:
            # Batch write
            print("\nBatch Write Pattern")
            print("-" * 40)
            print("""
async def batch_write(items):
    tasks = [db.put_float(k, v) for k, v in items]
    return await asyncio.gather(*tasks)
""")
            
            items = [(f"batch/{i}", float(i)) for i in range(100)]
            
            start = time.time()
            tasks = [db.put_float(k, v) for k, v in items]
            await asyncio.gather(*tasks)
            batch_time = time.time() - start
            
            print(f"✓ Batch wrote {len(items)} items in {batch_time*1000:.1f}ms")
            
            # Batch read
            print("\nBatch Read Pattern")
            print("-" * 40)
            print("""
async def batch_read(keys):
    tasks = [db.get_float(k) for k in keys]
    return await asyncio.gather(*tasks)
""")
            
            keys = [f"batch/{i}" for i in range(100)]
            
            start = time.time()
            tasks = [db.get_float(k) for k in keys]
            values = await asyncio.gather(*tasks)
            batch_time = time.time() - start
            
            print(f"✓ Batch read {len(keys)} items in {batch_time*1000:.1f}ms")
            print(f"✓ First 5 values: {values[:5]}")
            
            # Batch with error handling
            print("\nBatch with Error Handling")
            print("-" * 40)
            print("""
async def safe_read(key):
    try:
        return key, await db.get_float(key)
    except Exception as e:
        return key, None
""")
            
            async def safe_read(key: str):
                try:
                    value = await db.get_float(key)
                    return key, value
                except Exception:
                    return key, None
            
            # Mix of existing and non-existing keys
            mixed_keys = [f"batch/{i}" for i in range(50)] + [f"missing/{i}" for i in range(50)]
            
            tasks = [safe_read(k) for k in mixed_keys]
            results = await asyncio.gather(*tasks)
            
            found = sum(1 for _, v in results if v is not None)
            missing = sum(1 for _, v in results if v is None)
            
            print(f"✓ Found: {found}, Missing: {missing}")
    
    print()


async def demo_async_with_other_tasks():
    """Demonstrate async DB operations alongside other async tasks."""
    print("=" * 60)
    print("6. ASYNC DB WITH OTHER ASYNC TASKS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "async_demo.db")
        
        async with AsyncSynaDB(db_path) as db:
            print("\nRunning DB Operations Alongside Other Async Tasks")
            print("-" * 40)
            print("""
The key benefit of async DB operations is that they don't block
other async tasks. This is crucial for:
- Web servers handling multiple requests
- Real-time data pipelines
- Interactive applications
""")
            
            results = []
            
            async def db_task():
                """Database operations."""
                for i in range(10):
                    await db.put_float("async/value", float(i))
                    await asyncio.sleep(0.01)
                results.append("DB task completed")
            
            async def compute_task():
                """CPU-bound simulation (would normally use ProcessPoolExecutor)."""
                for i in range(10):
                    # Simulate some computation
                    _ = sum(range(1000))
                    await asyncio.sleep(0.01)
                results.append("Compute task completed")
            
            async def io_task():
                """I/O simulation."""
                for i in range(10):
                    await asyncio.sleep(0.01)
                results.append("I/O task completed")
            
            # Run all tasks concurrently
            start = time.time()
            await asyncio.gather(
                db_task(),
                compute_task(),
                io_task()
            )
            total_time = time.time() - start
            
            print(f"✓ All tasks completed in {total_time*1000:.1f}ms")
            print(f"✓ Results: {results}")
            print(f"\nNote: Tasks ran concurrently, not sequentially!")
    
    print()


async def main_async():
    """Run all async demos."""
    print("\n" + "=" * 60)
    print("   Syna ASYNC OPERATIONS DEMO")
    print("   Requirements: 2.4")
    print("=" * 60 + "\n")
    
    try:
        await demo_basic_async_operations()
        await demo_non_blocking_writes()
        await demo_concurrent_async_reads()
        await demo_mixed_async_workload()
        await demo_async_batch_operations()
        await demo_async_with_other_tasks()
        
        print("=" * 60)
        print("   ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Entry point."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())


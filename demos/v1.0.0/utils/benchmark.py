"""
Benchmarking utilities for SynaDB v1.0.0 Showcase.

Provides consistent benchmarking across all notebooks with:
- Warmup iterations
- Multiple measurement iterations
- Reproducible results via seeding
- Statistical analysis (mean, std, percentiles)
- Comparison tables for result aggregation
"""

import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Any, Dict
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.
    
    Attributes:
        name: System/operation name (e.g., "SynaDB", "Chroma")
        iterations: Number of benchmark iterations
        mean_ms: Mean latency in milliseconds
        std_ms: Standard deviation in milliseconds
        min_ms: Minimum latency
        max_ms: Maximum latency
        p50_ms: 50th percentile (median)
        p95_ms: 95th percentile
        p99_ms: 99th percentile
        throughput: Operations per second (optional)
        raw_times: Raw timing data for further analysis
    """
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput: Optional[float] = None
    raw_times: List[float] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name='{self.name}', "
            f"mean={self.mean_ms:.2f}ms, "
            f"std={self.std_ms:.2f}ms, "
            f"p95={self.p95_ms:.2f}ms)"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'throughput': self.throughput,
        }


class Benchmark:
    """Consistent benchmarking across all notebooks.
    
    Args:
        warmup: Number of warmup iterations (not measured)
        iterations: Number of measured iterations
        seed: Random seed for reproducibility
    
    Example:
        >>> bench = Benchmark(warmup=5, iterations=100, seed=42)
        >>> result = bench.run("SynaDB Insert", db.insert, "key", embedding)
        >>> print(f"Mean: {result.mean_ms:.2f}ms")
    """
    
    def __init__(self, warmup: int = 5, iterations: int = 100, seed: int = 42):
        self.warmup = warmup
        self.iterations = iterations
        self.seed = seed
        self._set_seed()
    
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def run(
        self, 
        name: str, 
        func: Callable, 
        *args, 
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark with warmup and multiple iterations.
        
        Args:
            name: Name for this benchmark
            func: Function to benchmark
            *args: Arguments to pass to func
            setup: Optional setup function called before each iteration
            teardown: Optional teardown function called after each iteration
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            BenchmarkResult with timing statistics
        """
        # Reset seed for reproducibility
        self._set_seed()
        
        # Warmup phase (not measured)
        for _ in range(self.warmup):
            if setup:
                setup()
            func(*args, **kwargs)
            if teardown:
                teardown()
        
        # Measurement phase
        times: List[float] = []
        for _ in range(self.iterations):
            if setup:
                setup()
            
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            
            times.append(elapsed)
            
            if teardown:
                teardown()
        
        # Calculate statistics
        times.sort()
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        
        return BenchmarkResult(
            name=name,
            iterations=self.iterations,
            mean_ms=mean,
            std_ms=std,
            min_ms=min(times),
            max_ms=max(times),
            p50_ms=self._percentile(times, 50),
            p95_ms=self._percentile(times, 95),
            p99_ms=self._percentile(times, 99),
            throughput=1000 / mean if mean > 0 else 0,
            raw_times=times
        )
    
    def run_batch(
        self,
        name: str,
        func: Callable,
        batch_size: int,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """Run benchmark for batch operations.
        
        Measures total time for batch_size operations and calculates
        per-operation metrics.
        
        Args:
            name: Name for this benchmark
            func: Function that processes a batch
            batch_size: Number of items in each batch
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            BenchmarkResult with per-operation timing
        """
        result = self.run(name, func, *args, **kwargs)
        
        # Adjust metrics to per-operation
        result.mean_ms /= batch_size
        result.std_ms /= batch_size
        result.min_ms /= batch_size
        result.max_ms /= batch_size
        result.p50_ms /= batch_size
        result.p95_ms /= batch_size
        result.p99_ms /= batch_size
        result.throughput = batch_size * 1000 / (result.mean_ms * batch_size) if result.mean_ms > 0 else 0
        
        return result
    
    @staticmethod
    def _percentile(sorted_data: List[float], p: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]
    
    def compare(self, results: List[BenchmarkResult]) -> 'ComparisonTable':
        """Generate comparison table from multiple results.
        
        Args:
            results: List of BenchmarkResult objects to compare
            
        Returns:
            ComparisonTable for display and analysis
        """
        return ComparisonTable(results)


class ComparisonTable:
    """Comparison table for benchmark results.
    
    Provides multiple output formats:
    - Markdown tables
    - Pandas DataFrames
    - Matplotlib charts
    """
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
    
    def to_markdown(self, highlight_best: bool = True) -> str:
        """Generate markdown table with optional highlighting.
        
        Args:
            highlight_best: Whether to highlight the best (lowest) values
            
        Returns:
            Markdown-formatted table string
        """
        if not self.results:
            return "No results to display."
        
        # Find best values for highlighting
        best_mean = min(r.mean_ms for r in self.results)
        best_p95 = min(r.p95_ms for r in self.results)
        
        # Build table
        lines = [
            "| System | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (ops/s) |",
            "|--------|-----------|----------|----------|----------|----------|-------------------|"
        ]
        
        for r in self.results:
            mean_str = f"**{r.mean_ms:.2f}**" if highlight_best and r.mean_ms == best_mean else f"{r.mean_ms:.2f}"
            p95_str = f"**{r.p95_ms:.2f}**" if highlight_best and r.p95_ms == best_p95 else f"{r.p95_ms:.2f}"
            throughput_str = f"{r.throughput:.0f}" if r.throughput else "N/A"
            
            lines.append(
                f"| {r.name} | {mean_str} | {r.std_ms:.2f} | {r.p50_ms:.2f} | "
                f"{p95_str} | {r.p99_ms:.2f} | {throughput_str} |"
            )
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame.
        
        Returns:
            pandas DataFrame or dict if pandas not available
        """
        data = [r.to_dict() for r in self.results]
        
        if HAS_PANDAS:
            return pd.DataFrame(data)
        else:
            return data
    
    def get_speedup(self, baseline: str, target: str) -> Optional[float]:
        """Calculate speedup of target vs baseline.
        
        Args:
            baseline: Name of baseline system
            target: Name of target system
            
        Returns:
            Speedup factor (baseline_time / target_time) or None if not found
        """
        baseline_result = next((r for r in self.results if r.name == baseline), None)
        target_result = next((r for r in self.results if r.name == target), None)
        
        if baseline_result and target_result and target_result.mean_ms > 0:
            return baseline_result.mean_ms / target_result.mean_ms
        return None
    
    def summary(self) -> str:
        """Generate a text summary of the comparison.
        
        Returns:
            Human-readable summary string
        """
        if not self.results:
            return "No results to summarize."
        
        # Find best performer
        best = min(self.results, key=lambda r: r.mean_ms)
        worst = max(self.results, key=lambda r: r.mean_ms)
        
        lines = [
            f"Benchmark Summary ({len(self.results)} systems compared)",
            f"  Best:  {best.name} ({best.mean_ms:.2f}ms mean)",
            f"  Worst: {worst.name} ({worst.mean_ms:.2f}ms mean)",
        ]
        
        if best.name != worst.name:
            speedup = worst.mean_ms / best.mean_ms
            lines.append(f"  Speedup: {speedup:.1f}x faster")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ComparisonTable({len(self.results)} results)"
    
    def _repr_markdown_(self) -> str:
        """Jupyter notebook markdown representation."""
        return self.to_markdown()

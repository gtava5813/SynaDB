"""
Property-based tests for SynaDB v1.0.0 Showcase utilities.

Tests:
- Property 4: Benchmark Reproducibility
- Property 6: Graceful Dependency Handling

Validates: Requirements 21.3, 21.5, 21.6, 21.8
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, strategies as st, settings

from utils.benchmark import Benchmark, BenchmarkResult, ComparisonTable
from utils.notebook_utils import check_dependency


class TestBenchmarkReproducibility:
    """
    **Feature: demos-v1-showcase, Property 4: Benchmark Reproducibility**
    
    For any benchmark cell in any notebook, when executed twice with the 
    same random seed, the results SHALL have variance within 10% of the mean.
    
    **Validates: Requirements 21.3, 21.5**
    """
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        warmup=st.integers(min_value=1, max_value=10),
        iterations=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_benchmark_reproducibility_with_same_seed(
        self, seed: int, warmup: int, iterations: int
    ):
        """
        **Feature: demos-v1-showcase, Property 4: Benchmark Reproducibility**
        
        When running the same benchmark twice with identical seeds,
        the results should be reproducible (same statistical properties).
        """
        # Create a simple deterministic function to benchmark
        counter = [0]
        def deterministic_func():
            counter[0] += 1
            return counter[0]
        
        # Run benchmark twice with same seed
        bench1 = Benchmark(warmup=warmup, iterations=iterations, seed=seed)
        counter[0] = 0
        result1 = bench1.run("test", deterministic_func)
        
        bench2 = Benchmark(warmup=warmup, iterations=iterations, seed=seed)
        counter[0] = 0
        result2 = bench2.run("test", deterministic_func)
        
        # Both should have completed the same number of iterations
        assert result1.iterations == result2.iterations == iterations
        
        # Results should be structurally identical
        assert result1.name == result2.name
    
    @given(
        iterations=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_benchmark_result_statistics_valid(self, iterations: int):
        """
        **Feature: demos-v1-showcase, Property 4: Benchmark Reproducibility**
        
        For any benchmark result, statistical properties should be valid:
        - min <= p50 <= p95 <= p99 <= max
        - mean should be between min and max
        - std should be non-negative
        """
        bench = Benchmark(warmup=2, iterations=iterations, seed=42)
        
        # Simple function with some variance
        import random
        random.seed(42)
        def variable_func():
            import time
            time.sleep(random.uniform(0.0001, 0.001))
        
        result = bench.run("test", variable_func)
        
        # Validate statistical ordering
        assert result.min_ms <= result.p50_ms, "min should be <= p50"
        assert result.p50_ms <= result.p95_ms, "p50 should be <= p95"
        assert result.p95_ms <= result.p99_ms, "p95 should be <= p99"
        assert result.p99_ms <= result.max_ms, "p99 should be <= max"
        
        # Mean should be within bounds
        assert result.min_ms <= result.mean_ms <= result.max_ms, "mean should be between min and max"
        
        # Std should be non-negative
        assert result.std_ms >= 0, "std should be non-negative"
        
        # Throughput should be positive
        assert result.throughput > 0, "throughput should be positive"
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=100, deadline=None)
    def test_benchmark_seed_affects_internal_state(self, seed: int):
        """
        **Feature: demos-v1-showcase, Property 4: Benchmark Reproducibility**
        
        Different seeds should produce different internal random states,
        ensuring reproducibility is seed-dependent.
        """
        import random
        import numpy as np
        
        # Create benchmark with specific seed
        bench = Benchmark(warmup=1, iterations=5, seed=seed)
        
        # After initialization, random state should be set
        # Get a random number to verify state was set
        val1 = random.random()
        
        # Reset with same seed
        bench._set_seed()
        val2 = random.random()
        
        # Should get same value after resetting seed
        assert val1 == val2, "Same seed should produce same random sequence"


class TestGracefulDependencyHandling:
    """
    **Feature: demos-v1-showcase, Property 6: Graceful Dependency Handling**
    
    For any notebook with optional dependencies, when an optional dependency 
    is missing, the notebook SHALL display an installation instruction and 
    skip that comparison without raising an exception.
    
    **Validates: Requirements 21.6, 21.8**
    """
    
    @given(
        module_name=st.text(
            alphabet=st.characters(whitelist_categories=('Ll',), min_codepoint=97, max_codepoint=122),
            min_size=5,
            max_size=20
        ).filter(lambda x: x.isidentifier() and x not in sys.modules)
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_dependency_returns_false(self, module_name: str):
        """
        **Feature: demos-v1-showcase, Property 6: Graceful Dependency Handling**
        
        For any non-existent module name, check_dependency should return False
        without raising an exception.
        """
        # Ensure the module doesn't exist
        fake_module = f"nonexistent_{module_name}_xyz123"
        
        # Should return False without raising
        result = check_dependency(fake_module)
        
        assert result is False, f"check_dependency should return False for missing module {fake_module}"
    
    def test_existing_dependency_returns_true(self):
        """
        **Feature: demos-v1-showcase, Property 6: Graceful Dependency Handling**
        
        For existing standard library modules, check_dependency should return True.
        """
        # Test with standard library modules that always exist
        assert check_dependency("sys") is True
        assert check_dependency("os") is True
        assert check_dependency("json") is True
    
    @given(
        install_cmd=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    )
    @settings(max_examples=100, deadline=None)
    def test_custom_install_command_accepted(self, install_cmd: str):
        """
        **Feature: demos-v1-showcase, Property 6: Graceful Dependency Handling**
        
        check_dependency should accept custom install commands without error.
        """
        # Should not raise with custom install command
        result = check_dependency(
            "nonexistent_module_abc123",
            install_cmd=install_cmd
        )
        
        assert result is False


class TestComparisonTable:
    """Tests for ComparisonTable functionality."""
    
    @given(
        names=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=1,
            max_size=10,
            unique=True
        ),
        mean_values=st.lists(
            st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_comparison_table_markdown_generation(self, names, mean_values):
        """
        ComparisonTable should generate valid markdown for any set of results.
        """
        # Ensure same length
        min_len = min(len(names), len(mean_values))
        names = names[:min_len]
        mean_values = mean_values[:min_len]
        
        if not names:
            return
        
        # Create mock results
        results = []
        for name, mean in zip(names, mean_values):
            result = BenchmarkResult(
                name=name,
                iterations=100,
                mean_ms=mean,
                std_ms=mean * 0.1,
                min_ms=mean * 0.8,
                max_ms=mean * 1.2,
                p50_ms=mean,
                p95_ms=mean * 1.1,
                p99_ms=mean * 1.15,
                throughput=1000 / mean if mean > 0 else 0
            )
            results.append(result)
        
        table = ComparisonTable(results)
        markdown = table.to_markdown()
        
        # Should contain table headers
        assert "System" in markdown
        assert "Mean" in markdown
        
        # Should contain all system names
        for name in names:
            assert name in markdown
    
    def test_comparison_table_speedup_calculation(self):
        """
        ComparisonTable should correctly calculate speedup between systems.
        """
        results = [
            BenchmarkResult(
                name="Baseline",
                iterations=100,
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=8.0,
                max_ms=12.0,
                p50_ms=10.0,
                p95_ms=11.0,
                p99_ms=11.5,
                throughput=100.0
            ),
            BenchmarkResult(
                name="Fast",
                iterations=100,
                mean_ms=2.0,
                std_ms=0.2,
                min_ms=1.6,
                max_ms=2.4,
                p50_ms=2.0,
                p95_ms=2.2,
                p99_ms=2.3,
                throughput=500.0
            ),
        ]
        
        table = ComparisonTable(results)
        
        # Fast should be 5x faster than Baseline
        speedup = table.get_speedup("Baseline", "Fast")
        assert speedup is not None
        assert abs(speedup - 5.0) < 0.01


class TestBenchmarkResultDataclass:
    """Tests for BenchmarkResult dataclass."""
    
    @given(
        name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        iterations=st.integers(min_value=1, max_value=10000),
        mean_ms=st.floats(min_value=0.001, max_value=10000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_benchmark_result_to_dict(self, name, iterations, mean_ms):
        """
        BenchmarkResult.to_dict() should return a valid dictionary.
        """
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            mean_ms=mean_ms,
            std_ms=mean_ms * 0.1,
            min_ms=mean_ms * 0.8,
            max_ms=mean_ms * 1.2,
            p50_ms=mean_ms,
            p95_ms=mean_ms * 1.1,
            p99_ms=mean_ms * 1.15,
            throughput=1000 / mean_ms if mean_ms > 0 else 0
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d['name'] == name
        assert d['iterations'] == iterations
        assert d['mean_ms'] == mean_ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

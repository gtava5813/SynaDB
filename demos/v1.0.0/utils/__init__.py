# SynaDB v1.0.0 Showcase Utilities
"""
Shared utilities for the SynaDB v1.0.0 showcase notebooks.

Modules:
- benchmark: Benchmarking utilities with reproducible results
- charts: Consistent matplotlib styling with SynaDB branding
- system_info: System specification reporting
- notebook_utils: TOC generation, branding, dependency checking
"""

from .benchmark import Benchmark, BenchmarkResult, ComparisonTable
from .charts import (
    setup_style, bar_comparison, latency_distribution, 
    throughput_comparison, memory_comparison, COLORS
)
from .system_info import (
    get_system_info, display_system_info, 
    get_system_summary, check_gpu_available
)
from .notebook_utils import (
    display_header, generate_toc, display_toc,
    highlight_advantage, highlight_disadvantage, highlight_neutral,
    check_dependency, check_dependencies,
    comparison_table, section_header, display_section,
    conclusion_box, info_box, warning_box
)

__all__ = [
    # Benchmark
    'Benchmark', 'BenchmarkResult', 'ComparisonTable',
    # Charts
    'setup_style', 'bar_comparison', 'latency_distribution', 
    'throughput_comparison', 'memory_comparison', 'COLORS',
    # System Info
    'get_system_info', 'display_system_info', 
    'get_system_summary', 'check_gpu_available',
    # Notebook Utils
    'display_header', 'generate_toc', 'display_toc',
    'highlight_advantage', 'highlight_disadvantage', 'highlight_neutral',
    'check_dependency', 'check_dependencies',
    'comparison_table', 'section_header', 'display_section',
    'conclusion_box', 'info_box', 'warning_box'
]

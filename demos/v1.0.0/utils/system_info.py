"""
System information utilities for SynaDB v1.0.0 Showcase.

Provides system specification reporting for benchmark reproducibility:
- CPU information
- RAM capacity
- GPU detection (if available)
- OS information
- Python version
"""

import platform
import sys
from typing import Dict, Any, Optional


def get_system_info() -> Dict[str, Any]:
    """Collect system specifications for benchmark reporting.
    
    Returns:
        Dictionary with system information including:
        - os: Operating system name
        - os_version: OS version string
        - python_version: Python version
        - cpu: CPU model/processor
        - cpu_cores: Physical CPU cores
        - cpu_threads: Logical CPU threads
        - ram_gb: Total RAM in GB
        - gpu: GPU name (if available)
        - gpu_memory_gb: GPU memory in GB (if available)
        
    Example:
        >>> info = get_system_info()
        >>> print(f"Running on {info['os']} with {info['ram_gb']}GB RAM")
    """
    info: Dict[str, Any] = {
        'os': platform.system(),
        'os_version': platform.version(),
        'os_release': platform.release(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'cpu': platform.processor() or 'Unknown',
        'machine': platform.machine(),
    }
    
    # Get CPU and RAM info via psutil if available
    try:
        import psutil
        info['cpu_cores'] = psutil.cpu_count(logical=False) or 'Unknown'
        info['cpu_threads'] = psutil.cpu_count(logical=True) or 'Unknown'
        info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info['cpu_cores'] = 'Unknown (psutil not installed)'
        info['cpu_threads'] = 'Unknown (psutil not installed)'
        info['ram_gb'] = 'Unknown (psutil not installed)'
    
    # Try to get more detailed CPU info on different platforms
    if info['cpu'] == 'Unknown' or info['cpu'] == '':
        info['cpu'] = _get_cpu_name()
    
    # GPU info via PyTorch if available
    gpu_info = _get_gpu_info_torch()
    if gpu_info:
        info.update(gpu_info)
    else:
        # Try TensorFlow as fallback
        gpu_info = _get_gpu_info_tensorflow()
        if gpu_info:
            info.update(gpu_info)
    
    return info


def _get_cpu_name() -> str:
    """Try to get a more descriptive CPU name."""
    try:
        if platform.system() == 'Windows':
            import subprocess
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
        elif platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        elif platform.system() == 'Darwin':  # macOS
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or 'Unknown'


def _get_gpu_info_torch() -> Optional[Dict[str, Any]]:
    """Get GPU info via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'gpu': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_gb': round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                ),
                'cuda_version': torch.version.cuda or 'Unknown',
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS
            return {
                'gpu': 'Apple Silicon (MPS)',
                'gpu_count': 1,
                'gpu_memory_gb': 'Shared',
            }
    except ImportError:
        pass
    return None


def _get_gpu_info_tensorflow() -> Optional[Dict[str, Any]]:
    """Get GPU info via TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return {
                'gpu': f'{len(gpus)} GPU(s) detected',
                'gpu_count': len(gpus),
            }
    except ImportError:
        pass
    return None


def display_system_info() -> None:
    """Display system info in notebook-friendly format.
    
    Uses IPython display for rich formatting in Jupyter notebooks,
    falls back to print for terminal usage.
    
    Example:
        >>> display_system_info()
        ### System Specifications
        | Specification | Value |
        |---|---|
        | OS | Linux |
        | ...
    """
    info = get_system_info()
    
    # Build markdown table
    md = "### 🖥️ System Specifications\n\n"
    md += "| Specification | Value |\n"
    md += "|---------------|-------|\n"
    
    # Define display order and labels
    display_items = [
        ('os', 'Operating System'),
        ('os_release', 'OS Release'),
        ('python_version', 'Python Version'),
        ('cpu', 'CPU'),
        ('cpu_cores', 'CPU Cores'),
        ('cpu_threads', 'CPU Threads'),
        ('ram_gb', 'RAM (GB)'),
        ('gpu', 'GPU'),
        ('gpu_count', 'GPU Count'),
        ('gpu_memory_gb', 'GPU Memory (GB)'),
        ('cuda_version', 'CUDA Version'),
    ]
    
    for key, label in display_items:
        if key in info:
            value = info[key]
            md += f"| {label} | {value} |\n"
    
    # Try IPython display, fall back to print
    try:
        from IPython.display import display, Markdown
        display(Markdown(md))
    except ImportError:
        # Convert markdown to plain text for terminal
        print("\nSystem Specifications")
        print("=" * 40)
        for key, label in display_items:
            if key in info:
                print(f"{label}: {info[key]}")
        print()


def get_system_summary() -> str:
    """Get a one-line system summary.
    
    Returns:
        Short string summarizing the system
        
    Example:
        >>> print(get_system_summary())
        'Linux, 8 cores, 32GB RAM, NVIDIA RTX 3080'
    """
    info = get_system_info()
    
    parts = [info['os']]
    
    if isinstance(info.get('cpu_cores'), int):
        parts.append(f"{info['cpu_cores']} cores")
    
    if isinstance(info.get('ram_gb'), (int, float)):
        parts.append(f"{info['ram_gb']}GB RAM")
    
    if 'gpu' in info:
        parts.append(info['gpu'])
    
    return ', '.join(parts)


def check_gpu_available() -> bool:
    """Check if GPU is available for computation.
    
    Returns:
        True if GPU is available via PyTorch or TensorFlow
    """
    # Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True
    except ImportError:
        pass
    
    # Try TensorFlow
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            return True
    except ImportError:
        pass
    
    return False

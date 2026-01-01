"""
Chart utilities for SynaDB v1.0.0 Showcase.

Provides consistent matplotlib styling with SynaDB branding:
- Brand colors
- Consistent chart styling
- Bar comparison charts
- Latency distribution plots
"""

from typing import Dict, List, Optional, Any, Union
import matplotlib.pyplot as plt
import numpy as np

# Try to import BenchmarkResult for type hints
try:
    from .benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any


# SynaDB brand colors
COLORS = {
    'synadb': '#4A90D9',           # Primary blue
    'synadb_light': '#7AB3E8',     # Light blue
    'synadb_dark': '#357ABD',      # Dark blue
    'competitor': '#888888',       # Gray for competitors
    'competitor_alt': '#A0A0A0',   # Alternate gray
    'competitor1': '#888888',      # Gray for first competitor
    'competitor2': '#A0A0A0',      # Light gray for second competitor
    'competitor3': '#606060',      # Dark gray for third competitor
    'highlight': '#2ECC71',        # Green for advantages
    'warning': '#E74C3C',          # Red for disadvantages
    'neutral': '#F39C12',          # Orange for neutral
    'background': '#F8F9FA',       # Light background
    'text': '#2C3E50',             # Dark text
    'grid': '#E0E0E0',             # Grid lines
}

# Color palette for multiple competitors
COMPETITOR_PALETTE = [
    '#888888',  # Gray
    '#A0A0A0',  # Light gray
    '#606060',  # Dark gray
    '#B8B8B8',  # Lighter gray
    '#707070',  # Medium gray
]


def setup_style() -> None:
    """Apply consistent SynaDB styling to matplotlib.
    
    Call this at the start of each notebook to ensure consistent styling.
    
    Example:
        >>> from utils.charts import setup_style
        >>> setup_style()
    """
    # Use a clean style as base
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')
    
    # Apply SynaDB customizations
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        
        # Title and label sizes
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Figure settings
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['grid'],
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Grid settings
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['grid'],
        
        # Save settings
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def get_color(name: str, is_synadb: bool = False) -> str:
    """Get appropriate color for a system.
    
    Args:
        name: System name
        is_synadb: Whether this is SynaDB (uses brand color)
        
    Returns:
        Hex color string
    """
    if is_synadb or 'synadb' in name.lower() or 'syna' in name.lower():
        return COLORS['synadb']
    return COLORS['competitor']


def bar_comparison(
    data: Dict[str, float],
    title: str,
    ylabel: str,
    highlight_synadb: bool = True,
    lower_is_better: bool = True,
    figsize: tuple = (10, 6),
    show_values: bool = True,
    value_format: str = '{:.2f}'
) -> plt.Figure:
    """Create bar chart comparing systems.
    
    Args:
        data: Dictionary mapping system names to values
        title: Chart title
        ylabel: Y-axis label
        highlight_synadb: Whether to highlight SynaDB in brand color
        lower_is_better: Whether lower values are better (affects highlighting)
        figsize: Figure size tuple
        show_values: Whether to show values on bars
        value_format: Format string for values
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> data = {'SynaDB': 1.5, 'Chroma': 3.2, 'FAISS': 2.1}
        >>> fig = bar_comparison(data, 'Search Latency', 'Latency (ms)')
        >>> plt.show()
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(data.keys())
    values = list(data.values())
    
    # Determine colors
    colors = []
    for name in names:
        if highlight_synadb and ('synadb' in name.lower() or 'syna' in name.lower()):
            colors.append(COLORS['synadb'])
        else:
            colors.append(COLORS['competitor'])
    
    # Create bars
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Highlight best performer
    if lower_is_better:
        best_idx = values.index(min(values))
    else:
        best_idx = values.index(max(values))
    
    # Add subtle highlight to best bar
    bars[best_idx].set_edgecolor(COLORS['highlight'])
    bars[best_idx].set_linewidth(2.5)
    
    # Add value labels on bars
    if show_values:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                value_format.format(val),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color=COLORS['text']
            )
    
    # Styling
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylim(0, max(values) * 1.15)  # Add headroom for labels
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def latency_distribution(
    results: List[BenchmarkResult],
    title: str = 'Latency Distribution',
    figsize: tuple = (12, 6),
    show_outliers: bool = True
) -> plt.Figure:
    """Create latency distribution comparison using box plots.
    
    Args:
        results: List of BenchmarkResult objects with raw_times
        title: Chart title
        figsize: Figure size tuple
        show_outliers: Whether to show outlier points
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> results = [synadb_result, chroma_result, faiss_result]
        >>> fig = latency_distribution(results, 'Search Latency Distribution')
        >>> plt.show()
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    names = [r.name for r in results]
    data = [r.raw_times if hasattr(r, 'raw_times') and r.raw_times else [r.mean_ms] for r in results]
    
    # Determine colors
    colors = []
    for name in names:
        if 'synadb' in name.lower() or 'syna' in name.lower():
            colors.append(COLORS['synadb'])
        else:
            colors.append(COLORS['competitor'])
    
    # Create box plot
    bp = ax.boxplot(
        data,
        labels=names,
        patch_artist=True,
        showfliers=show_outliers,
        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5}
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style median lines
    for median in bp['medians']:
        median.set_color(COLORS['text'])
        median.set_linewidth(2)
    
    # Styling
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid for y-axis only
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    
    plt.tight_layout()
    return fig


def throughput_comparison(
    data: Dict[str, float],
    title: str = 'Throughput Comparison',
    ylabel: str = 'Operations/sec',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """Create horizontal bar chart for throughput comparison.
    
    Higher values are better for throughput, so this uses a horizontal
    layout that naturally emphasizes larger bars.
    
    Args:
        data: Dictionary mapping system names to throughput values
        title: Chart title
        ylabel: Y-axis label (actually x-axis for horizontal bars)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(data.keys())
    values = list(data.values())
    
    # Determine colors
    colors = []
    for name in names:
        if 'synadb' in name.lower() or 'syna' in name.lower():
            colors.append(COLORS['synadb'])
        else:
            colors.append(COLORS['competitor'])
    
    # Create horizontal bars
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Highlight best performer (highest throughput)
    best_idx = values.index(max(values))
    bars[best_idx].set_edgecolor(COLORS['highlight'])
    bars[best_idx].set_linewidth(2.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f' {val:,.0f}',
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold',
            color=COLORS['text']
        )
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlim(0, max(values) * 1.2)  # Add headroom for labels
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def memory_comparison(
    data: Dict[str, float],
    title: str = 'Memory Usage Comparison',
    ylabel: str = 'Memory (MB)',
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """Create bar chart for memory usage comparison.
    
    Lower memory usage is better, so this highlights the lowest value.
    
    Args:
        data: Dictionary mapping system names to memory values (in MB)
        title: Chart title
        ylabel: Y-axis label
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    return bar_comparison(
        data=data,
        title=title,
        ylabel=ylabel,
        highlight_synadb=True,
        lower_is_better=True,
        figsize=figsize,
        value_format='{:.1f}'
    )


def create_legend_patch(label: str, color: str) -> plt.Rectangle:
    """Create a legend patch for custom legends.
    
    Args:
        label: Legend label
        color: Hex color string
        
    Returns:
        matplotlib Rectangle patch
    """
    return plt.Rectangle((0, 0), 1, 1, fc=color, label=label)

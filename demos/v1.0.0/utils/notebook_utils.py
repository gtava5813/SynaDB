"""
Notebook utilities for SynaDB v1.0.0 Showcase.

Provides consistent notebook formatting and branding:
- SynaDB branded headers
- Table of contents generation
- Advantage highlighting
- Dependency checking with graceful degradation
"""

from typing import List, Tuple, Optional, Dict, Any
import importlib
import sys


def display_header(
    title: str = "SynaDB v1.0.0 Showcase",
    subtitle: str = "The SQLite of AI"
) -> None:
    """Display SynaDB branded header.
    
    Creates a visually appealing header with SynaDB branding
    for consistent notebook presentation.
    
    Args:
        title: Main title text
        subtitle: Subtitle text
        
    Example:
        >>> display_header()
        >>> display_header("Vector Store Comparison", "SynaDB vs Chroma vs FAISS")
    """
    html = f"""
    <div style="background: linear-gradient(135deg, #4A90D9 0%, #357ABD 100%); 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            🗄️ {title}
        </h1>
        <p style="color: #E8F4FD; margin: 5px 0 0 0; font-size: 1.1em; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            {subtitle}
        </p>
    </div>
    """
    
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        # Fallback for non-notebook environments
        print(f"\n{'='*60}")
        print(f"🗄️ {title}")
        print(f"   {subtitle}")
        print(f"{'='*60}\n")


def generate_toc(sections: List[Tuple[str, str]]) -> str:
    """Generate clickable table of contents.
    
    Creates a markdown table of contents with anchor links
    for easy navigation within the notebook.
    
    Args:
        sections: List of (title, anchor) tuples
        
    Returns:
        Markdown string with clickable TOC
        
    Example:
        >>> sections = [
        ...     ("Introduction", "introduction"),
        ...     ("Setup", "setup"),
        ...     ("Benchmarks", "benchmarks"),
        ...     ("Conclusions", "conclusions"),
        ... ]
        >>> toc = generate_toc(sections)
        >>> print(toc)
    """
    toc = "## 📑 Table of Contents\n\n"
    
    for i, (title, anchor) in enumerate(sections, 1):
        toc += f"{i}. [{title}](#{anchor})\n"
    
    return toc


def display_toc(sections: List[Tuple[str, str]]) -> None:
    """Display table of contents in notebook.
    
    Args:
        sections: List of (title, anchor) tuples
    """
    toc = generate_toc(sections)
    
    try:
        from IPython.display import display, Markdown
        display(Markdown(toc))
    except ImportError:
        print(toc)


def highlight_advantage(text: str, color: str = "#2ECC71") -> str:
    """Highlight SynaDB advantage with visual indicator.
    
    Creates an HTML span with green checkmark and bold text
    to highlight SynaDB advantages in comparisons.
    
    Args:
        text: Text to highlight
        color: Hex color for the highlight (default: green)
        
    Returns:
        HTML string with highlighted text
        
    Example:
        >>> html = highlight_advantage("10x faster than competitors")
        >>> display(HTML(html))
    """
    return f'<span style="color: {color}; font-weight: bold;">✓ {text}</span>'


def highlight_disadvantage(text: str, color: str = "#E74C3C") -> str:
    """Highlight a disadvantage with visual indicator.
    
    Args:
        text: Text to highlight
        color: Hex color for the highlight (default: red)
        
    Returns:
        HTML string with highlighted text
    """
    return f'<span style="color: {color}; font-weight: bold;">✗ {text}</span>'


def highlight_neutral(text: str, color: str = "#F39C12") -> str:
    """Highlight neutral information with visual indicator.
    
    Args:
        text: Text to highlight
        color: Hex color for the highlight (default: orange)
        
    Returns:
        HTML string with highlighted text
    """
    return f'<span style="color: {color}; font-weight: bold;">○ {text}</span>'


def check_dependency(
    name: str,
    install_cmd: Optional[str] = None,
    package_name: Optional[str] = None,
    required: bool = True
) -> bool:
    """Check if dependency is available, show install instructions if not.
    
    Provides graceful degradation when optional dependencies are missing.
    Shows helpful installation instructions to the user.
    
    Args:
        name: Module name to import
        install_cmd: Custom install command (default: pip install {package_name})
        package_name: Package name for pip (default: same as name)
        required: If True, show warning message when missing. If False, silently return False.
        
    Returns:
        True if dependency is available, False otherwise
        
    Example:
        >>> if check_dependency("chromadb"):
        ...     import chromadb
        ...     # Run Chroma comparison
        ... else:
        ...     print("Skipping Chroma comparison")
    """
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        if not required:
            # Silently return False for optional dependencies
            return False
            
        pkg = package_name or name
        cmd = install_cmd or f"pip install {pkg}"
        
        message = f"""
⚠️ **Optional dependency `{name}` not found.**

To enable this comparison, install with:
```bash
{cmd}
```

Skipping `{name}` comparisons...
        """
        
        try:
            from IPython.display import display, Markdown
            display(Markdown(message))
        except ImportError:
            print(f"\n⚠️ Optional dependency '{name}' not found.")
            print(f"   Install with: {cmd}")
            print(f"   Skipping {name} comparisons...\n")
        
        return False


def check_dependencies(
    dependencies: Dict[str, Optional[str]]
) -> Dict[str, bool]:
    """Check multiple dependencies at once.
    
    Args:
        dependencies: Dict mapping module names to install commands
        
    Returns:
        Dict mapping module names to availability status
        
    Example:
        >>> deps = {
        ...     "chromadb": "pip install chromadb",
        ...     "faiss": "pip install faiss-cpu",
        ... }
        >>> available = check_dependencies(deps)
        >>> if available["chromadb"]:
        ...     # Run Chroma tests
    """
    results = {}
    for name, install_cmd in dependencies.items():
        results[name] = check_dependency(name, install_cmd)
    return results


def comparison_table(
    data: List[Dict[str, Any]],
    highlight_col: str = 'SynaDB',
    highlight_best: bool = True
) -> str:
    """Generate comparison table with highlighting.
    
    Creates a markdown table comparing features or metrics
    across different systems.
    
    Args:
        data: List of dicts with comparison data
        highlight_col: Column name to highlight (default: SynaDB)
        highlight_best: Whether to highlight best values
        
    Returns:
        Markdown table string
        
    Example:
        >>> data = [
        ...     {"Feature": "Embedded", "SynaDB": "✓", "Chroma": "✓", "Milvus": "✗"},
        ...     {"Feature": "Single File", "SynaDB": "✓", "Chroma": "✗", "Milvus": "✗"},
        ... ]
        >>> table = comparison_table(data)
    """
    if not data:
        return "No data to display."
    
    # Get column headers
    headers = list(data[0].keys())
    
    # Build table
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|"
    ]
    
    for row in data:
        values = []
        for h in headers:
            val = str(row.get(h, ""))
            # Highlight SynaDB column
            if h == highlight_col:
                val = f"**{val}**"
            values.append(val)
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def section_header(title: str, anchor: str, emoji: str = "📌") -> str:
    """Create a section header with anchor.
    
    Args:
        title: Section title
        anchor: Anchor ID for linking
        emoji: Emoji prefix (default: 📌)
        
    Returns:
        Markdown header string
    """
    return f'## {emoji} {title} <a id="{anchor}"></a>'


def display_section(title: str, anchor: str, emoji: str = "📌") -> None:
    """Display a section header in notebook.
    
    Args:
        title: Section title
        anchor: Anchor ID for linking
        emoji: Emoji prefix
    """
    header = section_header(title, anchor, emoji)
    
    try:
        from IPython.display import display, Markdown
        display(Markdown(header))
    except ImportError:
        print(f"\n{emoji} {title}")
        print("-" * (len(title) + 4))


def conclusion_box(
    title: str = "Key Takeaways",
    points: List[str] = None,
    summary: str = None
) -> None:
    """Display a conclusion box with key takeaways.
    
    Args:
        title: Box title
        points: List of bullet points
        summary: Optional summary paragraph
    """
    points = points or []
    
    html = f"""
    <div style="background: #E8F4FD; 
                border-left: 4px solid #4A90D9; 
                padding: 15px 20px; 
                border-radius: 0 8px 8px 0;
                margin: 20px 0;">
        <h3 style="color: #357ABD; margin: 0 0 10px 0;">🎯 {title}</h3>
    """
    
    if points:
        html += "<ul style='margin: 0; padding-left: 20px;'>"
        for point in points:
            html += f"<li style='margin: 5px 0;'>{point}</li>"
        html += "</ul>"
    
    if summary:
        html += f"<p style='margin: 10px 0 0 0; font-style: italic;'>{summary}</p>"
    
    html += "</div>"
    
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        print(f"\n🎯 {title}")
        for point in points:
            print(f"  • {point}")
        if summary:
            print(f"\n  {summary}")
        print()


def info_box(message: str, title: str = "Info") -> None:
    """Display an info box.
    
    Args:
        message: Message to display
        title: Box title
    """
    html = f"""
    <div style="background: #FFF3CD; 
                border-left: 4px solid #F39C12; 
                padding: 10px 15px; 
                border-radius: 0 8px 8px 0;
                margin: 10px 0;">
        <strong>ℹ️ {title}:</strong> {message}
    </div>
    """
    
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        print(f"ℹ️ {title}: {message}")


def warning_box(message: str, title: str = "Warning") -> None:
    """Display a warning box.
    
    Args:
        message: Warning message
        title: Box title
    """
    html = f"""
    <div style="background: #F8D7DA; 
                border-left: 4px solid #E74C3C; 
                padding: 10px 15px; 
                border-radius: 0 8px 8px 0;
                margin: 10px 0;">
        <strong>⚠️ {title}:</strong> {message}
    </div>
    """
    
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        print(f"⚠️ {title}: {message}")

# Author: gadwant
"""
Jupyter notebook integration.

Rich display of complexity analysis results in Jupyter notebooks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis


def _repr_html_(analysis: "Analysis") -> str:
    """Generate HTML representation for Jupyter notebooks."""
    # Build measurements table
    rows = ""
    for m in analysis.measurements:
        mem_col = f"<td>{m.memory_bytes:,}</td>" if m.memory_bytes else "<td>-</td>"
        rows += f"""
        <tr>
            <td>{m.size:,}</td>
            <td>{m.seconds:.6f}</td>
            <td>±{m.std_dev:.6f}</td>
            {mem_col}
        </tr>"""
    
    # Build fits table
    fits_rows = ""
    for f in analysis.fits[:5]:
        marker = "★" if f.label == analysis.best_label else ""
        style = 'background:#e8f5e9;' if marker else ''
        fits_rows += f"""
        <tr style="{style}">
            <td>{f.label} {marker}</td>
            <td>{f.error:.4f}</td>
        </tr>"""
    
    space_html = ""
    if analysis.space_label:
        space_html = f"""
        <div style="margin:5px 0;">
            <span style="color:#666;">Space Complexity:</span>
            <span style="color:#1a73e8;font-weight:bold;font-size:1.2em;margin-left:10px;">{analysis.space_label}</span>
        </div>"""
    
    return f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;padding:15px;background:#f8f9fa;border-radius:8px;max-width:600px;">
        <div style="border-bottom:2px solid #1a73e8;padding-bottom:10px;margin-bottom:15px;">
            <span style="color:#666;">Time Complexity:</span>
            <span style="color:#1a73e8;font-weight:bold;font-size:1.4em;margin-left:10px;">{analysis.best_label}</span>
            {space_html}
        </div>
        
        <details open>
            <summary style="cursor:pointer;font-weight:bold;margin-bottom:10px;">📊 Measurements</summary>
            <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
                <thead>
                    <tr style="background:#e8e8e8;">
                        <th style="padding:8px;text-align:left;">Size</th>
                        <th style="padding:8px;text-align:left;">Time (s)</th>
                        <th style="padding:8px;text-align:left;">Std Dev</th>
                        <th style="padding:8px;text-align:left;">Memory</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </details>
        
        <details style="margin-top:15px;">
            <summary style="cursor:pointer;font-weight:bold;margin-bottom:10px;">📈 Complexity Fits</summary>
            <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
                <thead>
                    <tr style="background:#e8e8e8;">
                        <th style="padding:8px;text-align:left;">Class</th>
                        <th style="padding:8px;text-align:left;">Error</th>
                    </tr>
                </thead>
                <tbody>{fits_rows}</tbody>
            </table>
        </details>
    </div>
    """


def enable_jupyter_display() -> None:
    """
    Enable rich Jupyter display for Analysis objects.
    
    Call this once to enable automatic rich display of Analysis
    objects in Jupyter notebooks.
    
    Example:
        >>> from bigocheck.jupyter import enable_jupyter_display
        >>> enable_jupyter_display()
        >>> 
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> analysis  # Will show rich HTML in Jupyter
    """
    from .core import Analysis
    
    # Add _repr_html_ method to Analysis class
    Analysis._repr_html_ = lambda self: _repr_html_(self)


def display_analysis(analysis: "Analysis") -> None:
    """
    Display Analysis with rich formatting in Jupyter.
    
    Works even without calling enable_jupyter_display().
    
    Example:
        >>> from bigocheck.jupyter import display_analysis
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> display_analysis(analysis)
    """
    try:
        from IPython.display import display, HTML
        display(HTML(_repr_html_(analysis)))
    except ImportError:
        # Not in Jupyter, just print
        print(f"Time Complexity: {analysis.best_label}")
        if analysis.space_label:
            print(f"Space Complexity: {analysis.space_label}")


def display_comparison(
    analyses: dict,
    title: str = "Complexity Comparison",
) -> None:
    """
    Display multiple analyses side-by-side in Jupyter.
    
    Args:
        analyses: Dict mapping names to Analysis objects.
        title: Comparison title.
    
    Example:
        >>> display_comparison({
        ...     "bubble_sort": analysis1,
        ...     "quick_sort": analysis2,
        ... })
    """
    rows = ""
    for name, analysis in analyses.items():
        rows += f"""
        <tr>
            <td style="padding:10px;font-weight:bold;">{name}</td>
            <td style="padding:10px;color:#1a73e8;font-weight:bold;">{analysis.best_label}</td>
            <td style="padding:10px;">{analysis.space_label or '-'}</td>
            <td style="padding:10px;">{analysis.fits[0].error:.4f}</td>
        </tr>"""
    
    html = f"""
    <div style="font-family:system-ui;padding:15px;background:#f8f9fa;border-radius:8px;">
        <h3 style="margin-top:0;color:#333;">{title}</h3>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#e8e8e8;">
                    <th style="padding:10px;text-align:left;">Function</th>
                    <th style="padding:10px;text-align:left;">Time</th>
                    <th style="padding:10px;text-align:left;">Space</th>
                    <th style="padding:10px;text-align:left;">Error</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """
    
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        for name, analysis in analyses.items():
            print(f"{name}: {analysis.best_label}")

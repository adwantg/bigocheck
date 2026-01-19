# Author: gadwant
"""
Export functionality for complexity analysis results.

Export to CSV, markdown, and other formats.
"""
from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from .core import Analysis


def to_csv(
    analysis: "Analysis",
    path: Optional[str] = None,
    include_fits: bool = True,
) -> str:
    """
    Export analysis results to CSV format.
    
    Args:
        analysis: Analysis object from benchmark_function.
        path: Optional path to save CSV file.
        include_fits: Include complexity fits in output.
    
    Returns:
        CSV string (also saves to file if path provided).
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> csv_str = to_csv(analysis, "results.csv")
    """
    output = io.StringIO()
    
    # Measurements section
    writer = csv.writer(output)
    writer.writerow(["# Measurements"])
    writer.writerow(["Size", "Time (s)", "Std Dev", "Memory (bytes)"])
    
    for m in analysis.measurements:
        writer.writerow([
            m.size,
            f"{m.seconds:.9f}",
            f"{m.std_dev:.9f}",
            m.memory_bytes or "",
        ])
    
    writer.writerow([])
    
    # Summary section
    writer.writerow(["# Summary"])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Time Complexity", analysis.best_label])
    if analysis.space_label:
        writer.writerow(["Space Complexity", analysis.space_label])
    
    if include_fits and analysis.fits:
        writer.writerow([])
        writer.writerow(["# Complexity Fits"])
        writer.writerow(["Class", "Error", "Scale"])
        for f in analysis.fits:
            writer.writerow([f.label, f"{f.error:.6f}", f"{f.scale:.9g}"])
    
    csv_str = output.getvalue()
    
    if path:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_str)
    
    return csv_str


def to_markdown_table(analysis: "Analysis") -> str:
    """
    Export analysis as a markdown table.
    
    Args:
        analysis: Analysis object.
    
    Returns:
        Markdown string.
    """
    lines = [
        f"**Time Complexity:** {analysis.best_label}",
    ]
    
    if analysis.space_label:
        lines.append(f"**Space Complexity:** {analysis.space_label}")
    
    lines.extend([
        "",
        "| Size | Time (s) | Std Dev | Memory |",
        "|------|----------|---------|--------|",
    ])
    
    for m in analysis.measurements:
        mem = f"{m.memory_bytes:,}" if m.memory_bytes else "-"
        lines.append(f"| {m.size:,} | {m.seconds:.6f} | ±{m.std_dev:.6f} | {mem} |")
    
    return "\n".join(lines)


def to_dict(analysis: "Analysis") -> Dict[str, Any]:
    """
    Convert analysis to a dictionary.
    
    Useful for JSON serialization or further processing.
    
    Args:
        analysis: Analysis object.
    
    Returns:
        Dictionary representation.
    """
    return {
        "time_complexity": analysis.best_label,
        "space_complexity": analysis.space_label,
        "measurements": [
            {
                "size": m.size,
                "seconds": m.seconds,
                "std_dev": m.std_dev,
                "memory_bytes": m.memory_bytes,
            }
            for m in analysis.measurements
        ],
        "fits": [
            {
                "label": f.label,
                "error": f.error,
                "scale": f.scale,
            }
            for f in analysis.fits
        ],
    }


def to_json(analysis: "Analysis", path: Optional[str] = None, indent: int = 2) -> str:
    """
    Export analysis to JSON format.
    
    Args:
        analysis: Analysis object.
        path: Optional path to save JSON file.
        indent: JSON indentation.
    
    Returns:
        JSON string.
    """
    import json
    
    data = to_dict(analysis)
    json_str = json.dumps(data, indent=indent)
    
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    return json_str


def to_dataframe(analysis: "Analysis"):
    """
    Convert analysis measurements to a pandas DataFrame.
    
    Requires pandas to be installed.
    
    Args:
        analysis: Analysis object.
    
    Returns:
        pandas DataFrame.
    
    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
    
    data = []
    for m in analysis.measurements:
        data.append({
            "size": m.size,
            "time_seconds": m.seconds,
            "std_dev": m.std_dev,
            "memory_bytes": m.memory_bytes,
        })
    
    df = pd.DataFrame(data)
    df.attrs["time_complexity"] = analysis.best_label
    df.attrs["space_complexity"] = analysis.space_label
    
    return df

# Author: gadwant
"""
HTML report generation.

Generate beautiful HTML reports for complexity analysis.
"""
from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis


def _escape(text: str) -> str:
    """Escape HTML entities."""
    return html.escape(str(text))


def _format_scale(scale: float) -> str:
    """Format scale in human-readable units."""
    if scale < 0:
        return f"{scale:.4g}"
    
    # Time units
    if scale >= 1:
        return f"{scale:.2f} s/elem"
    elif scale >= 1e-3:
        return f"{scale * 1e3:.2f} ms/elem"
    elif scale >= 1e-6:
        return f"{scale * 1e6:.2f} µs/elem"
    elif scale >= 1e-9:
        return f"{scale * 1e9:.2f} ns/elem"
    else:
        return f"{scale:.2e}"


def generate_html_report(
    analysis: "Analysis",
    *,
    title: str = "Complexity Analysis Report",
    include_chart: bool = True,
) -> str:
    """
    Generate an HTML report for complexity analysis.
    
    Args:
        analysis: Analysis object from benchmark_function.
        title: Report title.
        include_chart: Include SVG chart of measurements.
    
    Returns:
        HTML string.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> html_content = generate_html_report(analysis)
        >>> with open("report.html", "w") as f:
        ...     f.write(html_content)
    """
    # Build measurements table
    measurements_rows = ""
    for m in analysis.measurements:
        mem_col = f"<td>{m.memory_bytes:,}</td>" if m.memory_bytes else "<td>-</td>"
        measurements_rows += f"""
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
        fits_rows += f"""
        <tr class="{'best' if marker else ''}">
            <td>{_escape(f.label)} {marker}</td>
            <td>{f.error:.4f}</td>
            <td>{_format_scale(f.scale)}</td>
        </tr>"""
    
    # Generate simple SVG chart
    chart_svg = ""
    if include_chart and analysis.measurements:
        max_time = max(m.seconds for m in analysis.measurements)
        max_size = max(m.size for m in analysis.measurements)
        
        points = []
        for m in analysis.measurements:
            x = (m.size / max_size) * 280 + 40
            y = 160 - (m.seconds / max_time) * 140
            points.append(f"{x},{y}")
        
        chart_svg = f"""
        <svg width="350" height="200" style="background:#f8f9fa;border-radius:8px;">
            <text x="175" y="20" text-anchor="middle" font-size="12" fill="#333">Time vs Input Size</text>
            <line x1="40" y1="20" x2="40" y2="160" stroke="#ccc" stroke-width="1"/>
            <line x1="40" y1="160" x2="320" y2="160" stroke="#ccc" stroke-width="1"/>
            <polyline points="{' '.join(points)}" fill="none" stroke="#4285f4" stroke-width="2"/>
            {"".join(f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="4" fill="#4285f4"/>' for p in points)}
            <text x="40" y="180" font-size="10" fill="#666">0</text>
            <text x="320" y="180" font-size="10" fill="#666">{max_size:,}</text>
        </svg>
        """
    
    space_section = ""
    if analysis.space_label:
        space_section = f"""
        <div class="metric">
            <span class="label">Space Complexity</span>
            <span class="value">{_escape(analysis.space_label)}</span>
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #1a73e8; margin-bottom: 0.5em; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .label {{ color: #666; font-weight: 500; }}
        .value {{ font-weight: 700; color: #1a73e8; font-size: 1.2em; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
        tr.best {{ background: #e8f5e9; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>📊 {_escape(title)}</h1>
    
    <div class="card">
        <h2>Results</h2>
        <div class="metric">
            <span class="label">Time Complexity</span>
            <span class="value">{_escape(analysis.best_label)}</span>
        </div>
        {space_section}
    </div>
    
    <div class="card">
        <h2>Chart</h2>
        <div class="chart-container">
            {chart_svg}
        </div>
    </div>
    
    <div class="card">
        <h2>Measurements</h2>
        <table>
            <thead>
                <tr>
                    <th>Size</th>
                    <th>Time (s)</th>
                    <th>Std Dev</th>
                    <th>Memory</th>
                </tr>
            </thead>
            <tbody>
                {measurements_rows}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>Complexity Fits</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Error</th>
                    <th>Scale</th>
                </tr>
            </thead>
            <tbody>
                {fits_rows}
            </tbody>
        </table>
    </div>
    
    <footer>
        Generated by bigocheck • Zero-dependency complexity analysis
    </footer>
</body>
</html>"""


def save_html_report(html_content: str, path: str) -> None:
    """Save HTML report to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)

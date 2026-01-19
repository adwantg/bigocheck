# Author: gadwant
"""
Static Web Dashboard Generator for bigocheck.
Creates a folder with static HTML/JS files to visualize benchmarks.
Zero dependencies: Uses string templates, no Jinja2 or Flask required.
"""
import os
import datetime
from typing import List

from .core import Analysis

# Lightweight CSS/JS embedded to avoid external requests if needed
CSS_TEMPLATE = """
<style>
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }
    h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid #eee; }
    th { background: #f8f9fa; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold; }
    .O-1 { background: #e6fffa; color: #00a880; }
    .O-logn { background: #effcf6; color: #00a880; }
    .O-n { background: #fff8e1; color: #f59f00; }
    .O-nlogn { background: #fff3e0; color: #f59f00; }
    .O-n2 { background: #fff5f5; color: #e03131; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>bigocheck Dashboard</title>
    {css}
</head>
<body>
    <h1>🚀 bigocheck Dashboard</h1>
    <p>Last updated: {date}</p>
    
    <div class="card">
        <h2>Recent Benchmarks</h2>
        <table>
            <thead>
                <tr>
                    <th>Function</th>
                    <th>Complexity</th>
                    <th>Confidence</th>
                    <th>Last Run</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Trends</h2>
        <p><i>Visualization of complexity history (placeholder for chart.js implementation)</i></p>
    </div>
</body>
</html>
"""

def _get_badge_class(label: str) -> str:
    clean = label.replace("(", "").replace(")", "").replace(" ", "").replace("^","")
    return f"O-{clean}" if clean in ["1", "logn", "n", "nlogn", "n2"] else "O-n2"

def generate_dashboard(analyses: List[Analysis], output_dir: str = "dashboard"):
    """
    Generate a static HTML dashboard from a list of Analysis objects.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Generate Index
    rows = []
    for a in analyses:
        badge_cls = _get_badge_class(a.best_label)
        row = f"""
        <tr>
            <td><strong>{a.name or "Unknown"}</strong></td>
            <td><span class="badge {badge_cls}">{a.best_label}</span></td>
            <td>High</td>
            <td>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</td>
        </tr>
        """
        rows.append(row)

    html = INDEX_TEMPLATE.format(
        css=CSS_TEMPLATE,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rows="\n".join(rows)
    )

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Dashboard generated at: {output_dir}/index.html")

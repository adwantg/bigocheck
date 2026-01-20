# Author: gadwant
"""
Badge generation for complexity results.

Generate SVG badges for READMEs and documentation.
"""
from __future__ import annotations

from typing import Optional


# Color schemes based on complexity
COMPLEXITY_COLORS = {
    "O(1)": "#4c1",       # Bright green - excellent
    "O(log n)": "#97ca00", # Green - very good
    "O(√n)": "#a4a61d",    # Yellow-green - good
    "O(n)": "#dfb317",     # Yellow - acceptable
    "O(n log n)": "#fe7d37", # Orange - moderate
    "O(n^2)": "#e05d44",   # Red-orange - concerning
    "O(n^3)": "#cb2431",   # Red - bad
    "O(2^n)": "#b60205",   # Dark red - very bad
    "O(n!)": "#8b0000",    # Darkest red - worst
}

DEFAULT_COLOR = "#555"  # Gray for unknown


def _get_color(complexity: str) -> str:
    """Get color for a complexity class."""
    # Exact match
    if complexity in COMPLEXITY_COLORS:
        return COMPLEXITY_COLORS[complexity]
    
    # Partial match for polynomial
    if complexity.startswith("O(n^"):
        try:
            # Extract exponent
            exp_str = complexity[4:-1]
            exp = float(exp_str)
            if exp <= 1.5:
                return COMPLEXITY_COLORS["O(n)"]
            elif exp <= 2.5:
                return COMPLEXITY_COLORS["O(n^2)"]
            elif exp <= 3.5:
                return COMPLEXITY_COLORS["O(n^3)"]
            else:
                return COMPLEXITY_COLORS["O(2^n)"]
        except ValueError:
            pass
    
    return DEFAULT_COLOR


def generate_badge(
    complexity: str,
    *,
    label: str = "complexity",
    style: str = "flat",
) -> str:
    """
    Generate an SVG badge for a complexity class.
    
    Args:
        complexity: Complexity label (e.g., "O(n log n)").
        label: Left side text (default: "complexity").
        style: Badge style ("flat" or "flat-square").
    
    Returns:
        SVG string.
    
    Example:
        >>> svg = generate_badge("O(n log n)")
        >>> with open("complexity_badge.svg", "w") as f:
        ...     f.write(svg)
    """
    color = _get_color(complexity)
    
    # Calculate widths
    label_width = len(label) * 7 + 10
    value_width = len(complexity) * 7 + 10
    total_width = label_width + value_width
    
    # Border radius based on style
    radius = 0 if style == "flat-square" else 3
    
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label}: {complexity}">
  <title>{label}: {complexity}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="{radius}" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11">
    <text x="{label_width // 2}" y="14" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width // 2}" y="13" fill="#fff">{label}</text>
    <text x="{label_width + value_width // 2}" y="14" fill="#010101" fill-opacity=".3">{complexity}</text>
    <text x="{label_width + value_width // 2}" y="13" fill="#fff">{complexity}</text>
  </g>
</svg>'''


def generate_dual_badge(
    time_complexity: str,
    space_complexity: Optional[str] = None,
) -> str:
    """
    Generate a badge showing both time and space complexity.
    
    Args:
        time_complexity: Time complexity label.
        space_complexity: Space complexity label (optional).
    
    Returns:
        SVG string.
    """
    if not space_complexity:
        return generate_badge(time_complexity)
    
    time_color = _get_color(time_complexity)
    _space_color = _get_color(space_complexity)  # noqa: F841 - Reserved for future dual-color badge
    
    # Combined badge
    label = "complexity"
    value = f"time {time_complexity} | space {space_complexity}"
    
    label_width = len(label) * 7 + 10
    value_width = len(value) * 6 + 10
    total_width = label_width + value_width
    
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img">
  <title>Time: {time_complexity}, Space: {space_complexity}</title>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{time_color}"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width // 2}" y="14">{label}</text>
    <text x="{label_width + value_width // 2}" y="14">{value}</text>
  </g>
</svg>'''


def save_badge(svg: str, path: str) -> None:
    """Save SVG badge to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(svg)


def generate_badge_url(complexity: str, style: str = "flat") -> str:
    """
    Generate a shields.io URL for the badge.
    
    Useful when you want to use shields.io CDN instead of local SVG.
    
    Args:
        complexity: Complexity label.
        style: Badge style.
    
    Returns:
        shields.io URL.
    """
    color = _get_color(complexity).lstrip('#')
    # URL encode the complexity
    encoded = complexity.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"https://img.shields.io/badge/complexity-{encoded}-{color}?style={style}"

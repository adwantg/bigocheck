# Author: gadwant
"""
Report generation utilities for bigocheck.
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis
    from .compare import ComparisonResult
    from .assertions import VerificationResult


def generate_report(
    analysis: "Analysis",
    *,
    title: str = "Complexity Analysis Report",
    include_measurements: bool = True,
    include_all_fits: bool = True,
) -> str:
    """
    Generate a markdown report from an analysis.
    
    Args:
        analysis: Analysis object from benchmark_function.
        title: Report title.
        include_measurements: Include measurement table.
        include_all_fits: Include all complexity fits.
    
    Returns:
        Markdown formatted report string.
    
    Example:
        >>> report = generate_report(analysis, title="Sort Analysis")
        >>> print(report)
    """
    from .assertions import compute_confidence
    
    confidence = compute_confidence(analysis)
    lines = []
    
    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Best Fit** | `{analysis.best_label}` |")
    lines.append(f"| **Confidence** | {confidence.level} ({confidence.score:.0%}) |")
    lines.append(f"| **Measurements** | {len(analysis.measurements)} |")
    
    if analysis.measurements:
        sizes = [m.size for m in analysis.measurements]
        lines.append(f"| **Size Range** | {min(sizes):,} - {max(sizes):,} |")
    
    lines.append("")
    
    # Measurements
    if include_measurements:
        lines.append("## Measurements")
        lines.append("")
        
        has_memory = any(m.memory_bytes for m in analysis.measurements)
        
        if has_memory:
            lines.append("| Size | Time (s) | Std Dev | Memory |")
            lines.append("|------|----------|---------|--------|")
            for m in analysis.measurements:
                mem = f"{m.memory_bytes:,} B" if m.memory_bytes else "N/A"
                lines.append(f"| {m.size:,} | {m.seconds:.6f} | ±{m.std_dev:.6f} | {mem} |")
        else:
            lines.append("| Size | Time (s) | Std Dev |")
            lines.append("|------|----------|---------|")
            for m in analysis.measurements:
                lines.append(f"| {m.size:,} | {m.seconds:.6f} | ±{m.std_dev:.6f} |")
        
        lines.append("")
    
    # Fits
    if include_all_fits:
        lines.append("## Complexity Fits")
        lines.append("")
        lines.append("| Rank | Class | Error | Scale |")
        lines.append("|------|-------|-------|-------|")
        for i, f in enumerate(analysis.fits, 1):
            marker = " ⭐" if f.label == analysis.best_label else ""
            lines.append(f"| {i} | `{f.label}`{marker} | {f.error:.4f} | {f.scale:.6g} |")
        lines.append("")
    
    # Confidence details
    lines.append("## Confidence Analysis")
    lines.append("")
    for reason in confidence.reasons:
        lines.append(f"- {reason}")
    lines.append("")
    
    return "\n".join(lines)


def generate_comparison_report(
    result: "ComparisonResult",
    *,
    title: str = "Function Comparison Report",
) -> str:
    """
    Generate a markdown report from a comparison result.
    
    Args:
        result: ComparisonResult from compare_functions.
        title: Report title.
    
    Returns:
        Markdown formatted report string.
    """
    lines = []
    
    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"> {result.summary}")
    lines.append("")
    
    lines.append("| Function | Complexity |")
    lines.append("|----------|------------|")
    lines.append(f"| `{result.func_a_name}` | `{result.func_a_label}` |")
    lines.append(f"| `{result.func_b_name}` | `{result.func_b_label}` |")
    lines.append("")
    
    if result.winner:
        winner_name = result.func_a_name if result.winner == "func_a" else result.func_b_name
        lines.append(f"**Winner**: `{winner_name}` ({result.speedup:.2f}x faster)")
    else:
        lines.append("**Result**: Tie")
    lines.append("")
    
    if result.crossover_point:
        lines.append(f"**Crossover Point**: n = {result.crossover_point:,}")
        lines.append("")
    
    # Detailed comparison
    lines.append("## Size-by-Size Comparison")
    lines.append("")
    lines.append(f"| Size | {result.func_a_name} | {result.func_b_name} | Winner | Speedup |")
    lines.append("|------|----------|----------|--------|---------|")
    
    for comp in result.size_comparisons:
        winner_display = {
            "func_a": result.func_a_name,
            "func_b": result.func_b_name,
            "tie": "Tie",
        }.get(comp["winner"], "?")
        
        lines.append(
            f"| {comp['size']:,} | {comp['time_a']:.6f}s | {comp['time_b']:.6f}s | "
            f"{winner_display} | {comp['speedup']:.2f}x |"
        )
    
    lines.append("")
    
    return "\n".join(lines)


def generate_verification_report(
    result: "VerificationResult",
    *,
    title: str = "Complexity Verification Report",
) -> str:
    """
    Generate a markdown report from a verification result.
    
    Args:
        result: VerificationResult from verify_bounds.
        title: Report title.
    
    Returns:
        Markdown formatted report string.
    """
    lines = []
    
    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Status
    status = "✅ PASSED" if result.passes else "❌ FAILED"
    lines.append(f"## Status: {status}")
    lines.append("")
    
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Expected** | `{result.expected}` |")
    lines.append(f"| **Actual** | `{result.actual}` |")
    lines.append(f"| **Error** | {result.error:.4f} |")
    lines.append(f"| **Tolerance** | {result.tolerance:.4f} |")
    lines.append(f"| **Confidence** | {result.confidence} ({result.confidence_score:.0%}) |")
    lines.append("")
    
    lines.append(f"> {result.message}")
    lines.append("")
    
    return "\n".join(lines)


def save_report(content: str, path: str) -> None:
    """
    Save a report to a file.
    
    Args:
        content: Report content (markdown).
        path: File path to save to.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

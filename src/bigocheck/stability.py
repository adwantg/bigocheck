# Author: gadwant
"""
Instability detection for benchmark results.

Detects when benchmark results are too noisy to be reliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis


@dataclass
class StabilityResult:
    """Result of stability analysis."""
    is_stable: bool
    is_unstable: bool  # Inverse of is_stable for convenience
    stability_score: float  # 0.0 (unstable) to 1.0 (very stable)
    stability_level: str  # "high", "medium", "low"
    warnings: List[str]
    recommendations: List[str]


def compute_stability(analysis: "Analysis") -> StabilityResult:
    """
    Analyze the stability/reliability of benchmark results.
    
    Checks for:
    - High coefficient of variation (CV) in trial runs
    - Inconsistent scaling patterns
    - Outliers in measurements
    
    Args:
        analysis: Analysis object from benchmark_function.
    
    Returns:
        StabilityResult with stability score and warnings.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000], trials=5)
        >>> stability = compute_stability(analysis)
        >>> if stability.is_unstable:
        ...     print(f"⚠️ Results may be unreliable")
        ...     for warning in stability.warnings:
        ...         print(f"  - {warning}")
    """
    warnings: List[str] = []
    recommendations: List[str] = []
    deductions = 0.0  # Start at 1.0, deduct for issues
    
    measurements = analysis.measurements
    
    if len(measurements) < 3:
        warnings.append("Too few measurements for reliable stability analysis")
        recommendations.append("Use at least 5 different input sizes")
        deductions += 0.3
    
    # Check coefficient of variation (CV) for each measurement
    high_cv_count = 0
    for m in measurements:
        if m.std_dev > 0 and m.seconds > 0:
            cv = m.std_dev / m.seconds
            if cv > 0.3:  # More than 30% variation
                high_cv_count += 1
    
    if high_cv_count > 0:
        cv_ratio = high_cv_count / len(measurements)
        if cv_ratio > 0.5:
            warnings.append(f"High variance in {high_cv_count}/{len(measurements)} measurements (CV > 30%)")
            recommendations.append("Increase trials or reduce system load")
            deductions += 0.3
        elif cv_ratio > 0.2:
            warnings.append(f"Some variance detected in {high_cv_count} measurements")
            deductions += 0.1
    
    # Check for non-monotonic timing (times should generally increase)
    non_monotonic = 0
    for i in range(1, len(measurements)):
        if measurements[i].seconds < measurements[i - 1].seconds * 0.8:  # 20% tolerance
            non_monotonic += 1
    
    if non_monotonic > len(measurements) // 3:
        warnings.append("Non-monotonic timing pattern detected")
        recommendations.append("Results may be affected by caching or JIT compilation")
        deductions += 0.2
    
    # Check fit quality (error)
    if analysis.fits:
        best_error = analysis.fits[0].error
        if best_error > 0.5:
            warnings.append(f"Poor fit quality (error={best_error:.2f})")
            recommendations.append("Data doesn't match standard complexity classes well")
            deductions += 0.2
        elif best_error > 0.2:
            warnings.append(f"Moderate fit quality (error={best_error:.2f})")
            deductions += 0.1
    
    # Check gap between fits
    if len(analysis.fits) >= 2:
        gap = analysis.fits[1].error - analysis.fits[0].error
        if gap < 0.05:
            warnings.append("Ambiguous result: top fits are very similar")
            recommendations.append("Add more input sizes for clearer differentiation")
            deductions += 0.15
    
    # Calculate final score
    stability_score = max(0.0, 1.0 - deductions)
    
    # Determine level
    if stability_score >= 0.8:
        stability_level = "high"
    elif stability_score >= 0.5:
        stability_level = "medium"
    else:
        stability_level = "low"
    
    is_stable = stability_score >= 0.5
    
    if not warnings:
        recommendations.append("Results appear reliable ✓")
    
    return StabilityResult(
        is_stable=is_stable,
        is_unstable=not is_stable,
        stability_score=stability_score,
        stability_level=stability_level,
        warnings=warnings,
        recommendations=recommendations,
    )


def format_stability(stability: StabilityResult) -> str:
    """Format stability result for display."""
    status = "✓ Stable" if stability.is_stable else "⚠️ Unstable"
    lines = [
        f"Stability: {status} ({stability.stability_score:.0%})",
    ]
    
    if stability.warnings:
        lines.append("\nWarnings:")
        for w in stability.warnings:
            lines.append(f"  ⚠️ {w}")
    
    if stability.recommendations and not stability.is_stable:
        lines.append("\nRecommendations:")
        for r in stability.recommendations:
            lines.append(f"  → {r}")
    
    return "\n".join(lines)

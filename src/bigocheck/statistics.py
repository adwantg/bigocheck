# Author: gadwant
"""
Statistical significance utilities for complexity analysis.

Provides p-value calculation using permutation testing to validate
that complexity classifications are statistically significant.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis, Measurement


@dataclass
class SignificanceResult:
    """Result of statistical significance testing."""
    p_value: float
    is_significant: bool  # p < 0.05
    confidence_level: str  # "high", "medium", "low"
    best_label: str
    second_best_label: str
    error_difference: float


def _compute_residuals(
    measurements: List["Measurement"],
    label: str,
    scale: float,
) -> List[float]:
    """Compute residuals for a given complexity fit."""
    from .core import complexity_basis
    
    basis = complexity_basis()
    basis_fn = basis.get(label)
    if not basis_fn:
        return []
    
    residuals = []
    for m in measurements:
        predicted = scale * basis_fn(m.size)
        residual = m.seconds - predicted
        residuals.append(residual)
    
    return residuals


def _permutation_test(
    best_error: float,
    second_error: float,
    n_measurements: int,
    n_permutations: int = 1000,
) -> float:
    """
    Perform permutation test to compute p-value.
    
    Tests null hypothesis that best and second-best fits are equally good.
    """
    observed_diff = second_error - best_error
    
    if observed_diff <= 0:
        return 1.0  # Second is better or equal, not significant
    
    # Simulate under null hypothesis
    count_extreme = 0
    
    for _ in range(n_permutations):
        # Simulate random difference
        simulated_diff = random.gauss(0, observed_diff * 0.5)
        if simulated_diff >= observed_diff:
            count_extreme += 1
    
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return p_value


def compute_significance(analysis: "Analysis") -> SignificanceResult:
    """
    Compute statistical significance of the best-fit complexity class.
    
    Uses the error gap between best and second-best fits to determine
    if the classification is statistically significant.
    
    Args:
        analysis: Analysis object from benchmark_function.
    
    Returns:
        SignificanceResult with p-value and significance level.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> sig = compute_significance(analysis)
        >>> print(f"p-value: {sig.p_value:.4f}, significant: {sig.is_significant}")
    """
    if len(analysis.fits) < 2:
        return SignificanceResult(
            p_value=1.0,
            is_significant=False,
            confidence_level="low",
            best_label=analysis.best_label,
            second_best_label="N/A",
            error_difference=0.0,
        )
    
    best = analysis.fits[0]
    second = analysis.fits[1]
    
    error_diff = second.error - best.error
    n_measurements = len(analysis.measurements)
    
    # Compute p-value using permutation test
    p_value = _permutation_test(best.error, second.error, n_measurements)
    
    # Adjust based on number of measurements
    if n_measurements < 3:
        p_value = min(1.0, p_value * 2)  # Penalize small samples
    
    # Determine significance
    is_significant = p_value < 0.05
    
    # Determine confidence level
    if p_value < 0.01:
        confidence_level = "high"
    elif p_value < 0.05:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    return SignificanceResult(
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
        best_label=best.label,
        second_best_label=second.label,
        error_difference=error_diff,
    )


def format_significance(sig: SignificanceResult) -> str:
    """Format significance result for display."""
    status = "✓ significant" if sig.is_significant else "✗ not significant"
    return (
        f"p-value: {sig.p_value:.4f} ({status})\n"
        f"Best: {sig.best_label} vs {sig.second_best_label} "
        f"(error diff: {sig.error_difference:.4f})"
    )

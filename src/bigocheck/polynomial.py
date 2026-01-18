# Author: gadwant
"""
Polynomial complexity fitting - detect O(n^k) for arbitrary k.

Extends the standard complexity classes to detect polynomial complexity
with any exponent, e.g., O(n^2.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Measurement


@dataclass
class PolynomialFit:
    """Result of polynomial fitting."""
    exponent: float
    scale: float
    error: float
    label: str  # e.g., "O(n^2.34)"


def _fit_polynomial(
    n_values: List[int],
    y_values: List[float],
) -> Tuple[float, float, float]:
    """
    Fit y = a * n^k using log-log linear regression.
    
    Returns (exponent k, scale a, error).
    """
    # Filter out zero or negative values
    valid_pairs = [
        (n, y) for n, y in zip(n_values, y_values)
        if n > 0 and y > 0
    ]
    
    if len(valid_pairs) < 2:
        return 1.0, 1.0, float('inf')
    
    ns, ys = zip(*valid_pairs)
    
    # Log transform: log(y) = log(a) + k * log(n)
    log_ns = [math.log(n) for n in ns]
    log_ys = [math.log(y) for y in ys]
    
    # Linear regression on log values
    n = len(log_ns)
    sum_x = sum(log_ns)
    sum_y = sum(log_ys)
    sum_xy = sum(x * y for x, y in zip(log_ns, log_ys))
    sum_x2 = sum(x * x for x in log_ns)
    
    # Avoid division by zero
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 1.0, 1.0, float('inf')
    
    k = (n * sum_xy - sum_x * sum_y) / denom
    log_a = (sum_y - k * sum_x) / n
    a = math.exp(log_a)
    
    # Calculate error
    predictions = [a * (x ** k) for x in ns]
    error = 0.0
    for actual, pred in zip(ys, predictions):
        if actual > 0:
            error += ((actual - pred) / actual) ** 2
    error = math.sqrt(error / len(ys))
    
    return k, a, error


def fit_polynomial(measurements: List["Measurement"]) -> PolynomialFit:
    """
    Fit measurements to O(n^k) and find the best exponent k.
    
    Args:
        measurements: List of Measurement objects.
    
    Returns:
        PolynomialFit with detected exponent and error.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> poly = fit_polynomial(analysis.measurements)
        >>> print(f"Complexity: {poly.label}")
        Complexity: O(n^2.34)
    """
    n_values = [m.size for m in measurements]
    t_values = [m.seconds for m in measurements]
    
    k, a, error = _fit_polynomial(n_values, t_values)
    
    # Round exponent for display
    if abs(k - round(k)) < 0.1:
        # Close to integer, use integer
        label = f"O(n^{int(round(k))})"
    else:
        label = f"O(n^{k:.2f})"
    
    # Special cases
    if abs(k) < 0.1:
        label = "O(1)"
    elif abs(k - 0.5) < 0.1:
        label = "O(√n)"
    elif abs(k - 1.0) < 0.1:
        label = "O(n)"
    
    return PolynomialFit(
        exponent=k,
        scale=a,
        error=error,
        label=label,
    )


def fit_polynomial_space(measurements: List["Measurement"]) -> Optional[PolynomialFit]:
    """
    Fit memory measurements to O(n^k) for space complexity.
    
    Returns None if no memory data is available.
    """
    mem_measurements = [m for m in measurements if m.memory_bytes is not None]
    
    if len(mem_measurements) < 2:
        return None
    
    n_values = [m.size for m in mem_measurements]
    mem_values = [float(m.memory_bytes) for m in mem_measurements]  # type: ignore
    
    k, a, error = _fit_polynomial(n_values, mem_values)
    
    if abs(k - round(k)) < 0.1:
        label = f"O(n^{int(round(k))})"
    else:
        label = f"O(n^{k:.2f})"
    
    if abs(k) < 0.1:
        label = "O(1)"
    elif abs(k - 1.0) < 0.1:
        label = "O(n)"
    
    return PolynomialFit(
        exponent=k,
        scale=a,
        error=error,
        label=label,
    )

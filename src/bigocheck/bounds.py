# Author: gadwant
"""
Complexity bounds checking.

Assert complexity is within specified bounds (e.g., O(log n) ≤ f ≤ O(n²)).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Optional

from .core import Analysis, benchmark_function


# Complexity ordering
COMPLEXITY_ORDER = [
    "O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)",
    "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)",
]


class ComplexityBoundsError(Exception):
    """Raised when complexity is outside specified bounds."""


@dataclass
class BoundsResult:
    """Result of bounds checking."""
    in_bounds: bool
    actual: str
    lower_bound: Optional[str]
    upper_bound: Optional[str]
    message: str


def _get_index(label: str) -> int:
    """Get complexity index (higher = worse)."""
    try:
        return COMPLEXITY_ORDER.index(label)
    except ValueError:
        # Handle polynomial
        if label.startswith("O(n^"):
            try:
                exp = float(label[4:-1])
                if exp <= 2:
                    return COMPLEXITY_ORDER.index("O(n^2)")
                elif exp <= 3:
                    return COMPLEXITY_ORDER.index("O(n^3)")
                else:
                    return COMPLEXITY_ORDER.index("O(2^n)") - 1
            except ValueError:
                pass
        return len(COMPLEXITY_ORDER)


def check_bounds(
    analysis: Analysis,
    *,
    lower: Optional[str] = None,
    upper: Optional[str] = None,
) -> BoundsResult:
    """
    Check if complexity is within specified bounds.
    
    Args:
        analysis: Analysis object from benchmark_function.
        lower: Minimum acceptable complexity (e.g., "O(1)").
        upper: Maximum acceptable complexity (e.g., "O(n log n)").
    
    Returns:
        BoundsResult with in_bounds status.
    
    Example:
        >>> result = check_bounds(analysis, lower="O(log n)", upper="O(n^2)")
        >>> if not result.in_bounds:
        ...     print(result.message)
    """
    actual_idx = _get_index(analysis.best_label)
    
    in_bounds = True
    messages = []
    
    if lower is not None:
        lower_idx = _get_index(lower)
        if actual_idx < lower_idx:
            in_bounds = False
            messages.append(f"Complexity {analysis.best_label} is better than lower bound {lower}")
    
    if upper is not None:
        upper_idx = _get_index(upper)
        if actual_idx > upper_idx:
            in_bounds = False
            messages.append(f"Complexity {analysis.best_label} exceeds upper bound {upper}")
    
    if in_bounds:
        bounds_str = ""
        if lower and upper:
            bounds_str = f"{lower} ≤ {analysis.best_label} ≤ {upper}"
        elif upper:
            bounds_str = f"{analysis.best_label} ≤ {upper}"
        else:
            bounds_str = f"{analysis.best_label} ≥ {lower}"
        message = f"✓ Complexity in bounds: {bounds_str}"
    else:
        message = "; ".join(messages)
    
    return BoundsResult(
        in_bounds=in_bounds,
        actual=analysis.best_label,
        lower_bound=lower,
        upper_bound=upper,
        message=message,
    )


def assert_bounds(
    *,
    lower: Optional[str] = None,
    upper: Optional[str] = None,
    sizes: Optional[List[int]] = None,
    trials: int = 3,
) -> Callable:
    """
    Decorator to assert function complexity is within bounds.
    
    Args:
        lower: Minimum acceptable complexity.
        upper: Maximum acceptable complexity.
        sizes: Input sizes for benchmarking.
        trials: Number of trials per size.
    
    Returns:
        Decorated function.
    
    Raises:
        ComplexityBoundsError: If complexity is outside bounds.
    
    Example:
        >>> @assert_bounds(lower="O(log n)", upper="O(n log n)")
        ... def my_search(n):
        ...     # Should be O(n) or O(log n), not O(n^2)
        ...     pass
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000]
    
    def decorator(func: Callable) -> Callable:
        _verified = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _verified
            
            if not _verified:
                analysis = benchmark_function(func, sizes=sizes, trials=trials)
                result = check_bounds(analysis, lower=lower, upper=upper)
                
                if not result.in_bounds:
                    raise ComplexityBoundsError(
                        f"Function '{func.__name__}': {result.message}"
                    )
                
                _verified = True
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def format_bounds_check(result: BoundsResult) -> str:
    """Format bounds check result for display."""
    status = "✓ PASS" if result.in_bounds else "✗ FAIL"
    
    lines = [
        f"Bounds Check: {status}",
        f"Actual: {result.actual}",
    ]
    
    if result.lower_bound:
        lines.append(f"Lower Bound: {result.lower_bound}")
    if result.upper_bound:
        lines.append(f"Upper Bound: {result.upper_bound}")
    
    lines.append(f"Message: {result.message}")
    
    return "\n".join(lines)

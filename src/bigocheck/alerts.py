# Author: gadwant
"""
Threshold alerts for complexity analysis.

Alert when complexity exceeds acceptable thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Optional

from .core import Analysis, benchmark_function


# Complexity ordering for comparison
COMPLEXITY_ORDER = [
    "O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)",
    "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)",
]


class ComplexityThresholdError(Exception):
    """Raised when complexity exceeds threshold."""


@dataclass
class ThresholdResult:
    """Result of a threshold check."""
    passed: bool
    actual: str
    threshold: str
    message: str


def _complexity_index(label: str) -> int:
    """Get index of complexity in order (higher = worse)."""
    try:
        return COMPLEXITY_ORDER.index(label)
    except ValueError:
        # Unknown complexity - check if polynomial
        if label.startswith("O(n^"):
            try:
                exp = float(label[4:-1])
                if exp <= 1:
                    return COMPLEXITY_ORDER.index("O(n)")
                elif exp <= 2:
                    return COMPLEXITY_ORDER.index("O(n^2)")
                else:
                    return COMPLEXITY_ORDER.index("O(n^3)")
            except ValueError:
                pass
        return len(COMPLEXITY_ORDER)  # Assume worst


def check_threshold(
    analysis: Analysis,
    max_complexity: str,
) -> ThresholdResult:
    """
    Check if analysis meets complexity threshold.
    
    Args:
        analysis: Analysis object from benchmark_function.
        max_complexity: Maximum acceptable complexity (e.g., "O(n log n)").
    
    Returns:
        ThresholdResult with pass/fail status.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> result = check_threshold(analysis, "O(n)")
        >>> if not result.passed:
        ...     print(f"❌ {result.message}")
    """
    actual_idx = _complexity_index(analysis.best_label)
    threshold_idx = _complexity_index(max_complexity)
    
    passed = actual_idx <= threshold_idx
    
    if passed:
        message = f"✓ Complexity {analysis.best_label} is within threshold {max_complexity}"
    else:
        message = f"✗ Complexity {analysis.best_label} exceeds threshold {max_complexity}"
    
    return ThresholdResult(
        passed=passed,
        actual=analysis.best_label,
        threshold=max_complexity,
        message=message,
    )


def assert_threshold(
    max_complexity: str,
    sizes: Optional[List[int]] = None,
    trials: int = 3,
) -> Callable:
    """
    Decorator to assert function complexity is within threshold.
    
    Args:
        max_complexity: Maximum acceptable complexity (e.g., "O(n)").
        sizes: Input sizes for benchmarking.
        trials: Number of trials per size.
    
    Returns:
        Decorated function.
    
    Raises:
        ComplexityThresholdError: If complexity exceeds threshold.
    
    Example:
        >>> @assert_threshold("O(n log n)")
        ... def my_sort(n):
        ...     return sorted(range(n))
        >>> 
        >>> my_sort(100)  # First call triggers verification
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000]
    
    def decorator(func: Callable) -> Callable:
        _verified = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _verified
            
            if not _verified:
                # Run benchmark
                analysis = benchmark_function(func, sizes=sizes, trials=trials)
                result = check_threshold(analysis, max_complexity)
                
                if not result.passed:
                    raise ComplexityThresholdError(
                        f"Function '{func.__name__}' complexity {result.actual} "
                        f"exceeds threshold {result.threshold}"
                    )
                
                _verified = True
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def monitor_complexity(
    func: Callable,
    sizes: List[int],
    max_complexity: str,
    *,
    on_exceed: str = "warn",
    trials: int = 3,
) -> Analysis:
    """
    Monitor function complexity and take action if threshold exceeded.
    
    Args:
        func: Function to monitor.
        sizes: Input sizes for benchmarking.
        max_complexity: Maximum acceptable complexity.
        on_exceed: Action when exceeded - "warn", "error", or "ignore".
        trials: Number of trials.
    
    Returns:
        Analysis object.
    
    Raises:
        ComplexityThresholdError: If on_exceed="error" and threshold exceeded.
    
    Example:
        >>> analysis = monitor_complexity(
        ...     my_func,
        ...     sizes=[100, 500, 1000],
        ...     max_complexity="O(n)",
        ...     on_exceed="warn"
        ... )
    """
    import warnings
    
    analysis = benchmark_function(func, sizes=sizes, trials=trials)
    result = check_threshold(analysis, max_complexity)
    
    if not result.passed:
        if on_exceed == "error":
            raise ComplexityThresholdError(result.message)
        elif on_exceed == "warn":
            warnings.warn(result.message, UserWarning)
    
    return analysis


def format_threshold_report(results: List[ThresholdResult]) -> str:
    """
    Format multiple threshold results as a report.
    
    Args:
        results: List of ThresholdResult objects.
    
    Returns:
        Formatted report string.
    """
    lines = ["Threshold Check Report", "=" * 40]
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    for r in results:
        status = "✓" if r.passed else "✗"
        lines.append(f"{status} {r.actual} {'≤' if r.passed else '>'} {r.threshold}")
    
    lines.append("")
    lines.append(f"Passed: {passed}/{len(results)}")
    if failed > 0:
        lines.append(f"Failed: {failed}/{len(results)}")
    
    return "\n".join(lines)

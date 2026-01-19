# Author: gadwant
"""
Complexity assertions and verification utilities.

Provides decorators and functions for verifying that functions meet
expected complexity bounds - useful for CI/CD testing.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .core import Analysis, benchmark_function


@dataclass
class VerificationResult:
    """Result of complexity verification."""
    passes: bool
    expected: str
    actual: str
    error: float
    tolerance: float
    confidence: str
    confidence_score: float
    analysis: Analysis
    message: str


@dataclass
class ConfidenceResult:
    """Confidence assessment of a complexity fit."""
    level: str  # "high", "medium", "low"
    score: float  # 0.0 to 1.0
    reasons: List[str]


def compute_confidence(analysis: Analysis) -> ConfidenceResult:
    """
    Compute confidence level for a complexity analysis.
    
    Factors considered:
    - Error margin between best and second-best fit
    - Absolute error of best fit
    - Number of measurements
    - Spread of input sizes
    
    Returns:
        ConfidenceResult with level, score, and reasons.
    """
    reasons = []
    score = 1.0
    
    if len(analysis.fits) < 2:
        return ConfidenceResult(level="low", score=0.3, reasons=["Insufficient fits"])
    
    best = analysis.fits[0]
    second = analysis.fits[1]
    
    # Factor 1: Error gap between best and second best
    error_gap = second.error - best.error
    if error_gap < 0.05:
        score -= 0.3
        reasons.append(f"Small gap between top fits ({error_gap:.3f})")
    elif error_gap < 0.1:
        score -= 0.15
        reasons.append(f"Moderate gap between top fits ({error_gap:.3f})")
    else:
        reasons.append(f"Clear gap between fits ({error_gap:.3f})")
    
    # Factor 2: Absolute error of best fit
    if best.error > 0.3:
        score -= 0.3
        reasons.append(f"High best-fit error ({best.error:.3f})")
    elif best.error > 0.15:
        score -= 0.15
        reasons.append(f"Moderate best-fit error ({best.error:.3f})")
    else:
        reasons.append(f"Low best-fit error ({best.error:.3f})")
    
    # Factor 3: Number of measurements
    n_measurements = len(analysis.measurements)
    if n_measurements < 3:
        score -= 0.25
        reasons.append(f"Too few measurements ({n_measurements})")
    elif n_measurements < 5:
        score -= 0.1
        reasons.append(f"Few measurements ({n_measurements})")
    else:
        reasons.append(f"Good measurement count ({n_measurements})")
    
    # Factor 4: Size spread (ratio of max to min)
    sizes = [m.size for m in analysis.measurements]
    if sizes:
        size_ratio = max(sizes) / max(min(sizes), 1)
        if size_ratio < 4:
            score -= 0.2
            reasons.append(f"Limited size spread (ratio {size_ratio:.1f})")
        elif size_ratio < 10:
            score -= 0.1
            reasons.append(f"Moderate size spread (ratio {size_ratio:.1f})")
        else:
            reasons.append(f"Good size spread (ratio {size_ratio:.1f})")
    
    # Clamp score
    score = max(0.0, min(1.0, score))
    
    # Determine level
    if score >= 0.7:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"
    
    return ConfidenceResult(level=level, score=score, reasons=reasons)


def verify_bounds(
    func: Callable[..., Any],
    sizes: List[int],
    expected: str,
    *,
    tolerance: float = 0.3,
    trials: int = 3,
    warmup: int = 1,
) -> VerificationResult:
    """
    Verify that a function matches expected complexity bounds.
    
    Args:
        func: Function to verify.
        sizes: Input sizes to test.
        expected: Expected complexity class (e.g., "O(n)", "O(n log n)").
        tolerance: Maximum allowed error difference (default 0.3).
        trials: Number of trials per size.
        warmup: Warmup runs.
    
    Returns:
        VerificationResult with pass/fail status and details.
    
    Example:
        >>> result = verify_bounds(sorted, [100, 500, 1000], expected="O(n log n)")
        >>> assert result.passes, result.message
    """
    analysis = benchmark_function(func, sizes=sizes, trials=trials, warmup=warmup)
    confidence = compute_confidence(analysis)
    
    # Normalize expected format
    expected_normalized = expected.strip()
    
    # Check if expected matches best fit
    actual = analysis.best_label
    
    # Find error for expected complexity
    expected_fit = next((f for f in analysis.fits if f.label == expected_normalized), None)
    best_fit = analysis.fits[0]
    
    if expected_fit is None:
        # Try common aliases
        aliases = {
            "O(n^2)": "O(n^2)",
            "O(n**2)": "O(n^2)",
            "O(n²)": "O(n^2)",
            "O(n^3)": "O(n^3)",
            "O(n**3)": "O(n^3)",
            "O(n³)": "O(n^3)",
            "O(2**n)": "O(2^n)",
            "O(2ⁿ)": "O(2^n)",
            "O(sqrt(n))": "O(√n)",
            "O(n*log(n))": "O(n log n)",
            "O(nlogn)": "O(n log n)",
        }
        aliased = aliases.get(expected_normalized)
        if aliased:
            expected_fit = next((f for f in analysis.fits if f.label == aliased), None)
            expected_normalized = aliased
    
    if expected_fit is None:
        passes = False
        error = float('inf')
        message = f"Unknown complexity class: {expected}. Valid classes: {[f.label for f in analysis.fits]}"
    else:
        error_diff = expected_fit.error - best_fit.error
        passes = actual == expected_normalized or error_diff <= tolerance
        error = expected_fit.error
        
        if passes:
            message = f"✓ Complexity verified: {expected_normalized} (error={error:.4f})"
        else:
            message = f"✗ Expected {expected_normalized} but got {actual} (error diff={error_diff:.4f} > tolerance={tolerance})"
    
    return VerificationResult(
        passes=passes,
        expected=expected_normalized,
        actual=actual,
        error=error,
        tolerance=tolerance,
        confidence=confidence.level,
        confidence_score=confidence.score,
        analysis=analysis,
        message=message,
    )


class ComplexityAssertionError(AssertionError):
    """Raised when a complexity assertion fails."""


def assert_complexity(
    expected: str,
    *,
    sizes: Optional[List[int]] = None,
    tolerance: float = 0.3,
    trials: int = 3,
    warmup: int = 1,
    min_confidence: str = "low",
) -> Callable:
    """
    Decorator to assert that a function has expected complexity.
    
    The first call to the decorated function will benchmark it.
    Subsequent calls proceed normally without overhead.
    
    Args:
        expected: Expected complexity (e.g., "O(n)", "O(n log n)").
        sizes: Input sizes to test. Default: auto-generated.
        tolerance: Maximum error tolerance.
        trials: Number of trials per size.
        warmup: Warmup runs.
        min_confidence: Minimum confidence level ("high", "medium", "low").
    
    Raises:
        ComplexityAssertionError: If complexity doesn't match.
    
    Example:
        >>> @assert_complexity("O(n)")
        ... def linear_sum(n):
        ...     return sum(range(n))
    """
    def decorator(func: Callable) -> Callable:
        verified = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal verified
            
            if not verified:
                test_sizes = sizes or [100, 500, 1000, 5000]
                result = verify_bounds(
                    func,
                    sizes=test_sizes,
                    expected=expected,
                    tolerance=tolerance,
                    trials=trials,
                    warmup=warmup,
                )
                
                # Check confidence
                confidence_levels = {"low": 0, "medium": 1, "high": 2}
                if confidence_levels.get(result.confidence, 0) < confidence_levels.get(min_confidence, 0):
                    raise ComplexityAssertionError(
                        f"Confidence too low: {result.confidence} < {min_confidence}. "
                        f"Try larger or more varied input sizes."
                    )
                
                if not result.passes:
                    raise ComplexityAssertionError(result.message)
                
                verified = True
            
            return func(*args, **kwargs)
        
        # Store verification info
        wrapper._complexity_expected = expected
        wrapper._complexity_verified = lambda: verified
        
        return wrapper
    
    return decorator


def auto_select_sizes(
    func: Callable[..., Any],
    *,
    target_time: float = 5.0,
    min_sizes: int = 5,
    max_sizes: int = 10,
    initial_n: int = 10,
) -> List[int]:
    """
    Automatically select optimal input sizes for benchmarking.
    
    Starts with small sizes and increases until target time is reached
    or appropriate size range is found.
    
    Args:
        func: Function to analyze.
        target_time: Target total benchmark time in seconds.
        min_sizes: Minimum number of sizes to generate.
        max_sizes: Maximum number of sizes.
        initial_n: Starting input size.
    
    Returns:
        List of optimal input sizes.
    
    Example:
        >>> sizes = auto_select_sizes(my_func, target_time=3.0)
        >>> analysis = benchmark_function(my_func, sizes=sizes)
    """
    import time
    
    sizes = []
    n = initial_n
    total_time = 0.0
    
    # Phase 1: Find a size that takes measurable time
    while n < 10_000_000:
        start = time.perf_counter()
        func(n)
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.001:  # At least 1ms
            break
        n *= 2
    
    # Phase 2: Generate sizes with geometric progression
    min_n = max(initial_n, n // 4)
    max_n = n * 100
    
    # Create logarithmically spaced sizes
    log_min = math.log10(max(min_n, 1))
    log_max = math.log10(max_n)
    
    for i in range(max_sizes):
        if min_sizes <= len(sizes) and total_time >= target_time:
            break
        
        # Logarithmic spacing
        log_n = log_min + (log_max - log_min) * i / (max_sizes - 1)
        size = int(10 ** log_n)
        
        # Ensure unique and sorted
        if size not in sizes:
            sizes.append(size)
            
            # Estimate time for this size
            start = time.perf_counter()
            func(size)
            total_time += time.perf_counter() - start
    
    return sorted(set(sizes))

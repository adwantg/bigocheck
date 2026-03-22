# Author: gadwant
"""
Complexity assertions and verification utilities.

Provides decorators and functions for verifying that functions meet
expected complexity bounds - useful for CI/CD testing.
"""
from __future__ import annotations

import functools
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .async_bench import benchmark_async
from .bounds import _get_index
from .core import Analysis, benchmark_function
from .profiles import (
    DEFAULT_ASSERTION_PROFILE_NAME,
    DEFAULT_PROFILE_NAME,
    resolve_profile,
)
from .stability import compute_stability


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
    space_expected: Optional[str] = None
    space_actual: Optional[str] = None
    stability: Optional[str] = None
    stability_score: Optional[float] = None


@dataclass
class ConfidenceResult:
    """Confidence assessment of a complexity fit."""
    level: str  # "high", "medium", "low"
    score: float  # 0.0 to 1.0
    reasons: List[str]


CONFIDENCE_LEVELS = {"low": 0, "medium": 1, "high": 2}


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
    trials: Optional[int] = None,
    warmup: Optional[int] = None,
    profile: Optional[str] = None,
    setup: Optional[Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]] = None,
    arg_factory: Optional[Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]] = None,
    robust: Optional[bool] = None,
    memory: bool = False,
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
        profile: Optional benchmark profile name.
        setup: Optional callable returning (args, kwargs) outside the timed region.
        arg_factory: Optional callable returning (args, kwargs) inside the timed region.
        robust: Override profile robust aggregation setting.
        memory: If True, collect memory data for optional space assertions.
    
    Returns:
        VerificationResult with pass/fail status and details.
    
    Example:
        >>> result = verify_bounds(sorted, [100, 500, 1000], expected="O(n log n)")
        >>> assert result.passes, result.message
    """
    options = _resolve_benchmark_options(
        profile=profile,
        sizes=sizes,
        trials=trials,
        warmup=warmup,
        robust=robust,
    )
    analysis = benchmark_function(
        func,
        sizes=options.sizes,
        trials=options.trials,
        warmup=options.warmup,
        memory=memory,
        setup=setup,
        arg_factory=arg_factory,
        robust=options.robust,
    )
    confidence = compute_confidence(analysis)
    stability = compute_stability(analysis)
    
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
            message = _format_verification_message(
                expected=expected_normalized,
                actual=actual,
                expected_fit=expected_fit,
                best_fit=best_fit,
                analysis=analysis,
                confidence=confidence,
                stability=stability,
                passes=True,
            )
        else:
            message = _format_verification_message(
                expected=expected_normalized,
                actual=actual,
                expected_fit=expected_fit,
                best_fit=best_fit,
                analysis=analysis,
                confidence=confidence,
                stability=stability,
                passes=False,
                tolerance=tolerance,
            )
    
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
        stability=stability.stability_level,
        stability_score=stability.stability_score,
    )


class ComplexityAssertionError(AssertionError):
    """Raised when a complexity assertion fails."""


def assert_complexity(
    expected: str,
    *,
    sizes: Optional[List[int]] = None,
    tolerance: float = 0.3,
    trials: Optional[int] = None,
    warmup: Optional[int] = None,
    min_confidence: str = "low",
    profile: Optional[str] = None,
    setup: Optional[Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]] = None,
    arg_factory: Optional[Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]] = None,
    robust: Optional[bool] = None,
    space: Optional[str] = None,
    space_upper: Optional[str] = None,
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
        profile: Optional benchmark profile name.
        setup: Optional callable returning (args, kwargs) outside the timed region.
        arg_factory: Optional callable returning (args, kwargs) inside the timed region.
        robust: Override profile robust aggregation setting.
        space: Exact expected space complexity.
        space_upper: Maximum allowed space complexity.
    
    Raises:
        ComplexityAssertionError: If complexity doesn't match.
    
    Example:
        >>> @assert_complexity("O(n)")
        ... def linear_sum(n):
        ...     return sum(range(n))
    """
    def decorator(func: Callable) -> Callable:
        verified = False

        def _validate() -> None:
            nonlocal verified

            if not verified:
                options = _resolve_benchmark_options(
                    profile=profile,
                    sizes=sizes,
                    trials=trials,
                    warmup=warmup,
                    robust=robust,
                )
                result = verify_bounds(
                    func,
                    sizes=options.sizes,
                    expected=expected,
                    tolerance=tolerance,
                    trials=options.trials,
                    warmup=options.warmup,
                    profile=options.name,
                    setup=setup,
                    arg_factory=arg_factory,
                    robust=options.robust,
                    memory=bool(space or space_upper),
                )

                if CONFIDENCE_LEVELS.get(result.confidence, 0) < CONFIDENCE_LEVELS.get(min_confidence, 0):
                    raise ComplexityAssertionError(
                        f"Confidence too low: {result.confidence} < {min_confidence}. "
                        f"Try larger or more varied input sizes."
                    )

                _assert_space_expectation(
                    result.analysis,
                    expected=space,
                    upper=space_upper,
                    confidence=result.confidence,
                    stability=result.stability or "unknown",
                )

                if not result.passes:
                    raise ComplexityAssertionError(result.message)

                verified = True

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nonlocal verified

                if not verified:
                    options = _resolve_benchmark_options(
                        profile=profile,
                        sizes=sizes,
                        trials=trials,
                        warmup=warmup,
                        robust=robust,
                    )
                    analysis = await benchmark_async(
                        func,
                        sizes=options.sizes,
                        trials=options.trials,
                        warmup=options.warmup,
                        memory=bool(space or space_upper),
                        setup=setup,
                        arg_factory=arg_factory,
                        robust=options.robust,
                    )
                    _raise_for_async_assertion(
                        analysis=analysis,
                        expected=expected,
                        tolerance=tolerance,
                        min_confidence=min_confidence,
                        space=space,
                        space_upper=space_upper,
                    )
                    verified = True

                return await func(*args, **kwargs)

            async_wrapper._complexity_expected = expected
            async_wrapper._complexity_verified = lambda: verified
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _validate()
            return func(*args, **kwargs)
        
        # Store verification info
        wrapper._complexity_expected = expected
        wrapper._complexity_verified = lambda: verified
        
        return wrapper
    
    return decorator


def _resolve_benchmark_options(
    *,
    profile: Optional[str],
    sizes: Optional[List[int]],
    trials: Optional[int],
    warmup: Optional[int],
    robust: Optional[bool],
):
    base_profile = profile or (
        DEFAULT_ASSERTION_PROFILE_NAME
        if all(value is None for value in (sizes, trials, warmup, robust))
        else DEFAULT_PROFILE_NAME
    )
    return resolve_profile(
        profile=base_profile,
        sizes=sizes,
        trials=trials,
        warmup=warmup,
        robust=robust,
    )


def _format_top_fits(analysis: Analysis, limit: int = 3) -> str:
    return ", ".join(
        f"{fit.label} (rmse={fit.relative_rmse:.4f}, r2={fit.r_squared:.3f})"
        for fit in analysis.fits[:limit]
    )


def _format_measurements(analysis: Analysis, limit: int = 5) -> str:
    return "; ".join(
        f"n={measurement.size}: {measurement.seconds:.6f}s"
        for measurement in analysis.measurements[:limit]
    )


def _format_verification_message(
    *,
    expected: str,
    actual: str,
    expected_fit,
    best_fit,
    analysis: Analysis,
    confidence: ConfidenceResult,
    stability,
    passes: bool,
    tolerance: Optional[float] = None,
) -> str:
    status = "✓" if passes else "✗"
    lines = [f"{status} Expected {expected}, got {actual}."]

    if expected_fit is not None:
        lines.append(
            f"Expected fit rmse={expected_fit.relative_rmse:.4f}; "
            f"best fit rmse={best_fit.relative_rmse:.4f}"
        )
    if tolerance is not None and expected_fit is not None:
        lines.append(
            f"Tolerance check: Δrmse={expected_fit.relative_rmse - best_fit.relative_rmse:.4f} "
            f"(allowed {tolerance:.4f})"
        )

    lines.append(f"Top fits: {_format_top_fits(analysis)}")
    lines.append(f"Measurements: {_format_measurements(analysis)}")
    lines.append(
        f"Confidence: {confidence.level} ({confidence.score:.0%}); "
        f"stability: {stability.stability_level} ({stability.stability_score:.0%})"
    )

    if not passes and (confidence.level != "high" or stability.is_unstable):
        lines.append(
            "Hint: use larger sizes, a profile such as 'ci', setup=... to exclude construction, "
            "or robust=True for noisy timings."
        )

    return " ".join(lines)


def _assert_space_expectation(
    analysis: Analysis,
    *,
    expected: Optional[str],
    upper: Optional[str],
    confidence: str,
    stability: str,
) -> None:
    if expected is None and upper is None:
        return

    if analysis.space_label is None:
        raise ComplexityAssertionError(
            "Space complexity assertion requested but memory tracking was not available."
        )

    if expected is not None and analysis.space_label != expected:
        raise ComplexityAssertionError(
            f"Space complexity expected {expected}, got {analysis.space_label}. "
            f"Confidence={confidence}, stability={stability}."
        )

    if upper is not None and _get_index(analysis.space_label) > _get_index(upper):
        raise ComplexityAssertionError(
            f"Space complexity {analysis.space_label} exceeds upper bound {upper}. "
            f"Confidence={confidence}, stability={stability}."
        )


def _raise_for_async_assertion(
    *,
    analysis: Analysis,
    expected: str,
    tolerance: float,
    min_confidence: str,
    space: Optional[str],
    space_upper: Optional[str],
) -> None:
    confidence = compute_confidence(analysis)
    stability = compute_stability(analysis)
    expected_fit = next((fit for fit in analysis.fits if fit.label == expected), None)
    if expected_fit is None:
        aliases = {
            "O(n**2)": "O(n^2)",
            "O(n²)": "O(n^2)",
            "O(n**3)": "O(n^3)",
            "O(n³)": "O(n^3)",
            "O(2**n)": "O(2^n)",
            "O(2ⁿ)": "O(2^n)",
            "O(sqrt(n))": "O(√n)",
            "O(n*log(n))": "O(n log n)",
            "O(nlogn)": "O(n log n)",
        }
        aliased = aliases.get(expected.strip())
        if aliased:
            expected_fit = next((fit for fit in analysis.fits if fit.label == aliased), None)
            expected = aliased

    if expected_fit is None:
        raise ComplexityAssertionError(
            f"Unknown complexity class: {expected}. Valid classes: {[fit.label for fit in analysis.fits]}"
        )

    best_fit = analysis.fits[0]
    passes = analysis.best_label == expected or (expected_fit.relative_rmse - best_fit.relative_rmse) <= tolerance

    if CONFIDENCE_LEVELS.get(confidence.level, 0) < CONFIDENCE_LEVELS.get(min_confidence, 0):
        raise ComplexityAssertionError(
            f"Confidence too low: {confidence.level} < {min_confidence}. "
            f"Try larger or more varied input sizes."
        )

    _assert_space_expectation(
        analysis,
        expected=space,
        upper=space_upper,
        confidence=confidence.level,
        stability=stability.stability_level,
    )

    if not passes:
        raise ComplexityAssertionError(
            _format_verification_message(
                expected=expected,
                actual=analysis.best_label,
                expected_fit=expected_fit,
                best_fit=best_fit,
                analysis=analysis,
                confidence=confidence,
                stability=stability,
                passes=False,
                tolerance=tolerance,
            )
        )


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

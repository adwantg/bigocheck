# Author: gadwant
"""
Input size recommendations for better benchmarking.

Suggest optimal input sizes based on function characteristics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class SizeRecommendation:
    """Recommendation for input sizes."""
    sizes: List[int]
    reason: str
    estimated_time: float  # Total estimated benchmark time in seconds
    confidence: str  # "high", "medium", "low"


def suggest_sizes(
    func: Callable,
    *,
    time_budget: float = 10.0,
    min_sizes: int = 4,
    max_sizes: int = 8,
) -> SizeRecommendation:
    """
    Suggest optimal input sizes for benchmarking.
    
    Runs quick calibration to determine appropriate input sizes.
    
    Args:
        func: Function to benchmark (takes single int argument).
        time_budget: Maximum total benchmark time in seconds.
        min_sizes: Minimum number of sizes to recommend.
        max_sizes: Maximum number of sizes to recommend.
    
    Returns:
        SizeRecommendation with suggested sizes.
    
    Example:
        >>> def my_sort(n):
        ...     return sorted(range(n))
        >>> rec = suggest_sizes(my_sort, time_budget=5.0)
        >>> print(rec.sizes)  # e.g., [100, 500, 1000, 5000, 10000]
    """
    # Phase 1: Quick calibration with small sizes
    calibration_sizes = [10, 100, 1000]
    timings = []
    
    for size in calibration_sizes:
        try:
            start = time.perf_counter()
            func(size)
            elapsed = time.perf_counter() - start
            timings.append((size, elapsed))
        except Exception:
            # Function failed at this size
            break
    
    if len(timings) < 2:
        # Very fast function, use large sizes
        return SizeRecommendation(
            sizes=[1000, 5000, 10000, 50000, 100000],
            reason="Function is very fast, using larger sizes for accuracy.",
            estimated_time=1.0,
            confidence="medium",
        )
    
    # Phase 2: Estimate scaling
    t1, t2 = timings[-2][1], timings[-1][1]
    _s1, s2 = timings[-2][0], timings[-1][0]  # noqa: F841
    
    if t2 > 0 and t1 > 0:
        # Estimate growth rate (used to determine size range below)
        _ratio = t2 / t1 if t1 > 0.0001 else 10  # noqa: F841
    else:
        _ratio = 10  # noqa: F841
    
    # Phase 3: Determine appropriate range
    if t2 < 0.001:
        # Very fast - use larger sizes
        base_sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000]
        reason = "Very fast function - using larger input sizes."
        confidence = "high"
    elif t2 < 0.01:
        # Fast - medium sizes
        base_sizes = [500, 1000, 2500, 5000, 10000, 25000]
        reason = "Fast function - using medium to large sizes."
        confidence = "high"
    elif t2 < 0.1:
        # Medium - moderate sizes
        base_sizes = [100, 250, 500, 1000, 2500, 5000]
        reason = "Medium-speed function - using moderate sizes."
        confidence = "high"
    elif t2 < 1.0:
        # Slow - smaller sizes
        base_sizes = [50, 100, 200, 500, 1000, 2000]
        reason = "Slow function - using smaller sizes to fit time budget."
        confidence = "medium"
    else:
        # Very slow
        base_sizes = [10, 25, 50, 100, 200, 500]
        reason = "Very slow function - using small sizes."
        confidence = "low"
    
    # Trim to fit constraints
    sizes = base_sizes[:max_sizes]
    if len(sizes) < min_sizes:
        sizes = base_sizes[:min_sizes]
    
    # Estimate total time
    # Rough estimate: sum of (size/1000)^2 * base_time * trials
    base_time = t2 / (s2 / 1000) ** 2 if s2 > 0 else 0.001
    estimated = sum((s / 1000) ** 2 * base_time for s in sizes) * 3  # 3 trials
    
    return SizeRecommendation(
        sizes=sizes,
        reason=reason,
        estimated_time=min(estimated, time_budget * 2),
        confidence=confidence,
    )


def auto_calibrate(
    func: Callable,
    target_time: float = 0.1,
) -> int:
    """
    Find an input size that takes approximately target_time to execute.
    
    Useful for finding a good baseline size for benchmarking.
    
    Args:
        func: Function to calibrate.
        target_time: Target execution time in seconds.
    
    Returns:
        Input size that gives approximately target execution time.
    """
    # Binary search for the right size
    low, high = 10, 1000000
    
    while low < high:
        mid = (low + high) // 2
        
        try:
            start = time.perf_counter()
            func(mid)
            elapsed = time.perf_counter() - start
        except Exception:
            high = mid - 1
            continue
        
        if elapsed < target_time * 0.8:
            low = mid + 1
        elif elapsed > target_time * 1.2:
            high = mid - 1
        else:
            return mid
    
    return low


def detect_warmup_needed(func: Callable, size: int = 1000, runs: int = 5) -> bool:
    """
    Detect if function needs warmup (JIT compilation, caching effects).
    
    The first run is often slower due to:
    - JIT compilation (PyPy, Numba)
    - Cache warming
    - Lazy imports
    
    Args:
        func: Function to test.
        size: Input size to use.
        runs: Number of test runs.
    
    Returns:
        True if warmup is recommended (first run is >2x slower).
    """
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        try:
            func(size)
        except Exception:
            return False
        times.append(time.perf_counter() - start)
    
    if len(times) < 2:
        return False
    
    # Check if first run is significantly slower
    first = times[0]
    rest_avg = sum(times[1:]) / len(times[1:])
    
    return first > rest_avg * 2


def format_recommendation(rec: SizeRecommendation) -> str:
    """Format a size recommendation for display."""
    return f"""Input Size Recommendation
========================
Sizes: {rec.sizes}
Reason: {rec.reason}
Estimated Time: {rec.estimated_time:.1f}s
Confidence: {rec.confidence}"""

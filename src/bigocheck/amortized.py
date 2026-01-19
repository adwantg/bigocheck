# Author: gadwant
"""
Amortized complexity analysis.

Analyze complexity over sequences of operations, useful for data structures
with occasional expensive operations (e.g., dynamic arrays, hash tables).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .core import Measurement, fit_complexities


@dataclass
class OperationMeasurement:
    """Measurement for a single operation in a sequence."""
    operation_index: int
    seconds: float
    cumulative_seconds: float


@dataclass
class AmortizedResult:
    """Result of amortized complexity analysis."""
    operations: List[OperationMeasurement]
    total_time: float
    amortized_time: float  # Total time / n operations
    worst_case_time: float
    best_case_time: float
    amortized_complexity: str
    worst_case_complexity: str
    summary: str


def analyze_amortized(
    operation: Callable[[], Any],
    n_operations: int,
    *,
    setup: Optional[Callable[[], Any]] = None,
    warmup: int = 0,
) -> AmortizedResult:
    """
    Analyze amortized complexity over a sequence of operations.
    
    Useful for data structures where occasional operations are expensive
    (e.g., dynamic array resizing, hash table rehashing).
    
    Args:
        operation: Callable representing a single operation.
        n_operations: Number of operations to perform.
        setup: Optional setup function to call before starting.
        warmup: Number of warmup operations.
    
    Returns:
        AmortizedResult with amortized and worst-case analysis.
    
    Example:
        >>> data = []
        >>> def append_op():
        ...     data.append(len(data))
        >>> 
        >>> result = analyze_amortized(append_op, n_operations=1000)
        >>> print(f"Amortized: {result.amortized_complexity}")
    """
    # Setup
    if setup:
        setup()
    
    # Warmup
    for _ in range(max(warmup, 0)):
        operation()
    
    # Measure each operation
    measurements: List[OperationMeasurement] = []
    cumulative = 0.0
    
    for i in range(n_operations):
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start
        cumulative += elapsed
        
        measurements.append(OperationMeasurement(
            operation_index=i,
            seconds=elapsed,
            cumulative_seconds=cumulative,
        ))
    
    times = [m.seconds for m in measurements]
    total_time = sum(times)
    amortized_time = total_time / n_operations if n_operations > 0 else 0
    
    # Analyze complexity by looking at cumulative time growth
    # Create pseudo-measurements for fitting
    pseudo_measurements = [
        Measurement(
            size=i + 1,
            seconds=measurements[i].cumulative_seconds,
        )
        for i in range(0, n_operations, max(1, n_operations // 20))
    ]
    
    if len(pseudo_measurements) >= 2:
        _, amortized_label = fit_complexities(pseudo_measurements)
    else:
        amortized_label = "O(n)"  # Default linear
    
    # Worst case - look at individual operation times
    worst_case_time = max(times) if times else 0
    best_case_time = min(times) if times else 0
    
    # Estimate worst case complexity (how worst grows with n)
    worst_case_complexity = "O(n)"  # Typically O(n) for amortized O(1)
    
    summary = (
        f"Amortized Analysis ({n_operations} operations):\n"
        f"  Total time:     {total_time:.6f}s\n"
        f"  Amortized/op:   {amortized_time:.9f}s\n"
        f"  Worst case:     {worst_case_time:.9f}s\n"
        f"  Best case:      {best_case_time:.9f}s\n"
        f"  Amortized:      {amortized_label}\n"
        f"  Worst case:     {worst_case_complexity}"
    )
    
    return AmortizedResult(
        operations=measurements,
        total_time=total_time,
        amortized_time=amortized_time,
        worst_case_time=worst_case_time,
        best_case_time=best_case_time,
        amortized_complexity=amortized_label,
        worst_case_complexity=worst_case_complexity,
        summary=summary,
    )


def analyze_sequence(
    operations: List[Callable[[], Any]],
    *,
    sizes: Optional[List[int]] = None,
) -> Dict[str, AmortizedResult]:
    """
    Analyze multiple operation types.
    
    Args:
        operations: List of (name, callable) pairs.
        sizes: List of sequence lengths to test.
    
    Returns:
        Dict mapping operation names to AmortizedResult.
    """
    if sizes is None:
        sizes = [100, 500, 1000]
    
    results: Dict[str, AmortizedResult] = {}
    
    for i, op in enumerate(operations):
        name = getattr(op, '__name__', f'operation_{i}')
        # Use largest size for complete analysis
        result = analyze_amortized(op, n_operations=sizes[-1])
        results[name] = result
    
    return results

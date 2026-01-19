# Author: gadwant
"""
Async function benchmarking support.

Allows benchmarking of async functions and coroutines.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Tuple

from .core import Analysis, FitResult, Measurement, fit_complexities, fit_space_complexity


async def _run_async_trials(
    func: Callable[..., Coroutine[Any, Any, Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    trials: int,
) -> Tuple[List[float], Optional[int]]:
    """Run async function multiple times and return timings."""
    import tracemalloc
    import gc
    
    times: List[float] = []
    peak_memory: Optional[int] = None
    
    for trial_idx in range(max(trials, 1)):
        if trial_idx == 0:
            gc.collect()
            tracemalloc.start()
            start = time.perf_counter()
            await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            start = time.perf_counter()
            await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
        
        times.append(elapsed)
    
    return times, peak_memory


async def benchmark_async(
    func: Callable[..., Coroutine[Any, Any, Any]],
    sizes: Iterable[int],
    *,
    trials: int = 3,
    warmup: int = 0,
    memory: bool = False,
    arg_factory: Optional[Callable[[int], Tuple[Tuple[Any, ...], Dict[str, Any]]]] = None,
) -> Analysis:
    """
    Benchmark an async callable across input sizes.
    
    Args:
        func: Async callable to benchmark (async def function).
        sizes: Iterable of input sizes.
        trials: Number of repetitions per size (averaged).
        warmup: Number of warmup runs before timing.
        memory: If True, track peak memory usage.
        arg_factory: Optional callable returning (args, kwargs) for each n.
    
    Returns:
        Analysis object with measurements and complexity fits.
    
    Example:
        >>> async def async_sum(n):
        ...     await asyncio.sleep(0.001)
        ...     return sum(range(n))
        >>> 
        >>> import asyncio
        >>> analysis = asyncio.run(benchmark_async(async_sum, sizes=[100, 500, 1000]))
        >>> print(f"Best fit: {analysis.best_label}")
    """
    measurements: List[Measurement] = []
    sizes_list = list(sizes)
    
    for n in sizes_list:
        # Prepare arguments
        if arg_factory:
            args, kwargs = arg_factory(n)
        else:
            args, kwargs = (n,), {}
        
        # Warmup runs
        for _ in range(max(warmup, 0)):
            await func(*args, **kwargs)
        
        # Timed runs
        times, peak_memory = await _run_async_trials(
            func, args, kwargs, trials
        )
        
        if not memory:
            peak_memory = None
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        measurements.append(Measurement(
            size=int(n),
            seconds=avg_time,
            std_dev=std_dev,
            memory_bytes=peak_memory,
        ))
    
    # Fit complexities
    fits, best_label = fit_complexities(measurements)
    
    # Fit space complexity
    space_fits: List[FitResult] = []
    space_label: Optional[str] = None
    if memory:
        space_fits, space_label = fit_space_complexity(measurements)
    
    return Analysis(
        measurements=measurements,
        fits=fits,
        best_label=best_label,
        space_fits=space_fits,
        space_label=space_label,
    )


def run_benchmark_async(
    func: Callable[..., Coroutine[Any, Any, Any]],
    sizes: Iterable[int],
    **kwargs: Any,
) -> Analysis:
    """
    Synchronous wrapper for benchmark_async.
    
    Convenience function that handles asyncio.run() for you.
    
    Example:
        >>> async def async_sum(n):
        ...     return sum(range(n))
        >>> 
        >>> analysis = run_benchmark_async(async_sum, sizes=[100, 500, 1000])
    """
    return asyncio.run(benchmark_async(func, sizes, **kwargs))

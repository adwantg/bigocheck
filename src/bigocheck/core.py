# Author: gadwant
"""
Core benchmarking and complexity fitting utilities.
"""
from __future__ import annotations

import gc
import importlib
import math
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class Measurement:
    """Represents a single timing measurement at a given input size."""
    size: int
    seconds: float
    std_dev: float = 0.0
    memory_bytes: Optional[int] = None


@dataclass
class FitResult:
    """Result of fitting a complexity class to measurements."""
    label: str
    scale: float
    error: float


@dataclass
class Analysis:
    """Complete analysis result with measurements and complexity fits."""
    measurements: List[Measurement]
    fits: List[FitResult]
    best_label: str
    name: Optional[str] = None
    # Space complexity (only populated when memory=True)
    space_fits: List[FitResult] = field(default_factory=list)
    space_label: Optional[str] = None


def _safe_log(n: int) -> float:
    """Log base 2 with safe handling for n <= 1."""
    return math.log(max(n, 1) + 1, 2)


def _safe_exp2(n: int) -> float:
    """2^n with clamping to avoid overflow while preserving growth ordering."""
    return math.exp(min(n * math.log(2), 40.0))


def _safe_sqrt(n: int) -> float:
    """Square root of n."""
    return math.sqrt(max(n, 0))


def _safe_factorial(n: int) -> float:
    """Factorial with clamping to avoid overflow."""
    # Clamp to avoid massive computation; 20! is already ~ 2.4e18
    clamped = min(max(n, 0), 20)
    result = 1.0
    for i in range(2, clamped + 1):
        result *= i
    return result


def complexity_basis() -> Dict[str, Callable[[int], float]]:
    """
    Return dictionary of complexity class labels to their basis functions.
    
    Includes: O(1), O(log n), O(√n), O(n), O(n log n), O(n²), O(n³), O(2^n), O(n!)
    """
    return {
        "O(1)": lambda n: 1.0,
        "O(log n)": _safe_log,
        "O(√n)": _safe_sqrt,
        "O(n)": lambda n: float(n),
        "O(n log n)": lambda n: float(n) * _safe_log(n),
        "O(n^2)": lambda n: float(n) * float(n),
        "O(n^3)": lambda n: float(n) * float(n) * float(n),
        "O(2^n)": _safe_exp2,
        "O(n!)": _safe_factorial,
    }


def _relative_rmse(actual: List[float], predicted: List[float]) -> float:
    """Calculate relative root mean squared error."""
    eps = 1e-12
    num = 0.0
    for a, p in zip(actual, predicted):
        denom = max(abs(a), eps)
        num += ((a - p) / denom) ** 2
    return math.sqrt(num / max(len(actual), 1))


def fit_complexities(measurements: List[Measurement]) -> Tuple[List[FitResult], str]:
    """
    Fit measurements to complexity classes using least-squares scaling.
    
    Returns a tuple of (sorted list of FitResults, best matching label).
    """
    if not measurements:
        raise ValueError("No measurements provided.")

    n_values = [m.size for m in measurements]
    t_values = [m.seconds for m in measurements]

    results: List[FitResult] = []
    for label, basis_fn in complexity_basis().items():
        basis_vals = [max(basis_fn(n), 1e-12) for n in n_values]
        denom = sum(b * b for b in basis_vals)
        scale = sum(b * t for b, t in zip(basis_vals, t_values)) / denom if denom else 0.0
        preds = [scale * b for b in basis_vals]
        err = _relative_rmse(t_values, preds)
        results.append(FitResult(label=label, scale=scale, error=err))

    results.sort(key=lambda r: r.error)
    best = results[0].label
    return results, best


def fit_space_complexity(measurements: List[Measurement]) -> Tuple[List[FitResult], Optional[str]]:
    """
    Fit memory measurements to complexity classes using least-squares scaling.
    
    Returns a tuple of (sorted list of FitResults, best matching label).
    Returns ([], None) if no memory data is available.
    """
    # Filter measurements with memory data
    mem_measurements = [m for m in measurements if m.memory_bytes is not None]
    
    if len(mem_measurements) < 2:
        return [], None
    
    n_values = [m.size for m in mem_measurements]
    mem_values = [float(m.memory_bytes) for m in mem_measurements]  # type: ignore
    
    results: List[FitResult] = []
    for label, basis_fn in complexity_basis().items():
        basis_vals = [max(basis_fn(n), 1e-12) for n in n_values]
        denom = sum(b * b for b in basis_vals)
        scale = sum(b * m for b, m in zip(basis_vals, mem_values)) / denom if denom else 0.0
        preds = [scale * b for b in basis_vals]
        err = _relative_rmse(mem_values, preds)
        results.append(FitResult(label=label, scale=scale, error=err))
    
    results.sort(key=lambda r: r.error)
    best = results[0].label if results else None
    return results, best


def benchmark_function(
    func: Callable[..., Any],
    sizes: Iterable[int],
    *,
    trials: int = 3,
    warmup: int = 0,
    verbose: bool = False,
    memory: bool = False,
    arg_factory: Callable[[int], Tuple[Tuple[Any, ...], Dict[str, Any]]] | None = None,
) -> Analysis:
    """
    Benchmark a callable across input sizes and fit its empirical complexity.

    Args:
        func: Callable that accepts arguments produced by arg_factory or a single `n`.
        sizes: Iterable of input sizes to benchmark.
        trials: Number of repetitions per size (averaged).
        warmup: Number of warmup runs before timing (to warm caches/JIT).
        verbose: If True, print progress to stderr.
        memory: If True, track peak memory usage and fit space complexity.
        arg_factory: Optional callable returning (args, kwargs) for each n.

    Returns:
        Analysis object containing measurements, time complexity fits,
        and space complexity fits (when memory=True).
    """
    measurements: List[Measurement] = []
    sizes_list = list(sizes)
    
    for idx, n in enumerate(sizes_list):
        if verbose:
            print(f"[{idx + 1}/{len(sizes_list)}] Benchmarking n={n}...", file=sys.stderr)
        
        # Prepare arguments
        args: Tuple[Any, ...]
        kwargs: Dict[str, Any]
        if arg_factory:
            args, kwargs = arg_factory(n)
        else:
            args, kwargs = (n,), {}
        
        # Warmup runs (not timed)
        for _ in range(max(warmup, 0)):
            func(*args, **kwargs)
        
        # Timed runs
        times: List[float] = []
        peak_memory: Optional[int] = None
        
        for trial_idx in range(max(trials, 1)):
            # Regenerate args for each trial if using factory
            if arg_factory:
                args, kwargs = arg_factory(n)
            
            if memory and trial_idx == 0:
                # Force garbage collection before measuring memory
                gc.collect()
                tracemalloc.start()
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
            
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        measurements.append(Measurement(
            size=int(n),
            seconds=avg_time,
            std_dev=std_dev,
            memory_bytes=peak_memory,
        ))

    # Fit time complexity
    fits, best_label = fit_complexities(measurements)
    
    
    # Fit space complexity (if memory tracking was enabled)
    space_fits: List[FitResult] = []
    space_label: Optional[str] = None
    if memory:
        space_fits, space_best = fit_space_complexity(measurements)
        space_label = space_best

    return Analysis(
        measurements=measurements,
        fits=fits,
        best_label=best_label,
        name=getattr(func, "__name__", str(func)),
        space_fits=space_fits,
        space_label=space_label,
    )


def resolve_callable(target: str) -> Callable[..., Any]:
    """
    Import a callable from a string like 'module.sub:func'.
    
    Args:
        target: Import path in the form 'module.submodule:callable_name'.
    
    Returns:
        The resolved callable.
    
    Raises:
        ValueError: If the target format is invalid or the callable doesn't exist.
    """
    if ":" not in target:
        raise ValueError("Target must be in the form module.sub:callable")
    module_name, func_name = target.split(":", 1)
    # nosec B102 - Module path comes from CLI target argument, validated above
    module: ModuleType = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"{target!r} is not callable")
    return func

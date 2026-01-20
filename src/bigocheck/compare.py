# Author: gadwant
"""
A/B comparison utilities for comparing algorithm implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .core import Analysis, benchmark_function


@dataclass
class ComparisonResult:
    """Result of comparing two functions."""
    winner: Optional[str]  # "func_a", "func_b", or None (tie)
    func_a_name: str
    func_b_name: str
    func_a_label: str
    func_b_label: str
    func_a_analysis: Analysis
    func_b_analysis: Analysis
    speedup: float  # How much faster winner is (ratio)
    crossover_point: Optional[int]  # Size where one beats the other
    size_comparisons: List[Dict[str, Any]]
    summary: str


def compare_functions(
    func_a: Callable[..., Any],
    func_b: Callable[..., Any],
    sizes: List[int],
    *,
    trials: int = 3,
    warmup: int = 1,
    func_a_name: Optional[str] = None,
    func_b_name: Optional[str] = None,
) -> ComparisonResult:
    """
    Compare the performance of two functions across input sizes.
    
    Args:
        func_a: First function to compare.
        func_b: Second function to compare.
        sizes: Input sizes to test.
        trials: Number of trials per size.
        warmup: Warmup runs.
        func_a_name: Optional name for func_a.
        func_b_name: Optional name for func_b.
    
    Returns:
        ComparisonResult with winner, speedup, and details.
    
    Example:
        >>> result = compare_functions(linear_search, binary_search, [100, 1000, 10000])
        >>> print(result.winner)  # "func_b"
        >>> print(result.speedup)  # 5.2 (binary_search is 5.2x faster)
    """
    # Get function names
    name_a = func_a_name or getattr(func_a, '__name__', 'func_a')
    name_b = func_b_name or getattr(func_b, '__name__', 'func_b')
    
    # Benchmark both functions
    analysis_a = benchmark_function(func_a, sizes=sizes, trials=trials, warmup=warmup)
    analysis_b = benchmark_function(func_b, sizes=sizes, trials=trials, warmup=warmup)
    
    # Compare at each size
    size_comparisons = []
    a_wins = 0
    b_wins = 0
    crossover_point = None
    prev_winner = None
    
    total_time_a = 0.0
    total_time_b = 0.0
    
    for m_a, m_b in zip(analysis_a.measurements, analysis_b.measurements):
        total_time_a += m_a.seconds
        total_time_b += m_b.seconds
        
        if m_a.seconds < m_b.seconds:
            winner = "func_a"
            a_wins += 1
            ratio = m_b.seconds / max(m_a.seconds, 1e-12)
        elif m_b.seconds < m_a.seconds:
            winner = "func_b"
            b_wins += 1
            ratio = m_a.seconds / max(m_b.seconds, 1e-12)
        else:
            winner = "tie"
            ratio = 1.0
        
        # Detect crossover
        if prev_winner and prev_winner != winner and winner != "tie" and crossover_point is None:
            crossover_point = m_a.size
        
        if winner != "tie":
            prev_winner = winner
        
        size_comparisons.append({
            "size": m_a.size,
            "time_a": m_a.seconds,
            "time_b": m_b.seconds,
            "winner": winner,
            "speedup": ratio,
        })
    
    # Determine overall winner
    if a_wins > b_wins:
        overall_winner = "func_a"
        speedup = total_time_b / max(total_time_a, 1e-12)
    elif b_wins > a_wins:
        overall_winner = "func_b"
        speedup = total_time_a / max(total_time_b, 1e-12)
    else:
        overall_winner = None
        speedup = 1.0
    
    # Generate summary
    if overall_winner == "func_a":
        summary = f"{name_a} is {speedup:.2f}x faster than {name_b} overall ({a_wins}/{len(sizes)} sizes)"
    elif overall_winner == "func_b":
        summary = f"{name_b} is {speedup:.2f}x faster than {name_a} overall ({b_wins}/{len(sizes)} sizes)"
    else:
        summary = f"{name_a} and {name_b} are roughly equivalent"
    
    if crossover_point:
        summary += f". Crossover at n={crossover_point}"
    
    return ComparisonResult(
        winner=overall_winner,
        func_a_name=name_a,
        func_b_name=name_b,
        func_a_label=analysis_a.best_label,
        func_b_label=analysis_b.best_label,
        func_a_analysis=analysis_a,
        func_b_analysis=analysis_b,
        speedup=speedup,
        crossover_point=crossover_point,
        size_comparisons=size_comparisons,
        summary=summary,
    )


def compare_to_baseline(
    func: Callable[..., Any],
    baseline_label: str,
    sizes: List[int],
    *,
    trials: int = 3,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Compare a function's actual complexity to a baseline complexity.
    
    Args:
        func: Function to analyze.
        baseline_label: Expected complexity (e.g., "O(n)").
        sizes: Input sizes.
        trials: Trials per size.
        warmup: Warmup runs.
    
    Returns:
        Dictionary with comparison results.
    """
    
    analysis = benchmark_function(func, sizes=sizes, trials=trials, warmup=warmup)
    
    # Get baseline fit
    baseline_fit = next((f for f in analysis.fits if f.label == baseline_label), None)
    best_fit = analysis.fits[0]
    
    if baseline_fit is None:
        return {
            "matches_baseline": False,
            "actual": analysis.best_label,
            "expected": baseline_label,
            "error": f"Unknown baseline: {baseline_label}",
        }
    
    # Compare errors
    error_diff = baseline_fit.error - best_fit.error
    matches = error_diff <= 0.3  # tolerance
    
    return {
        "matches_baseline": matches,
        "actual": analysis.best_label,
        "expected": baseline_label,
        "actual_error": best_fit.error,
        "baseline_error": baseline_fit.error,
        "error_difference": error_diff,
        "analysis": analysis,
    }

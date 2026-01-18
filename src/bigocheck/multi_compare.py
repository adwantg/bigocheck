# Author: gadwant
"""
Multiple algorithm comparison.

Compare N algorithms at once with summary tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .core import Analysis, benchmark_function


@dataclass
class AlgorithmResult:
    """Result for a single algorithm."""
    name: str
    analysis: Analysis
    time_complexity: str
    space_complexity: Optional[str]
    avg_time: float
    rank: int = 0


@dataclass
class ComparisonSummary:
    """Summary of multiple algorithm comparison."""
    results: List[AlgorithmResult]
    winner: str
    fastest: str
    summary_table: str


def compare_algorithms(
    algorithms: Dict[str, Callable],
    sizes: List[int],
    *,
    trials: int = 3,
    memory: bool = False,
    rank_by: str = "complexity",  # "complexity" or "time"
) -> ComparisonSummary:
    """
    Compare multiple algorithms at once.
    
    Args:
        algorithms: Dict mapping names to functions.
        sizes: Input sizes to benchmark.
        trials: Number of trials per size.
        memory: Track memory usage.
        rank_by: Ranking criterion - "complexity" or "time".
    
    Returns:
        ComparisonSummary with ranked results and summary table.
    
    Example:
        >>> results = compare_algorithms({
        ...     "bubble_sort": bubble_sort,
        ...     "quick_sort": quick_sort,
        ...     "merge_sort": merge_sort,
        ... }, sizes=[100, 500, 1000])
        >>> print(results.summary_table)
    """
    results: List[AlgorithmResult] = []
    
    for name, func in algorithms.items():
        analysis = benchmark_function(func, sizes=sizes, trials=trials, memory=memory)
        avg_time = sum(m.seconds for m in analysis.measurements) / len(analysis.measurements)
        
        results.append(AlgorithmResult(
            name=name,
            analysis=analysis,
            time_complexity=analysis.best_label,
            space_complexity=analysis.space_label,
            avg_time=avg_time,
        ))
    
    # Rank results
    if rank_by == "complexity":
        # Rank by complexity class (lower is better)
        complexity_order = [
            "O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)",
            "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)",
        ]
        
        def get_order(r: AlgorithmResult) -> Tuple[int, float]:
            try:
                idx = complexity_order.index(r.time_complexity)
            except ValueError:
                idx = len(complexity_order)
            return (idx, r.avg_time)
        
        results.sort(key=get_order)
    else:
        # Rank by average time
        results.sort(key=lambda r: r.avg_time)
    
    # Assign ranks
    for i, r in enumerate(results):
        r.rank = i + 1
    
    # Determine winners
    winner = results[0].name
    fastest = min(results, key=lambda r: r.avg_time).name
    
    # Generate summary table
    table = _generate_summary_table(results)
    
    return ComparisonSummary(
        results=results,
        winner=winner,
        fastest=fastest,
        summary_table=table,
    )


def _generate_summary_table(results: List[AlgorithmResult]) -> str:
    """Generate a formatted summary table."""
    lines = [
        "Algorithm Comparison Summary",
        "=" * 70,
        "",
        f"{'Rank':<6} {'Algorithm':<20} {'Time Complexity':<15} {'Avg Time':<12}",
        "-" * 70,
    ]
    
    for r in results:
        medal = "🥇" if r.rank == 1 else "🥈" if r.rank == 2 else "🥉" if r.rank == 3 else "  "
        lines.append(
            f"{medal} {r.rank:<3} {r.name:<20} {r.time_complexity:<15} {r.avg_time:.6f}s"
        )
    
    lines.extend([
        "",
        f"Winner (by complexity): {results[0].name} ({results[0].time_complexity})",
    ])
    
    return "\n".join(lines)


def generate_markdown_comparison(results: ComparisonSummary) -> str:
    """Generate a markdown table for the comparison."""
    lines = [
        "## Algorithm Comparison",
        "",
        "| Rank | Algorithm | Time Complexity | Space Complexity | Avg Time |",
        "|------|-----------|-----------------|------------------|----------|",
    ]
    
    for r in results.results:
        medal = "🥇" if r.rank == 1 else "🥈" if r.rank == 2 else "🥉" if r.rank == 3 else ""
        space = r.space_complexity or "-"
        lines.append(
            f"| {medal} {r.rank} | {r.name} | {r.time_complexity} | {space} | {r.avg_time:.6f}s |"
        )
    
    lines.extend([
        "",
        f"**Winner**: {results.winner} ({results.results[0].time_complexity})",
    ])
    
    return "\n".join(lines)

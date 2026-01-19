# Author: gadwant
"""
Best/Worst/Average case analysis utilities.

Test algorithms with different input arrangements to understand
how they perform in best, worst, and average scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis


@dataclass
class CaseResult:
    """Result for a single case (best/worst/average)."""
    case_name: str
    analysis: "Analysis"
    avg_time: float
    time_complexity: str
    space_complexity: Optional[str]


@dataclass
class CasesAnalysis:
    """Complete analysis across all cases."""
    results: Dict[str, CaseResult]
    best_case: CaseResult
    worst_case: CaseResult
    average_case: CaseResult
    summary: str


def _default_generators() -> Dict[str, Callable[[int], Any]]:
    """Default input generators for different cases."""
    import random  # nosec B311 - Not used for cryptographic purposes
    
    def sorted_list(n: int) -> List[int]:
        return list(range(n))
    
    def reversed_list(n: int) -> List[int]:
        return list(range(n, 0, -1))
    
    def random_list(n: int) -> List[int]:
        return [random.randint(0, n * 10) for _ in range(n)]
    
    def nearly_sorted(n: int) -> List[int]:
        data = list(range(n))
        # Swap a few random elements
        swaps = max(1, n // 10)
        for _ in range(swaps):
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            data[i], data[j] = data[j], data[i]
        return data
    
    return {
        "best": sorted_list,
        "worst": reversed_list,
        "average": random_list,
        "nearly_sorted": nearly_sorted,
    }


def analyze_cases(
    func: Callable[..., Any],
    sizes: List[int],
    *,
    cases: Optional[Dict[str, Callable[[int], Any]]] = None,
    trials: int = 3,
    warmup: int = 1,
    memory: bool = False,
) -> CasesAnalysis:
    """
    Analyze a function across best, worst, and average cases.
    
    This is useful for understanding how algorithms perform with different
    input arrangements (e.g., sorted vs reversed vs random).
    
    Args:
        func: Function to analyze. Should accept the generator's output.
        sizes: Input sizes to test.
        cases: Dict mapping case names to generator functions.
               Defaults to {"best": sorted, "worst": reversed, "average": random}.
        trials: Number of trials per size.
        warmup: Warmup runs before timing.
        memory: Track memory usage.
    
    Returns:
        CasesAnalysis with results for each case.
    
    Example:
        >>> def my_sort(arr):
        ...     return sorted(arr)
        >>> 
        >>> results = analyze_cases(my_sort, sizes=[1000, 5000, 10000])
        >>> print(results.summary)
    """
    from .core import benchmark_function
    
    if cases is None:
        cases = {
            "best": _default_generators()["best"],
            "worst": _default_generators()["worst"],
            "average": _default_generators()["average"],
        }
    
    results: Dict[str, CaseResult] = {}
    
    for case_name, generator in cases.items():
        # Create arg factory for this case
        def make_factory(gen: Callable[[int], Any]):
            def factory(n: int) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
                return (gen(n),), {}
            return factory
        
        analysis = benchmark_function(
            func,
            sizes=sizes,
            trials=trials,
            warmup=warmup,
            memory=memory,
            arg_factory=make_factory(generator),
        )
        
        avg_time = sum(m.seconds for m in analysis.measurements) / len(analysis.measurements)
        
        results[case_name] = CaseResult(
            case_name=case_name,
            analysis=analysis,
            avg_time=avg_time,
            time_complexity=analysis.best_label,
            space_complexity=analysis.space_label,
        )
    
    # Identify best, worst, and average cases
    sorted_by_time = sorted(results.values(), key=lambda r: r.avg_time)
    best_case = sorted_by_time[0]
    worst_case = sorted_by_time[-1]
    average_case = results.get("average", sorted_by_time[len(sorted_by_time) // 2])
    
    # Generate summary
    summary_lines = [
        "Case Analysis Summary:",
        f"  Best case:    {best_case.case_name} - {best_case.time_complexity}",
        f"  Worst case:   {worst_case.case_name} - {worst_case.time_complexity}",
        f"  Average case: {average_case.case_name} - {average_case.time_complexity}",
        "",
        "Time by case:",
    ]
    for case_name, result in results.items():
        summary_lines.append(
            f"  {case_name:<15} avg={result.avg_time:.6f}s  complexity={result.time_complexity}"
        )
    
    # Check if complexity differs across cases
    complexities = {r.time_complexity for r in results.values()}
    if len(complexities) > 1:
        summary_lines.append("")
        summary_lines.append(f"⚠️  Complexity varies by input: {', '.join(complexities)}")
    
    return CasesAnalysis(
        results=results,
        best_case=best_case,
        worst_case=worst_case,
        average_case=average_case,
        summary="\n".join(summary_lines),
    )


def format_cases_result(cases_result: CasesAnalysis) -> str:
    """Format cases analysis for display."""
    lines = [
        "═" * 50,
        "CASE ANALYSIS",
        "═" * 50,
        "",
    ]
    
    for case_name, result in cases_result.results.items():
        lines.append(f"【{case_name.upper()}】")
        lines.append(f"  Time Complexity:  {result.time_complexity}")
        if result.space_complexity:
            lines.append(f"  Space Complexity: {result.space_complexity}")
        lines.append(f"  Average Time:     {result.avg_time:.6f}s")
        lines.append("")
    
    lines.append("─" * 50)
    lines.append(
        f"Best:  {cases_result.best_case.case_name} "
        f"({cases_result.best_case.time_complexity})"
    )
    lines.append(
        f"Worst: {cases_result.worst_case.case_name} "
        f"({cases_result.worst_case.time_complexity})"
    )
    
    return "\n".join(lines)

# Author: gadwant
"""
Complexity explanations for educational purposes.

Human-readable explanations of what complexity classes mean.
"""
from __future__ import annotations

from typing import Dict, Optional


# Comprehensive complexity explanations
COMPLEXITY_INFO: Dict[str, Dict[str, str]] = {
    "O(1)": {
        "name": "Constant",
        "description": "Execution time doesn't change with input size.",
        "example": "Array index lookup, hash table access",
        "scaling": "Excellent - handles any input size instantly",
        "real_world": "Looking up a key in a dictionary",
    },
    "O(log n)": {
        "name": "Logarithmic",
        "description": "Execution time grows slowly as input doubles.",
        "example": "Binary search, balanced tree operations",
        "scaling": "Excellent - 1M elements only ~20 operations",
        "real_world": "Finding a word in a dictionary by halving",
    },
    "O(√n)": {
        "name": "Square Root",
        "description": "Execution time grows with the square root of input.",
        "example": "Jump search, some number theory algorithms",
        "scaling": "Very Good - 1M elements = 1000 operations",
        "real_world": "Checking if a number is prime (trial division)",
    },
    "O(n)": {
        "name": "Linear",
        "description": "Execution time grows proportionally with input size.",
        "example": "Linear search, array sum, simple iteration",
        "scaling": "Good - 2x input = 2x time",
        "real_world": "Reading every email in your inbox",
    },
    "O(n log n)": {
        "name": "Linearithmic",
        "description": "Slightly more than linear, common in efficient sorting.",
        "example": "Merge sort, heap sort, quick sort (average)",
        "scaling": "Good - optimal for comparison-based sorting",
        "real_world": "Sorting a playlist by song name",
    },
    "O(n^2)": {
        "name": "Quadratic",
        "description": "Execution time squares with input - nested loops.",
        "example": "Bubble sort, selection sort, comparing all pairs",
        "scaling": "Poor - 10x input = 100x time",
        "real_world": "Checking if any two people share a birthday (brute force)",
    },
    "O(n^3)": {
        "name": "Cubic",
        "description": "Three nested loops over input.",
        "example": "Naive matrix multiplication, some DP algorithms",
        "scaling": "Bad - 10x input = 1000x time",
        "real_world": "Comparing all possible triangles from n points",
    },
    "O(2^n)": {
        "name": "Exponential",
        "description": "Execution time doubles with each additional element.",
        "example": "Generating all subsets, naive Fibonacci",
        "scaling": "Terrible - only practical for n < 30",
        "real_world": "Trying every combination of on/off switches",
    },
    "O(n!)": {
        "name": "Factorial",
        "description": "Execution time explodes - all permutations.",
        "example": "Generating all permutations, brute force TSP",
        "scaling": "Catastrophic - only for n < 12",
        "real_world": "Trying every possible ordering of cities to visit",
    },
}


def explain_complexity(label: str) -> str:
    """
    Get a human-readable explanation of a complexity class.
    
    Args:
        label: Complexity label (e.g., "O(n log n)").
    
    Returns:
        Multi-line explanation string.
    
    Example:
        >>> print(explain_complexity("O(n log n)"))
        O(n log n) - Linearithmic
        
        Description: Slightly more than linear, common in efficient sorting.
        Example: Merge sort, heap sort, quick sort (average)
        Scaling: Good - optimal for comparison-based sorting
        Real World: Sorting a playlist by song name
    """
    info = COMPLEXITY_INFO.get(label)
    
    if not info:
        # Check for polynomial O(n^k)
        if label.startswith("O(n^"):
            try:
                exp = float(label[4:-1])
                return f"""{label} - Polynomial (degree {exp:.1f})

Description: Execution time grows as n^{exp:.1f}.
Scaling: {'Acceptable' if exp <= 2 else 'Poor' if exp <= 3 else 'Bad'}
Note: Detected via empirical polynomial fitting."""
            except ValueError:
                pass
        return f"{label} - Unknown complexity class"
    
    return f"""{label} - {info['name']}

Description: {info['description']}
Example: {info['example']}
Scaling: {info['scaling']}
Real World: {info['real_world']}"""


def get_complexity_comparison() -> str:
    """
    Get a comparison table of all complexity classes.
    
    Returns:
        Formatted comparison table.
    """
    lines = [
        "Complexity Class Comparison",
        "=" * 60,
        "",
        f"{'Class':<12} {'Name':<14} {'n=1000':<12} {'n=1M':<15}",
        "-" * 60,
    ]
    
    # Approximate operations for each complexity at n=1000 and n=1M
    operations = {
        "O(1)": ("1", "1"),
        "O(log n)": ("10", "20"),
        "O(√n)": ("32", "1,000"),
        "O(n)": ("1,000", "1,000,000"),
        "O(n log n)": ("10,000", "20,000,000"),
        "O(n^2)": ("1,000,000", "1,000,000,000,000"),
        "O(n^3)": ("1,000,000,000", "10^18"),
        "O(2^n)": ("10^301", "∞"),
        "O(n!)": ("∞", "∞"),
    }
    
    for label, info in COMPLEXITY_INFO.items():
        ops = operations.get(label, ("?", "?"))
        lines.append(f"{label:<12} {info['name']:<14} {ops[0]:<12} {ops[1]:<15}")
    
    return "\n".join(lines)


def suggest_improvement(current: str) -> Optional[str]:
    """
    Suggest a better complexity class if possible.
    
    Args:
        current: Current complexity label.
    
    Returns:
        Suggestion string or None if already optimal.
    """
    suggestions = {
        "O(n^2)": "Consider using a hash set for O(n) or sorting for O(n log n).",
        "O(n^3)": "Look for dynamic programming or matrix optimization techniques.",
        "O(2^n)": "Consider memoization or dynamic programming to reduce complexity.",
        "O(n!)": "Look for greedy algorithms or approximation methods.",
    }
    
    return suggestions.get(current)


def format_complexity_report(label: str, include_suggestions: bool = True) -> str:
    """
    Generate a complete complexity report with explanation and suggestions.
    
    Args:
        label: Complexity label.
        include_suggestions: Include improvement suggestions.
    
    Returns:
        Formatted report string.
    """
    lines = [explain_complexity(label)]
    
    if include_suggestions:
        suggestion = suggest_improvement(label)
        if suggestion:
            lines.extend([
                "",
                "💡 Optimization Tip:",
                f"   {suggestion}",
            ])
    
    return "\n".join(lines)

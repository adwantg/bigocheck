# Author: gadwant
"""
Documentation generator for complexity analysis.

Auto-generate docstrings with complexity information.
"""
from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

from .core import Analysis, benchmark_function


def generate_complexity_docstring(
    analysis: Analysis,
    func_name: str = "function",
) -> str:
    """
    Generate a docstring section with complexity information.
    
    Args:
        analysis: Analysis object from benchmark_function.
        func_name: Function name for the docstring.
    
    Returns:
        Docstring section to append.
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> docstring = generate_complexity_docstring(analysis, "my_func")
        >>> print(docstring)
    """
    lines = [
        "",
        "Complexity:",
        f"    Time: {analysis.best_label}",
    ]
    
    if analysis.space_label:
        lines.append(f"    Space: {analysis.space_label}")
    
    lines.extend([
        "",
        "Note:",
        "    Complexity measured empirically by bigocheck.",
    ])
    
    return "\n".join(lines)


def document_complexity(
    sizes: Optional[list] = None,
    trials: int = 3,
    update_docstring: bool = True,
) -> Callable:
    """
    Decorator to automatically document function complexity.
    
    On first call, benchmarks the function and appends complexity
    information to its docstring.
    
    Args:
        sizes: Input sizes for benchmarking.
        trials: Number of trials.
        update_docstring: Whether to modify __doc__.
    
    Returns:
        Decorated function.
    
    Example:
        >>> @document_complexity()
        ... def my_sort(n):
        ...     '''Sort numbers from 0 to n.'''
        ...     return sorted(range(n))
        >>> 
        >>> my_sort(100)  # Triggers complexity analysis
        >>> print(my_sort.__doc__)
        # Sort numbers from 0 to n.
        # 
        # Complexity:
        #     Time: O(n log n)
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000]
    
    def decorator(func: Callable) -> Callable:
        _documented = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _documented
            
            if not _documented and update_docstring:
                analysis = benchmark_function(func, sizes=sizes, trials=trials)
                complexity_doc = generate_complexity_docstring(analysis, func.__name__)
                
                original_doc = func.__doc__ or ""
                wrapper.__doc__ = original_doc + complexity_doc
                wrapper._complexity = analysis.best_label
                wrapper._analysis = analysis
                
                _documented = True
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_complexity_annotation(analysis: Analysis) -> str:
    """
    Get a complexity annotation string for type hints.
    
    Returns a comment that can be added to function signature.
    
    Args:
        analysis: Analysis object.
    
    Returns:
        Annotation string.
    """
    space = f", space={analysis.space_label}" if analysis.space_label else ""
    return f"# Complexity: time={analysis.best_label}{space}"


def generate_readme_entry(
    func: Callable,
    analysis: Analysis,
) -> str:
    """
    Generate a README documentation entry for a function.
    
    Args:
        func: Function to document.
        analysis: Analysis results.
    
    Returns:
        Markdown string for README.
    """
    name = func.__name__
    doc = func.__doc__ or "No description available."
    
    # Get first line of docstring
    first_line = doc.split("\n")[0].strip()
    
    return f"""### `{name}`

{first_line}

- **Time Complexity:** {analysis.best_label}
- **Space Complexity:** {analysis.space_label or "Not measured"}
"""


def generate_api_docs(
    functions: dict,
    sizes: Optional[list] = None,
) -> str:
    """
    Generate API documentation with complexity for multiple functions.
    
    Args:
        functions: Dict mapping names to functions.
        sizes: Input sizes for benchmarking.
    
    Returns:
        Markdown documentation string.
    """
    if sizes is None:
        sizes = [100, 500, 1000]
    
    lines = ["# API Reference", "", "## Functions", ""]
    
    for name, func in functions.items():
        analysis = benchmark_function(func, sizes=sizes, trials=2)
        entry = generate_readme_entry(func, analysis)
        lines.append(entry)
    
    return "\n".join(lines)

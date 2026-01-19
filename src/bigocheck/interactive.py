# Author: gadwant
"""
Interactive REPL mode for bigocheck.

Provides a simple REPL for quick complexity analysis.
"""
from __future__ import annotations

import code
from typing import Any, Callable, List, Optional


def start_repl() -> None:
    """
    Start an interactive REPL for complexity analysis.
    
    Provides a Python REPL with bigocheck functions pre-imported.
    
    Example:
        >>> # From command line:
        >>> # bigocheck repl
        >>> # 
        >>> # Or from Python:
        >>> from bigocheck.interactive import start_repl
        >>> start_repl()
    """
    # Pre-import common functions
    from bigocheck import (
        benchmark_function,
        compare_functions,
        verify_bounds,
        compute_confidence,
        compute_significance,
        analyze_cases,
        fit_polynomial,
        generate_report,
        generate_html_report,
    )
    
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  bigocheck Interactive Mode                                       ║
║  Zero-dependency complexity analysis                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Quick start:                                                     ║
║  >>> def my_func(n): return sum(range(n))                        ║
║  >>> a = benchmark_function(my_func, sizes=[100, 500, 1000])     ║
║  >>> print(a.best_label)                                         ║
║                                                                   ║
║  Available functions:                                             ║
║    benchmark_function   - Run complexity analysis                 ║
║    compare_functions    - Compare two implementations             ║
║    verify_bounds        - Verify expected complexity              ║
║    compute_confidence   - Get confidence score                    ║
║    compute_significance - Get p-value                             ║
║    analyze_cases        - Best/worst/avg case analysis            ║
║    fit_polynomial       - Detect O(n^k)                           ║
║    generate_report      - Generate markdown report                ║
║    generate_html_report - Generate HTML report                    ║
║                                                                   ║
║  Type 'exit()' to quit.                                          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    local_vars = {
        'benchmark_function': benchmark_function,
        'compare_functions': compare_functions,
        'verify_bounds': verify_bounds,
        'compute_confidence': compute_confidence,
        'compute_significance': compute_significance,
        'analyze_cases': analyze_cases,
        'fit_polynomial': fit_polynomial,
        'generate_report': generate_report,
        'generate_html_report': generate_html_report,
    }
    
    print(banner)
    code.interact(local=local_vars, banner='', exitmsg='Goodbye!')


def quick_check(func: Callable[..., Any], sizes: Optional[List[int]] = None) -> str:
    """
    Quick one-liner complexity check.
    
    Args:
        func: Function to analyze.
        sizes: Optional list of sizes. Default: [100, 500, 1000, 5000].
    
    Returns:
        Best-fit complexity label as string.
    
    Example:
        >>> def my_func(n): return sum(range(n))
        >>> quick_check(my_func)
        'O(n)'
    """
    from bigocheck import benchmark_function
    
    if sizes is None:
        sizes = [100, 500, 1000, 5000]
    
    analysis = benchmark_function(func, sizes=sizes, trials=1)
    return analysis.best_label

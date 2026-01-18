# Author: gadwant
"""
bigocheck: empirical complexity regression checker.
"""

# Core functionality
from .core import (
    Analysis,
    FitResult,
    Measurement,
    benchmark_function,
    fit_complexities,
    complexity_basis,
    resolve_callable,
)

# Assertions and verification
from .assertions import (
    ComplexityAssertionError,
    ConfidenceResult,
    VerificationResult,
    assert_complexity,
    auto_select_sizes,
    compute_confidence,
    verify_bounds,
)

# Comparison
from .compare import (
    ComparisonResult,
    compare_functions,
    compare_to_baseline,
)

# Reports
from .reports import (
    generate_report,
    generate_comparison_report,
    generate_verification_report,
    save_report,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Analysis",
    "FitResult",
    "Measurement",
    "benchmark_function",
    "fit_complexities",
    "complexity_basis",
    "resolve_callable",
    # Assertions
    "ComplexityAssertionError",
    "ConfidenceResult",
    "VerificationResult",
    "assert_complexity",
    "auto_select_sizes",
    "compute_confidence",
    "verify_bounds",
    # Comparison
    "ComparisonResult",
    "compare_functions",
    "compare_to_baseline",
    # Reports
    "generate_report",
    "generate_comparison_report",
    "generate_verification_report",
    "save_report",
]

__version__ = "0.2.0"

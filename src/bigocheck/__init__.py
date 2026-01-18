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
    fit_space_complexity,
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

# Statistics (p-values)
from .statistics import (
    SignificanceResult,
    compute_significance,
    format_significance,
)

# Regression detection
from .regression import (
    Baseline,
    RegressionResult,
    save_baseline,
    load_baseline,
    detect_regression,
    compare_to_baseline_file,
)

# Case analysis (best/worst/avg)
from .cases import (
    CaseResult,
    CasesAnalysis,
    analyze_cases,
    format_cases_result,
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
    "fit_space_complexity",
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
    # Statistics
    "SignificanceResult",
    "compute_significance",
    "format_significance",
    # Regression
    "Baseline",
    "RegressionResult",
    "save_baseline",
    "load_baseline",
    "detect_regression",
    "compare_to_baseline_file",
    # Cases
    "CaseResult",
    "CasesAnalysis",
    "analyze_cases",
    "format_cases_result",
]

__version__ = "0.4.0"

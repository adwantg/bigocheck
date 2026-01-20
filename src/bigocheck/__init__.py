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

# Polynomial fitting
from .polynomial import (
    PolynomialFit,
    fit_polynomial,
    fit_polynomial_space,
)

# Async benchmarking
from .async_bench import (
    benchmark_async,
    run_benchmark_async,
)

# Amortized analysis
from .amortized import (
    AmortizedResult,
    OperationMeasurement,
    analyze_amortized,
    analyze_sequence,
)

# Parallel benchmarking
from .parallel import (
    benchmark_parallel,
    speedup_estimate,
)

# HTML reports
from .html_report import (
    generate_html_report,
    save_html_report,
)

# Interactive mode
from .interactive import (
    start_repl,
    quick_check,
)

# Git tracking
from .git_tracking import (
    CommitResult,
    TrackingResult,
    track_commits,
    find_regression_commit,
)

# Stability detection
from .stability import (
    StabilityResult,
    compute_stability,
    format_stability,
)

# Badge generation
from .badges import (
    generate_badge,
    generate_dual_badge,
    save_badge,
    generate_badge_url,
)

# Jupyter integration
from .jupyter import (
    enable_jupyter_display,
    display_analysis,
    display_comparison,
)

# Export functionality
from .export import (
    to_csv,
    to_markdown_table,
    to_dict,
    to_json,
)

# Threshold alerts
from .alerts import (
    ComplexityThresholdError,
    ThresholdResult,
    check_threshold,
    assert_threshold,
    monitor_complexity,
)

# Explanations
from .explanations import (
    explain_complexity,
    get_complexity_comparison,
    suggest_improvement,
    format_complexity_report,
)

# Recommendations
from .recommendations import (
    SizeRecommendation,
    suggest_sizes,
    auto_calibrate,
    detect_warmup_needed,
)

# Multi-algorithm comparison
from .multi_compare import (
    AlgorithmResult,
    ComparisonSummary,
    compare_algorithms,
    generate_markdown_comparison,
)

# Bounds checking
from .bounds import (
    ComplexityBoundsError,
    BoundsResult,
    check_bounds,
    assert_bounds,
)

# Benchmark profiles
from .profiles import (
    BenchmarkProfile,
    get_profile,
    benchmark_with_profile,
    profile_decorator,
    list_profiles,
)

# Documentation generation
from .docgen import (
    generate_complexity_docstring,
    document_complexity,
    generate_readme_entry,
)

# Differentiation Features (v0.8.0-dev)
from .ast_analysis import (
    predict_complexity,
    verify_hybrid,
)

from .dashboard import (
    generate_dashboard,
)

from .cloud import (
    generate_github_action,
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
    # Polynomial
    "PolynomialFit",
    "fit_polynomial",
    "fit_polynomial_space",
    # Async
    "benchmark_async",
    "run_benchmark_async",
    # Amortized
    "AmortizedResult",
    "OperationMeasurement",
    "analyze_amortized",
    "analyze_sequence",
    # Parallel
    "benchmark_parallel",
    "speedup_estimate",
    # HTML
    "generate_html_report",
    "save_html_report",
    # Interactive
    "start_repl",
    "quick_check",
    # Git Tracking
    "CommitResult",
    "TrackingResult",
    "track_commits",
    "find_regression_commit",
    # Stability
    "StabilityResult",
    "compute_stability",
    "format_stability",
    # Badges
    "generate_badge",
    "generate_dual_badge",
    "save_badge",
    "generate_badge_url",
    # Jupyter
    "enable_jupyter_display",
    "display_analysis",
    "display_comparison",
    # Export
    "to_csv",
    "to_markdown_table",
    "to_dict",
    "to_json",
    # Alerts
    "ComplexityThresholdError",
    "ThresholdResult",
    "check_threshold",
    "assert_threshold",
    "monitor_complexity",
    # Explanations
    "explain_complexity",
    "get_complexity_comparison",
    "suggest_improvement",
    "format_complexity_report",
    # Recommendations
    "SizeRecommendation",
    "suggest_sizes",
    "auto_calibrate",
    "detect_warmup_needed",
    # Multi-compare
    "AlgorithmResult",
    "ComparisonSummary",
    "compare_algorithms",
    "generate_markdown_comparison",
    # Bounds
    "ComplexityBoundsError",
    "BoundsResult",
    "check_bounds",
    "assert_bounds",
    # Profiles
    "BenchmarkProfile",
    "get_profile",
    "benchmark_with_profile",
    "profile_decorator",
    "list_profiles",
    # Docgen
    "generate_complexity_docstring",
    "document_complexity",
    "generate_readme_entry",
    # Differentiation (v0.8.0)
    "predict_complexity",
    "verify_hybrid",
    "generate_dashboard",
    "generate_github_action",
]

__version__ = "0.8.0"




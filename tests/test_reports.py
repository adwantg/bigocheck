# Author: gadwant
"""
Tests for reports module: generate_report, generate_comparison_report.
"""

from bigocheck import (
    benchmark_function,
    compare_functions,
    verify_bounds,
    generate_report,
    generate_comparison_report,
    generate_verification_report,
)


def linear_func(n):
    """O(n): Linear time."""
    return sum(range(n))


def constant_func(n):
    """O(1): Constant time."""
    return 42


class TestGenerateReport:
    """Tests for generate_report function."""
    
    def test_report_is_string(self):
        """Report generation returns string."""
        analysis = benchmark_function(linear_func, sizes=[100, 200, 400], trials=1)
        report = generate_report(analysis)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_report_contains_title(self):
        """Report contains the title."""
        analysis = benchmark_function(linear_func, sizes=[100, 200], trials=1)
        report = generate_report(analysis, title="My Test Report")
        
        assert "My Test Report" in report
    
    def test_report_contains_best_fit(self):
        """Report contains best fit label."""
        analysis = benchmark_function(linear_func, sizes=[100, 200, 400], trials=1)
        report = generate_report(analysis)
        
        assert analysis.best_label in report
    
    def test_report_contains_measurements(self):
        """Report contains measurements table."""
        analysis = benchmark_function(linear_func, sizes=[100, 200], trials=1)
        report = generate_report(analysis, include_measurements=True)
        
        assert "100" in report
        assert "200" in report
    
    def test_report_markdown_format(self):
        """Report is in markdown format."""
        analysis = benchmark_function(linear_func, sizes=[100, 200], trials=1)
        report = generate_report(analysis)
        
        assert report.startswith("#")
        assert "|" in report  # Tables


class TestGenerateComparisonReport:
    """Tests for generate_comparison_report function."""
    
    def test_comparison_report_is_string(self):
        """Comparison report returns string."""
        result = compare_functions(
            constant_func,
            linear_func,
            sizes=[100, 500],
            trials=1,
        )
        report = generate_comparison_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_comparison_report_contains_functions(self):
        """Comparison report contains function names."""
        result = compare_functions(
            constant_func,
            linear_func,
            sizes=[100, 500],
            trials=1,
        )
        report = generate_comparison_report(result)
        
        assert "constant_func" in report
        assert "linear_func" in report
    
    def test_comparison_report_contains_winner(self):
        """Comparison report mentions winner."""
        result = compare_functions(
            constant_func,
            linear_func,
            sizes=[100, 500],
            trials=1,
        )
        report = generate_comparison_report(result)
        
        assert "Winner" in report or "Tie" in report


class TestGenerateVerificationReport:
    """Tests for generate_verification_report function."""
    
    def test_verification_report_is_string(self):
        """Verification report returns string."""
        result = verify_bounds(linear_func, sizes=[100, 200], expected="O(n)")
        report = generate_verification_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_verification_report_contains_status(self):
        """Verification report contains pass/fail status."""
        result = verify_bounds(linear_func, sizes=[100, 200], expected="O(n)")
        report = generate_verification_report(result)
        
        assert "PASSED" in report or "FAILED" in report
    
    def test_verification_report_contains_expected(self):
        """Verification report contains expected complexity."""
        result = verify_bounds(linear_func, sizes=[100, 200], expected="O(n)")
        report = generate_verification_report(result)
        
        assert "O(n)" in report

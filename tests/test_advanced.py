# Author: gadwant
"""Tests for new advanced features."""
import os
import tempfile


from bigocheck import (
    benchmark_function,
    fit_polynomial,
    PolynomialFit,
    run_benchmark_async,
    analyze_amortized,
    AmortizedResult,
    benchmark_parallel,
    generate_html_report,
    save_html_report,
    quick_check,
)


class TestPolynomialFitting:
    def test_fit_polynomial_linear(self):
        """Test polynomial fitting detects linear."""
        def linear(n):
            return sum(range(n))
        
        analysis = benchmark_function(linear, sizes=[100, 500, 1000], trials=1)
        poly = fit_polynomial(analysis.measurements)
        
        assert isinstance(poly, PolynomialFit)
        assert 0.5 < poly.exponent < 1.5  # Should be close to 1
    
    def test_fit_polynomial_quadratic(self):
        """Test polynomial fitting detects quadratic."""
        from bigocheck.core import Measurement
        
        measurements = [
            Measurement(size=10, seconds=0.01),
            Measurement(size=20, seconds=0.04),
            Measurement(size=40, seconds=0.16),
        ]
        
        poly = fit_polynomial(measurements)
        assert 1.5 < poly.exponent < 2.5  # Should be close to 2


class TestAsyncBenchmark:
    def test_run_benchmark_async(self):
        """Test async function benchmarking."""
        async def async_sum(n):
            return sum(range(n))
        
        analysis = run_benchmark_async(async_sum, sizes=[100, 500], trials=1)
        # Small inputs can produce various complexity classes due to timing noise
        assert analysis.best_label in ["O(1)", "O(n)", "O(log n)", "O(√n)", "O(n log n)"]


class TestAmortizedAnalysis:
    def test_analyze_amortized(self):
        """Test amortized analysis."""
        data = []
        
        def append_op():
            data.append(len(data))
        
        result = analyze_amortized(append_op, n_operations=100)
        
        assert isinstance(result, AmortizedResult)
        assert len(result.operations) == 100
        assert result.total_time > 0


class TestParallelBenchmark:
    def test_benchmark_parallel(self):
        """Test parallel benchmarking."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_parallel(simple, sizes=[100, 500, 1000], trials=1)
        
        assert analysis.best_label is not None
        assert len(analysis.measurements) == 3


class TestHTMLReport:
    def test_generate_html_report(self):
        """Test HTML report generation."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        html = generate_html_report(analysis, title="Test Report")
        
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert analysis.best_label in html
    
    def test_save_html_report(self):
        """Test saving HTML report to file."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        html = generate_html_report(analysis)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        
        try:
            save_html_report(html, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)


class TestQuickCheck:
    def test_quick_check(self):
        """Test quick_check function."""
        def linear(n):
            return sum(range(n))
        
        result = quick_check(linear, sizes=[100, 500, 1000])
        assert result in ["O(1)", "O(n)", "O(log n)", "O(√n)", "O(n log n)"]

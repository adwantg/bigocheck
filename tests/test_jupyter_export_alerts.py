# Author: gadwant
"""Tests for Jupyter, Export, and Alerts features."""
import os
import tempfile
import pytest

from bigocheck import benchmark_function
from bigocheck.jupyter import _repr_html_
from bigocheck.export import to_csv, to_markdown_table, to_dict, to_json
from bigocheck.alerts import (
    check_threshold,
    assert_threshold,
    monitor_complexity,
    ComplexityThresholdError,
    ThresholdResult,
)


class TestJupyterIntegration:
    def test_repr_html_returns_string(self):
        """Test _repr_html_ returns HTML string."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000], trials=1)
        html = _repr_html_(analysis)
        
        assert isinstance(html, str)
        assert "<div" in html
        assert analysis.best_label in html
    
    def test_repr_html_includes_measurements(self):
        """Test HTML includes measurement data."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        html = _repr_html_(analysis)
        
        assert "100" in html
        assert "500" in html


class TestExportCSV:
    def test_to_csv_returns_string(self):
        """Test to_csv returns CSV string."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        csv_str = to_csv(analysis)
        
        assert isinstance(csv_str, str)
        assert "Size" in csv_str
        assert "Time" in csv_str
    
    def test_to_csv_saves_file(self):
        """Test to_csv saves to file."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        
        try:
            to_csv(analysis, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "100" in content
        finally:
            os.unlink(path)
    
    def test_to_markdown_table(self):
        """Test to_markdown_table returns markdown."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        md = to_markdown_table(analysis)
        
        assert "|" in md
        assert "Size" in md
        assert analysis.best_label in md
    
    def test_to_dict(self):
        """Test to_dict returns dictionary."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        d = to_dict(analysis)
        
        assert isinstance(d, dict)
        assert "time_complexity" in d
        assert "measurements" in d
        assert d["time_complexity"] == analysis.best_label
    
    def test_to_json(self):
        """Test to_json returns JSON string."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500], trials=1)
        json_str = to_json(analysis)
        
        assert isinstance(json_str, str)
        assert analysis.best_label in json_str


class TestThresholdAlerts:
    def test_check_threshold_passes(self):
        """Test check_threshold passes when within limit."""
        def simple(n):
            return sum(range(n))
        
        analysis = benchmark_function(simple, sizes=[100, 500, 1000], trials=1)
        result = check_threshold(analysis, "O(n^2)")  # Should pass
        
        assert isinstance(result, ThresholdResult)
        assert result.passed
        assert "✓" in result.message
    
    def test_check_threshold_fails(self):
        """Test check_threshold fails when over limit."""
        def quadratic(n):
            for i in range(n):
                for j in range(n):
                    pass
        
        analysis = benchmark_function(quadratic, sizes=[50, 100, 200], trials=1)
        result = check_threshold(analysis, "O(log n)")  # Should fail
        
        assert not result.passed
        assert "✗" in result.message
    
    def test_assert_threshold_decorator(self):
        """Test assert_threshold decorator works."""
        @assert_threshold("O(n^2)", sizes=[1000, 5000, 10000])
        def simple(n):
            return sum(range(n))
        
        # Should not raise
        simple(10)
    
    def test_assert_threshold_raises(self):
        """Test assert_threshold raises on violation."""
        with pytest.raises(ComplexityThresholdError):
            @assert_threshold("O(1)", sizes=[100, 200, 400])
            def linear(n):
                return sum(range(n))
            
            linear(10)  # Triggers verification
    
    def test_monitor_complexity_warn(self):
        """Test monitor_complexity with warn mode."""
        def simple(n):
            return sum(range(n))
        
        import warnings
        with warnings.catch_warnings(record=True) as _:  # noqa: F841
            warnings.simplefilter("always")
            analysis = monitor_complexity(
                simple,
                sizes=[100, 500, 1000],
                max_complexity="O(1)",  # Too strict
                on_exceed="warn",
            )
            # Should warn but not raise
            assert analysis.best_label is not None

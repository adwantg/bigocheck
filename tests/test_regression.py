# Author: gadwant
"""Tests for regression module (baseline save/load, regression detection)."""
import json
import os
import tempfile
import pytest

from bigocheck import (
    benchmark_function,
    save_baseline,
    load_baseline,
    detect_regression,
    Baseline,
    RegressionResult,
)


class TestSaveLoadBaseline:
    def test_save_baseline_creates_file(self):
        """Test that save_baseline creates a JSON file."""
        def simple(n):
            return n ** 2
        
        analysis = benchmark_function(simple, sizes=[10, 20], trials=1)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            save_baseline(analysis, path, name="test_baseline")
            
            assert os.path.exists(path)
            
            with open(path) as f:
                data = json.load(f)
            
            assert data["name"] == "test_baseline"
            assert data["best_label"] == analysis.best_label
            assert len(data["measurements"]) == 2
        finally:
            os.unlink(path)
    
    def test_load_baseline(self):
        """Test that load_baseline reads correctly."""
        def simple(n):
            return n ** 2
        
        analysis = benchmark_function(simple, sizes=[10, 20], trials=1)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            save_baseline(analysis, path, name="load_test", metadata={"version": "1.0"})
            
            baseline = load_baseline(path)
            
            assert isinstance(baseline, Baseline)
            assert baseline.name == "load_test"
            assert baseline.best_label == analysis.best_label
            assert baseline.metadata["version"] == "1.0"
        finally:
            os.unlink(path)
    
    def test_load_baseline_not_found(self):
        """Test that load_baseline raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/path/baseline.json")


class TestDetectRegression:
    def test_no_regression_same_complexity(self):
        """Test no regression when complexity matches."""
        def simple(n):
            return n ** 2
        
        analysis = benchmark_function(simple, sizes=[10, 20, 40], trials=1)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        
        try:
            save_baseline(analysis, path)
            baseline = load_baseline(path)
            
            result = detect_regression(analysis, baseline)
            
            assert isinstance(result, RegressionResult)
            assert not result.time_regression
            assert result.current_time_label == result.baseline_time_label
        finally:
            os.unlink(path)
    
    def test_detect_complexity_regression(self):
        """Test detection when complexity class changes."""
        from bigocheck.core import Analysis, Measurement, FitResult
        
        # Baseline: O(n)
        baseline = Baseline(
            name="old",
            best_label="O(n)",
            space_label=None,
            measurements=[{"size": 100, "seconds": 0.01, "std_dev": 0.001, "memory_bytes": None}],
            timestamp="2024-01-01",
            metadata={},
        )
        
        # Current: O(n^2) - regression!
        current = Analysis(
            measurements=[Measurement(100, 0.01)],
            fits=[FitResult("O(n^2)", 1.0, 0.01)],
            best_label="O(n^2)",
        )
        
        result = detect_regression(current, baseline)
        
        assert result.time_regression
        assert result.has_regression
        assert "O(n) → O(n^2)" in result.message

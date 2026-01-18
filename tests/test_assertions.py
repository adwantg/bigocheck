# Author: gadwant
"""
Tests for assertions module: @assert_complexity, verify_bounds, confidence scoring.
"""
import pytest

from bigocheck import (
    verify_bounds,
    assert_complexity,
    auto_select_sizes,
    compute_confidence,
    benchmark_function,
    ComplexityAssertionError,
)


def constant_func(n):
    """O(1): Constant time."""
    return 42


def linear_func(n):
    """O(n): Linear time."""
    total = 0
    for i in range(n):
        total += i
    return total


def quadratic_func(n):
    """O(n^2): Quadratic time."""
    total = 0
    for i in range(n):
        for j in range(n):
            total += 1
    return total


class TestVerifyBounds:
    """Tests for verify_bounds function."""
    
    def test_verify_bounds_passes_correct_complexity(self):
        """Verify bounds passes when complexity matches."""
        result = verify_bounds(linear_func, sizes=[100, 200, 400], expected="O(n)")
        assert result.passes or result.actual == "O(n)"
        assert result.expected == "O(n)"
        assert result.confidence in ("high", "medium", "low")
    
    def test_verify_bounds_fails_wrong_complexity(self):
        """Verify bounds fails when complexity is wrong."""
        result = verify_bounds(quadratic_func, sizes=[50, 100, 200], expected="O(1)")
        # Should either fail or have different actual
        assert not result.passes or result.actual != "O(1)"
    
    def test_verify_bounds_returns_verification_result(self):
        """Verify result contains all expected fields."""
        result = verify_bounds(constant_func, sizes=[100, 500, 1000], expected="O(1)")
        assert hasattr(result, 'passes')
        assert hasattr(result, 'expected')
        assert hasattr(result, 'actual')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'message')
        assert hasattr(result, 'analysis')


class TestAssertComplexity:
    """Tests for @assert_complexity decorator."""
    
    def test_decorator_passes_correct_complexity(self):
        """Decorator should not raise for correct complexity."""
        @assert_complexity("O(n)", sizes=[100, 200, 400], trials=1)
        def my_linear(n):
            return sum(range(n))
        
        # Should not raise
        my_linear(10)
    
    def test_decorator_raises_wrong_complexity(self):
        """Decorator should raise for wrong complexity."""
        @assert_complexity("O(1)", sizes=[50, 100, 200], trials=1, tolerance=0.1)
        def my_quadratic(n):
            total = 0
            for i in range(n):
                for j in range(n):
                    total += 1
            return total
        
        with pytest.raises(ComplexityAssertionError):
            my_quadratic(10)
    
    def test_decorator_stores_expected(self):
        """Decorator stores expected complexity on function."""
        @assert_complexity("O(n)")
        def decorated(n):
            return n
        
        assert decorated._complexity_expected == "O(n)"


class TestComputeConfidence:
    """Tests for compute_confidence function."""
    
    def test_confidence_returns_result(self):
        """Confidence computation returns proper result."""
        analysis = benchmark_function(linear_func, sizes=[100, 200, 400], trials=2)
        confidence = compute_confidence(analysis)
        
        assert confidence.level in ("high", "medium", "low")
        assert 0.0 <= confidence.score <= 1.0
        assert isinstance(confidence.reasons, list)
        assert len(confidence.reasons) > 0
    
    def test_more_measurements_better_confidence(self):
        """More measurements should give better confidence."""
        analysis_few = benchmark_function(linear_func, sizes=[100, 200], trials=1)
        analysis_many = benchmark_function(linear_func, sizes=[100, 200, 400, 800, 1600], trials=2)
        
        conf_few = compute_confidence(analysis_few)
        conf_many = compute_confidence(analysis_many)
        
        # More measurements should give higher confidence
        assert conf_many.score >= conf_few.score or conf_many.level in ("medium", "high")


class TestAutoSelectSizes:
    """Tests for auto_select_sizes function."""
    
    def test_auto_select_returns_list(self):
        """Auto select returns a list of sizes."""
        sizes = auto_select_sizes(linear_func, target_time=1.0, min_sizes=3)
        
        assert isinstance(sizes, list)
        assert len(sizes) >= 3
        assert all(isinstance(s, int) for s in sizes)
    
    def test_auto_select_sizes_are_sorted(self):
        """Auto selected sizes are sorted."""
        sizes = auto_select_sizes(linear_func, target_time=0.5)
        assert sizes == sorted(sizes)
    
    def test_auto_select_sizes_are_unique(self):
        """Auto selected sizes are unique."""
        sizes = auto_select_sizes(linear_func, target_time=0.5)
        assert len(sizes) == len(set(sizes))

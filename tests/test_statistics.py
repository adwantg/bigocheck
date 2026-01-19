# Author: gadwant
"""Tests for statistics module (p-values)."""

from bigocheck import benchmark_function, compute_significance, SignificanceResult


def test_compute_significance_returns_result():
    """Test that compute_significance returns a SignificanceResult."""
    def linear(n):
        return sum(range(n))
    
    analysis = benchmark_function(linear, sizes=[100, 500, 1000], trials=1)
    sig = compute_significance(analysis)
    
    assert isinstance(sig, SignificanceResult)
    assert 0 <= sig.p_value <= 1
    assert sig.confidence_level in ["high", "medium", "low"]
    assert sig.best_label == analysis.best_label


def test_compute_significance_single_fit():
    """Test significance with edge case of single fit."""
    from bigocheck.core import Analysis, Measurement, FitResult
    
    analysis = Analysis(
        measurements=[Measurement(100, 0.01)],
        fits=[FitResult("O(n)", 1.0, 0.01)],
        best_label="O(n)",
    )
    
    sig = compute_significance(analysis)
    assert sig.p_value == 1.0
    assert not sig.is_significant


def test_significance_format():
    """Test format_significance function."""
    from bigocheck import format_significance
    
    sig = SignificanceResult(
        p_value=0.01,
        is_significant=True,
        confidence_level="high",
        best_label="O(n)",
        second_best_label="O(n^2)",
        error_difference=0.1,
    )
    
    formatted = format_significance(sig)
    assert "0.01" in formatted
    assert "significant" in formatted
    assert "O(n)" in formatted

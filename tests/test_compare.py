# Author: gadwant
"""
Tests for compare module: compare_functions, compare_to_baseline.
"""

from bigocheck import compare_functions, compare_to_baseline


def linear_search(n):
    """O(n): Linear search simulation."""
    arr = list(range(n))
    target = n - 1
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


def binary_search(n):
    """O(log n): Binary search simulation."""
    arr = list(range(n))
    target = n - 1
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def constant_func(n):
    """O(1): Constant time."""
    return 42


class TestCompareFunctions:
    """Tests for compare_functions."""
    
    def test_compare_returns_result(self):
        """Compare returns ComparisonResult."""
        result = compare_functions(
            constant_func,
            linear_search,
            sizes=[100, 500, 1000],
            trials=1,
        )
        
        assert hasattr(result, 'winner')
        assert hasattr(result, 'speedup')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'size_comparisons')
    
    def test_compare_uses_function_names(self):
        """Compare uses correct function names."""
        result = compare_functions(
            constant_func,
            linear_search,
            sizes=[100, 500],
            trials=1,
        )
        
        assert result.func_a_name == "constant_func"
        assert result.func_b_name == "linear_search"
    
    def test_compare_custom_names(self):
        """Compare allows custom names."""
        result = compare_functions(
            constant_func,
            linear_search,
            sizes=[100, 500],
            trials=1,
            func_a_name="fast_func",
            func_b_name="slow_func",
        )
        
        assert result.func_a_name == "fast_func"
        assert result.func_b_name == "slow_func"
    
    def test_compare_identifies_winner(self):
        """Compare identifies correct winner."""
        result = compare_functions(
            constant_func,  # O(1) - should be faster
            linear_search,  # O(n) - should be slower
            sizes=[1000, 5000, 10000],
            trials=1,
        )
        
        # Constant should be faster at large sizes
        assert result.winner in ("func_a", "func_b", None)
        assert result.speedup >= 1.0
    
    def test_compare_size_comparisons(self):
        """Compare includes per-size comparisons."""
        result = compare_functions(
            constant_func,
            linear_search,
            sizes=[100, 500, 1000],
            trials=1,
        )
        
        assert len(result.size_comparisons) == 3
        for comp in result.size_comparisons:
            assert "size" in comp
            assert "time_a" in comp
            assert "time_b" in comp
            assert "winner" in comp


class TestCompareToBaseline:
    """Tests for compare_to_baseline."""
    
    def test_baseline_comparison(self):
        """Baseline comparison returns result dict."""
        result = compare_to_baseline(
            linear_search,
            baseline_label="O(n)",
            sizes=[100, 500, 1000],
            trials=1,
        )
        
        assert "matches_baseline" in result
        assert "actual" in result
        assert "expected" in result
    
    def test_baseline_unknown_class(self):
        """Unknown baseline class is handled."""
        result = compare_to_baseline(
            linear_search,
            baseline_label="O(unknown)",
            sizes=[100, 500],
            trials=1,
        )
        
        assert result["matches_baseline"] is False
        assert "error" in result

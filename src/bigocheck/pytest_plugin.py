# Author: gadwant
"""
pytest plugin for bigocheck.

Provides fixtures and markers for complexity testing in pytest.

Usage:
    # In conftest.py or test file
    pytest_plugins = ["bigocheck.pytest_plugin"]
    
    # In test
    @pytest.mark.complexity("O(n)")
    def test_linear_function():
        from mymodule import my_func
        result = complexity_check(my_func, sizes=[100, 500, 1000])
        assert result.passes
"""
from __future__ import annotations

import pytest
from typing import Any, Callable, List, Optional

from .assertions import verify_bounds, VerificationResult, ComplexityAssertionError
from .core import Analysis, benchmark_function


class ComplexityChecker:
    """Helper class for complexity checking in tests."""
    
    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        trials: int = 3,
        warmup: int = 1,
        tolerance: float = 0.3,
    ):
        self.sizes = sizes or [100, 500, 1000, 5000]
        self.trials = trials
        self.warmup = warmup
        self.tolerance = tolerance
    
    def check(
        self,
        func: Callable[..., Any],
        expected: str,
        *,
        sizes: Optional[List[int]] = None,
    ) -> VerificationResult:
        """
        Check that a function has expected complexity.
        
        Args:
            func: Function to check.
            expected: Expected complexity (e.g., "O(n)").
            sizes: Override default sizes.
        
        Returns:
            VerificationResult with pass/fail and details.
        """
        return verify_bounds(
            func,
            sizes=sizes or self.sizes,
            expected=expected,
            tolerance=self.tolerance,
            trials=self.trials,
            warmup=self.warmup,
        )
    
    def benchmark(
        self,
        func: Callable[..., Any],
        *,
        sizes: Optional[List[int]] = None,
    ) -> Analysis:
        """
        Benchmark a function without assertions.
        
        Args:
            func: Function to benchmark.
            sizes: Override default sizes.
        
        Returns:
            Analysis object with results.
        """
        return benchmark_function(
            func,
            sizes=sizes or self.sizes,
            trials=self.trials,
            warmup=self.warmup,
        )
    
    def assert_complexity(
        self,
        func: Callable[..., Any],
        expected: str,
        *,
        sizes: Optional[List[int]] = None,
        msg: Optional[str] = None,
    ) -> None:
        """
        Assert that a function has expected complexity.
        
        Args:
            func: Function to check.
            expected: Expected complexity.
            sizes: Override default sizes.
            msg: Custom failure message.
        
        Raises:
            ComplexityAssertionError: If check fails.
        """
        result = self.check(func, expected, sizes=sizes)
        if not result.passes:
            raise ComplexityAssertionError(msg or result.message)


@pytest.fixture
def complexity_checker() -> ComplexityChecker:
    """
    Pytest fixture providing a ComplexityChecker instance.
    
    Example:
        def test_sorting(complexity_checker):
            result = complexity_checker.check(sorted, "O(n log n)")
            assert result.passes
    """
    return ComplexityChecker()


@pytest.fixture
def assert_complexity_fixture() -> Callable:
    """
    Pytest fixture for asserting complexity.
    
    Example:
        def test_linear(assert_complexity_fixture):
            def my_func(n):
                return sum(range(n))
            assert_complexity_fixture(my_func, "O(n)")
    """
    checker = ComplexityChecker()
    return checker.assert_complexity


def pytest_configure(config):
    """Register the complexity marker."""
    config.addinivalue_line(
        "markers",
        "complexity(expected): mark test as a complexity test with expected complexity"
    )


def pytest_collection_modifyitems(config, items):
    """Handle complexity markers."""
    for item in items:
        complexity_marker = item.get_closest_marker("complexity")
        if complexity_marker:
            expected = complexity_marker.args[0] if complexity_marker.args else None
            if expected:
                item.user_properties.append(("expected_complexity", expected))

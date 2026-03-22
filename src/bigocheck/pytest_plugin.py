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

import json
from pathlib import Path
import pytest
from typing import Any, Callable, Dict, List, Optional

from .assertions import (
    ComplexityAssertionError,
    VerificationResult,
    _assert_space_expectation,
    verify_bounds,
)
from .bounds import BoundsResult, check_bounds
from .core import Analysis, benchmark_function
from .profiles import DEFAULT_ASSERTION_PROFILE_NAME, resolve_profile


class ComplexityChecker:
    """Helper class for complexity checking in tests."""
    
    def __init__(
        self,
        sizes: Optional[List[int]] = None,
        trials: Optional[int] = None,
        warmup: Optional[int] = None,
        tolerance: float = 0.3,
        profile: Optional[str] = None,
        robust: Optional[bool] = None,
        expected: Optional[str] = None,
        lower: Optional[str] = None,
        upper: Optional[str] = None,
        space: Optional[str] = None,
        space_upper: Optional[str] = None,
        setup: Optional[Callable[[int], tuple[tuple[Any, ...], dict[str, Any]]]] = None,
        reporter: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        options = resolve_profile(
            profile=profile or DEFAULT_ASSERTION_PROFILE_NAME,
            sizes=sizes,
            trials=trials,
            warmup=warmup,
            robust=robust,
            default_profile=DEFAULT_ASSERTION_PROFILE_NAME,
        )
        self.profile = options.name
        self.sizes = options.sizes
        self.trials = options.trials
        self.warmup = options.warmup
        self.robust = options.robust
        self.tolerance = tolerance
        self.expected = expected
        self.lower = lower
        self.upper = upper
        self.space = space
        self.space_upper = space_upper
        self.setup = setup
        self.reporter = reporter
    
    def check(
        self,
        func: Callable[..., Any],
        expected: Optional[str] = None,
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
        result = verify_bounds(
            func,
            sizes=sizes or self.sizes,
            expected=expected or self.expected or _raise_missing_expected(),
            tolerance=self.tolerance,
            trials=self.trials,
            warmup=self.warmup,
            profile=self.profile,
            setup=self.setup,
            robust=self.robust,
            memory=bool(self.space or self.space_upper),
        )
        _assert_space_expectation(
            result.analysis,
            expected=self.space,
            upper=self.space_upper,
            confidence=result.confidence,
            stability=result.stability or "unknown",
        )
        self._report(
            {
                "kind": "complexity",
                "expected": result.expected,
                "actual": result.actual,
                "passed": result.passes,
                "space_expected": self.space,
                "space_upper": self.space_upper,
                "space_actual": result.analysis.space_label,
                "confidence": result.confidence,
                "stability": result.stability,
                "message": result.message,
            }
        )
        return result
    
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
            setup=self.setup,
            robust=self.robust,
        )

    def check_bounds(
        self,
        func: Callable[..., Any],
        *,
        lower: Optional[str] = None,
        upper: Optional[str] = None,
        sizes: Optional[List[int]] = None,
    ) -> BoundsResult:
        analysis = self.benchmark(func, sizes=sizes)
        result = check_bounds(
            analysis,
            lower=lower if lower is not None else self.lower,
            upper=upper if upper is not None else self.upper,
        )
        self._report(
            {
                "kind": "bounds",
                "lower": result.lower_bound,
                "upper": result.upper_bound,
                "actual": result.actual,
                "passed": result.in_bounds,
                "message": result.message,
            }
        )
        return result
    
    def assert_complexity(
        self,
        func: Callable[..., Any],
        expected: Optional[str] = None,
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

    def assert_bounds(
        self,
        func: Callable[..., Any],
        *,
        lower: Optional[str] = None,
        upper: Optional[str] = None,
        sizes: Optional[List[int]] = None,
        msg: Optional[str] = None,
    ) -> None:
        result = self.check_bounds(func, lower=lower, upper=upper, sizes=sizes)
        if not result.in_bounds:
            raise ComplexityAssertionError(msg or result.message)

    def _report(self, record: Dict[str, Any]) -> None:
        if self.reporter is not None:
            self.reporter(record)


def _raise_missing_expected() -> str:
    raise ValueError("Expected complexity must be provided directly or via @pytest.mark.complexity")


def _marker_config(request) -> Dict[str, Any]:
    marker = request.node.get_closest_marker("complexity")
    if marker is None:
        return {}

    config: Dict[str, Any] = dict(marker.kwargs)
    if marker.args and "expected" not in config:
        config["expected"] = marker.args[0]
    return config


def _reporter_for(request) -> Callable[[Dict[str, Any]], None]:
    config = request.config

    def emit(record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload["test"] = request.node.nodeid
        config._bigocheck_records.append(payload)

    return emit


@pytest.fixture
def complexity_checker(request) -> ComplexityChecker:
    """
    Pytest fixture providing a ComplexityChecker instance.
    
    Example:
        def test_sorting(complexity_checker):
            result = complexity_checker.check(sorted, "O(n log n)")
            assert result.passes
    """
    marker_config = _marker_config(request)
    return ComplexityChecker(
        sizes=marker_config.get("sizes"),
        trials=marker_config.get("trials"),
        warmup=marker_config.get("warmup"),
        tolerance=marker_config.get("tolerance", 0.3),
        profile=marker_config.get("profile"),
        robust=marker_config.get("robust"),
        expected=marker_config.get("expected"),
        lower=marker_config.get("lower"),
        upper=marker_config.get("upper"),
        space=marker_config.get("space"),
        space_upper=marker_config.get("space_upper"),
        setup=marker_config.get("setup"),
        reporter=_reporter_for(request),
    )


@pytest.fixture
def assert_complexity_fixture(complexity_checker: ComplexityChecker) -> Callable:
    """
    Pytest fixture for asserting complexity.
    
    Example:
        def test_linear(assert_complexity_fixture):
            def my_func(n):
                return sum(range(n))
            assert_complexity_fixture(my_func, "O(n)")
    """
    return complexity_checker.assert_complexity


def pytest_addoption(parser):
    """Register bigocheck plugin command-line options."""
    parser.addoption(
        "--bigocheck-report",
        action="store",
        default=None,
        help="Write executed bigocheck checks to a JSON report.",
    )


def pytest_configure(config):
    """Register the complexity marker."""
    config._bigocheck_records = []
    config.addinivalue_line(
        "markers",
        "complexity(expected=None, **kwargs): mark test as a complexity test and provide defaults"
    )


def pytest_collection_modifyitems(config, items):
    """Handle complexity markers."""
    for item in items:
        complexity_marker = item.get_closest_marker("complexity")
        if complexity_marker:
            expected = complexity_marker.args[0] if complexity_marker.args else None
            if expected:
                item.user_properties.append(("expected_complexity", expected))


def pytest_sessionfinish(session, exitstatus):
    """Write the optional per-suite bigocheck report."""
    report_path = session.config.getoption("--bigocheck-report")
    if not report_path:
        return

    payload = {
        "exitstatus": exitstatus,
        "checks": session.config._bigocheck_records,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

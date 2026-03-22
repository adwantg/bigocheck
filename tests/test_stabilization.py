# Author: gadwant
"""Tests for CI-stability and expanded assertion/reporting paths."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

from bigocheck import (
    Analysis,
    ComplexityAssertionError,
    FitResult,
    Measurement,
    VerificationResult,
    assert_bounds,
    assert_complexity,
    assert_threshold,
    benchmark_function,
    verify_bounds,
)
from bigocheck.pytest_plugin import ComplexityChecker


def _install_fake_timer(monkeypatch):
    state = {"now": 0.0, "pending": 0.0}

    def perf_counter() -> float:
        state["now"] += state["pending"]
        state["pending"] = 0.0
        return state["now"]

    monkeypatch.setattr("bigocheck.core.time.perf_counter", perf_counter)
    monkeypatch.setattr("bigocheck.async_bench.time.perf_counter", perf_counter)
    return state


def _linear_analysis(*, space_label: str | None = None) -> Analysis:
    return Analysis(
        measurements=[
            Measurement(size=100, seconds=1.0, std_dev=0.01),
            Measurement(size=200, seconds=2.0, std_dev=0.01),
            Measurement(size=400, seconds=4.0, std_dev=0.02),
        ],
        fits=[
            FitResult(label="O(n)", scale=1.0, error=0.01, r_squared=0.999),
            FitResult(label="O(n log n)", scale=1.2, error=0.12, r_squared=0.97),
            FitResult(label="O(n^2)", scale=0.02, error=0.5, r_squared=0.2),
        ],
        best_label="O(n)",
        space_fits=[FitResult(label="O(n)", scale=1.0, error=0.02, r_squared=0.998)]
        if space_label
        else [],
        space_label=space_label,
    )


def test_setup_excludes_construction_from_timing(monkeypatch):
    state = _install_fake_timer(monkeypatch)

    def setup(n):
        state["pending"] += float(n * n)
        return ((n,), {})

    def arg_factory(n):
        state["pending"] += float(n * n)
        return ((n,), {})

    def linear(n):
        state["pending"] += float(n)
        return n

    setup_analysis = benchmark_function(linear, sizes=[10, 20, 40], trials=1, setup=setup)
    arg_factory_analysis = benchmark_function(
        linear,
        sizes=[10, 20, 40],
        trials=1,
        arg_factory=arg_factory,
    )

    assert [m.seconds for m in setup_analysis.measurements] == [10.0, 20.0, 40.0]
    assert setup_analysis.best_label == "O(n)"
    assert [m.seconds for m in arg_factory_analysis.measurements] == [110.0, 420.0, 1640.0]
    assert arg_factory_analysis.best_label == "O(n^2)"


def test_robust_benchmarking_uses_median_after_outlier_filter(monkeypatch):
    state = _install_fake_timer(monkeypatch)
    durations = {
        10: [10.0, 10.0, 10.0, 10.0],
        20: [20.0, 20.0, 20.0, 2000.0],
        40: [40.0, 40.0, 40.0, 40.0],
    }

    def make_func():
        counters = {size: 0 for size in durations}

        def func(n):
            idx = counters[n]
            counters[n] += 1
            state["pending"] += durations[n][idx]
            return n

        return func

    standard = benchmark_function(make_func(), sizes=[10, 20, 40], trials=4, robust=False)
    robust = benchmark_function(make_func(), sizes=[10, 20, 40], trials=4, robust=True)

    assert [m.seconds for m in standard.measurements] == pytest.approx([10.0, 515.0, 40.0])
    assert [m.seconds for m in robust.measurements] == pytest.approx([10.0, 20.0, 40.0])
    assert robust.best_label == "O(n)"


def test_assert_complexity_supports_async_functions(monkeypatch):
    async def fake_benchmark_async(*args, **kwargs):
        assert kwargs["memory"] is False
        return _linear_analysis()

    monkeypatch.setattr("bigocheck.assertions.benchmark_async", fake_benchmark_async)

    @assert_complexity("O(n)", profile="ci")
    async def async_linear(n):
        return n

    assert asyncio.run(async_linear(5)) == 5
    assert async_linear._complexity_verified() is True


def test_space_assertions_request_memory_and_enforce_bounds(monkeypatch):
    calls = []

    def fake_verify_bounds(func, sizes, expected, **kwargs):
        calls.append(kwargs["memory"])
        analysis = _linear_analysis(space_label="O(n)")
        return VerificationResult(
            passes=True,
            expected=expected,
            actual="O(n)",
            error=0.01,
            tolerance=kwargs["tolerance"],
            confidence="high",
            confidence_score=0.9,
            analysis=analysis,
            message="ok",
            space_actual=analysis.space_label,
            stability="high",
            stability_score=0.9,
        )

    monkeypatch.setattr("bigocheck.assertions.verify_bounds", fake_verify_bounds)

    @assert_complexity("O(n)", space="O(n)")
    def linear_space(n):
        return [0] * n

    linear_space(10)
    assert calls == [True]

    @assert_complexity("O(n)", space_upper="O(1)")
    def too_much_space(n):
        return [0] * n

    with pytest.raises(ComplexityAssertionError, match="Space complexity O\\(n\\) exceeds upper bound O\\(1\\)"):
        too_much_space(10)


def test_default_assertion_helpers_use_ci_profile(monkeypatch):
    captured = {}

    def fake_verify_bounds(func, sizes, expected, **kwargs):
        captured["assert_complexity"] = {
            "sizes": sizes,
            "profile": kwargs["profile"],
            "trials": kwargs["trials"],
            "warmup": kwargs["warmup"],
            "robust": kwargs["robust"],
        }
        return VerificationResult(
            passes=True,
            expected=expected,
            actual="O(n)",
            error=0.01,
            tolerance=kwargs["tolerance"],
            confidence="high",
            confidence_score=0.9,
            analysis=_linear_analysis(),
            message="ok",
            stability="high",
            stability_score=0.9,
        )

    def fake_benchmark(func, *, sizes, trials, warmup, setup, arg_factory, robust):
        captured.setdefault("benchmarks", []).append(
            {
                "sizes": sizes,
                "trials": trials,
                "warmup": warmup,
                "robust": robust,
            }
        )
        return _linear_analysis()

    monkeypatch.setattr("bigocheck.assertions.verify_bounds", fake_verify_bounds)
    monkeypatch.setattr("bigocheck.bounds.benchmark_function", fake_benchmark)
    monkeypatch.setattr("bigocheck.alerts.benchmark_function", fake_benchmark)

    @assert_complexity("O(n)")
    def linear(n):
        return n

    @assert_bounds(upper="O(n)")
    def bounded(n):
        return n

    @assert_threshold("O(n)")
    def thresholded(n):
        return n

    linear(1)
    bounded(1)
    thresholded(1)

    assert captured["assert_complexity"] == {
        "sizes": [200, 1000, 5000, 20000],
        "profile": "ci",
        "trials": 5,
        "warmup": 2,
        "robust": True,
    }
    assert captured["benchmarks"] == [
        {"sizes": [200, 1000, 5000, 20000], "trials": 5, "warmup": 2, "robust": True},
        {"sizes": [200, 1000, 5000, 20000], "trials": 5, "warmup": 2, "robust": True},
    ]


def test_complexity_checker_defaults_to_ci_profile():
    checker = ComplexityChecker()
    assert checker.profile == "ci"
    assert checker.sizes == [200, 1000, 5000, 20000]
    assert checker.trials == 5
    assert checker.warmup == 2
    assert checker.robust is True


def test_verification_failure_message_includes_actionable_details():
    def quadratic(n):
        total = 0
        for _ in range(n):
            for _ in range(n):
                total += 1
        return total

    result = verify_bounds(quadratic, sizes=[30, 60, 120], expected="O(1)", tolerance=0.0)
    assert result.passes is False
    assert "Expected O(1), got" in result.message
    assert "Top fits:" in result.message
    assert "Measurements:" in result.message
    assert "Confidence:" in result.message
    assert "stability:" in result.message


def test_fit_result_exposes_relative_rmse_and_r_squared():
    fit = FitResult(label="O(n)", scale=1.0, error=0.125, r_squared=0.98)
    assert fit.relative_rmse == pytest.approx(0.125)
    assert fit.error == pytest.approx(0.125)
    assert fit.r_squared == pytest.approx(0.98)


def test_pytest_plugin_marker_defaults_and_report_export(pytester):
    report_path = Path(pytester.path) / "bigocheck-report.json"
    src_path = Path(__file__).resolve().parents[1] / "src"

    pytester.makeconftest(
        f"import sys\nsys.path.insert(0, {str(src_path)!r})\n"
    )
    pytester.makepyfile(
        test_plugin="""
import pytest

from bigocheck import Analysis, FitResult, Measurement, VerificationResult
import bigocheck.pytest_plugin as plugin


@pytest.mark.complexity("O(n)", profile="ci", space_upper="O(n)")
def test_marker_defaults(complexity_checker, monkeypatch):
    calls = {}

    def fake_verify_bounds(func, sizes, expected, **kwargs):
        calls["sizes"] = sizes
        calls["expected"] = expected
        calls["kwargs"] = kwargs
        analysis = Analysis(
            measurements=[
                Measurement(size=100, seconds=1.0, std_dev=0.01),
                Measurement(size=200, seconds=2.0, std_dev=0.01),
                Measurement(size=400, seconds=4.0, std_dev=0.02),
            ],
            fits=[
                FitResult(label="O(n)", scale=1.0, error=0.01, r_squared=0.999),
                FitResult(label="O(n^2)", scale=0.1, error=0.2, r_squared=0.85),
            ],
            best_label="O(n)",
            space_fits=[FitResult(label="O(n)", scale=1.0, error=0.02, r_squared=0.99)],
            space_label="O(n)",
        )
        return VerificationResult(
            passes=True,
            expected=expected,
            actual="O(n)",
            error=0.01,
            tolerance=kwargs["tolerance"],
            confidence="high",
            confidence_score=0.9,
            analysis=analysis,
            message="ok",
            space_actual="O(n)",
            stability="high",
            stability_score=0.9,
        )

    monkeypatch.setattr(plugin, "verify_bounds", fake_verify_bounds)

    def linear(n):
        return n

    result = complexity_checker.check(linear)
    assert result.passes
    assert calls["expected"] == "O(n)"
    assert calls["sizes"] == [200, 1000, 5000, 20000]
    assert calls["kwargs"]["profile"] == "ci"
    assert calls["kwargs"]["trials"] == 5
    assert calls["kwargs"]["warmup"] == 2
    assert calls["kwargs"]["robust"] is True
    assert calls["kwargs"]["memory"] is True
"""
    )

    result = pytester.runpytest(
        "-q",
        "-p",
        "bigocheck.pytest_plugin",
        "--bigocheck-report",
        str(report_path),
    )
    result.assert_outcomes(passed=1)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["exitstatus"] == 0
    assert len(payload["checks"]) == 1
    assert payload["checks"][0]["kind"] == "complexity"
    assert payload["checks"][0]["expected"] == "O(n)"
    assert payload["checks"][0]["actual"] == "O(n)"
    assert payload["checks"][0]["space_upper"] == "O(n)"

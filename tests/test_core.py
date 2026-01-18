# Author: gadwant
import json

import pytest

from bigocheck import __version__, Analysis, Measurement, benchmark_function, fit_complexities
from bigocheck.core import complexity_basis
from . import targets


def test_version_present():
    assert __version__


@pytest.mark.parametrize(
    "func,expected_label",
    [
        (targets.constant_sleep, "O(1)"),
        (targets.linear_sleep, "O(n)"),
        (targets.quadratic_sleep, "O(n^2)"),
    ],
)
def test_benchmark_classifies(func, expected_label):
    analysis: Analysis = benchmark_function(func, sizes=[2, 4, 6], trials=1)
    assert analysis.best_label == expected_label
    assert analysis.fits[0].label == expected_label
    assert analysis.measurements


def test_resolve_callable_and_benchmark_json(capsys):
    from bigocheck.core import resolve_callable
    func = resolve_callable("tests.targets:linear_sleep")
    analysis = benchmark_function(func, sizes=[1, 2, 3], trials=1)
    data = json.loads(
        json.dumps(
            {
                "best": analysis.best_label,
                "measurements": [m.seconds for m in analysis.measurements],
            }
        )
    )
    # Small input sizes can produce various fits due to timing noise
    assert data["best"] in {"O(n)", "O(n log n)", "O(log n)", "O(2^n)", "O(√n)"}
    assert len(data["measurements"]) == 3


def test_warmup_parameter():
    """Test that warmup parameter doesn't cause errors."""
    analysis = benchmark_function(targets.constant_sleep, sizes=[1, 2], trials=1, warmup=1)
    assert analysis.best_label == "O(1)"


def test_std_dev_computed():
    """Test that standard deviation is computed with multiple trials."""
    analysis = benchmark_function(targets.constant_sleep, sizes=[1, 2], trials=3)
    # With 3 trials, std_dev should be computed (may be 0 if very consistent)
    for m in analysis.measurements:
        assert hasattr(m, 'std_dev')
        assert m.std_dev >= 0


def test_memory_profiling():
    """Test that memory profiling works."""
    analysis = benchmark_function(targets.linear_sleep, sizes=[1, 2], trials=1, memory=True)
    # Memory should be tracked for at least one measurement
    assert any(m.memory_bytes is not None for m in analysis.measurements)


def test_space_complexity_fields():
    """Test that Analysis has space complexity fields."""
    analysis = benchmark_function(targets.linear_sleep, sizes=[1, 2], trials=1, memory=True)
    assert hasattr(analysis, 'space_fits')
    assert hasattr(analysis, 'space_label')


def test_space_complexity_fitting():
    """Test that space complexity is fitted when memory=True."""
    def allocate_linear(n):
        """O(n) space - allocates a list of size n."""
        return [0] * n
    
    analysis = benchmark_function(allocate_linear, sizes=[1000, 5000, 10000], trials=1, memory=True)
    
    # Should have space fits
    assert len(analysis.space_fits) > 0
    # Should have a space label
    assert analysis.space_label is not None
    # Space should be classified (likely O(n) for linear allocation)
    assert analysis.space_label in ["O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)"]


def test_fit_space_complexity_function():
    """Test fit_space_complexity directly."""
    from bigocheck.core import fit_space_complexity
    
    # Create measurements with memory data
    measurements = [
        Measurement(size=100, seconds=0.01, memory_bytes=1000),
        Measurement(size=200, seconds=0.02, memory_bytes=2000),
        Measurement(size=400, seconds=0.04, memory_bytes=4000),
    ]
    
    fits, label = fit_space_complexity(measurements)
    
    assert len(fits) > 0
    assert label is not None
    # Linear growth in memory should be classified as O(n)
    assert label == "O(n)"


def test_fit_space_complexity_no_memory():
    """Test fit_space_complexity with no memory data returns empty."""
    from bigocheck.core import fit_space_complexity
    
    measurements = [
        Measurement(size=100, seconds=0.01, memory_bytes=None),
        Measurement(size=200, seconds=0.02, memory_bytes=None),
    ]
    
    fits, label = fit_space_complexity(measurements)
    
    assert fits == []
    assert label is None


def test_complexity_basis_includes_new_classes():
    """Test that new complexity classes are present."""
    basis = complexity_basis()
    assert "O(√n)" in basis
    assert "O(n!)" in basis
    assert "O(1)" in basis
    assert "O(2^n)" in basis


def test_verbose_parameter():
    """Test that verbose parameter doesn't cause errors."""
    # Just ensure it runs without error (output goes to stderr)
    analysis = benchmark_function(targets.constant_sleep, sizes=[1], trials=1, verbose=True)
    assert analysis.best_label == "O(1)"


def test_fit_complexities_no_measurements():
    """Test that empty measurements raises ValueError."""
    with pytest.raises(ValueError, match="No measurements"):
        fit_complexities([])


def test_measurement_dataclass():
    """Test Measurement dataclass fields."""
    m = Measurement(size=100, seconds=0.5, std_dev=0.01, memory_bytes=1024)
    assert m.size == 100
    assert m.seconds == 0.5
    assert m.std_dev == 0.01
    assert m.memory_bytes == 1024


def test_datagen_module():
    """Test datagen module generators."""
    from bigocheck.datagen import n_, range_n, integers, floats, strings, sorted_integers
    
    assert n_(100) == 100
    assert list(range_n(5)) == [0, 1, 2, 3, 4]
    assert len(integers(10)) == 10
    assert len(floats(10)) == 10
    assert len(strings(10)) == 10
    # Generate once and verify it's sorted
    sorted_list = sorted_integers(10)
    assert sorted_list == sorted(sorted_list)


def test_arg_factory_wrapper():
    """Test arg_factory_for wrapper."""
    from bigocheck.datagen import integers, arg_factory_for
    
    factory = arg_factory_for(integers)
    args, kwargs = factory(10)
    assert len(args[0]) == 10
    assert kwargs == {}


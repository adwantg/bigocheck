<!-- Author: gadwant -->
# bigocheck

> **The only zero-dependency, CLI-first Big-O complexity checker for Python**

Empirical complexity regression checker: run a target function across input sizes, measure runtimes, and fit against common complexity classes. Ships as both a library and CLI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)]()
[![Author: gadwant](https://img.shields.io/badge/author-gadwant-purple.svg)](https://github.com/adwantg)

---

## 🎯 Features at a Glance

| Feature | Description |
|---------|-------------|
| **🧮 Complexity Fitting** | Fits to 9 classes: O(1), O(log n), O(√n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ), O(n!) |
| **✅ Complexity Assertions** | `@assert_complexity("O(n)")` decorator for CI/CD testing |
| **🔍 Bounds Verification** | `verify_bounds()` to check if function meets expected complexity |
| **📊 Confidence Scoring** | Know how reliable your results are |
| **🔀 A/B Comparison** | Compare two implementations head-to-head |
| **📄 Report Generation** | Generate markdown reports automatically |
| **🔧 pytest Plugin** | Integration with pytest for testing |
| **📈 Plotting** | Optional matplotlib visualization |
| **💾 Memory Profiling** | Track peak memory usage |
| **🚀 Auto Size Selection** | Automatically choose optimal input sizes |
| **📦 Zero Dependencies** | Pure standard library, no numpy required |
| **💻 CLI-First** | Full command-line interface |

---

## 📦 Installation

```bash
pip install bigocheck
```

### Development Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

---

## 🚀 Quick Start

### CLI Usage

```bash
# Basic benchmark
bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --trials 3

# With warmup and verbose output
bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --warmup 2 --verbose

# Output as JSON for CI/CD
bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --json
```

### Library Usage

```python
from bigocheck import benchmark_function

def my_func(n):
    return sum(range(n))

analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
print(f"Best fit: {analysis.best_label}")  # 'O(n)'
```

---

## 📚 Feature Guide

### 1️⃣ Basic Benchmarking

Measure a function's complexity across input sizes.

```python
from bigocheck import benchmark_function

def bubble_sort(n):
    arr = list(range(n, 0, -1))
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

analysis = benchmark_function(bubble_sort, sizes=[100, 200, 400, 800], trials=2)
print(f"Best fit: {analysis.best_label}")  # O(n^2)

for fit in analysis.fits[:3]:
    print(f"  {fit.label}: error={fit.error:.4f}")
```

---

### 2️⃣ Complexity Assertions (CI/CD Testing)

Assert that functions have expected complexity. Perfect for CI/CD pipelines.

```python
from bigocheck import assert_complexity, ComplexityAssertionError

@assert_complexity("O(n)", sizes=[100, 500, 1000])
def linear_sum(n):
    return sum(range(n))

# First call triggers verification
linear_sum(10)  # Passes silently

# If complexity is wrong, raises ComplexityAssertionError
@assert_complexity("O(1)")  # Wrong!
def actually_linear(n):
    return sum(range(n))

try:
    actually_linear(10)
except ComplexityAssertionError as e:
    print(f"Caught: {e}")
```

---

### 3️⃣ Bounds Verification

Verify complexity without decorators.

```python
from bigocheck import verify_bounds

def my_sort(arr):
    return sorted(arr)

# Verify using wrapper
def test_wrapper(n):
    return my_sort(list(range(n)))

result = verify_bounds(test_wrapper, sizes=[1000, 5000, 10000], expected="O(n log n)")

if result.passes:
    print(f"✓ Verified: {result.expected}")
else:
    print(f"✗ Expected {result.expected}, got {result.actual}")
    
print(f"Confidence: {result.confidence} ({result.confidence_score:.0%})")
```

---

### 4️⃣ Confidence Scoring

Know how reliable your results are.

```python
from bigocheck import benchmark_function, compute_confidence

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000, 10000])
confidence = compute_confidence(analysis)

print(f"Confidence: {confidence.level} ({confidence.score:.0%})")
print("Reasons:")
for reason in confidence.reasons:
    print(f"  - {reason}")
```

Output:
```
Confidence: high (85%)
Reasons:
  - Clear gap between fits (0.234)
  - Low best-fit error (0.012)
  - Good measurement count (5)
  - Good size spread (ratio 100.0)
```

---

### 5️⃣ A/B Comparison

Compare two implementations head-to-head.

```python
from bigocheck import compare_functions

def linear_search(n):
    arr = list(range(n))
    return n - 1 in arr

def binary_search(n):
    arr = list(range(n))
    target = n - 1
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False

result = compare_functions(
    linear_search,
    binary_search,
    sizes=[1000, 5000, 10000, 50000],
)

print(result.summary)
# "binary_search is 15.23x faster than linear_search overall (4/4 sizes)"

print(f"Complexities: {result.func_a_label} vs {result.func_b_label}")
# "Complexities: O(n) vs O(log n)"
```

---

### 6️⃣ Report Generation

Generate beautiful markdown reports.

```python
from bigocheck import benchmark_function, generate_report, save_report

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000])
report = generate_report(analysis, title="My Analysis Report")

print(report)  # Print to console
save_report(report, "analysis_report.md")  # Save to file
```

**Comparison reports:**
```python
from bigocheck import compare_functions, generate_comparison_report

result = compare_functions(func_a, func_b, sizes=[100, 500, 1000])
report = generate_comparison_report(result)
save_report(report, "comparison.md")
```

**Verification reports:**
```python
from bigocheck import verify_bounds, generate_verification_report

result = verify_bounds(my_func, sizes=[100, 500], expected="O(n)")
report = generate_verification_report(result)
```

---

### 7️⃣ Auto Size Selection

Let bigocheck choose optimal input sizes automatically.

```python
from bigocheck import auto_select_sizes, benchmark_function

# Automatically find good sizes
sizes = auto_select_sizes(my_func, target_time=3.0, min_sizes=5)
print(f"Selected sizes: {sizes}")

# Use the auto-selected sizes
analysis = benchmark_function(my_func, sizes=sizes)
```

---

### 8️⃣ pytest Integration

Use the pytest plugin for testing.

```python
# In your test file
import pytest
from bigocheck.pytest_plugin import ComplexityChecker

def test_with_fixture(complexity_checker):
    def my_func(n):
        return sum(range(n))
    
    result = complexity_checker.check(my_func, expected="O(n)")
    assert result.passes, result.message

def test_with_assertion(complexity_checker):
    def my_func(n):
        return sum(range(n))
    
    # Raises ComplexityAssertionError if fails
    complexity_checker.assert_complexity(my_func, "O(n)")
```

**Register the plugin in conftest.py:**
```python
pytest_plugins = ["bigocheck.pytest_plugin"]
```

---

### 9️⃣ Data Generators

Use built-in data generators for testing.

```python
from bigocheck import benchmark_function
from bigocheck.datagen import integers, sorted_integers, arg_factory_for

def my_sort(arr):
    return sorted(arr)

# Benchmark with random integer lists
analysis = benchmark_function(
    my_sort,
    sizes=[1000, 5000, 10000],
    arg_factory=arg_factory_for(integers),
)
```

**Available generators:**

| Generator | Description |
|-----------|-------------|
| `n_(n)` | Returns N itself |
| `range_n(n)` | Returns `range(n)` |
| `integers(n, lo, hi)` | Random integers |
| `floats(n, lo, hi)` | Random floats |
| `strings(n, length)` | Random strings |
| `sorted_integers(n)` | Sorted random integers |
| `reversed_integers(n)` | Reverse-sorted integers |

---

### 🔟 Memory Profiling

Track peak memory usage.

```bash
bigocheck run --target mymodule:myfunc --sizes 1000 5000 10000 --memory
```

```python
analysis = benchmark_function(my_func, sizes=[1000, 5000, 10000], memory=True)

for m in analysis.measurements:
    print(f"n={m.size}: {m.seconds:.4f}s, memory={m.memory_bytes:,} bytes")
```

---

### 1️⃣1️⃣ Plotting (Optional)

Requires matplotlib: `pip install matplotlib`

```python
from bigocheck import benchmark_function
from bigocheck.plotting import plot_analysis, plot_all_fits

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000])

# Plot best fit
plot_analysis(analysis, title="My Analysis")

# Plot all complexity curves
plot_all_fits(analysis, save_path="all_fits.png", show=False)
```

---

## 🖥️ CLI Reference

```bash
bigocheck run --target MODULE:FUNC --sizes N1 N2 N3 [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--target` | Import path `module:func` (required) |
| `--sizes` | Input sizes to test (required) |
| `--trials` | Runs per size, averaged (default: 3) |
| `--warmup` | Warmup runs before timing (default: 0) |
| `--verbose`, `-v` | Show progress |
| `--memory` | Track memory usage |
| `--json` | JSON output for CI/CD |
| `--plot` | Show plot (requires matplotlib) |
| `--plot-save PATH` | Save plot to file |

---

## 🧮 Supported Complexity Classes

| Class | Notation | Example Use Case |
|-------|----------|------------------|
| Constant | O(1) | Hash table lookup |
| Logarithmic | O(log n) | Binary search |
| Square Root | O(√n) | Prime checking |
| Linear | O(n) | Linear search |
| Linearithmic | O(n log n) | Efficient sorting |
| Quadratic | O(n²) | Bubble sort |
| Cubic | O(n³) | Matrix multiplication |
| Exponential | O(2ⁿ) | Naive Fibonacci |
| Factorial | O(n!) | Permutations |

---

## 🔧 API Reference

### Core Functions

```python
from bigocheck import (
    # Core
    benchmark_function,    # Main benchmarking function
    fit_complexities,      # Fit measurements to complexity classes
    complexity_basis,      # Get all complexity basis functions
    
    # Assertions
    assert_complexity,     # Decorator for complexity assertions
    verify_bounds,         # Verify against expected complexity
    compute_confidence,    # Compute confidence score
    auto_select_sizes,     # Auto-select optimal sizes
    
    # Comparison
    compare_functions,     # A/B comparison
    compare_to_baseline,   # Compare to baseline complexity
    
    # Reports
    generate_report,       # Generate markdown report
    generate_comparison_report,
    generate_verification_report,
    save_report,
    
    # Data Classes
    Analysis,
    Measurement,
    FitResult,
    VerificationResult,
    ComparisonResult,
    ConfidenceResult,
)
```

---

## 📁 Project Structure

```
bigocheck/
├── src/bigocheck/
│   ├── __init__.py      # Package exports
│   ├── core.py          # Benchmarking and fitting
│   ├── cli.py           # Command-line interface
│   ├── assertions.py    # Assertions and verification
│   ├── compare.py       # A/B comparison
│   ├── reports.py       # Report generation
│   ├── datagen.py       # Data generators
│   ├── plotting.py      # Optional plotting
│   └── pytest_plugin.py # pytest integration
├── tests/               # Test suite
├── pyproject.toml
└── LICENSE
```

---

## 🧪 Testing

```bash
pip install -e '.[dev]'
pytest -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

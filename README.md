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
| **🧮 Time Complexity** | Fits to 9 classes: O(1), O(log n), O(√n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ), O(n!) |
| **📐 Space Complexity** | Classifies memory usage to complexity classes |
| **📏 Polynomial Fitting** | Detect O(n^k) for arbitrary k (e.g., O(n^2.34)) |
| **📊 Statistical Significance** | P-values to validate complexity classification |
| **🔄 Regression Detection** | CLI: `bigocheck regression --baseline file.json` |
| **📉 Best/Worst/Avg Cases** | Analyze with sorted, reversed, and random inputs |
| **⚡ Async Support** | Benchmark `async def` functions |
| **📊 Amortized Analysis** | Track complexity over sequences of operations |
| **🚀 Parallel Benchmarking** | Run sizes in parallel for faster results |
| **📑 HTML Reports** | Generate beautiful HTML reports with SVG charts |
| **💻 Interactive REPL** | CLI: `bigocheck repl` for quick analysis |
| **✅ Complexity Assertions** | `@assert_complexity("O(n)")` decorator |
| **🔍 Bounds Verification** | `verify_bounds()` to check expected complexity |
| **📊 Confidence Scoring** | Know how reliable your results are |
| **🔀 A/B Comparison** | Compare two implementations head-to-head |
| **📄 Report Generation** | Generate markdown reports automatically |
| **🔧 pytest Plugin** | Integration with pytest for testing |
| **📈 Plotting** | Optional matplotlib visualization |
| **💾 Memory Profiling** | Track peak memory usage with `--memory` flag |
| **🚀 Auto Size Selection** | Automatically choose optimal input sizes |
| **📦 Zero Dependencies** | Pure standard library, no numpy required |
| **💻 CLI-First** | Full command-line interface |
| **⚙️ GitHub Actions** | Pre-built CI workflow template |

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

### 🔟 Memory & Space Complexity

Track memory usage and automatically classify space complexity.

```bash
bigocheck run --target mymodule:myfunc --sizes 1000 5000 10000 --memory
```

**Example Output:**
```
Time Complexity:  O(n)
Space Complexity: O(n)

Measurements:
  n=1000     time=0.001234s ±0.000001s  mem=81,920B
  n=5000     time=0.006789s ±0.000005s  mem=409,600B
  n=10000    time=0.012345s ±0.000012s  mem=819,200B
```

**Library Usage:**
```python
analysis = benchmark_function(my_func, sizes=[1000, 5000, 10000], memory=True)

print(f"Time Complexity:  {analysis.best_label}")
print(f"Space Complexity: {analysis.space_label}")

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
### 1️⃣2️⃣ Polynomial Fitting

Detect O(n^k) for arbitrary k values.

```python
from bigocheck import benchmark_function, fit_polynomial

def my_func(n):
    return sum(i * i for i in range(n))

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000])
poly = fit_polynomial(analysis.measurements)

print(f"Detected: {poly.label}")      # e.g., "O(n^1.23)"
print(f"Exponent: {poly.exponent:.2f}")  # 1.23
print(f"Error: {poly.error:.4f}")
```

---

### 1️⃣3️⃣ Statistical Significance

Get p-values to validate complexity classification.

```python
from bigocheck import benchmark_function, compute_significance

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000, 10000])
sig = compute_significance(analysis)

print(f"p-value: {sig.p_value:.4f}")
print(f"Significant: {sig.is_significant}")  # True if p < 0.05
print(f"Confidence: {sig.confidence_level}")  # "high", "medium", "low"
```

---

### 1️⃣4️⃣ Regression Detection (CI/CD)

Save baselines and detect performance regressions.

**CLI:**
```bash
# Save a baseline
bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --save-baseline baseline.json

# Check for regressions
bigocheck regression --target mymodule:myfunc --baseline baseline.json
```

**Library:**
```python
from bigocheck import benchmark_function, save_baseline, load_baseline, detect_regression

# Save baseline
analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
save_baseline(analysis, "baseline.json", name="v1.0")

# Later: detect regressions
current = benchmark_function(my_func, sizes=[100, 500, 1000])
baseline = load_baseline("baseline.json")
result = detect_regression(current, baseline, time_threshold=0.2)

if result.has_regression:
    print(f"❌ REGRESSION: {result.message}")
else:
    print("✅ No regression")
```

---

### 1️⃣5️⃣ Best/Worst/Average Case Analysis

Test performance with different input arrangements.

```python
from bigocheck import analyze_cases

def my_sort(arr):
    return sorted(arr)

result = analyze_cases(my_sort, sizes=[1000, 5000, 10000])

print(result.summary)
# Case Analysis Summary:
#   Best case:    best - O(n log n)
#   Worst case:   worst - O(n log n)
#   Average case: average - O(n log n)

print(f"Best:  {result.best_case.time_complexity}")
print(f"Worst: {result.worst_case.time_complexity}")
```

---

### 1️⃣6️⃣ Async Function Support

Benchmark `async def` functions.

```python
import asyncio
from bigocheck import run_benchmark_async, benchmark_async

async def fetch_data(n):
    await asyncio.sleep(0.001 * n)
    return list(range(n))

# Synchronous wrapper (easier)
analysis = run_benchmark_async(fetch_data, sizes=[10, 50, 100])
print(f"Complexity: {analysis.best_label}")

# Or use async directly
async def main():
    analysis = await benchmark_async(fetch_data, sizes=[10, 50, 100])
    print(f"Complexity: {analysis.best_label}")

asyncio.run(main())
```

---

### 1️⃣7️⃣ Amortized Analysis

Analyze complexity over sequences of operations.

```python
from bigocheck import analyze_amortized

# Example: dynamic array append
data = []
def append_op():
    data.append(len(data))

result = analyze_amortized(append_op, n_operations=1000)

print(result.summary)
print(f"Amortized: {result.amortized_complexity}")
print(f"Total time: {result.total_time:.6f}s")
print(f"Per operation: {result.amortized_time:.9f}s")
```

---

### 1️⃣8️⃣ Parallel Benchmarking

Run benchmarks faster using parallel execution.

```python
from bigocheck import benchmark_parallel, benchmark_function
import time

def slow_func(n):
    time.sleep(0.01)
    return sum(range(n))

# Sequential (slower)
start = time.time()
analysis = benchmark_function(slow_func, sizes=[100, 500, 1000, 5000], trials=1)
print(f"Sequential: {time.time() - start:.2f}s")

# Parallel (faster)
start = time.time()
analysis = benchmark_parallel(slow_func, sizes=[100, 500, 1000, 5000], trials=1)
print(f"Parallel: {time.time() - start:.2f}s")
```

---

### 1️⃣9️⃣ HTML Reports

Generate beautiful HTML reports with SVG charts.

**CLI:**
```bash
bigocheck run --target mymodule:myfunc --sizes 100 500 1000 --html report.html
```

**Library:**
```python
from bigocheck import benchmark_function, generate_html_report, save_html_report

analysis = benchmark_function(my_func, sizes=[100, 500, 1000, 5000], memory=True)
html = generate_html_report(analysis, title="My Analysis")
save_html_report(html, "report.html")
```

**Example Output:**

![HTML Report Example](docs/html_report_screenshot.png)

---

### 2️⃣0️⃣ Interactive REPL

Quick analysis from the command line.

**CLI:**
```bash
bigocheck repl
```

**Output:**
```
╔══════════════════════════════════════════════════════════════════╗
║  bigocheck Interactive Mode                                       ║
║  Zero-dependency complexity analysis                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Quick start:                                                     ║
║  >>> def my_func(n): return sum(range(n))                        ║
║  >>> a = benchmark_function(my_func, sizes=[100, 500, 1000])     ║
║  >>> print(a.best_label)                                         ║
╚══════════════════════════════════════════════════════════════════╝
>>> 
```

**One-liner complexity check:**
```python
from bigocheck import quick_check

result = quick_check(lambda n: sum(range(n)))
print(result)  # "O(n)"
```

---

## 🖥️ CLI Reference

```bash
bigocheck run --target MODULE:FUNC --sizes N1 N2 N3 [OPTIONS]
bigocheck regression --target MODULE:FUNC --baseline FILE [OPTIONS]
bigocheck repl
```

| Option | Description |
|--------|-------------|
| `--target` | Import path `module:func` (required) |
| `--sizes` | Input sizes to test (required for `run`) |
| `--trials` | Runs per size, averaged (default: 3) |
| `--warmup` | Warmup runs before timing (default: 0) |
| `--verbose`, `-v` | Show progress |
| `--memory` | Track memory usage |
| `--json` | JSON output for CI/CD |
| `--plot` | Show plot (requires matplotlib) |
| `--plot-save PATH` | Save plot to file |
| `--html PATH` | Generate HTML report |
| `--save-baseline PATH` | Save baseline for regression detection |
| `--baseline PATH` | Baseline file for regression check |
| `--threshold` | Slowdown threshold for regression (default: 0.2) |

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
| **Polynomial** | O(n^k) | Detected via `fit_polynomial()` |

---

## 🔧 API Reference

### Core Functions

```python
from bigocheck import (
    # Core
    benchmark_function,    # Main benchmarking function
    fit_complexities,      # Fit measurements to complexity classes
    fit_space_complexity,  # Fit memory to complexity classes
    complexity_basis,      # Get all complexity basis functions
    
    # Assertions
    assert_complexity,     # Decorator for complexity assertions
    verify_bounds,         # Verify against expected complexity
    compute_confidence,    # Compute confidence score
    auto_select_sizes,     # Auto-select optimal sizes
    
    # Comparison
    compare_functions,     # A/B comparison
    compare_to_baseline,   # Compare to baseline complexity
    
    # Reports (Markdown)
    generate_report,       # Generate markdown report
    generate_comparison_report,
    generate_verification_report,
    save_report,
    
    # Reports (HTML)
    generate_html_report,  # Generate HTML report with charts
    save_html_report,
    
    # Statistics
    compute_significance,  # P-values for classification
    SignificanceResult,
    
    # Regression Detection
    save_baseline,         # Save baseline JSON
    load_baseline,         # Load baseline JSON
    detect_regression,     # Detect regressions
    Baseline,
    RegressionResult,
    
    # Case Analysis
    analyze_cases,         # Best/worst/avg case
    CasesAnalysis,
    CaseResult,
    
    # Polynomial Fitting
    fit_polynomial,        # Detect O(n^k)
    fit_polynomial_space,
    PolynomialFit,
    
    # Async Benchmarking
    benchmark_async,       # Async version
    run_benchmark_async,   # Sync wrapper for async
    
    # Amortized Analysis
    analyze_amortized,     # Sequence analysis
    analyze_sequence,
    AmortizedResult,
    
    # Parallel Benchmarking
    benchmark_parallel,    # Parallel execution
    
    # Interactive
    start_repl,            # Start REPL
    quick_check,           # One-liner check
    
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
│   ├── __init__.py       # Package exports (25+ functions)
│   ├── core.py           # Benchmarking and fitting
│   ├── cli.py            # CLI (run, regression, repl)
│   ├── assertions.py     # @assert_complexity, verify_bounds
│   ├── compare.py        # A/B comparison
│   ├── reports.py        # Markdown report generation
│   ├── html_report.py    # HTML report with SVG charts
│   ├── statistics.py     # P-values and significance
│   ├── regression.py     # Baseline save/load, regression detection
│   ├── cases.py          # Best/worst/average case analysis
│   ├── polynomial.py     # O(n^k) polynomial fitting
│   ├── async_bench.py    # Async function benchmarking
│   ├── amortized.py      # Amortized complexity analysis
│   ├── parallel.py       # Parallel benchmarking
│   ├── interactive.py    # REPL mode
│   ├── datagen.py        # Data generators
│   ├── plotting.py       # Optional matplotlib plots
│   ├── pytest_plugin.py  # pytest integration
│   └── pre_commit.py     # Pre-commit hook template
├── .github/workflows/    # CI/CD templates
├── docs/                 # Documentation assets
├── tests/                # Test suite (77+ tests)
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

---

## 🧪 Testing

```bash
pip install -e '.[dev]'
pytest -v
```

**Test Coverage:**
- Core benchmarking and fitting
- All complexity assertions
- Comparison and reports
- Statistical significance
- Regression detection
- Case analysis
- Polynomial fitting
- Async benchmarking
- Amortized analysis
- Parallel benchmarking
- HTML report generation

---


## 📄 License

MIT — see [LICENSE](LICENSE).

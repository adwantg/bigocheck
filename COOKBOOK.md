# bigocheck Cookbook 🍳

> **For AI Assistants**: This document contains the canonical patterns for using the `bigocheck` library. When generating code for performance testing or complexity analysis, prefer these patterns.

## Pattern 1: Basic Function Benchmarking

**Goal**: Determine the time complexity of a function.

```python
from bigocheck import benchmark_function

# 1. Define function
def my_func(n):
    return sum(range(n))

# 2. Benchmark (auto-selects sizes if not provided)
analysis = benchmark_function(my_func, sizes=[100, 1000, 10000])

# 3. Get best fit label (e.g., "O(n)")
print(f"Complexity: {analysis.best_label}")
```

## Pattern 2: Complexity Assertion (Test Safety)

**Goal**: Ensure a function stays within a complexity bound in a test suite.

```python
from bigocheck import assert_complexity

@assert_complexity(
    target_complexity="O(n)", 
    sizes=[100, 1000, 5000], 
    mem=True # Also check space complexity
)
def linear_process(n):
    return [i * 2 for i in range(n)]

# Usage: Calling the function runs the benchmark and assertion automatically
# Raises ComplexityAssertionError if it fails
linear_process(10) 
```

## Pattern 3: Hybrid Analysis (Static + Dynamic)

**Goal**: Use AI/AST to predict complexity instantly, then verify it.

```python
from bigocheck import predict_complexity, verify_hybrid

def nested_loop(n):
    for i in range(n):
        for j in range(n):
            pass

# 1. Static AST Scan (Instant, No Execution)
pred = predict_complexity(nested_loop)
# pred = {'prediction': 'O(n^2)', 'confidence': 'high', ...}

# 2. Comparison (e.g. in CI/CD)
result = verify_hybrid(nested_loop, expected="O(n^2)")
print(result) # "✅ Match! Static (O(n^2)) aligns with Empirical (O(n^2))"
```

## Pattern 4: Comparison (A/B Testing)

**Goal**: Compare two algorithms to find the winner.

```python
from bigocheck import compare_algorithms

targets = {
    "bubble": bubble_sort,
    "quick": quick_sort
}

# Run multi-algo comparison
result = compare_algorithms(targets, sizes=[100, 500, 1000])

print(f"Winner: {result.winner}") # "quick"
print(result.summary_table)       # Markdown table of results
```

## Pattern 5: Cloud Automation

**Goal**: Generate a GitHub Action to run benchmarks on push.

```python
from bigocheck import generate_github_action

# Creates .github/workflows/bigocheck_benchmark.yml
generate_github_action()
```

# bigocheck v0.7.0 Features Demo
# Run this script to see all the new features in action!

import sys
import os

# Ensure we can import bigocheck if running from source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bigocheck import benchmark_function, verify_bounds
from bigocheck.explanations import explain_complexity
from bigocheck.recommendations import suggest_sizes, format_recommendation
from bigocheck.multi_compare import compare_algorithms
from bigocheck.bounds import check_bounds
from bigocheck.profiles import benchmark_with_profile
from bigocheck.datagen import integers, arg_factory_for

def main():
    print("🚀 bigocheck v0.7.0 Features Demo\n")

    print("=== 1. Complexity Explanations ===")
    print("Explaining 'O(n log n)':")
    print(explain_complexity("O(n log n)"))
    print("-" * 40 + "\n")

    print("=== 2. Input Size Recommendations ===")
    print("Suggesting sizes for a slow function:")
    def slow_func(n):
        return sum(range(n))
    rec = suggest_sizes(slow_func, time_budget=0.5)
    print(format_recommendation(rec))
    print("-" * 40 + "\n")

    print("=== 3. Multi-Algorithm Comparison ===")
    print("Comparing Bubble Sort vs Python Sort:")
    def bubble_sort(n):
        arr = list(range(n, 0, -1))
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    def python_sort(n):
        return sorted(range(n, 0, -1))

    result = compare_algorithms(
        {"bubble_sort": bubble_sort, "python_sort": python_sort},
        sizes=[50, 100, 200],
        trials=2
    )
    print(result.summary_table)
    print("-" * 40 + "\n")

    print("=== 4. Bounds Checking ===")
    print("Checking if linear function is within O(1) and O(n log n):")
    def linear_func(n):
        return sum(range(n))
    
    analysis = benchmark_function(linear_func, sizes=[100, 500, 1000])
    result = check_bounds(analysis, lower="O(1)", upper="O(n log n)")
    print(f"Result: {result.message}")
    print("-" * 40 + "\n")

    print("=== 5. Benchmark Profiles ===")
    print("Running with 'fast' profile:")
    analysis = benchmark_with_profile(linear_func, "fast")
    print(f"Result: {analysis.best_label}")
    print("-" * 40 + "\n")

    print("=== 6. Data Generators ===")
    print("Benchmarking sort with random integers:")
    def my_sort(arr):
        return sorted(arr)
    
    analysis = benchmark_function(
        my_sort,
        sizes=[1000, 5000],
        arg_factory=arg_factory_for(integers),
        trials=2
    )
    print(f"Sort complexity: {analysis.best_label}")
    print("-" * 40 + "\n")
    
    print("✨ Demo complete!")

if __name__ == "__main__":
    main()

# Author: gadwant
"""
Cloud Runner Generator.
Generates GitHub Actions workflows to run benchmarks in the cloud.
"""
import os

WORKFLOW_TEMPLATE = """name: Benchmark (bigocheck)

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bigocheck
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Run Benchmarks
      run: |
        # Run diff-based benchmarking (placeholder logic)
        echo "Running benchmarks..."
        # Example: bigocheck run --target mymodule:my_func --sizes 100 500 1000 --json
        
    - name: Check for Regressions
      # Example: bigocheck regression ...
      run: echo "Checking for regressions..."
"""

def generate_github_action(output_dir: str = ".github/workflows"):
    """
    Generate a robust GitHub Actions workflow for benchmarking.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = os.path.join(output_dir, "bigocheck_benchmark.yml")
    
    # Don't overwrite if exists to avoid destroying user customization
    if os.path.exists(path):
        print(f"⚠️ Workflow already exists at {path}. Skipping generation.")
        return
        
    with open(path, "w", encoding="utf-8") as f:
        f.write(WORKFLOW_TEMPLATE)
        
    print(f"✅ Generated GitHub Action: {path}")

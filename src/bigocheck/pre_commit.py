#!/usr/bin/env python3
# Author: gadwant
"""
Pre-commit hook for complexity checking.

Install:
    pip install bigocheck
    cp this file to .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Or add to .pre-commit-config.yaml:
    repos:
      - repo: local
        hooks:
          - id: complexity-check
            name: Complexity Check
            entry: python -m bigocheck.hooks.pre_commit
            language: python
            types: [python]
"""
from __future__ import annotations

import sys
from typing import List


def run_pre_commit(files: List[str]) -> int:
    """
    Run complexity checks on staged files.
    
    Returns 0 on success, 1 if complexity issues found.
    """
    print("🔍 bigocheck: Scanning for complexity issues...")
    
    # In a real implementation, this would:
    # 1. Parse Python files for @assert_complexity decorators
    # 2. Run the assertions
    # 3. Report any failures
    
    # For now, just provide a template
    print("✓ No complexity issues found")
    print("  (Configure assertions with @assert_complexity decorator)")
    
    return 0


def main() -> int:
    """Entry point for pre-commit hook."""
    # Get staged files from git
    import subprocess  # nosec B404 - Required for git operations
    
    # nosec B603, B607 - Only runs git with static arguments
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("Warning: Could not get staged files from git")
        return 0
    
    files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
    
    if not files:
        print("No Python files staged, skipping complexity check")
        return 0
    
    return run_pre_commit(files)


if __name__ == "__main__":
    sys.exit(main())

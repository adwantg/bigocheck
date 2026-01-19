# Author: gadwant
"""
Git commit tracking for complexity analysis.

Track complexity changes across git commits and find regression commits.
"""
from __future__ import annotations

import subprocess  # nosec B404 - Required for git operations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import Analysis, benchmark_function, resolve_callable


@dataclass
class CommitResult:
    """Result for a single commit."""
    commit_hash: str
    commit_message: str
    time_complexity: str
    space_complexity: Optional[str]
    avg_time: float
    analysis: Analysis


@dataclass
class TrackingResult:
    """Result of tracking complexity across commits."""
    results: List[CommitResult]
    has_regression: bool
    regression_commit: Optional[str]
    regression_from: Optional[str]
    regression_to: Optional[str]
    summary: str


def _run_git_command(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run a git command and return output."""
    # nosec B603, B607 - Only runs git with controlled arguments
    result = subprocess.run(
        ["git"] + cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {result.stderr}")
    return result.stdout.strip()


def _get_commit_info(commit: str, cwd: Optional[str] = None) -> Tuple[str, str]:
    """Get commit hash and message."""
    full_hash = _run_git_command(["rev-parse", commit], cwd)
    message = _run_git_command(["log", "-1", "--format=%s", commit], cwd)
    return full_hash[:8], message


def _checkout_commit(commit: str, cwd: Optional[str] = None) -> None:
    """Checkout a specific commit."""
    _run_git_command(["checkout", commit, "--quiet"], cwd)


def _get_current_branch(cwd: Optional[str] = None) -> str:
    """Get current branch name."""
    return _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd)


def track_commits(
    target: str,
    commits: List[str],
    sizes: List[int],
    *,
    trials: int = 3,
    memory: bool = False,
    cwd: Optional[str] = None,
) -> TrackingResult:
    """
    Track complexity changes across git commits.
    
    Args:
        target: Import path in form "module:func".
        commits: List of commit refs (e.g., ["HEAD~10", "HEAD~5", "HEAD"]).
        sizes: Input sizes to benchmark.
        trials: Number of trials per size.
        memory: Track memory usage.
        cwd: Working directory (git repo root).
    
    Returns:
        TrackingResult with per-commit results and regression info.
    
    Example:
        >>> result = track_commits(
        ...     "mymodule:myfunc",
        ...     commits=["v1.0", "v1.1", "v2.0"],
        ...     sizes=[100, 500, 1000]
        ... )
        >>> if result.has_regression:
        ...     print(f"Regression at {result.regression_commit}")
    """
    original_branch = _get_current_branch(cwd)
    results: List[CommitResult] = []
    
    try:
        for commit in commits:
            # Get commit info
            commit_hash, commit_message = _get_commit_info(commit, cwd)
            
            # Checkout commit
            _checkout_commit(commit, cwd)
            
            # Resolve and benchmark
            try:
                func = resolve_callable(target)
                analysis = benchmark_function(
                    func, sizes=sizes, trials=trials, memory=memory
                )
                
                avg_time = sum(m.seconds for m in analysis.measurements) / len(analysis.measurements)
                
                results.append(CommitResult(
                    commit_hash=commit_hash,
                    commit_message=commit_message,
                    time_complexity=analysis.best_label,
                    space_complexity=analysis.space_label,
                    avg_time=avg_time,
                    analysis=analysis,
                ))
            except Exception as e:
                # Function may not exist in older commits
                print(f"Warning: Could not benchmark {commit}: {e}")
    
    finally:
        # Always restore original branch
        _checkout_commit(original_branch, cwd)
    
    # Detect regressions
    has_regression = False
    regression_commit = None
    regression_from = None
    regression_to = None
    
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        
        # Check for complexity class regression
        if _is_worse_complexity(prev.time_complexity, curr.time_complexity):
            has_regression = True
            regression_commit = curr.commit_hash
            regression_from = prev.time_complexity
            regression_to = curr.time_complexity
            break
    
    # Generate summary
    summary_lines = [
        f"Tracked {len(results)} commits:",
    ]
    for r in results:
        summary_lines.append(
            f"  {r.commit_hash}: {r.time_complexity} ({r.commit_message[:40]}...)"
        )
    
    if has_regression:
        summary_lines.append(f"\n❌ Regression detected at {regression_commit}:")
        summary_lines.append(f"   {regression_from} → {regression_to}")
    else:
        summary_lines.append("\n✅ No complexity regressions detected")
    
    return TrackingResult(
        results=results,
        has_regression=has_regression,
        regression_commit=regression_commit,
        regression_from=regression_from,
        regression_to=regression_to,
        summary="\n".join(summary_lines),
    )


def _is_worse_complexity(before: str, after: str) -> bool:
    """Check if 'after' is worse (higher) complexity than 'before'."""
    order = [
        "O(1)", "O(log n)", "O(√n)", "O(n)", "O(n log n)",
        "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)",
    ]
    
    try:
        before_idx = order.index(before)
        after_idx = order.index(after)
        return after_idx > before_idx
    except ValueError:
        # Unknown complexity, compare lexically
        return after > before


def find_regression_commit(
    target: str,
    good_commit: str,
    bad_commit: str,
    sizes: List[int],
    *,
    cwd: Optional[str] = None,
) -> Optional[str]:
    """
    Binary search to find the exact commit that introduced a regression.
    
    Similar to git bisect but for complexity regressions.
    
    Args:
        target: Import path "module:func".
        good_commit: Known good commit (lower complexity).
        bad_commit: Known bad commit (higher complexity).
        sizes: Input sizes to benchmark.
        cwd: Working directory.
    
    Returns:
        Commit hash of first bad commit, or None if no regression found.
    """
    # Get list of commits between good and bad
    commits_output = _run_git_command(
        ["rev-list", "--reverse", f"{good_commit}..{bad_commit}"],
        cwd
    )
    commits = commits_output.split('\n') if commits_output else []
    
    if not commits:
        return None
    
    # Binary search
    left, right = 0, len(commits) - 1
    
    # Get baseline complexity from good commit
    good_result = track_commits(target, [good_commit], sizes, cwd=cwd)
    if not good_result.results:
        return None
    baseline = good_result.results[0].time_complexity
    
    while left < right:
        mid = (left + right) // 2
        
        result = track_commits(target, [commits[mid]], sizes, cwd=cwd)
        if not result.results:
            left = mid + 1
            continue
        
        if _is_worse_complexity(baseline, result.results[0].time_complexity):
            right = mid
        else:
            left = mid + 1
    
    return commits[left][:8] if commits else None

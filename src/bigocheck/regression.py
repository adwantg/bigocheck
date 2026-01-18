# Author: gadwant
"""
Regression detection utilities for tracking performance over time.

Allows saving baselines and detecting complexity regressions in CI/CD pipelines.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Analysis


@dataclass
class Baseline:
    """Saved baseline for regression detection."""
    name: str
    best_label: str
    space_label: Optional[str]
    measurements: List[Dict[str, Any]]
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class RegressionResult:
    """Result of regression comparison."""
    has_regression: bool
    time_regression: bool
    space_regression: bool
    current_time_label: str
    baseline_time_label: str
    current_space_label: Optional[str]
    baseline_space_label: Optional[str]
    time_slowdown: float  # Multiplier (> 1.0 means slower)
    message: str


def save_baseline(
    analysis: "Analysis",
    path: str,
    name: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save an analysis as a baseline for future comparison.
    
    Args:
        analysis: Analysis object to save.
        path: File path to save baseline JSON.
        name: Optional name for the baseline.
        metadata: Optional metadata (e.g., git commit, version).
    
    Example:
        >>> analysis = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> save_baseline(analysis, "baseline.json", name="v1.0")
    """
    baseline = Baseline(
        name=name,
        best_label=analysis.best_label,
        space_label=analysis.space_label,
        measurements=[
            {
                "size": m.size,
                "seconds": m.seconds,
                "std_dev": m.std_dev,
                "memory_bytes": m.memory_bytes,
            }
            for m in analysis.measurements
        ],
        timestamp=datetime.now().isoformat(),
        metadata=metadata or {},
    )
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(baseline), f, indent=2)


def load_baseline(path: str) -> Baseline:
    """
    Load a baseline from file.
    
    Args:
        path: Path to baseline JSON file.
    
    Returns:
        Baseline object.
    
    Raises:
        FileNotFoundError: If baseline file doesn't exist.
        ValueError: If file is not valid baseline JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Baseline(
        name=data.get("name", "unknown"),
        best_label=data["best_label"],
        space_label=data.get("space_label"),
        measurements=data["measurements"],
        timestamp=data.get("timestamp", "unknown"),
        metadata=data.get("metadata", {}),
    )


def detect_regression(
    current: "Analysis",
    baseline: Baseline,
    *,
    time_threshold: float = 0.2,
    memory_threshold: float = 0.3,
) -> RegressionResult:
    """
    Detect performance regressions by comparing current analysis to baseline.
    
    Args:
        current: Current analysis from benchmark_function.
        baseline: Previously saved baseline.
        time_threshold: Time slowdown threshold (0.2 = 20% slower triggers).
        memory_threshold: Memory increase threshold.
    
    Returns:
        RegressionResult with regression status and details.
    
    Example:
        >>> current = benchmark_function(my_func, sizes=[100, 500, 1000])
        >>> baseline = load_baseline("baseline.json")
        >>> result = detect_regression(current, baseline)
        >>> assert not result.has_regression, result.message
    """
    messages = []
    
    # Check complexity class regression
    time_regression = current.best_label != baseline.best_label
    if time_regression:
        messages.append(
            f"Time complexity changed: {baseline.best_label} → {current.best_label}"
        )
    
    # Check space complexity regression
    space_regression = False
    if current.space_label and baseline.space_label:
        space_regression = current.space_label != baseline.space_label
        if space_regression:
            messages.append(
                f"Space complexity changed: {baseline.space_label} → {current.space_label}"
            )
    
    # Calculate time slowdown by comparing measurements
    time_slowdown = 1.0
    baseline_times = {m["size"]: m["seconds"] for m in baseline.measurements}
    
    slowdowns = []
    for m in current.measurements:
        if m.size in baseline_times:
            baseline_time = baseline_times[m.size]
            if baseline_time > 0:
                slowdown = m.seconds / baseline_time
                slowdowns.append(slowdown)
    
    if slowdowns:
        time_slowdown = sum(slowdowns) / len(slowdowns)
        
        if time_slowdown > (1 + time_threshold):
            messages.append(
                f"Performance slowdown: {time_slowdown:.2f}x slower (threshold: {1 + time_threshold:.2f}x)"
            )
    
    has_regression = time_regression or space_regression or time_slowdown > (1 + time_threshold)
    
    if not messages:
        messages.append("No regressions detected ✓")
    
    return RegressionResult(
        has_regression=has_regression,
        time_regression=time_regression,
        space_regression=space_regression,
        current_time_label=current.best_label,
        baseline_time_label=baseline.best_label,
        current_space_label=current.space_label,
        baseline_space_label=baseline.space_label,
        time_slowdown=time_slowdown,
        message="\n".join(messages),
    )


def compare_to_baseline_file(
    analysis: "Analysis",
    baseline_path: str,
    *,
    time_threshold: float = 0.2,
) -> RegressionResult:
    """
    Compare analysis to a baseline file.
    
    Convenience function that combines load_baseline and detect_regression.
    
    Args:
        analysis: Current analysis.
        baseline_path: Path to baseline JSON file.
        time_threshold: Slowdown threshold.
    
    Returns:
        RegressionResult.
    """
    baseline = load_baseline(baseline_path)
    return detect_regression(analysis, baseline, time_threshold=time_threshold)

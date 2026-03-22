# Author: gadwant
"""
Benchmark profiles for different use cases.

Preset configurations: fast, accurate, thorough.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional, Union

from .core import Analysis, benchmark_function


@dataclass
class BenchmarkProfile:
    """Configuration profile for benchmarking."""
    name: str
    sizes: List[int]
    trials: int
    warmup: int
    robust: bool = False
    description: str


DEFAULT_PROFILE_NAME = "balanced"
DEFAULT_ASSERTION_PROFILE_NAME = "ci"


# Preset profiles
PROFILES: Dict[str, BenchmarkProfile] = {
    "fast": BenchmarkProfile(
        name="fast",
        sizes=[100, 500, 1000],
        trials=2,
        warmup=0,
        robust=False,
        description="Quick check - minimal sizes, fewer trials",
    ),
    "balanced": BenchmarkProfile(
        name="balanced",
        sizes=[100, 500, 1000, 5000],
        trials=3,
        warmup=1,
        robust=False,
        description="Default - good balance of speed and accuracy",
    ),
    "accurate": BenchmarkProfile(
        name="accurate",
        sizes=[100, 250, 500, 1000, 2500, 5000],
        trials=5,
        warmup=2,
        robust=False,
        description="More data points and trials for better accuracy",
    ),
    "thorough": BenchmarkProfile(
        name="thorough",
        sizes=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        trials=7,
        warmup=3,
        robust=False,
        description="Maximum accuracy - many sizes and trials",
    ),
    "large": BenchmarkProfile(
        name="large",
        sizes=[1000, 5000, 10000, 50000, 100000],
        trials=3,
        warmup=1,
        robust=False,
        description="For functions that need large inputs",
    ),
    "small": BenchmarkProfile(
        name="small",
        sizes=[10, 25, 50, 100, 200],
        trials=5,
        warmup=1,
        robust=False,
        description="For slow functions - smaller input sizes",
    ),
    "ci": BenchmarkProfile(
        name="ci",
        sizes=[200, 1000, 5000, 20000],
        trials=5,
        warmup=2,
        robust=True,
        description="CI-safe profile - larger sizes, more trials, robust aggregation",
    ),
}


def get_profile(name: str) -> BenchmarkProfile:
    """
    Get a benchmark profile by name.
    
    Available profiles: fast, balanced, accurate, thorough, large, small
    
    Args:
        name: Profile name.
    
    Returns:
        BenchmarkProfile configuration.
    
    Raises:
        ValueError: If profile name is not recognized.
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def benchmark_with_profile(
    func: Callable,
    profile: str = DEFAULT_PROFILE_NAME,
    *,
    memory: bool = False,
) -> Analysis:
    """
    Run benchmark using a predefined profile.
    
    Args:
        func: Function to benchmark.
        profile: Profile name (fast, balanced, accurate, thorough, large, small).
        memory: Track memory usage.
    
    Returns:
        Analysis object.
    
    Example:
        >>> analysis = benchmark_with_profile(my_func, "accurate")
        >>> print(analysis.best_label)
    """
    p = get_profile(profile)
    return benchmark_function(
        func,
        sizes=p.sizes,
        trials=p.trials,
        warmup=p.warmup,
        memory=memory,
        robust=p.robust,
    )


def profile_decorator(
    profile: str = DEFAULT_PROFILE_NAME,
    *,
    memory: bool = False,
    print_result: bool = True,
) -> Callable:
    """
    Decorator to benchmark a function with a profile.
    
    Args:
        profile: Profile name.
        memory: Track memory.
        print_result: Print result on first call.
    
    Returns:
        Decorated function.
    
    Example:
        >>> @profile_decorator("accurate")
        ... def my_sort(n):
        ...     return sorted(range(n))
        >>> 
        >>> my_sort(100)  # First call prints complexity
    """
    def decorator(func: Callable) -> Callable:
        _analyzed = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _analyzed
            
            if not _analyzed and print_result:
                analysis = benchmark_with_profile(func, profile, memory=memory)
                print(f"📊 {func.__name__}: {analysis.best_label}")
                _analyzed = True
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def list_profiles() -> str:
    """List all available profiles with descriptions."""
    lines = [
        "Available Benchmark Profiles",
        "=" * 50,
        "",
    ]
    
    for name, p in PROFILES.items():
        lines.append(f"• {name}")
        lines.append(f"  Sizes: {p.sizes}")
        lines.append(f"  Trials: {p.trials}, Warmup: {p.warmup}, Robust: {p.robust}")
        lines.append(f"  {p.description}")
        lines.append("")
    
    return "\n".join(lines)


def create_custom_profile(
    name: str,
    sizes: List[int],
    trials: int = 3,
    warmup: int = 1,
    robust: bool = False,
    description: str = "Custom profile",
) -> BenchmarkProfile:
    """
    Create a custom benchmark profile.
    
    Args:
        name: Profile name.
        sizes: Input sizes.
        trials: Trials per size.
        warmup: Warmup runs.
        description: Profile description.
    
    Returns:
        BenchmarkProfile object.
    
    Example:
        >>> profile = create_custom_profile("my_profile", [100, 1000, 10000], trials=10)
    """
    return BenchmarkProfile(
        name=name,
        sizes=sizes,
        trials=trials,
        warmup=warmup,
        robust=robust,
        description=description,
    )


def resolve_profile(
    *,
    profile: Optional[Union[str, BenchmarkProfile]] = None,
    sizes: Optional[List[int]] = None,
    trials: Optional[int] = None,
    warmup: Optional[int] = None,
    robust: Optional[bool] = None,
    default_profile: str = DEFAULT_PROFILE_NAME,
) -> BenchmarkProfile:
    """Resolve benchmark options against a named profile."""
    base = profile if isinstance(profile, BenchmarkProfile) else get_profile(profile or default_profile)
    return BenchmarkProfile(
        name=base.name,
        sizes=list(sizes or base.sizes),
        trials=base.trials if trials is None else trials,
        warmup=base.warmup if warmup is None else warmup,
        robust=base.robust if robust is None else robust,
        description=base.description,
    )

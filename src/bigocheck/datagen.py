# Author: gadwant
"""
Data generators for benchmarking common data structures.

These generators produce test inputs of size N for use with benchmark_function's
arg_factory parameter.
"""
from __future__ import annotations

import random  # nosec B311 - Not used for cryptographic purposes
import string
from typing import Any, Dict, List, Tuple


def n_(n: int) -> int:
    """
    Identity generator that returns N itself.
    
    Useful for functions that take an integer directly.
    
    Example:
        >>> from bigocheck.datagen import n_
        >>> n_(100)
        100
    """
    return n


def range_n(n: int) -> range:
    """
    Generate a range object of size N.
    
    Example:
        >>> from bigocheck.datagen import range_n
        >>> list(range_n(5))
        [0, 1, 2, 3, 4]
    """
    return range(n)


def integers(n: int, lo: int = 0, hi: int = 10000) -> List[int]:
    """
    Generate a list of N random integers in the range [lo, hi].
    
    Args:
        n: Number of integers to generate.
        lo: Minimum value (inclusive).
        hi: Maximum value (inclusive).
    
    Example:
        >>> from bigocheck.datagen import integers
        >>> len(integers(100, 0, 1000))
        100
    """
    return [random.randint(lo, hi) for _ in range(n)]


def floats(n: int, lo: float = 0.0, hi: float = 1.0) -> List[float]:
    """
    Generate a list of N random floats in the range [lo, hi].
    
    Args:
        n: Number of floats to generate.
        lo: Minimum value.
        hi: Maximum value.
    
    Example:
        >>> from bigocheck.datagen import floats
        >>> len(floats(100))
        100
    """
    return [random.uniform(lo, hi) for _ in range(n)]


def strings(n: int, length: int = 10) -> List[str]:
    """
    Generate a list of N random strings of specified length.
    
    Args:
        n: Number of strings to generate.
        length: Length of each string.
    
    Example:
        >>> from bigocheck.datagen import strings
        >>> all(len(s) == 10 for s in strings(5, 10))
        True
    """
    chars = string.ascii_letters + string.digits
    return [''.join(random.choices(chars, k=length)) for _ in range(n)]


def sorted_integers(n: int, lo: int = 0, hi: int = 10000) -> List[int]:
    """
    Generate a sorted list of N random integers.
    
    Args:
        n: Number of integers to generate.
        lo: Minimum value (inclusive).
        hi: Maximum value (inclusive).
    
    Example:
        >>> from bigocheck.datagen import sorted_integers
        >>> lst = sorted_integers(100)
        >>> lst == sorted(lst)
        True
    """
    return sorted(integers(n, lo, hi))


def reversed_integers(n: int, lo: int = 0, hi: int = 10000) -> List[int]:
    """
    Generate a reverse-sorted list of N random integers.
    
    Args:
        n: Number of integers to generate.
        lo: Minimum value (inclusive).
        hi: Maximum value (inclusive).
    
    Example:
        >>> from bigocheck.datagen import reversed_integers
        >>> lst = reversed_integers(100)
        >>> lst == sorted(lst, reverse=True)
        True
    """
    return sorted(integers(n, lo, hi), reverse=True)


def arg_factory_for(generator) -> callable:
    """
    Create an arg_factory wrapper for a generator function.
    
    This wraps a generator so it can be used with benchmark_function's arg_factory.
    
    Args:
        generator: A callable that takes n and returns the input data.
    
    Returns:
        An arg_factory callable returning ((generated_data,), {}).
    
    Example:
        >>> from bigocheck.datagen import integers, arg_factory_for
        >>> factory = arg_factory_for(integers)
        >>> args, kwargs = factory(10)
        >>> len(args[0])
        10
    """
    def factory(n: int) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (generator(n),), {}
    return factory

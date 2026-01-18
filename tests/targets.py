# Author: gadwant
"""
Target functions used by CLI tests.
"""
import time


def linear_sleep(n: int) -> None:
    # Sleep scales linearly with n to give stable measurements.
    time.sleep(0.002 * n)


def quadratic_sleep(n: int) -> None:
    # Sleep scales quadratically with n.
    time.sleep(0.0004 * n * n)


def constant_sleep(n: int) -> None:
    # Constant-time sleep independent of n.
    time.sleep(0.01)

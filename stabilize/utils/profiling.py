"""Simple timing utilities."""

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    """Context manager for timing a block of code."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[TIMER] {label}: {elapsed:.2f}s")

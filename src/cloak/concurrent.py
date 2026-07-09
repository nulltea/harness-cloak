"""Small concurrency helpers for remote LLM calls."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Iterable[T], workers: int = 8) -> list[R]:
    """Map fn over items concurrently, preserving order.

    Intended for remote LLM calls, where the proxy batches concurrent requests. Avoid
    using this around GPU probes; those call paths are not thread-safe.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

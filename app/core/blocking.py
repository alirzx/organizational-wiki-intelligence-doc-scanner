"""Bounded process-local executors for blocking model and storage work."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from functools import partial
from threading import Lock
from typing import ParamSpec, TypeVar


class BlockingPool(StrEnum):
    OCR = "ocr"
    FIGURE_TABLE = "figure_table"
    STAMP_SIGNATURE = "stamp_signature"
    IO = "io"


_WORKERS = {
    BlockingPool.OCR: 1,
    BlockingPool.FIGURE_TABLE: 1,
    BlockingPool.STAMP_SIGNATURE: 1,
    BlockingPool.IO: 4,
}
_executors: dict[BlockingPool, ThreadPoolExecutor] = {}
_executor_pid = os.getpid()
_executor_lock = Lock()

Params = ParamSpec("Params")
Result = TypeVar("Result")


def _reset_after_fork() -> None:
    global _executor_lock, _executor_pid, _executors
    _executors = {}
    _executor_pid = os.getpid()
    _executor_lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_executor(pool: BlockingPool) -> ThreadPoolExecutor:
    global _executor_pid, _executors
    pid = os.getpid()
    with _executor_lock:
        if _executor_pid != pid:
            _executors = {}
            _executor_pid = pid
        executor = _executors.get(pool)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=_WORKERS[pool],
                thread_name_prefix=f"wiki-hami-{pool.value}",
            )
            _executors[pool] = executor
        return executor


async def run_blocking(
    pool: BlockingPool,
    function: Callable[Params, Result],
    *args: Params.args,
    **kwargs: Params.kwargs,
) -> Result:
    loop = asyncio.get_running_loop()
    call = partial(function, *args, **kwargs)
    future = loop.run_in_executor(_get_executor(pool), call)
    while not future.done():
        await asyncio.wait({future}, timeout=0.05)
    return future.result()

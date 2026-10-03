import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from app.core.blocking import BlockingPool, _get_executor, run_blocking


class RejectingDefaultExecutor(ThreadPoolExecutor):
    def submit(self, *args, **kwargs):
        raise AssertionError("blocking work used asyncio's default executor")


def test_run_blocking_uses_an_explicit_executor_across_asyncio_run():
    async def run_once():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(RejectingDefaultExecutor(max_workers=1))
        return [
            await run_blocking(BlockingPool.IO, threading.current_thread)
            for _ in range(3)
        ]

    worker_threads = asyncio.run(run_once())

    assert all(thread.name.startswith("wiki-hami-io") for thread in worker_threads)


def test_executor_pools_are_bounded_and_reused():
    for pool in (
        BlockingPool.OCR,
        BlockingPool.FIGURE_TABLE,
        BlockingPool.STAMP_SIGNATURE,
    ):
        executor = _get_executor(pool)
        assert executor is _get_executor(pool)
        assert executor._max_workers == 1

    assert _get_executor(BlockingPool.IO)._max_workers == 4


def test_application_runtime_has_no_asyncio_to_thread_calls():
    from pathlib import Path

    offenders = [
        path
        for path in Path("app").rglob("*.py")
        if "asyncio.to_thread" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []

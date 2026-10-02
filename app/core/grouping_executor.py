"""Bounded process-local grouping executor, reusable across Celery event loops."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock

_registry_lock = Lock()
_executor = None
_slots = None
_workers = None


async def run_grouping(fn, *args, workers=2):
    global _executor, _slots, _workers
    with _registry_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=workers,thread_name_prefix='ocr-grouping')
            _slots = BoundedSemaphore(workers*2)
            _workers = workers
        if workers != _workers:
            raise RuntimeError('grouping worker configuration must be consistent within a process')
        executor,slots = _executor,_slots
    # Admission never blocks the event loop. Running + queued work is bounded.
    while not slots.acquire(blocking=False):
        await asyncio.sleep(.005)
    try:
        future = executor.submit(fn,*args)
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda completed:slots.release())
    return await asyncio.wrap_future(future)


def close_grouping_executor():
    global _executor,_slots,_workers
    with _registry_lock:
        executor = _executor
        _executor = _slots = _workers = None
    if executor is not None:
        executor.shutdown(wait=True,cancel_futures=True)

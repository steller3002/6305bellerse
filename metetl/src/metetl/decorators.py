import time
import functools
import asyncio
from logging import getLogger

logger = getLogger(__name__)

def measure_time(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        logger.info("Время %s: %.4f сек.", func.__name__, time.perf_counter() - start)
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info("Время %s: %.4f сек.", func.__name__, time.perf_counter() - start)
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
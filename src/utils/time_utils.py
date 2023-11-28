import time
from typing import TypeVar, Callable, Tuple, Any

from loguru import logger


def print_time_perf(func) -> Callable[..., Any]:
    """
    Decorator that calculates the time taken to execute a function.

    :param func: function to calculate execution time for
    :return:
    """

    # args and kwargs are passed to the function
    def inner(*args, **kwargs):
        begin = time.time()

        result = func(*args, **kwargs)

        end = time.time()
        logger.debug(f"Total time taken in '{func.__name__}': {end - begin}[s]")

        return result

    return inner


_T = TypeVar("_T")


def get_execution_time(func: Callable[..., _T]) -> Callable[..., Tuple[float, _T]]:
    """
    Decorator that returns the time that a function takes to execute,
    along with its result

    :param func: function to retrieve the execution time for
    :return:
    """

    def inner(*args, **kwargs):
        begin = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        execution_time = (end - begin)
        return execution_time, result

    return inner


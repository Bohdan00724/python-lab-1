import functools
import logging
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для вимірювання часу виконання функцій зі збереженням метаданих."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info("[TIMED] %s: %.3f ms", func.__qualname__, elapsed * 1000)

    return wrapper

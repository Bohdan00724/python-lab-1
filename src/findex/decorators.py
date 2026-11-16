import functools
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Декоратор для замера времени выполнения функций с сохранением метаданных."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            wrapper.last_elapsed = elapsed  # type: ignore[attr-defined]
            wrapper.call_count += 1  # type: ignore[attr-defined]
            print(f"[TIMED] {func.__qualname__}: {elapsed * 1000:.3f} ms")

    wrapper.last_elapsed = 0.0  # type: ignore[attr-defined]
    wrapper.call_count = 0  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
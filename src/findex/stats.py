import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.core import iter_documents, tokenize


def run_generator_test(path: Path) -> tuple[float, float, list[tuple[str, int]]]:
    """Підрахунок статистики корпусу через лінивий генератор."""
    tracemalloc.start()
    start_time = time.perf_counter()

    word_counts: Counter[str] = Counter()
    for document in iter_documents(path):
        for word in tokenize(document):
            word_counts[word] += 1

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start_time, peak_mem / (1024 * 1024), word_counts.most_common(5)


def run_list_test(path: Path) -> tuple[float, float]:
    """Жадібний підрахунок статистики корпусу через завантаження всіх документів у список."""
    tracemalloc.start()
    start_time = time.perf_counter()

    all_documents = list(iter_documents(path))
    all_words = [word for doc in all_documents for word in tokenize(doc)]
    _ = Counter(all_words)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start_time, peak_mem / (1024 * 1024)

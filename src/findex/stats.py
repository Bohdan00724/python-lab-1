import time
import tracemalloc
from collections import Counter
from src.findex.core import stream_documents, extract_tokens

def run_generator_test(path):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    word_counts = Counter()
    for document in stream_documents(path):
        for word in extract_tokens(document):
            word_counts[word] += 1
            
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start_time, peak_mem / (1024 * 1024), word_counts.most_common(5)

def run_list_test(path):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Жадібний метод: завантажуємо все в оперативну пам'ять
    all_documents = list(stream_documents(path))
    all_words = [word for doc in all_documents for word in extract_tokens(doc)]
    word_counts = Counter(all_words)
    
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return time.perf_counter() - start_time, peak_mem / (1024 * 1024)
from pathlib import Path
from src.findex.core import iter_document_files, tokenize
from src.findex.index import Index, Posting, open_index
from src.findex.ranking import BM25, TfIdf
from src.findex.search import clear_search_cache, search

CORPUS_DOCS: dict[str, str] = {
    "doc01_python_asyncio.txt": (
        "Python asyncio provides infrastructure for writing concurrent code using async and await syntax. "
        "At the heart of asyncio is the event loop, which schedules coroutines and handles non-blocking I/O."
    ),
    "doc02_event_loop_deep_dive.txt": (
        "Understanding the event loop in Python requires knowing how tasks, futures, and callbacks work. "
        "When an async function hits an await expression, control returns to the event loop so other tasks run."
    ),
    "doc03_python_java_interop.txt": (
        "Integrating Python with Java can be done through JPype or GraalVM. "
        "While Python uses an async event loop for concurrency, Java traditionally relies on JVM threads and executors."
    ),
    "doc04_python_basics.txt": (
        "Python is a readable high-level programming language with dynamic typing, lists, dictionaries, and generators. "
        "Developers value Python for clean syntax and rapid prototyping across automation and scripting tasks."
    ),
    "doc05_python_long_guide.txt": (
        "Comprehensive software engineering guide covering architecture, testing, packaging, deployment, observability, "
        "containers, continuous integration, version control, code review, profiling, caching, logging, security, "
        "configuration management, dependency injection, API design, documentation, modularity, refactoring, "
        "microservices, message queues, load balancing, fault tolerance, distributed tracing, and one brief note "
        "that python supports clear readable modules for backend services and automation pipelines."
    ),
    "doc06_python_spam.txt": (
        "python python python python python python python python python python "
        "python python python python python python python python python python "
        "keywords repeated without real tutorial value."
    ),
    "doc07_async_js_rust.txt": (
        "Modern JavaScript and Rust both implement async and await abstractions over an event loop or executor. "
        "In Node.js the single-threaded event loop processes timers, promises, and network callbacks efficiently."
    ),
    "doc08_java_jvm.txt": (
        "Java runs on the Java Virtual Machine and powers enterprise backend systems with Spring Boot. "
        "Garbage collection, JIT compilation, and virtual threads make Java strong for high-throughput workloads."
    ),
    "doc09_search_engines.txt": (
        "Information retrieval engines rely on an inverted index to map terms to postings lists. "
        "Ranking algorithms such as TF-IDF and BM25 score documents using term frequency and inverse document frequency."
    ),
    "doc10_linux_networking.txt": (
        "Linux network servers use non-blocking sockets with epoll to multiplex thousands of connections. "
        "An OS-level event loop reacts to socket readiness events without spawning a thread per client."
    ),
    "doc11_python_decorators.txt": (
        "Python decorators wrap functions using closures and functools.wraps to preserve metadata. "
        "Combined with dunder methods and context managers, Python provides an expressive object model."
    ),
    "doc12_databases_sql.txt": (
        "Relational databases store records on disk and accelerate lookups using B-tree indexes. "
        "SQL transactions guarantee ACID properties through write-ahead logging and concurrency control."
    ),
}

BENCHMARK_QUERIES: list[tuple[str, set[str]]] = [
    ('python AND (async OR await) NOT java "event loop"', {"doc01_python_asyncio.txt", "doc02_event_loop_deep_dive.txt"}),
    ("python OR async OR decorators", {"doc01_python_asyncio.txt", "doc02_event_loop_deep_dive.txt", "doc04_python_basics.txt", "doc11_python_decorators.txt"}),
    ("python AND (decorators OR functools)", {"doc11_python_decorators.txt"}),
    ('"event loop" AND (async OR epoll)', {"doc01_python_asyncio.txt", "doc02_event_loop_deep_dive.txt", "doc07_async_js_rust.txt", "doc10_linux_networking.txt"}),
    ('bm25 OR tfidf OR "inverted index"', {"doc09_search_engines.txt"}),
    ("java AND (jvm OR threads) NOT python", {"doc08_java_jvm.txt"}),
    ("python OR syntax OR modules", {"doc01_python_asyncio.txt", "doc04_python_basics.txt", "doc05_python_long_guide.txt", "doc11_python_decorators.txt"}),
    ("sql AND transactions", {"doc12_databases_sql.txt"}),
    ("async AND await NOT python", {"doc07_async_js_rust.txt"}),
    ("linux AND (sockets OR epoll)", {"doc10_linux_networking.txt"}),
]


def setup_corpus(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_file in data_dir.glob("*.txt"):
        old_file.unlink()
    for filename, content in CORPUS_DOCS.items():
        (data_dir / filename).write_text(content, encoding="utf-8")


def precision_at_k(retrieved_paths: list[str], relevant_names: set[str], k: int = 5) -> float:
    top_names = [Path(p).name for p in retrieved_paths[:k]]
    hits = sum(1 for name in top_names if name in relevant_names)
    return hits / k


def main() -> None:
    data_dir = Path("data")
    index_path = Path("index.json")

    setup_corpus(data_dir)
    idx = Index(store_positions=True)

    def pipeline():
        for doc_id, (file_path, text) in enumerate(iter_document_files(data_dir)):
            yield doc_id, list(tokenize(text)), str(file_path)

    idx.build_from_pipeline(pipeline())
    idx.save_json(index_path)

    with open_index(index_path) as index:
        print("\n=== 1. ДЕМОНСТРАЦИЯ СЛОЖНОГО ЗАПРОСА (BM25) + CACHE ===")
        demo_q = 'python AND (async OR await) NOT java "event loop"'
        res1 = search(index, demo_q, scorer=BM25(), top_k=5)
        for rank, r in enumerate(res1, 1):
            print(f"  {rank}. {Path(r.path).name} | score={r.score:.4f} | snippet: {r.snippet}")
        print("--- Повторный вызов (ожидаем CACHE HIT) ---")
        search(index, demo_q, scorer=BM25(), top_k=5)

        print("\n=== 2. ТРИ SANITY-CHECK РАНЖИРОВАНИЯ ===")
        bm25 = BM25()
        tfidf = TfIdf()

        # Check 1: Редкий термин выше частого
        # В doc11_python_decorators.txt есть редкий 'functools' (df=1) и частый 'python' (df=6)
        doc11_id = next(d for d in index.all_doc_ids() if Path(index.doc_path(d)).name == "doc11_python_decorators.txt")
        p_rare = next(p for p in index["functools"] if p.doc_id == doc11_id)
        p_freq = next(p for p in index["python"] if p.doc_id == doc11_id)
        print(
            f"Check 1 (редкий vs частый в одном документе): "
            f"BM25('functools', df={index.df('functools')})={bm25.score('functools', p_rare, index):.4f} > "
            f"BM25('python', df={index.df('python')})={bm25.score('python', p_freq, index):.4f}"
        )

        # Check 2: 20-е повторение почти ничего не добавляет (насыщение BM25)
        s1 = bm25.score("python", Posting(doc_id=doc11_id, frequency=1), index)
        s2 = bm25.score("python", Posting(doc_id=doc11_id, frequency=2), index)
        s19 = bm25.score("python", Posting(doc_id=doc11_id, frequency=19), index)
        s20 = bm25.score("python", Posting(doc_id=doc11_id, frequency=20), index)
        gain_1st = s1
        gain_2nd = s2 - s1
        gain_20th = s20 - s19
        print(
            f"Check 2 (насыщение BM25): 1-е вхождение=+{gain_1st:.4f}, "
            f"2-е=+{gain_2nd:.4f}, 20-е=+{gain_20th:.4f} (всего {gain_20th / gain_1st * 100:.2f}% от 1-го)"
        )

        # Check 3: Короткий документ с 1 вхождением выше очень длинного с 1 вхождением
        doc05_id = next(d for d in index.all_doc_ids() if Path(index.doc_path(d)).name == "doc05_python_long_guide.txt")
        p_short = Posting(doc_id=doc11_id, frequency=1)
        p_long = next(p for p in index["python"] if p.doc_id == doc05_id)
        print(
            f"Check 3 (короткий len={index.doc_length(doc11_id)} vs длинный len={index.doc_length(doc05_id)}): "
            f"BM25 short={bm25.score('python', p_short, index):.4f} > long={bm25.score('python', p_long, index):.4f} | "
            f"TF-IDF short={tfidf.score('python', p_short, index):.4f} > long={tfidf.score('python', p_long, index):.4f}"
        )

        print("\n=== 3. ТАБЛИЦА PRECISION@5 (TF-IDF vs BM25) ===")
        tfidf_scores: list[float] = []
        bm25_scores: list[float] = []
        print("| # | Запит | Релевантні документи | P@5 (TF-IDF) | P@5 (BM25) |")
        print("|---|-------|----------------------|--------------|------------|")
        for i, (q, rel_docs) in enumerate(BENCHMARK_QUERIES, 1):
            clear_search_cache()
            res_tfidf = search(index, q, scorer=tfidf, top_k=5, with_snippets=False)
            clear_search_cache()
            res_bm25 = search(index, q, scorer=bm25, top_k=5, with_snippets=False)

            p_tfidf = precision_at_k([r.path for r in res_tfidf], rel_docs, k=5)
            p_bm25 = precision_at_k([r.path for r in res_bm25], rel_docs, k=5)
            tfidf_scores.append(p_tfidf)
            bm25_scores.append(p_bm25)
            rel_str = ", ".join(sorted(rel_docs))
            print(f"| {i} | `{q}` | {rel_str} | {p_tfidf:.2f} | {p_bm25:.2f} |")

        avg_tfidf = sum(tfidf_scores) / len(tfidf_scores)
        avg_bm25 = sum(bm25_scores) / len(bm25_scores)
        print(f"| **Avg** | **Середній Precision@5** | — | **{avg_tfidf:.3f}** | **{avg_bm25:.3f}** |")


if __name__ == "__main__":
    main()
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from findex.index import Index, build_index
from findex.ranking import BM25, TFIDF, NumPyCorpusArrays, rank_documents, top_k_indices
from findex.search import search


class _LegacyLab3BM25:
    """Еталонний цикл по об'єктах Python із Лаби 3 для перевірки порядку та швидкості."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_documents(
        self,
        index: Index,
        query_terms: list[str],
        candidate_doc_ids: Iterable[int],
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        avgdl = index.avg_doc_length if index.avg_doc_length > 0 else 1.0
        for doc_id in candidate_doc_ids:
            total = 0.0
            dl = float(index.doc_length(doc_id))
            for term in dict.fromkeys(query_terms):
                plist = index[term] if term in index else []
                df_val = len(plist)
                if df_val == 0:
                    continue
                tf = 0
                for p in plist:
                    if p.doc_id == doc_id:
                        tf = p.frequency
                        break
                if tf > 0:
                    import math

                    idf = math.log((index.num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
                    total += idf * (tf * (self.k1 + 1.0)) / (
                        tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    )
            if total > 0.0:
                scores[doc_id] = total
        return scores


def _build_scaled_index(num_docs: int = 300) -> Index:
    idx = Index(store_positions=True)
    vocab = [
        "python",
        "asyncio",
        "event",
        "loop",
        "numpy",
        "vector",
        "bm25",
        "tfidf",
        "index",
        "query",
        "rareterm",
        "shard",
    ]
    for doc_id in range(num_docs):
        tokens = [vocab[(doc_id + k * 3) % len(vocab)] for k in range(25)]
        if doc_id % 50 == 0:
            tokens.append("rareterm")
        idx.add_document(doc_id, tokens, path=f"doc_{doc_id:04d}.txt")
    return idx


def test_numpy_ranking_order_matches_lab3(built_index: tuple[Index, Path]) -> None:
    idx, _ = built_index
    for query in ["python", "asyncio", "python AND asyncio"]:
        numpy_hits = search(idx, query, scorer=BM25(), top_k=5, with_snippets=False)
        tfidf_hits = search(idx, query, scorer=TFIDF(), top_k=5, with_snippets=False)
        assert [h.doc_id for h in numpy_hits]
        assert [h.doc_id for h in tfidf_hits]


def test_numpy_arrays_and_argpartition_top_k() -> None:
    idx = _build_scaled_index(num_docs=120)
    arrays = NumPyCorpusArrays.from_index(idx)
    assert arrays.doc_ids_sorted.dtype == np.int32
    assert arrays.doc_lengths.dtype == np.int32

    scores = np.array([1.2, 9.5, 4.1, 9.5, 0.3], dtype=np.float64)
    top2 = top_k_indices(scores, 2)
    assert top2.tolist() == [1, 3]

    ranked = rank_documents(idx, ["python", "numpy", "rareterm"], BM25(), top_k=5)
    assert len(ranked) <= 5
    assert all(ranked[i].score >= ranked[i + 1].score for i in range(len(ranked) - 1))


def test_benchmark_table_and_speedup_comparison() -> None:
    idx = _build_scaled_index(num_docs=400)
    arrays = NumPyCorpusArrays.from_index(idx)
    legacy = _LegacyLab3BM25()
    bm25_np = BM25()
    all_ids = sorted(idx.all_doc_ids())

    print("\n=== LAB 08 SCORER SPEEDUP TABLE (400 DOCS) ===")
    print("| Query type | Lab 3 Python loop (ms) | NumPy scorer (ms) | Speedup |")
    print("|---|---:|---:|---:|")

    for label, terms in [
        ("Frequent term (python)", ["python"]),
        ("Rare term (rareterm)", ["rareterm"]),
        ("3-term query (python, numpy, rareterm)", ["python", "numpy", "rareterm"]),
    ]:
        t0 = time.perf_counter()
        for _ in range(25):
            legacy.score_documents(idx, terms, all_ids)
        legacy_ms = ((time.perf_counter() - t0) / 25.0) * 1000.0

        t1 = time.perf_counter()
        for _ in range(25):
            bm25_np.score_with_numpy(arrays, terms, all_ids)
        np_ms = ((time.perf_counter() - t1) / 25.0) * 1000.0

        speedup = legacy_ms / max(np_ms, 1e-9)
        print(f"| {label} | {legacy_ms:.3f} | {np_ms:.3f} | {speedup:.2f}x |")


def test_pytest_benchmark_search(benchmark: object, built_index: tuple[Index, Path]) -> None:
    idx, _ = built_index
    bench_fn = getattr(benchmark, "pedantic", None)
    if callable(bench_fn):
        bench_fn(
            lambda: search(idx, "python AND asyncio", scorer=BM25(), top_k=5, with_snippets=False),
            iterations=10,
            rounds=3,
        )
    else:
        assert callable(benchmark)
        benchmark(lambda: search(idx, "python AND asyncio", scorer=BM25(), top_k=5, with_snippets=False))


def test_pytest_benchmark_build_index(benchmark: object, mini_corpus_dir: Path) -> None:
    files = sorted(mini_corpus_dir.rglob("*.txt"))
    bench_fn = getattr(benchmark, "pedantic", None)
    if callable(bench_fn):
        bench_fn(
            lambda: build_index(files, workers=1, executor="serial"),
            iterations=5,
            rounds=2,
        )
    else:
        assert callable(benchmark)
        benchmark(lambda: build_index(files, workers=1, executor="serial"))

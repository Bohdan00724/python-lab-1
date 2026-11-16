import argparse
from dataclasses import replace
from functools import lru_cache

from src.findex.decorators import timed
from src.findex.index import Index, open_index
from src.findex.parser import parse_query
from src.findex.ranking import BM25, Scorer, SearchResult, TfIdf, rank_documents
from src.findex.snippets import build_snippet

_INDEX_REGISTRY: dict[tuple[int, int, int], Index] = {}


@lru_cache(maxsize=128)
def _cached_query_eval(
    index_key: tuple[int, int, int], query: str
) -> tuple[frozenset[int], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Кэширует путь: строка запроса -> множество подходящих doc_id."""
    index = _INDEX_REGISTRY[index_key]
    ast = parse_query(query)
    matched_ids = frozenset(ast.evaluate(index))
    return matched_ids, tuple(ast.positive_terms()), tuple(ast.phrases())


def clear_search_cache() -> None:
    _cached_query_eval.cache_clear()


def get_search_cache_info():
    return _cached_query_eval.cache_info()


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    top_k: int = 5,
    with_snippets: bool = True,
) -> list[SearchResult]:
    """Выполняет разбор запроса, кэшируемую фильтрацию, ранжирование и генерацию сниппетов."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    index_key = (id(index), index.num_docs, len(index))
    _INDEX_REGISTRY[index_key] = index

    hits_before = _cached_query_eval.cache_info().hits
    matched_ids, pos_terms, phrases = _cached_query_eval(index_key, query)
    hits_after = _cached_query_eval.cache_info().hits

    cache_status = "CACHE HIT" if hits_after > hits_before else "CACHE MISS"
    print(f"[CACHE] {cache_status} для запиту {query!r} ({_cached_query_eval.cache_info()})")

    if not matched_ids:
        return []

    score_terms = list(dict.fromkeys(pos_terms))
    ranked = rank_documents(
        index=index,
        query_terms=score_terms,
        scorer=active_scorer,
        allowed_docs=set(matched_ids),
        top_k=top_k,
    )

    if not with_snippets:
        return ranked

    enriched: list[SearchResult] = []
    for item in ranked:
        snippet = build_snippet(item.path, score_terms, phrases=list(phrases), window=80)
        enriched.append(replace(item, snippet=snippet))
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Ранжований пошук у findex")
    parser.add_argument("index_file", help="Шлях до файлу індексу (.json)")
    parser.add_argument("query", help="Пошуковий запит (підтримує AND, OR, NOT, дужки, \"фрази\")")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Алгоритм ранжування")
    parser.add_argument("--top-k", type=int, default=5, help="Кількість результатів у топі")
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Виконати запит двічі для демонстрації @lru_cache (CACHE HIT)",
    )
    args = parser.parse_args()

    scorer: Scorer = BM25() if args.scorer == "bm25" else TfIdf()

    with open_index(args.index_file) as idx:
        print(f"Відкрито індекс: {idx!r}")
        results = search(idx, args.query, scorer=scorer, top_k=args.top_k)

        if args.repeat:
            print("\n--- Повторний запуск того ж запиту ---")
            results = search(idx, args.query, scorer=scorer, top_k=args.top_k)

        if not results:
            print("\nНічого не знайдено.")
            return

        print(f"\nТоп-{len(results)} результатів ({args.scorer.upper()}):")
        for rank, res in enumerate(results, start=1):
            print(f"{rank}. Doc #{res.doc_id} ({res.path}) | score={res.score:.4f}")
            if res.snippet:
                print(f"   Сніпет: {res.snippet}")


if __name__ == "__main__":
    main()
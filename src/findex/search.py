import logging
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Literal, NamedTuple

from findex.decorators import timed
from findex.index import Index, merge_postings
from findex.parser import parse_query
from findex.ranking import BM25, Scorer, SearchResult, rank_documents
from findex.snippets import build_snippet

EngineName = Literal["merge", "set"]


class CacheStats(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


logger = logging.getLogger(__name__)
_INDEX_REGISTRY: dict[tuple[int, int, int], Index] = {}


def intersect_terms(index: Index, terms: Sequence[str], engine: EngineName = "merge") -> set[int]:
    """Перетин документів за списком термінів через 'merge' або 'set'."""
    if not terms:
        return set()
    postings_lists = [index[t] if t in index else [] for t in terms]
    if any(not plist for plist in postings_lists):
        return set()

    if engine == "merge":
        return set(merge_postings(postings_lists))
    doc_sets = [{p.doc_id for p in plist} for plist in postings_lists]
    return doc_sets[0].intersection(*doc_sets[1:])


@lru_cache(maxsize=128)
def _cached_query_eval(
    index_key: tuple[int, int, int], query: str
) -> tuple[frozenset[int], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Кешує шлях: рядок запиту -> множина відповідних doc_id."""
    index = _INDEX_REGISTRY[index_key]
    ast = parse_query(query)
    matched_ids = frozenset(ast.evaluate(index))
    return matched_ids, tuple(ast.positive_terms()), tuple(ast.phrases())


def clear_search_cache() -> None:
    _cached_query_eval.cache_clear()


def get_search_cache_info() -> CacheStats:
    info = _cached_query_eval.cache_info()
    return CacheStats(info.hits, info.misses, info.maxsize, info.currsize)


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    engine: EngineName = "merge",
    top_k: int = 5,
    with_snippets: bool = True,
) -> list[SearchResult]:
    """Виконує розбір запиту, кешовану фільтрацію, ранжування та генерацію сніпетів."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    index_key = (id(index), index.num_docs, len(index))
    _INDEX_REGISTRY[index_key] = index

    hits_before = _cached_query_eval.cache_info().hits
    matched_ids, pos_terms, phrases = _cached_query_eval(index_key, query)
    hits_after = _cached_query_eval.cache_info().hits

    cache_status = "CACHE HIT" if hits_after > hits_before else "CACHE MISS"
    logger.debug(
        "[CACHE] %s для запиту %r (engine=%s, %s)",
        cache_status,
        query,
        engine,
        _cached_query_eval.cache_info(),
    )

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

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from findex.index import Index, Posting

ScorerName = Literal["bm25", "tfidf"]


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Результат пошуку, впорядкований за спаданням score (через heapq.nlargest)."""

    score: float
    doc_id: int
    path: str = field(default="", compare=False)
    snippet: str = field(default="", compare=False)


@runtime_checkable
class Scorer(Protocol):
    """Протокол для взаємозамінних алгоритмів ранжування (duck typing)."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """Класичне ранжування TF-IDF з нормалізацією за довжиною документа."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        n_docs = index.num_docs
        df = index.df(term)
        if n_docs == 0 or df == 0 or posting.frequency == 0:
            return 0.0

        doc_len = max(index.doc_length(posting.doc_id), 1)
        tf = posting.frequency / doc_len
        idf = math.log((n_docs + 1) / (df + 1)) + 1.0
        return tf * idf


@dataclass(frozen=True, slots=True)
class BM25:
    """Ранжування Okapi BM25 з насиченням частоти та нормалізацією довжини."""

    k1: float = 1.5
    b: float = 0.75

    def score(self, term: str, posting: Posting, index: Index) -> float:
        n_docs = index.num_docs
        df = index.df(term)
        if n_docs == 0 or df == 0 or posting.frequency == 0:
            return 0.0

        doc_len = index.doc_length(posting.doc_id)
        avg_dl = index.avg_doc_length or 1.0

        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf = posting.frequency
        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
        return idf * ((tf * (self.k1 + 1.0)) / denom)


def get_scorer(name: ScorerName) -> Scorer:
    """Фабрика скорерів за рядковим Literal-іменем."""
    if name == "bm25":
        return BM25()
    return TfIdf()


def rank_documents(
    index: Index,
    query_terms: Iterable[str],
    scorer: Scorer,
    allowed_docs: set[int] | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """Накопичує бали за термінами запиту та повертає топ-k через heapq.nlargest."""
    scores: defaultdict[int, float] = defaultdict(float)

    for term in query_terms:
        if term not in index:
            continue
        for posting in index[term]:
            if allowed_docs is not None and posting.doc_id not in allowed_docs:
                continue
            scores[posting.doc_id] += scorer.score(term, posting, index)

    candidates = [
        SearchResult(score=total_score, doc_id=doc_id, path=index.doc_path(doc_id))
        for doc_id, total_score in scores.items()
    ]
    return heapq.nlargest(top_k, candidates)

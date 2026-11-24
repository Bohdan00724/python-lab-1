import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.index import Index, Posting

ScorerName = Literal["tfidf", "bm25"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: int
    score: float
    path: str = ""
    snippet: str = ""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.score < other.score

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.score > other.score


@dataclass(slots=True)
class NumPyCorpusArrays:
    """Компактне представлення корпусу та постінгів у масивах NumPy int32."""

    doc_ids_sorted: NDArray[np.int32]
    doc_lengths: NDArray[np.int32]
    doc_id_to_pos: dict[int, int]
    term_doc_ids: dict[str, NDArray[np.int32]]
    term_freqs: dict[str, NDArray[np.int32]]
    avg_doc_length: float
    num_docs: int

    @classmethod
    def from_index(cls, index: Index) -> "NumPyCorpusArrays":
        all_ids = sorted(index.all_doc_ids())
        doc_ids_arr = np.asarray(all_ids, dtype=np.int32)
        doc_lengths_arr = np.asarray(
            [index.doc_length(doc_id) for doc_id in all_ids], dtype=np.int32
        )
        doc_id_to_pos = {int(doc_id): pos for pos, doc_id in enumerate(all_ids)}

        term_doc_ids: dict[str, NDArray[np.int32]] = {}
        term_freqs: dict[str, NDArray[np.int32]] = {}
        for term in index:
            plist: list[Posting] = index[term]
            if plist:
                term_doc_ids[term] = np.asarray([p.doc_id for p in plist], dtype=np.int32)
                term_freqs[term] = np.asarray([p.frequency for p in plist], dtype=np.int32)

        return cls(
            doc_ids_sorted=doc_ids_arr,
            doc_lengths=doc_lengths_arr,
            doc_id_to_pos=doc_id_to_pos,
            term_doc_ids=term_doc_ids,
            term_freqs=term_freqs,
            avg_doc_length=float(index.avg_doc_length),
            num_docs=int(index.num_docs),
        )


def top_k_indices(scores: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    """Повертає індекси top-k елементів (за спаданням score) через np.argpartition."""
    n = int(scores.shape[0])
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.intp)
    effective_k = min(k, n)
    if effective_k < n:
        part_idx = np.argpartition(-scores, effective_k - 1)[:effective_k]
    else:
        part_idx = np.arange(n, dtype=np.intp)
    order = np.argsort(-scores[part_idx], kind="mergesort")
    return part_idx[order]


def _term_frequency_in_doc(index: Index, term: str, doc_id: int) -> int:
    if term not in index:
        return 0
    for posting in index[term]:
        if posting.doc_id == doc_id:
            return posting.frequency
    return 0


def _resolve_score_args(
    first: Index | int,
    second: int | Sequence[str],
    third: Sequence[str] | Index | None,
) -> tuple[Index, int, Sequence[str]]:
    """Підтримує виклики score(index, doc_id, terms) та score(doc_id, terms, index)."""
    if isinstance(first, Index) and isinstance(second, int) and third is not None:
        return first, second, list(third)  # type: ignore[arg-type]
    if isinstance(first, int) and isinstance(third, Index):
        terms_seq = list(second) if isinstance(second, (list, tuple, set)) else [str(second)]
        return third, first, terms_seq
    raise TypeError("Unsupported arguments for score()")


class Scorer(ABC):
    @abstractmethod
    def score(
        self,
        first: Index | int,
        second: int | Sequence[str],
        third: Sequence[str] | Index | None = None,
    ) -> float:
        """Обчислює оцінку релевантності документа за термами запиту."""

    def score_documents(
        self,
        index: Index,
        query_terms: Sequence[str],
        candidate_doc_ids: Iterable[int],
    ) -> dict[int, float]:
        """Повертає словник {doc_id: score} для кандидатів."""
        candidates = list(candidate_doc_ids)
        if not candidates or not query_terms or index.num_docs == 0:
            return {}
        arrays = NumPyCorpusArrays.from_index(index)
        doc_arr, score_arr = self.score_with_numpy(arrays, query_terms, candidates)
        return {
            int(d_id): float(sc)
            for d_id, sc in zip(doc_arr.tolist(), score_arr.tolist(), strict=False)
            if float(sc) > 0.0
        }

    def score_with_numpy(
        self,
        arrays: NumPyCorpusArrays,
        query_terms: Sequence[str],
        candidate_doc_ids: Sequence[int],
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Векторизований підрахунок балів над масивами NumPy для кандидатів."""
        raise NotImplementedError


class TfIdf(Scorer):
    """TF-IDF скорер з векторизованим обчисленням над масивами NumPy."""

    def score(
        self,
        first: Index | int,
        second: int | Sequence[str],
        third: Sequence[str] | Index | None = None,
    ) -> float:
        index, doc_id, query_terms = _resolve_score_args(first, second, third)
        if index.num_docs == 0 or not query_terms:
            return 0.0
        doc_len = max(index.doc_length(doc_id), 1)
        total = 0.0
        for term in dict.fromkeys(query_terms):
            df_val = index.df(term)
            if df_val == 0:
                continue
            tf_raw = _term_frequency_in_doc(index, term, doc_id)
            if tf_raw > 0:
                idf = math.log((index.num_docs + 1.0) / (df_val + 0.5)) + 1.0
                total += (float(tf_raw) / float(doc_len)) * idf
        return total

    def score_with_numpy(
        self,
        arrays: NumPyCorpusArrays,
        query_terms: Sequence[str],
        candidate_doc_ids: Sequence[int],
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        cand_ids = np.asarray(list(candidate_doc_ids), dtype=np.int32)
        if cand_ids.size == 0 or arrays.num_docs == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        scores = np.zeros(cand_ids.shape[0], dtype=np.float64)
        cand_pos_map = {int(doc_id): idx for idx, doc_id in enumerate(cand_ids.tolist())}

        for term in dict.fromkeys(query_terms):
            t_docs = arrays.term_doc_ids.get(term)
            t_freqs = arrays.term_freqs.get(term)
            if t_docs is None or t_freqs is None or t_docs.size == 0:
                continue
            df_val = int(t_docs.size)
            idf = math.log((arrays.num_docs + 1.0) / (df_val + 0.5)) + 1.0
            for d_id, freq in zip(t_docs.tolist(), t_freqs.tolist(), strict=False):
                pos_in_cand = cand_pos_map.get(int(d_id))
                if pos_in_cand is not None:
                    doc_pos = arrays.doc_id_to_pos.get(int(d_id))
                    doc_len = (
                        max(int(arrays.doc_lengths[doc_pos]), 1)
                        if doc_pos is not None
                        else 1
                    )
                    tf = float(freq) / float(doc_len)
                    scores[pos_in_cand] += tf * idf

        return cand_ids, scores


TFIDF = TfIdf


class BM25(Scorer):
    """Okapi BM25 скорер з векторизованим обчисленням над масивами NumPy."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        first: Index | int,
        second: int | Sequence[str],
        third: Sequence[str] | Index | None = None,
    ) -> float:
        index, doc_id, query_terms = _resolve_score_args(first, second, third)
        if index.num_docs == 0 or not query_terms:
            return 0.0
        avgdl = index.avg_doc_length if index.avg_doc_length > 0.0 else 1.0
        dl = float(index.doc_length(doc_id))
        denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
        total = 0.0
        for term in dict.fromkeys(query_terms):
            df_val = index.df(term)
            if df_val == 0:
                continue
            tf = float(_term_frequency_in_doc(index, term, doc_id))
            if tf > 0.0:
                idf = math.log((index.num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
                total += idf * (tf * (self.k1 + 1.0)) / (tf + denom_norm)
        return total

    def score_with_numpy(
        self,
        arrays: NumPyCorpusArrays,
        query_terms: Sequence[str],
        candidate_doc_ids: Sequence[int],
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        cand_ids = np.asarray(list(candidate_doc_ids), dtype=np.int32)
        if cand_ids.size == 0 or arrays.num_docs == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)

        avgdl = arrays.avg_doc_length if arrays.avg_doc_length > 0.0 else 1.0
        doc_positions = np.asarray(
            [arrays.doc_id_to_pos.get(int(d_id), 0) for d_id in cand_ids.tolist()],
            dtype=np.intp,
        )
        dl = arrays.doc_lengths[doc_positions].astype(np.float64)
        denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / avgdl))

        scores = np.zeros(cand_ids.shape[0], dtype=np.float64)
        cand_pos_map = {int(doc_id): idx for idx, doc_id in enumerate(cand_ids.tolist())}

        for term in dict.fromkeys(query_terms):
            t_docs = arrays.term_doc_ids.get(term)
            t_freqs = arrays.term_freqs.get(term)
            if t_docs is None or t_freqs is None or t_docs.size == 0:
                continue
            df_val = int(t_docs.size)
            idf = math.log(
                (arrays.num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0
            )
            tf_vec = np.zeros(cand_ids.shape[0], dtype=np.float64)
            for d_id, freq in zip(t_docs.tolist(), t_freqs.tolist(), strict=False):
                pos_in_cand = cand_pos_map.get(int(d_id))
                if pos_in_cand is not None:
                    tf_vec[pos_in_cand] = float(freq)

            term_score = idf * (tf_vec * (self.k1 + 1.0)) / (tf_vec + denom_norm)
            scores += term_score

        return cand_ids, scores


def get_scorer(name: ScorerName) -> Scorer:
    if name == "tfidf":
        return TfIdf()
    if name == "bm25":
        return BM25()
    raise ValueError(f"Unknown scorer: {name!r}")


def rank_documents(
    index: Index,
    query_terms: Sequence[str],
    scorer: Scorer,
    allowed_docs: set[int] | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """Оцінює документи через NumPy-масиви та відбирає top-k через np.argpartition."""
    if top_k <= 0 or index.num_docs == 0:
        return []

    candidates_list = (
        sorted(allowed_docs) if allowed_docs is not None else sorted(index.all_doc_ids())
    )
    if not candidates_list:
        return []

    arrays = NumPyCorpusArrays.from_index(index)
    doc_arr, score_arr = scorer.score_with_numpy(arrays, query_terms, candidates_list)

    positive_mask = score_arr > 0.0
    pos_indices = np.flatnonzero(positive_mask)
    if pos_indices.size == 0:
        return []

    filtered_docs = doc_arr[pos_indices]
    filtered_scores = score_arr[pos_indices]

    selected_local = top_k_indices(filtered_scores, top_k)

    pairs: list[tuple[int, float]] = [
        (int(filtered_docs[i]), float(filtered_scores[i])) for i in selected_local.tolist()
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))

    return [
        SearchResult(doc_id=doc_id, score=sc, path=index.doc_path(doc_id))
        for doc_id, sc in pairs[:top_k]
    ]

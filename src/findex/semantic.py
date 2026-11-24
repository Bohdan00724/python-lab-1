import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.core import iter_document_files, tokenize
from findex.index import Index
from findex.ranking import BM25, SearchResult, rank_documents

SearchMode = Literal["lexical", "semantic", "hybrid"]

EMBEDDING_DIM = 256

_SEMANTIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "async": ("asynchronous", "coroutine", "event", "loop", "await", "concurrent"),
    "asynchronous": ("async", "coroutine", "event", "loop", "await"),
    "coroutine": ("async", "await", "generator", "event", "loop"),
    "event": ("loop", "async", "callback", "scheduler"),
    "loop": ("event", "async", "iteration", "cycle"),
    "search": ("query", "find", "retrieval", "ranking", "index"),
    "query": ("search", "filter", "expression", "term"),
    "ranking": ("score", "bm25", "tfidf", "relevance", "sort"),
    "relevance": ("ranking", "score", "match", "similarity"),
    "index": ("inverted", "postings", "corpus", "document", "lookup"),
    "document": ("file", "text", "page", "article", "record"),
    "speed": ("fast", "performance", "latency", "throughput", "optimization"),
    "fast": ("speed", "performance", "quick", "numpy", "vectorized"),
    "performance": ("speed", "latency", "throughput", "benchmark", "numpy"),
    "vector": ("embedding", "numpy", "array", "matrix", "cosine"),
    "embedding": ("vector", "semantic", "meaning", "representation"),
    "parallel": ("concurrent", "threads", "processes", "workers", "gil"),
    "thread": ("parallel", "concurrent", "lock", "gil", "worker"),
    "process": ("parallel", "worker", "spawn", "multiprocessing"),
    "network": ("http", "socket", "crawler", "request", "client"),
    "crawler": ("spider", "http", "url", "fetch", "html"),
    "error": ("exception", "timeout", "retry", "failure", "status"),
    "config": ("settings", "env", "environment", "options"),
    "web": ("http", "api", "fastapi", "server", "endpoint"),
}


@dataclass(slots=True)
class ChunkMeta:
    chunk_id: int
    doc_id: int
    path: str
    text: str


@dataclass(slots=True)
class SemanticStore:
    """Хранилище нормализованных матриц эмбеддингов (.npy) и метаданных фрагментов."""

    embeddings: NDArray[np.float32]
    doc_ids: NDArray[np.int32]
    chunks: list[ChunkMeta]

    def save(self, npy_path: Path) -> None:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), self.embeddings)
        meta_path = npy_path.with_suffix(".meta.json")
        payload = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": c.path, "text": c.text}
            for c in self.chunks
        ]
        meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, npy_path: Path) -> "SemanticStore":
        raw_arr = np.load(str(npy_path))
        matrix = np.asarray(raw_arr, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normalized = (matrix / norms).astype(np.float32)

        meta_path = npy_path.with_suffix(".meta.json")
        chunks: list[ChunkMeta] = []
        if meta_path.is_file():
            items = json.loads(meta_path.read_text(encoding="utf-8"))
            for item in items:
                chunks.append(
                    ChunkMeta(
                        chunk_id=int(item["chunk_id"]),
                        doc_id=int(item["doc_id"]),
                        path=str(item["path"]),
                        text=str(item["text"]),
                    )
                )
        doc_ids = np.asarray([c.doc_id for c in chunks], dtype=np.int32)
        return cls(embeddings=normalized, doc_ids=doc_ids, chunks=chunks)


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Делит текст документа на перекрывающиеся фрагменты для семантического поиска."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]
    step = max(1, chunk_size - overlap)
    result: list[str] = []
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + chunk_size].strip()
        if piece:
            result.append(piece)
        if start + chunk_size >= len(cleaned):
            break
    return result


def _token_bucket(token: str, dim: int = EMBEDDING_DIM) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, "little")
    return val % dim


def _token_sign(token: str) -> float:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return 1.0 if (digest[0] & 1) == 0 else -1.0


def embed_texts(texts: Sequence[str], dim: int = EMBEDDING_DIM) -> NDArray[np.float32]:
    """Вычисляет и L2-нормализует матрицу эмбеддингов (N, dim) для списка текстов."""
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, raw_text in enumerate(texts):
        vec = np.zeros(dim, dtype=np.float32)
        base_tokens = list(tokenize(raw_text))
        for tok in base_tokens:
            vec[_token_bucket(tok, dim)] += _token_sign(tok)
            for syn in _SEMANTIC_SYNONYMS.get(tok, ()):
                vec[_token_bucket(syn, dim)] += 0.55 * _token_sign(syn)

        # Биграммы для учёта локального контекста фраз
        for i in range(len(base_tokens) - 1):
            bg = f"{base_tokens[i]}_{base_tokens[i + 1]}"
            vec[_token_bucket(bg, dim)] += 0.4 * _token_sign(bg)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        matrix[row_idx] = vec

    return matrix


def build_semantic_store(
    corpus_path: Path,
    index: Index | None = None,
    chunk_size: int = 220,
    overlap: int = 40,
) -> SemanticStore:
    """Формирует матрицу эмбеддингов по фрагментам документов корпуса."""
    path_to_doc_id: dict[str, int] = {}
    if index is not None:
        for d_id in index.all_doc_ids():
            p = index.doc_path(d_id)
            if p:
                path_to_doc_id[str(Path(p))] = d_id
                path_to_doc_id[p] = d_id

    chunks: list[ChunkMeta] = []
    texts_to_embed: list[str] = []

    for fallback_id, (doc_path, text) in enumerate(iter_document_files(corpus_path)):
        p_str = str(doc_path)
        doc_id = path_to_doc_id.get(p_str, path_to_doc_id.get(doc_path.name, fallback_id))
        for piece in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                ChunkMeta(
                    chunk_id=len(chunks),
                    doc_id=doc_id,
                    path=p_str,
                    text=piece,
                )
            )
            texts_to_embed.append(piece)

    emb = embed_texts(texts_to_embed)
    doc_ids = np.asarray([c.doc_id for c in chunks], dtype=np.int32)
    return SemanticStore(embeddings=emb, doc_ids=doc_ids, chunks=chunks)


def semantic_search(
    store: SemanticStore,
    query: str,
    top_k: int = 5,
    index: Index | None = None,
) -> list[SearchResult]:
    """Умножает матрицу фрагментов на вектор запроса и выбирает лучший фрагмент документа."""
    if store.embeddings.size == 0 or not query.strip() or top_k <= 0:
        return []

    q_vec = embed_texts([query])[0]
    similarities: NDArray[np.float32] = store.embeddings @ q_vec

    best_by_doc: dict[int, tuple[float, ChunkMeta]] = {}
    for idx_pos, sim_val in enumerate(similarities.tolist()):
        score_f = float(sim_val)
        if score_f <= 0.0:
            continue
        meta = store.chunks[idx_pos]
        prev = best_by_doc.get(meta.doc_id)
        if prev is None or score_f > prev[0]:
            best_by_doc[meta.doc_id] = (score_f, meta)

    ordered = sorted(best_by_doc.values(), key=lambda item: (-item[0], item[1].doc_id))[:top_k]
    results: list[SearchResult] = []
    for sc, meta in ordered:
        resolved_path = index.doc_path(meta.doc_id) if index is not None else meta.path
        snippet = meta.text[:160] + ("..." if len(meta.text) > 160 else "")
        results.append(
            SearchResult(
                doc_id=meta.doc_id,
                score=sc,
                path=resolved_path or meta.path,
                snippet=snippet,
            )
        )
    return results


def rrf_merge(
    ranked_lists: Sequence[Sequence[SearchResult]],
    k_rrf: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Объединяет рейтинги через Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank))."""
    rrf_scores: dict[int, float] = {}
    best_meta: dict[int, SearchResult] = {}

    for rlist in ranked_lists:
        for rank_idx, item in enumerate(rlist, start=1):
            rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + 1.0 / (k_rrf + rank_idx)
            if item.doc_id not in best_meta:
                best_meta[item.doc_id] = item

    sorted_ids = sorted(rrf_scores.keys(), key=lambda d: (-rrf_scores[d], d))[:top_k]
    merged: list[SearchResult] = []
    for d_id in sorted_ids:
        base = best_meta[d_id]
        merged.append(
            SearchResult(
                doc_id=d_id,
                score=rrf_scores[d_id],
                path=base.path,
                snippet=base.snippet,
            )
        )
    return merged


def hybrid_search(
    index: Index,
    store: SemanticStore,
    query: str,
    query_terms: Iterable[str],
    top_k: int = 5,
    k_rrf: int = 60,
) -> list[SearchResult]:
    """Гибридный поиск: сливает BM25 и семантический поиск по RRF."""
    terms_list = list(dict.fromkeys(query_terms))
    bm25_hits = rank_documents(
        index=index,
        query_terms=terms_list,
        scorer=BM25(),
        top_k=max(top_k * 3, 10),
    )
    sem_hits = semantic_search(
        store=store,
        query=query,
        top_k=max(top_k * 3, 10),
        index=index,
    )
    return rrf_merge([bm25_hits, sem_hits], k_rrf=k_rrf, top_k=top_k)


def precision_at_k(retrieved_doc_ids: Sequence[int], relevant_doc_ids: set[int], k: int = 5) -> float:
    """Вычисляет precision@k."""
    if k <= 0:
        return 0.0
    top_slice = list(retrieved_doc_ids)[:k]
    if not top_slice:
        return 0.0
    hits = sum(1 for d_id in top_slice if d_id in relevant_doc_ids)
    return hits / float(k)


def round_metric(val: float) -> float:
    return math.floor(val * 10000.0 + 0.5) / 10000.0

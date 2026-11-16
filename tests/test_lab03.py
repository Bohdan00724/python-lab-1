import heapq
from pathlib import Path

import pytest

from findex.core import tokenize
from findex.decorators import timed
from findex.index import Index, Posting, open_index
from findex.parser import And, Not, Or, Phrase, QuerySyntaxError, Term, parse_query
from findex.ranking import BM25, Scorer, SearchResult, TfIdf
from findex.search import clear_search_cache, get_search_cache_info, search
from findex.snippets import build_snippet


@pytest.fixture
def sample_index(tmp_path: Path) -> tuple[Index, Path]:
    docs = {
        "d0.txt": "Python asyncio uses async and await with an event loop.",
        "d1.txt": "Java uses JVM threads instead of a Python event loop.",
        "d2.txt": "Event loop scheduling in Python async frameworks.",
        "d3.txt": (
            "A very long engineering handbook covering architecture, databases, "
            "networking, deployment, testing, security, monitoring, and a single "
            "mention of python near the end of this long paragraph."
        ),
    }
    idx = Index(store_positions=True)
    for doc_id, (name, text) in enumerate(docs.items()):
        file_path = tmp_path / name
        file_path.write_text(text, encoding="utf-8")
        idx.add_document(doc_id, list(tokenize(text)), path=str(file_path))

    index_file = tmp_path / "test_index.json"
    idx.save_json(index_file)
    return idx, index_file


def test_index_mapping_and_dunders(sample_index: tuple[Index, Path]) -> None:
    idx, _ = sample_index
    assert idx.num_docs == 4
    assert len(idx) > 10
    assert "python" in idx
    assert "nonexistent_word" not in idx
    assert len(idx["python"]) == 4
    assert idx.df("java") == 1
    assert set(iter(idx)) == set(idx.keys())
    assert "Index(docs=4" in repr(idx)
    assert idx.avg_doc_length > 0


def test_open_index_releases_resources_on_exception(sample_index: tuple[Index, Path]) -> None:
    _, index_file = sample_index
    leaked_ref: Index | None = None

    with pytest.raises(RuntimeError):
        with open_index(index_file) as idx:
            leaked_ref = idx
            assert idx.num_docs == 4
            raise RuntimeError("Simulated failure inside with-block")

    assert leaked_ref is not None
    assert leaked_ref.num_docs == 0
    assert len(leaked_ref) == 0


def test_parser_builds_exact_ast_and_supports_operators() -> None:
    query = 'python AND (async OR await) NOT java "event loop"'
    parsed_tree = parse_query(query)

    expected_tree = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert parsed_tree == expected_tree

    dsl_tree = (
        Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(("event", "loop"))
    )
    assert parsed_tree == dsl_tree


def test_parser_phrase_and_boolean_evaluation(sample_index: tuple[Index, Path]) -> None:
    idx, _ = sample_index
    ast = parse_query('python AND (async OR await) NOT java "event loop"')
    assert ast.evaluate(idx) == {0, 2}
    assert Phrase(("loop", "event")).evaluate(idx) == set()


def test_parser_syntax_error_and_missing_positions() -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query("python AND (async OR await")

    no_pos_idx = Index(store_positions=False)
    no_pos_idx.add_document(0, ["event", "loop"])
    with pytest.raises(ValueError):
        Phrase(("event", "loop")).evaluate(no_pos_idx)


def test_scorer_protocol_and_search_result_order() -> None:
    assert isinstance(TfIdf(), Scorer)
    assert isinstance(BM25(), Scorer)

    r_low = SearchResult(score=0.5, doc_id=1)
    r_high = SearchResult(score=2.5, doc_id=2)
    assert r_high > r_low
    assert heapq.nlargest(1, [r_low, r_high])[0] == r_high


def test_ranking_three_sanity_checks(sample_index: tuple[Index, Path]) -> None:
    idx, _ = sample_index
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()

    p_rare = Posting(doc_id=0, frequency=1)
    p_freq = Posting(doc_id=0, frequency=1)
    assert bm25.score("asyncio", p_rare, idx) > bm25.score("python", p_freq, idx)
    assert tfidf.score("asyncio", p_rare, idx) > tfidf.score("python", p_freq, idx)

    s1 = bm25.score("python", Posting(doc_id=0, frequency=1), idx)
    s19 = bm25.score("python", Posting(doc_id=0, frequency=19), idx)
    s20 = bm25.score("python", Posting(doc_id=0, frequency=20), idx)
    assert (s20 - s19) < 0.05 * s1

    p_short = Posting(doc_id=0, frequency=1)
    p_long = Posting(doc_id=3, frequency=1)
    assert bm25.score("python", p_short, idx) > bm25.score("python", p_long, idx)
    assert tfidf.score("python", p_short, idx) > tfidf.score("python", p_long, idx)


def test_timed_wraps_and_lru_cache_hits(sample_index: tuple[Index, Path]) -> None:
    @timed
    def dummy_fn(x: int) -> int:
        """Sample docstring."""
        return x * 2

    assert dummy_fn.__name__ == "dummy_fn"
    assert dummy_fn.__doc__ == "Sample docstring."
    assert dummy_fn(5) == 10

    idx, _ = sample_index
    clear_search_cache()
    search(idx, "python AND async", scorer=BM25(), top_k=3)
    assert get_search_cache_info().misses == 1
    assert get_search_cache_info().hits == 0

    search(idx, "python AND async", scorer=BM25(), top_k=3)
    assert get_search_cache_info().hits == 1


def test_snippet_highlighting(sample_index: tuple[Index, Path]) -> None:
    idx, _ = sample_index
    snippet = build_snippet(
        idx.doc_path(0), terms=["python"], phrases=[("event", "loop")], window=80
    )
    assert "**Python**" in snippet
    assert "**event loop**" in snippet

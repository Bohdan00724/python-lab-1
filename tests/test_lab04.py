import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.core import tokenize
from findex.index import Index, Posting, merge_postings
from findex.parser import And, Not, Or, Phrase, QuerySyntaxError, Term, parse_query
from findex.ranking import BM25, ScorerName, TfIdf, get_scorer
from findex.search import clear_search_cache, intersect_terms, search

runner = CliRunner()


# --- 1. Параметризовані тести токенізатора (включно з Unicode) ---
@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("async/await & event-loop", ["async", "await", "event-loop"]),
        ("don't stop believing", ["don't", "stop", "believing"]),
        ("Привіт, Київ! Тест-кейс №1.", ["привіт", "київ", "тест-кейс", "no1"]),
        ("Café naïve résumé über", ["café", "naïve", "résumé", "über"]),
        ("   \n\t  ", []),
    ],
)
def test_tokenizer_parametrized_unicode(raw_text: str, expected: list[str]) -> None:
    assert list(tokenize(raw_text)) == expected


# --- 2. Тести збірки індексу та merge ---
def test_index_build_from_mini_corpus(built_index: tuple[Index, Path]) -> None:
    idx, _ = built_index
    assert idx.num_docs == 5
    assert "python" in idx
    assert "київ" in idx
    assert idx.df("python") == 5
    assert idx.df("java") == 1


def test_merge_postings_matches_set_engine(built_index: tuple[Index, Path]) -> None:
    idx, _ = built_index
    terms = ["python", "async", "loop"]
    by_merge = intersect_terms(idx, terms, engine="merge")
    by_set = intersect_terms(idx, terms, engine="set")
    assert by_merge == by_set
    assert intersect_terms(idx, ["python", "nonexistent"], engine="merge") == set()
    assert merge_postings([]) == []


# --- 3. Тести парсера запитів (порівняння AST-дерев) ---
@pytest.mark.parametrize(
    ("query_str", "expected_ast"),
    [
        ("python", Term("python")),
        ("python AND async", And(Term("python"), Term("async"))),
        ("python OR java", Or(Term("python"), Term("java"))),
        ("NOT java", Not(Term("java"))),
        ('"event loop"', Phrase(("event", "loop"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(
                    And(Term("python"), Or(Term("async"), Term("await"))),
                    Not(Term("java")),
                ),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_parser_ast_trees(query_str: str, expected_ast: object) -> None:
    assert parse_query(query_str) == expected_ast


@pytest.mark.parametrize(
    "bad_query",
    ["", "   ", "(python AND async", "python AND OR async", '""'],
)
def test_parser_invalid_queries(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


# --- 4. Тести скорерів та save/load ---
@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
def test_scorers_three_invariants(built_index: tuple[Index, Path], scorer_name: ScorerName) -> None:
    idx, _ = built_index
    scorer = get_scorer(scorer_name)

    rare_score = scorer.score("asyncio", Posting(0, 1), idx)
    freq_score = scorer.score("python", Posting(0, 1), idx)
    assert rare_score > freq_score

    short_score = scorer.score("python", Posting(0, 1), idx)
    long_score = scorer.score("python", Posting(4, 1), idx)
    assert short_score > long_score


def test_bm25_saturation(built_index: tuple[Index, Path]) -> None:
    idx, _ = built_index
    bm25 = BM25(k1=1.5, b=0.75)
    s1 = bm25.score("python", Posting(0, 1), idx)
    s19 = bm25.score("python", Posting(0, 19), idx)
    s20 = bm25.score("python", Posting(0, 20), idx)
    assert (s20 - s19) < 0.05 * s1


def test_save_and_load_roundtrip(built_index: tuple[Index, Path], tmp_path: Path) -> None:
    idx, _ = built_index
    out_file = tmp_path / "copy.json"
    idx.save_json(out_file)
    loaded = Index.load_json(out_file)

    assert loaded.num_docs == idx.num_docs
    assert list(loaded.keys()) == list(idx.keys())
    for term in idx:
        assert loaded[term] == idx[term]


# --- 5. Тести CLI через typer.testing.CliRunner ---
def test_cli_index_and_search_table(mini_corpus_dir: Path, tmp_path: Path) -> None:
    idx_file = tmp_path / "cli_idx.json"
    res_index = runner.invoke(app, ["index", str(mini_corpus_dir), "--out", str(idx_file)])
    assert res_index.exit_code == 0
    assert idx_file.exists()

    clear_search_cache()
    res_search = runner.invoke(
        app,
        ["search", 'python AND (async OR await) NOT java "event loop"', "--index", str(idx_file)],
    )
    assert res_search.exit_code == 0
    assert "01_async.txt" in res_search.stdout


def test_cli_search_json_stdout_clean(built_index: tuple[Index, Path]) -> None:
    _, idx_file = built_index
    clear_search_cache()
    res = runner.invoke(app, ["search", "python AND async", "--index", str(idx_file), "--json"])
    assert res.exit_code == 0
    parsed: list[dict[str, Any]] = json.loads(res.stdout)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert "doc_id" in parsed[0] and "score" in parsed[0]


def test_cli_stats_and_verbosity(built_index: tuple[Index, Path], mini_corpus_dir: Path) -> None:
    _, idx_file = built_index
    res = runner.invoke(
        app,
        ["stats", "--index", str(idx_file), "--corpus", str(mini_corpus_dir), "-vv"],
    )
    assert res.exit_code == 0
    assert "Статистика індексу" in res.stdout


def test_cli_expected_errors_exit_nonzero_without_traceback(
    built_index: tuple[Index, Path], tmp_path: Path
) -> None:
    _, idx_file = built_index
    missing_file = tmp_path / "missing.json"

    res_missing = runner.invoke(app, ["search", "python", "--index", str(missing_file)])
    assert res_missing.exit_code != 0
    assert "Traceback" not in res_missing.output

    res_bad_query = runner.invoke(app, ["search", "python AND (", "--index", str(idx_file)])
    assert res_bad_query.exit_code != 0
    assert "Traceback" not in res_bad_query.output


# --- 6. Три властивості Hypothesis ---
_DOCS_STRATEGY = st.lists(
    st.lists(
        st.text(alphabet="abcxyz", min_size=1, max_size=5),
        min_size=1,
        max_size=12,
    ),
    min_size=1,
    max_size=8,
)


@given(docs=_DOCS_STRATEGY)
@settings(max_examples=30, deadline=None)
def test_hypothesis_save_load_preserves_index(docs: list[list[str]]) -> None:
    idx = Index(store_positions=True)
    for doc_id, tokens in enumerate(docs):
        idx.add_document(doc_id, tokens)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "idx.json"
        idx.save_json(tmp_file)
        restored = Index.load_json(tmp_file)

    assert restored.num_docs == idx.num_docs
    assert set(restored.keys()) == set(idx.keys())
    for term in idx:
        assert restored[term] == idx[term]


@given(
    doc_ids=st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=1,
        max_size=15,
        unique=True,
    ),
)
@settings(max_examples=30)
def test_hypothesis_postings_always_sorted(doc_ids: list[int]) -> None:
    idx = Index(store_positions=True)
    for doc_id in doc_ids:
        idx.add_document(doc_id, ["shared", f"doc{doc_id}"])

    posting_ids = [p.doc_id for p in idx["shared"]]
    assert posting_ids == sorted(doc_ids)


@given(docs=_DOCS_STRATEGY)
@settings(max_examples=30)
def test_hypothesis_merge_equals_set_intersection(docs: list[list[str]]) -> None:
    idx = Index(store_positions=True)
    for doc_id, tokens in enumerate(docs):
        idx.add_document(doc_id, tokens)

    terms = list(idx.keys())[:3]
    if terms:
        by_merge = intersect_terms(idx, terms, engine="merge")
        by_set = intersect_terms(idx, terms, engine="set")
        assert by_merge == by_set


# --- 7. Повільний тест із маркером slow ---
@pytest.mark.slow
def test_slow_large_corpus_indexing_and_search() -> None:
    idx = Index(store_positions=True)
    for doc_id in range(400):
        idx.add_document(doc_id, ["python", "async", "event", "loop", f"item{doc_id}"])
    time.sleep(1.05)
    results = search(idx, "python AND async", scorer=TfIdf(), top_k=5, with_snippets=False)
    assert len(results) == 5

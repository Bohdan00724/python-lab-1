from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import (
    ExecutorType,
    Index,
    build_index,
    build_partial,
    merge,
    split_into_chunks,
)

runner = CliRunner()


def _assert_indices_equal(actual: Index, expected: Index) -> None:
    assert actual.num_docs == expected.num_docs
    assert actual.store_positions == expected.store_positions
    assert actual.avg_doc_length == pytest.approx(expected.avg_doc_length)
    assert set(actual.keys()) == set(expected.keys())
    for term in expected:
        assert actual[term] == expected[term]


def test_merge_single_build_partial_matches_lab4(
    built_index: tuple[Index, Path], mini_corpus_dir: Path
) -> None:
    """Послідовний варіант merge([build_partial(усі шляхи)]) дає той самий індекс."""
    expected_idx, _ = built_index
    all_paths = sorted(mini_corpus_dir.rglob("*.txt"))

    merged_idx = merge([build_partial(all_paths)])
    _assert_indices_equal(merged_idx, expected_idx)


def test_split_into_chunks_assigns_unique_doc_ids(mini_corpus_dir: Path) -> None:
    all_paths = sorted(mini_corpus_dir.rglob("*.txt"))
    chunks = split_into_chunks(all_paths, num_chunks=3)

    assigned_ids: list[int] = []
    for start_id, chunk_paths in chunks:
        for offset, _ in enumerate(chunk_paths):
            assigned_ids.append(start_id + offset)

    assert assigned_ids == list(range(len(all_paths)))


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_produce_identical_index(
    built_index: tuple[Index, Path],
    mini_corpus_dir: Path,
    executor: ExecutorType,
    workers: int,
) -> None:
    expected_idx, _ = built_index
    all_paths = sorted(mini_corpus_dir.rglob("*.txt"))

    parallel_idx = build_index(all_paths, workers=workers, executor=executor)
    _assert_indices_equal(parallel_idx, expected_idx)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_worker_crash_propagates_without_hanging(
    mini_corpus_dir: Path, executor: ExecutorType
) -> None:
    """Якщо воркер падає (неіснуючий файл), збірка теж падає і показує виняток, а не зависає."""
    valid_paths = sorted(mini_corpus_dir.rglob("*.txt"))
    broken_paths = [*valid_paths, mini_corpus_dir / "non_existent_worker_crash.txt"]

    with pytest.raises(FileNotFoundError):
        build_index(broken_paths, workers=2, executor=executor)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_cli_index_workers_and_executor(
    mini_corpus_dir: Path, tmp_path: Path, executor: ExecutorType
) -> None:
    out_file = tmp_path / f"idx_{executor}.json"
    res = runner.invoke(
        app,
        [
            "index",
            str(mini_corpus_dir),
            "--out",
            str(out_file),
            "--workers",
            "2",
            "--executor",
            executor,
        ],
    )
    assert res.exit_code == 0
    assert out_file.exists()

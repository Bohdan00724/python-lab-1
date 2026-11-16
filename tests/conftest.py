from pathlib import Path

import pytest

from findex.core import iter_document_files, tokenize
from findex.index import Index


@pytest.fixture
def mini_corpus_dir(tmp_path: Path) -> Path:
    docs = {
        "01_async.txt": "Python asyncio uses async and await inside an event loop.",
        "02_java.txt": "Java uses JVM threads instead of a Python event loop.",
        "03_loop.txt": "Event loop scheduling in Python async web frameworks.",
        "04_unicode.txt": "Привіт, Київ! Python підтримує Unicode та токенізацію.",
        "05_long.txt": (
            "Extensive software engineering manual covering architecture, networking, "
            "databases, deployment, security, monitoring, containers, testing, and "
            "one brief mention of python near the end of this long document."
        ),
    }
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for name, text in docs.items():
        (corpus_dir / name).write_text(text, encoding="utf-8")
    return corpus_dir


@pytest.fixture
def built_index(mini_corpus_dir: Path, tmp_path: Path) -> tuple[Index, Path]:
    idx = Index(store_positions=True)

    def pipeline():
        for doc_id, (file_path, text) in enumerate(iter_document_files(mini_corpus_dir)):
            yield doc_id, list(tokenize(text)), str(file_path)

    idx.build_from_pipeline(pipeline())
    index_path = tmp_path / "index.json"
    idx.save_json(index_path)
    return idx, index_path

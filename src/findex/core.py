import pathlib
import re
import unicodedata
from collections.abc import Iterator


def iter_document_files(corpus_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    """Ліниво читає .txt файли у відсортованому порядку та повертає (шлях, текст)."""
    for file_path in sorted(corpus_path.rglob("*.txt")):
        with open(file_path, encoding="utf-8") as f:
            yield file_path, f.read()


def iter_documents(corpus_path: pathlib.Path) -> Iterator[str]:
    """Ліниво повертає лише тексти документів."""
    for _, text in iter_document_files(corpus_path):
        yield text


def tokenize(text: str) -> Iterator[str]:
    """Нормалізує текст і витягує токени."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    pattern = re.compile(r"[\w'-]+\b")
    for match in pattern.finditer(normalized):
        yield match.group()

import pathlib
import re
import unicodedata
from typing import Iterator


def iter_document_files(corpus_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    """Лениво читает .txt файлы в отсортированном порядке и возвращает (путь, текст)."""
    for file_path in sorted(corpus_path.rglob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            yield file_path, f.read()


def iter_documents(corpus_path: pathlib.Path) -> Iterator[str]:
    """Лениво возвращает только тексты документов (совместимость с лаб 1–2)."""
    for _, text in iter_document_files(corpus_path):
        yield text


def tokenize(text: str) -> Iterator[str]:
    """Нормализует текст и извлекает токены."""
    text = unicodedata.normalize("NFKC", text.lower())
    pattern = re.compile(r"[\w'-]+\b")
    for match in pattern.finditer(text):
        yield match.group()
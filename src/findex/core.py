import json
import pathlib
import re
import unicodedata
from collections.abc import Iterator
from typing import Any


def _iter_jsonl_file(jsonl_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    """Ліниво читає JSONL-файл рядок за рядком без завантаження всього файлу в пам'ять."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            url = str(record.get("url") or f"{jsonl_path}:{line_no}")
            title = str(record.get("title") or "").strip()
            body = str(record.get("text") or "").strip()
            full_text = f"{title}\n{body}".strip() if title else body
            yield pathlib.Path(url), full_text


def iter_document_files(corpus_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    """Ліниво читає .txt або .jsonl документи та повертає (шлях/URL, текст)."""
    if corpus_path.is_file():
        if corpus_path.suffix.lower() == ".jsonl":
            yield from _iter_jsonl_file(corpus_path)
        else:
            with open(corpus_path, encoding="utf-8") as f:
                yield corpus_path, f.read()
        return

    for file_path in sorted(corpus_path.rglob("*.txt")):
        with open(file_path, encoding="utf-8") as f:
            yield file_path, f.read()

    for jsonl_path in sorted(corpus_path.rglob("*.jsonl")):
        yield from _iter_jsonl_file(jsonl_path)


def iter_documents(corpus_path: pathlib.Path) -> Iterator[str]:
    """Ліниво повертає лише тексти документів (з .txt або .jsonl)."""
    for _, text in iter_document_files(corpus_path):
        yield text


def tokenize(text: str) -> Iterator[str]:
    """Нормалізує текст і витягує токени."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    pattern = re.compile(r"[\w'-]+\b")
    for match in pattern.finditer(normalized):
        yield match.group()

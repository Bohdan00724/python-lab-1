from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import TextIO
import tracemalloc

from src.findex.core import iter_document_files, tokenize
from src.findex.decorators import timed


@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int
    path: str = ""


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: tuple[int, ...] = ()


class Index(Mapping[str, list[Posting]]):
    """Инвертированный индекс с интерфейсом Python Mapping."""

    def __init__(self, store_positions: bool = True) -> None:
        self.store_positions = store_positions
        self._postings: defaultdict[str, list[Posting]] = defaultdict(list)
        self._doc_meta: dict[int, DocMeta] = {}

    # --- Dunder-методы для Mapping и модели объектов ---
    def __len__(self) -> int:
        return len(self._postings)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return (
            f"Index(docs={self.num_docs}, terms={len(self)}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"store_positions={self.store_positions})"
        )

    # --- Свойства и статистика корпуса ---
    @property
    def num_docs(self) -> int:
        return len(self._doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_meta:
            return 0.0
        total_len = sum(meta.length for meta in self._doc_meta.values())
        return total_len / len(self._doc_meta)

    def df(self, term: str) -> int:
        """Document frequency: в скольких документах встречается термин."""
        return len(self._postings.get(term, ()))

    def doc_length(self, doc_id: int) -> int:
        return self._doc_meta[doc_id].length

    def doc_path(self, doc_id: int) -> str:
        return self._doc_meta[doc_id].path

    def all_doc_ids(self) -> set[int]:
        return set(self._doc_meta.keys())

    # --- Построение и сериализация ---
    def add_document(self, doc_id: int, tokens: list[str], path: str = "") -> None:
        self.__dict__.pop("avg_doc_length", None)

        self._doc_meta[doc_id] = DocMeta(length=len(tokens), path=path)
        pos_map: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            pos_map[token].append(pos)

        for term, positions in pos_map.items():
            self._postings[term].append(
                Posting(
                    doc_id=doc_id,
                    frequency=len(positions),
                    positions=tuple(positions) if self.store_positions else (),
                )
            )

    @timed
    def build_from_pipeline(self, pipeline) -> None:
        for item in pipeline:
            if len(item) == 3:
                doc_id, tokens, path = item
            else:
                doc_id, tokens = item
                path = ""
            self.add_document(doc_id, tokens, path=path)

    def save_json(self, filepath: str | Path) -> None:
        data = {
            "store_positions": self.store_positions,
            "doc_meta": {
                str(k): {"length": v.length, "path": v.path}
                for k, v in self._doc_meta.items()
            },
            "postings": {
                term: [
                    {
                        "doc_id": p.doc_id,
                        "frequency": p.frequency,
                        **(
                            {"positions": list(p.positions)}
                            if self.store_positions
                            else {}
                        ),
                    }
                    for p in plist
                ]
                for term, plist in self._postings.items()
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    @timed
    def from_stream(cls, stream: TextIO) -> "Index":
        data = json.load(stream)
        store_positions = data.get("store_positions", True)
        index = cls(store_positions=store_positions)
        index._doc_meta = {
            int(k): DocMeta(length=v["length"], path=v.get("path", ""))
            for k, v in data["doc_meta"].items()
        }
        index._postings = defaultdict(
            list,
            {
                term: [
                    Posting(
                        doc_id=p["doc_id"],
                        frequency=p["frequency"],
                        positions=tuple(p.get("positions", ())),
                    )
                    for p in plist
                ]
                for term, plist in data["postings"].items()
            },
        )
        return index

    @classmethod
    def load_json(cls, filepath: str | Path) -> "Index":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_stream(f)


@contextmanager
def open_index(path: str | Path) -> Iterator[Index]:
    """Context manager для безопасной загрузки и освобождения ресурсов индекса."""
    file_obj: TextIO | None = None
    index: Index | None = None
    try:
        file_obj = open(path, "r", encoding="utf-8")
        index = Index.from_stream(file_obj)
        yield index
    finally:
        if file_obj is not None and not file_obj.closed:
            file_obj.close()
        if index is not None:
            index._postings.clear()
            index._doc_meta.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Построение инвертированного индекса")
    parser.add_argument("data_dir", help="Шлях до папки з документами")
    parser.add_argument("--out", required=True, help="Файл для збереження (.json)")
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Не зберігати позиції слів (економія пам'яті, вимикає фразовий пошук)",
    )
    args = parser.parse_args()

    print(f"Будуємо індекс з {args.data_dir}...")
    tracemalloc.start()

    idx = Index(store_positions=not args.no_positions)

    def pipeline():
        for doc_id, (file_path, text) in enumerate(iter_document_files(Path(args.data_dir))):
            yield doc_id, list(tokenize(text)), str(file_path)

    idx.build_from_pipeline(pipeline())

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Об'єкт індексу: {idx!r}")
    print(f"Пікове використання пам'яті: {peak / 1024 / 1024:.4f} MB")
    idx.save_json(args.out)
    print(f"Збережено у {args.out}!")
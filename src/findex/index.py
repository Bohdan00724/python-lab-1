import json
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, TextIO

from findex.decorators import timed


@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int
    path: str = ""


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: tuple[int, ...] = ()


def merge_postings(postings_lists: Sequence[Sequence[Posting]]) -> list[int]:
    """Перетин відсортованих списків постінгів за doc_id (алгоритм merge)."""
    if not postings_lists:
        return []
    if any(len(plist) == 0 for plist in postings_lists):
        return []

    ordered = sorted(postings_lists, key=len)
    result = [p.doc_id for p in ordered[0]]

    for plist in ordered[1:]:
        next_result: list[int] = []
        i = 0
        j = 0
        while i < len(result) and j < len(plist):
            doc_a = result[i]
            doc_b = plist[j].doc_id
            if doc_a == doc_b:
                next_result.append(doc_a)
                i += 1
                j += 1
            elif doc_a < doc_b:
                i += 1
            else:
                j += 1
        result = next_result
        if not result:
            break

    return result


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс з інтерфейсом Python Mapping."""

    def __init__(self, store_positions: bool = True) -> None:
        self.store_positions = store_positions
        self._postings: dict[str, list[Posting]] = {}
        self._doc_meta: dict[int, DocMeta] = {}

    def __len__(self) -> int:
        return len(self._postings)

    def __getitem__(self, key: str, /) -> list[Posting]:
        return self._postings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __repr__(self) -> str:
        return (
            f"Index(docs={self.num_docs}, terms={len(self)}, "
            f"avg_doc_length={self.avg_doc_length:.2f}, "
            f"store_positions={self.store_positions})"
        )

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
        """Document frequency: у скількох документах зустрічається термін."""
        return len(self._postings.get(term, ()))

    def doc_length(self, doc_id: int) -> int:
        return self._doc_meta[doc_id].length

    def doc_path(self, doc_id: int) -> str:
        return self._doc_meta[doc_id].path

    def all_doc_ids(self) -> set[int]:
        return set(self._doc_meta.keys())

    def clear(self) -> None:
        """Очищає всі структури індексу та звільняє пам'ять."""
        self.__dict__.pop("avg_doc_length", None)
        self._postings.clear()
        self._doc_meta.clear()

    def add_document(self, doc_id: int, tokens: Sequence[str], path: str = "") -> None:
        self.__dict__.pop("avg_doc_length", None)

        if doc_id in self._doc_meta:
            for plist in self._postings.values():
                plist[:] = [p for p in plist if p.doc_id != doc_id]

        self._doc_meta[doc_id] = DocMeta(length=len(tokens), path=path)
        pos_map: dict[str, list[int]] = {}
        for pos, token in enumerate(tokens):
            pos_map.setdefault(token, []).append(pos)

        for term, positions in pos_map.items():
            plist = self._postings.setdefault(term, [])
            new_posting = Posting(
                doc_id=doc_id,
                frequency=len(positions),
                positions=tuple(positions) if self.store_positions else (),
            )
            if plist and plist[-1].doc_id > doc_id:
                plist.append(new_posting)
                plist.sort(key=lambda p: p.doc_id)
            else:
                plist.append(new_posting)

    @timed
    def build_from_pipeline(
        self,
        pipeline: Iterable[tuple[int, Sequence[str], str]],
    ) -> None:
        for doc_id, tokens, path in pipeline:
            self.add_document(doc_id, tokens, path=path)

    def save_json(self, filepath: str | Path) -> None:
        serialized_postings: dict[str, list[dict[str, object]]] = {}
        for term, plist in self._postings.items():
            entries: list[dict[str, object]] = []
            for p in plist:
                entry: dict[str, object] = {
                    "doc_id": p.doc_id,
                    "frequency": p.frequency,
                }
                if self.store_positions:
                    entry["positions"] = list(p.positions)
                entries.append(entry)
            serialized_postings[term] = entries

        data: dict[str, object] = {
            "store_positions": self.store_positions,
            "doc_meta": {
                str(k): {"length": v.length, "path": v.path} for k, v in self._doc_meta.items()
            },
            "postings": serialized_postings,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    @timed
    def from_stream(cls, stream: TextIO) -> "Index":
        raw_data: dict[str, Any] = json.load(stream)
        store_positions = bool(raw_data.get("store_positions", True))
        index = cls(store_positions=store_positions)

        doc_meta_raw: dict[str, dict[str, Any]] = raw_data["doc_meta"]
        for k, v in doc_meta_raw.items():
            index._doc_meta[int(k)] = DocMeta(
                length=int(v["length"]),
                path=str(v.get("path", "")),
            )

        postings_raw: dict[str, list[dict[str, Any]]] = raw_data["postings"]
        for term, plist in postings_raw.items():
            items: list[Posting] = []
            for p in plist:
                raw_positions: list[int] = p.get("positions", [])
                items.append(
                    Posting(
                        doc_id=int(p["doc_id"]),
                        frequency=int(p["frequency"]),
                        positions=tuple(int(pos) for pos in raw_positions),
                    )
                )
            items.sort(key=lambda item: item.doc_id)
            index._postings[str(term)] = items

        return index

    @classmethod
    def load_json(cls, filepath: str | Path) -> "Index":
        with open(filepath, encoding="utf-8") as f:
            return cls.from_stream(f)


@contextmanager
def open_index(path: str | Path) -> Generator[Index, None, None]:
    """Context manager для безпечного завантаження та звільнення ресурсів індексу."""
    file_obj: TextIO | None = None
    index: Index | None = None
    try:
        file_obj = open(path, encoding="utf-8")
        index = Index.from_stream(file_obj)
        yield index
    finally:
        if file_obj is not None and not file_obj.closed:
            file_obj.close()
        if index is not None:
            index.clear()

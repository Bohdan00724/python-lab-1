import argparse
import time
import tracemalloc
import json
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int

class InvertedIndex:
    def __init__(self):
        self.postings: defaultdict[str, list[Posting]] = defaultdict(list)
        self.doc_meta: dict[int, DocMeta] = {}

    @classmethod
    def load_json(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        index = cls()
        index.doc_meta = {int(k): DocMeta(length=v["length"]) for k, v in data["doc_meta"].items()}
        index.postings = defaultdict(list, {
            term: [Posting(doc_id=p["doc_id"], frequency=p["frequency"]) for p in plist]
            for term, plist in data["postings"].items()
        })
        return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file", help="Файл індексу (.json)")
    parser.add_argument("query", help="Пошуковий запит (слова через пробіл)")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    args = parser.parse_args()

    print(f"Завантажуємо індекс з {args.index_file}...")
    tracemalloc.start()
    start = time.time()
    idx = InvertedIndex.load_json(args.index_file)
    load_time = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Завантажено за {load_time:.4f} сек.")
    print(f"Пам'ять під час завантаження: {peak / 1024 / 1024:.4f} MB")

    terms = args.query.lower().split()
    if not terms:
        return

    print(f"\nШукаємо: {terms} (рушій: {args.engine})")
    start_search = time.time()
    postings_lists = [idx.postings.get(term, []) for term in terms]
    
    if any(not p for p in postings_lists):
        print("Нічого не знайдено.")
        return

    if args.engine == "set":
        sets = [set(p.doc_id for p in plist) for plist in postings_lists]
        common_docs = set.intersection(*sets)
        results = [p for plist in postings_lists for p in plist if p.doc_id in common_docs]
    else:
        doc_sets = [set(p.doc_id for p in plist) for plist in postings_lists]
        common_ids = set.intersection(*doc_sets)
        results = [p for plist in postings_lists for p in plist if p.doc_id in common_ids]

    unique_results = {p.doc_id: p for p in results}.values()
    search_time = time.time() - start_search
    
    top = sorted(unique_results, key=lambda x: x.frequency, reverse=True)[:5]
    print(f"Знайдено документів: {len(unique_results)} (за {search_time:.6f} сек)")
    print("\nТоп-5 результатів (за частотою):")
    for r in top:
        print(f"  Doc ID: {r.doc_id} | Частота: {r.frequency}")

if __name__ == '__main__':
    main()
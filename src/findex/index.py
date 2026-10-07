from collections import defaultdict, Counter
from dataclasses import dataclass
import json
import tracemalloc
import time
from pathlib import Path
from src.findex.core import iter_documents, tokenize

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

    def add_document(self, doc_id: int, tokens: list[str]):
        self.doc_meta[doc_id] = DocMeta(length=len(tokens))
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            self.postings[term].append(Posting(doc_id=doc_id, frequency=count))

    def build_from_pipeline(self, pipeline):
        for doc_id, tokens in pipeline:
            self.add_document(doc_id, tokens)

    def save_json(self, filepath: str):
        data = {
            "doc_meta": {str(k): {"length": v.length} for k, v in self.doc_meta.items()},
            "postings": {
                term: [{"doc_id": p.doc_id, "frequency": p.frequency} for p in plist]
                for term, plist in self.postings.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Шлях до папки з документами")
    parser.add_argument("--out", required=True, help="Файл для збереження (.json)")
    args = parser.parse_args()

    print(f"Будуємо індекс з {args.data_dir}...")
    tracemalloc.start()
    start = time.time()

    idx = InvertedIndex()
    def pipeline():
        for doc_id, text in enumerate(iter_documents(Path(args.data_dir))):
            yield doc_id, list(tokenize(text))
            
    idx.build_from_pipeline(pipeline())
    
    build_time = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Побудовано за {build_time:.4f} сек.")
    print(f"Пікове використання пам'яті: {peak / 1024 / 1024:.4f} MB")
    idx.save_json(args.out)
    print(f"Збережено у {args.out}!")
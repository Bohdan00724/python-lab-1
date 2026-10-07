import pathlib
import re
import unicodedata

def iter_documents(corpus_path: pathlib.Path):
    for file_path in corpus_path.rglob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            yield f.read()

def tokenize(text: str):
    text = unicodedata.normalize('NFKC', text.lower())
    pattern = re.compile(r"[\w'-]+\b")
    for match in pattern.finditer(text):
        yield match.group()
import re
from pathlib import Path

def stream_documents(folder_path: Path):
    """Ліниве читання текстових файлів по одному."""
    for file_path in folder_path.glob("*.txt"):
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as file:
                yield file.read()

def extract_tokens(text: str):
    """Очищення та токенізація тексту."""
    clean_text = text.lower()
    pattern = re.compile(r'[a-z0-9]+')
    for match in pattern.finditer(clean_text):
        yield match.group()
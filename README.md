# findex — Пошуковий рушій на Python (Лабораторні роботи 1–3)

Пошуковий рушій з об'єктною моделлю Python (`Index` як `Mapping`), булевим парсером запитів, сніпетами та ранжуванням **TF-IDF** і **BM25**.

## Команди запуску

```bash
# Побудова індексу
python -m src.findex.index data --out index.json

# Пошук зі складним запитом, сніпетами та перевіркою @lru_cache
python -m src.findex.search index.json 'python AND (async OR await) NOT java "event loop"' --scorer bm25 --repeat

# Оцінка якості (3 sanity-check + таблиця Precision@5)
python -m src.findex.evaluate

# Запуск тестів
pytest -v
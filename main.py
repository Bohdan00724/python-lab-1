from pathlib import Path
from src.findex.stats import run_generator_test, run_list_test

if __name__ == "__main__":
    data_directory = Path("data")
    
    print("=== Тест 1: Використання генераторів (Лінивий підхід) ===")
    gen_time, gen_mem, top_words = run_generator_test(data_directory)
    print(f"Витрачено часу: {gen_time:.4f} сек | Пікова пам'ять: {gen_mem:.4f} МБ")
    print(f"Топ-5 слів: {top_words}\n")
    
    print("=== Тест 2: Використання списків (Жадібний підхід) ===")
    list_time, list_mem = run_list_test(data_directory)
    print(f"Витрачено часу: {list_time:.4f} сек | Пікова пам'ять: {list_mem:.4f} МБ")
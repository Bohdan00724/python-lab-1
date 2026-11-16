import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.core import tokenize
from findex.index import Index, open_index
from findex.parser import QuerySyntaxError
from findex.ranking import ScorerName, get_scorer
from findex.search import EngineName, search
from findex.stats import run_generator_test, run_list_test

app = typer.Typer(
    name="findex",
    help="Пошуковий рушій findex з інвертованим індексом та ранжуванням TF-IDF / BM25.",
    no_args_is_help=True,
)

err_console = Console(stderr=True)
out_console = Console()
_global_verbosity: int = 0


def _configure_logging(verbosity: int) -> None:
    """Налаштовує логер у точці входу CLI; уся діагностика йде в stderr."""
    effective = max(verbosity, _global_verbosity)
    if effective >= 2:
        level = logging.DEBUG
    elif effective == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


VerboseOption = Annotated[
    int,
    typer.Option(
        "-v",
        "--verbose",
        count=True,
        help="Рівень деталізації логів у stderr (-v для INFO, -vv для DEBUG).",
    ),
]


@app.callback()
def main_callback(verbose: VerboseOption = 0) -> None:
    global _global_verbosity
    _global_verbosity = verbose
    _configure_logging(verbose)


@app.command("index")
def index_cmd(
    corpus_dir: Annotated[Path, typer.Argument(help="Каталог із .txt документами корпусу")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Шлях до вихідного JSON-індексу")] = Path(
        "index.json"
    ),
    no_positions: Annotated[
        bool,
        typer.Option("--no-positions", help="Не зберігати позиції слів в індексі"),
    ] = False,
    verbose: VerboseOption = 0,
) -> None:
    """Будує інвертований індекс із корпусу документів зі смугою прогресу."""
    _configure_logging(verbose)
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        err_console.print(f"[red]Помилка:[/red] каталог корпусу не знайдено: {corpus_dir}")
        raise typer.Exit(code=1)

    files = sorted(corpus_dir.rglob("*.txt"))
    if not files:
        err_console.print(f"[red]Помилка:[/red] у каталозі {corpus_dir} немає .txt файлів.")
        raise typer.Exit(code=1)

    idx = Index(store_positions=not no_positions)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=err_console,
    ) as progress:
        task_id = progress.add_task("Індексація документів...", total=len(files))

        def pipeline() -> Iterator[tuple[int, list[str], str]]:
            for doc_id, file_path in enumerate(files):
                text = file_path.read_text(encoding="utf-8")
                tokens = list(tokenize(text))
                progress.advance(task_id)
                yield doc_id, tokens, str(file_path)

        idx.build_from_pipeline(pipeline())

    idx.save_json(out)
    err_console.print(f"[green]Готово:[/green] {idx!r} збережено у {out}")


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help='Пошуковий запит (AND, OR, NOT, дужки, "фрази")')],
    index_file: Annotated[
        Path, typer.Option("--index", "-i", help="Шлях до файлу індексу (.json)")
    ] = Path("index.json"),
    scorer: Annotated[
        ScorerName, typer.Option("--scorer", "-s", help="Алгоритм ранжування: bm25 або tfidf")
    ] = "bm25",
    engine: Annotated[
        EngineName, typer.Option("--engine", "-e", help="Рушій перетину: merge або set")
    ] = "merge",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Кількість результатів")] = 5,
    as_json: Annotated[
        bool, typer.Option("--json", help="Вивести результати у форматі JSON у stdout")
    ] = False,
    verbose: VerboseOption = 0,
) -> None:
    """Виконує ранжований пошук та виводить таблицю Rich або JSON у stdout."""
    _configure_logging(verbose)
    if not index_file.exists() or not index_file.is_file():
        err_console.print(f"[red]Помилка:[/red] файл індексу не знайдено: {index_file}")
        raise typer.Exit(code=1)

    try:
        with open_index(index_file) as idx:
            results = search(
                index=idx,
                query=query,
                scorer=get_scorer(scorer),
                engine=engine,
                top_k=top_k,
                with_snippets=True,
            )
    except (QuerySyntaxError, ValueError) as exc:
        err_console.print(f"[red]Помилка запиту:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except OSError as exc:
        err_console.print(f"[red]Помилка читання індексу:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if as_json:
        payload = [
            {
                "doc_id": r.doc_id,
                "score": round(r.score, 6),
                "path": r.path,
                "snippet": r.snippet,
            }
            for r in results
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    if not results:
        err_console.print("Нічого не знайдено.")
        return

    table = Table(title=f"Результати пошуку ({scorer.upper()}): {query}")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Doc ID", justify="right", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Файл", style="yellow")
    table.add_column("Сніпет", style="white")

    for rank, res in enumerate(results, start=1):
        table.add_row(
            str(rank),
            str(res.doc_id),
            f"{res.score:.4f}",
            Path(res.path).name if res.path else "-",
            res.snippet,
        )

    out_console.print(table)


@app.command("stats")
def stats_cmd(
    index_file: Annotated[
        Path, typer.Option("--index", "-i", help="Шлях до файлу індексу (.json)")
    ] = Path("index.json"),
    corpus_dir: Annotated[
        Path | None,
        typer.Option("--corpus", "-c", help="Опційний каталог корпусу для заміру пам'яті"),
    ] = None,
    verbose: VerboseOption = 0,
) -> None:
    """Показує статистику індексу та корпусу."""
    _configure_logging(verbose)
    if not index_file.exists() or not index_file.is_file():
        err_console.print(f"[red]Помилка:[/red] файл індексу не знайдено: {index_file}")
        raise typer.Exit(code=1)

    with open_index(index_file) as idx:
        table = Table(title=f"Статистика індексу: {index_file}")
        table.add_column("Метрика", style="cyan")
        table.add_column("Значення", style="green")

        table.add_row("Документів у корпусі", str(idx.num_docs))
        table.add_row("Унікальних термінів", str(len(idx)))
        table.add_row("Середня довжина документа", f"{idx.avg_doc_length:.2f}")
        table.add_row("Збереження позицій", str(idx.store_positions))

        top_terms = sorted(idx.keys(), key=lambda t: idx.df(t), reverse=True)[:5]
        if top_terms:
            top_str = ", ".join(f"{t} (df={idx.df(t)})" for t in top_terms)
            table.add_row("Топ-5 термінів за df", top_str)

        if corpus_dir is not None:
            if not corpus_dir.exists() or not corpus_dir.is_dir():
                err_console.print(f"[red]Помилка:[/red] каталог не знайдено: {corpus_dir}")
                raise typer.Exit(code=1)
            gen_time, gen_mem, _ = run_generator_test(corpus_dir)
            list_time, list_mem = run_list_test(corpus_dir)
            table.add_row("Генератор (час / пік RAM)", f"{gen_time:.4f} с / {gen_mem:.4f} MB")
            table.add_row("Список (час / пік RAM)", f"{list_time:.4f} с / {list_mem:.4f} MB")

        out_console.print(table)


if __name__ == "__main__":
    app()

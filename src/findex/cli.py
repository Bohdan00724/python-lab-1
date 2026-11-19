import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.core import iter_document_files, tokenize
from findex.crawler import CrawlStats, crawl
from findex.index import ExecutorType, Index, build_index, open_index
from findex.parser import QuerySyntaxError
from findex.ranking import ScorerName, get_scorer
from findex.search import EngineName, search
from findex.stats import run_generator_test, run_list_test

app = typer.Typer(
    name="findex",
    help="Пошуковий рушій findex з інвертованим індексом, TF-IDF/BM25, краулером та веб-UI.",
    no_args_is_help=True,
)

err_console = Console(stderr=True)
out_console = Console()
_global_verbosity: int = 0


def _normalize_doc_ref(doc_ref: Path) -> str:
    """Відновлює коректний URL (https://...) після перетворення у Windows Path."""
    posix_str = doc_ref.as_posix()
    if posix_str.startswith("https:/") and not posix_str.startswith("https://"):
        return "https://" + posix_str[len("https:/") :]
    if posix_str.startswith("http:/") and not posix_str.startswith("http://"):
        return "http://" + posix_str[len("http:/") :]
    return str(doc_ref)


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


def _render_crawl_table(stats: CrawlStats, max_pages: int, out_path: Path) -> Table:
    table = Table(title=f"Асинхронний краулер findex -> {out_path}")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значення", style="green", justify="right")
    table.add_row("Зібрано сторінок", f"{stats.pages_fetched} / {max_pages}")
    table.add_row("Сторінок / сек", f"{stats.pages_per_sec:.2f}")
    table.add_row("Активні запити (in-flight)", str(stats.in_flight))
    table.add_row("Помилки", str(stats.errors))
    table.add_row("Глибина черги", str(stats.queue_depth))
    table.add_row("Завантажено (KB)", f"{stats.bytes_downloaded / 1024:.1f}")
    return table


async def _run_crawl_cli(
    start_url: str,
    out: Path,
    log_file: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    timeout: float,
    same_host: bool,
    debug: bool,
) -> CrawlStats:
    if debug:
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = 0.05

    stats = CrawlStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    stop_ui = asyncio.Event()

    async def ui_refresher(live: Live) -> None:
        while not stop_ui.is_set():
            live.update(_render_crawl_table(stats, max_pages, out))
            await asyncio.sleep(0.1)
        live.update(_render_crawl_table(stats, max_pages, out))

    initial_table = _render_crawl_table(stats, max_pages, out)
    with (
        open(out, "w", encoding="utf-8") as jsonl_file,
        Live(initial_table, console=err_console, refresh_per_second=10) as live,
    ):
        ui_task = asyncio.create_task(ui_refresher(live))
        try:
            async for page in crawl(
                start_url,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_concurrency=per_host,
                min_delay_per_host=delay,
                timeout=timeout,
                same_host_only=same_host,
                stats=stats,
                log_path=log_file,
            ):
                record = {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "status": page.status_code,
                    "bytes": page.bytes_read,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_file.flush()
        finally:
            stop_ui.set()
            await ui_task

    return stats


@app.command("crawl")
def crawl_cmd(
    start_url: Annotated[str, typer.Argument(help="Початковий URL для обходу")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Вихідний JSONL-файл зі сторінками")
    ] = Path("crawled.jsonl"),
    log_file: Annotated[
        Path, typer.Option("--log", help="Файл журналу статусів кожного URL")
    ] = Path("crawl.log"),
    max_pages: Annotated[
        int, typer.Option("--max-pages", "-m", min=1, help="Максимум сторінок")
    ] = 50,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Ліміт одночасних запитів")
    ] = 5,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="Ліміт запитів на один хост")
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Пауза (с) між запитами до одного хоста"),
    ] = 0.05,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.1, help="Таймаут HTTP-запиту (с)")
    ] = 10.0,
    same_host: Annotated[
        bool,
        typer.Option("--same-host/--all-hosts", help="Обмежувати обхід доменом старту"),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Увімкнути asyncio debug mode"),
    ] = False,
    verbose: VerboseOption = 0,
) -> None:
    """Асинхронно обходить веб-сторінки та потоково записує результат у JSONL."""
    _configure_logging(verbose)
    stats = asyncio.run(
        _run_crawl_cli(
            start_url=start_url,
            out=out,
            log_file=log_file,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            delay=delay,
            timeout=timeout,
            same_host=same_host,
            debug=debug,
        ),
        debug=debug,
    )
    err_console.print(
        f"[green]Готово:[/green] зібрано {stats.pages_fetched} сторінок за "
        f"{stats.elapsed_s:.2f} с ({stats.pages_per_sec:.2f} стор/с), "
        f"помилок={stats.errors}. Лог: {log_file}"
    )


@app.command("index")
def index_cmd(
    corpus_path: Annotated[
        Path, typer.Argument(help="Каталог із .txt/.jsonl або один .jsonl файл")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Вихідний JSON-індекс")] = Path(
        "index.json"
    ),
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Кількість паралельних воркерів"),
    ] = 1,
    executor: Annotated[
        ExecutorType,
        typer.Option(
            "--executor",
            "-e",
            help="Виконавець: serial, threads або processes",
        ),
    ] = "processes",
    no_positions: Annotated[
        bool,
        typer.Option("--no-positions", help="Не зберігати позиції слів в індексі"),
    ] = False,
    verbose: VerboseOption = 0,
) -> None:
    """Будує інвертований індекс із каталогу документів або JSONL-файлу."""
    _configure_logging(verbose)
    if not corpus_path.exists():
        err_console.print(f"[red]Помилка:[/red] шлях корпусу не знайдено: {corpus_path}")
        raise typer.Exit(code=1)

    if corpus_path.is_file() and corpus_path.suffix.lower() == ".jsonl":
        idx = Index(store_positions=not no_positions)

        def jsonl_pipeline() -> Iterator[tuple[int, list[str], str]]:
            for doc_id, (doc_ref, text) in enumerate(iter_document_files(corpus_path)):
                yield doc_id, list(tokenize(text)), _normalize_doc_ref(doc_ref)

        idx.build_from_pipeline(jsonl_pipeline())
        idx.save_json(out)
        err_console.print(f"[green]Готово:[/green] {idx!r} збережено у {out}")
        return

    if not corpus_path.is_dir():
        err_console.print(f"[red]Помилка:[/red] очікувався каталог або .jsonl: {corpus_path}")
        raise typer.Exit(code=1)

    files = sorted(corpus_path.rglob("*.txt"))
    if not files:
        err_console.print(f"[red]Помилка:[/red] у каталозі {corpus_path} немає .txt файлів.")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=err_console,
    ) as progress:
        task_id = progress.add_task(
            f"Індексація ({executor}, workers={workers})...", total=len(files)
        )

        def _advance(count: int) -> None:
            progress.advance(task_id, advance=count)

        idx = build_index(
            files,
            workers=workers,
            executor=executor,
            store_positions=not no_positions,
            on_chunk_done=_advance,
        )

    idx.save_json(out)
    err_console.print(f"[green]Готово:[/green] {idx!r} збережено у {out}")


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help='Пошуковий запит (AND, OR, NOT, "фрази")')],
    index_file: Annotated[
        Path, typer.Option("--index", "-i", help="Шлях до файлу індексу (.json)")
    ] = Path("index.json"),
    scorer: Annotated[
        ScorerName, typer.Option("--scorer", "-s", help="Алгоритм: bm25 або tfidf")
    ] = "bm25",
    engine: Annotated[
        EngineName, typer.Option("--engine", "-e", help="Перетин: merge або set")
    ] = "merge",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Кількість результатів")] = 5,
    as_json: Annotated[
        bool, typer.Option("--json", help="Вивести результати у форматі JSON")
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
    table.add_column("Файл / URL", style="yellow", no_wrap=True)
    table.add_column("Сніпет", style="white")

    for rank, res in enumerate(results, start=1):
        display_source = (
            res.path
            if res.path.startswith(("http://", "https://"))
            else (Path(res.path).name if res.path else "-")
        )
        table.add_row(
            str(rank),
            str(res.doc_id),
            f"{res.score:.4f}",
            display_source,
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
        typer.Option("--corpus", "-c", help="Опційний каталог для заміру пам'яті"),
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


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host", help="Хост для прослуховування")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535, help="Порт")] = 8000,
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Кількість воркерів uvicorn")
    ] = 1,
    index_file: Annotated[
        Path | None,
        typer.Option("--index", "-i", help="Опційний шлях до індексу (встановлює INDEX_PATH)"),
    ] = None,
    verbose: VerboseOption = 0,
) -> None:
    """Запускає HTTP-сервер FastAPI через uvicorn."""
    _configure_logging(verbose)
    if index_file is not None:
        os.environ["INDEX_PATH"] = str(index_file)
    effective_port = int(os.getenv("PORT", str(port)))
    err_console.print(
        f"[green]Запуск findex web:[/green] http://{host}:{effective_port} "
        f"(workers={workers}, INDEX_PATH={os.getenv('INDEX_PATH', 'index.json')})"
    )
    uvicorn.run(
        "findex.web.app:create_app",
        factory=True,
        host=host,
        port=effective_port,
        workers=workers,
    )


if __name__ == "__main__":
    app()

import asyncio
import contextlib
import logging
import posixpath
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "findex-crawler/0.6.0 (+https://github.com/Bohdan00724/python-lab-1; "
    "contact: bohdan.starchenko@student.kai.edu.ua)"
)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int | None
    bytes_read: int
    elapsed_s: float
    text: str = ""
    attempts: int = 1
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CrawledPage:
    url: str
    title: str
    text: str
    links: tuple[str, ...]
    status_code: int
    bytes_read: int
    elapsed_s: float


@dataclass(slots=True)
class CrawlStats:
    pages_fetched: int = 0
    in_flight: int = 0
    errors: int = 0
    queue_depth: int = 0
    bytes_downloaded: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        return max(time.perf_counter() - self.start_time, 1e-6)

    @property
    def pages_per_sec(self) -> float:
        return self.pages_fetched / self.elapsed_s


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Канонізує URL: нижній регістр схеми/хоста, без #fragment, нормальний шлях і query."""
    joined = urljoin(base_url, raw_url.strip()) if base_url else raw_url.strip()
    if not joined:
        return None

    parts = urlsplit(joined)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    hostname = (parts.hostname or "").lower()
    if not hostname:
        return None

    port = parts.port
    is_default_port = (
        port is None
        or (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    )
    netloc = hostname if is_default_port else f"{hostname}:{port}"

    raw_path = parts.path or "/"
    norm_path = posixpath.normpath(raw_path)
    if not norm_path.startswith("/"):
        norm_path = "/" + norm_path
    if norm_path != "/" and norm_path.endswith("/"):
        norm_path = norm_path.rstrip("/")

    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    norm_query = urlencode(query_pairs, doseq=True)

    return urlunsplit((scheme, netloc, norm_path, norm_query, ""))


class LinkAndTextExtractor(HTMLParser):
    """Витягує заголовок, видимий текст і посилання <a href> через стандартний html.parser."""

    _SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth: int = 0
        self._in_title: bool = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag_l == "title":
            self._in_title = True
        elif tag_l == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    self.links.append(attr_val)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag_l == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self._title_parts.append(cleaned)
        else:
            self._text_parts.append(cleaned)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_parts).strip()


def extract_page_content(html: str, base_url: str) -> tuple[str, str, list[str]]:
    parser = LinkAndTextExtractor()
    parser.feed(html)
    parser.close()

    canonical_links: list[str] = []
    for href in parser.links:
        canon = canonicalize_url(href, base_url=base_url)
        if canon is not None:
            canonical_links.append(canon)
    return parser.title, parser.text, canonical_links


def parse_retry_after(header_val: str | None) -> float | None:
    """Розбирає заголовок Retry-After (секунди або HTTP-дата) у секунди очікування."""
    if not header_val:
        return None
    stripped = header_val.strip()
    try:
        seconds = float(stripped)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        target_dt = parsedate_to_datetime(stripped)
        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=UTC)
        delta = (target_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.2,
    jitter_ratio: float = 0.25,
) -> FetchResult:
    """Завантажує URL з asyncio.timeout та повторами (429, 5xx, таймаут) + backoff."""
    t_start = time.perf_counter()
    last_status: int | None = None
    last_error: str | None = None

    for attempt in range(1, max_retries + 2):
        retry_wait: float | None = None
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, follow_redirects=True)
            status = response.status_code
            last_status = status
            raw_bytes = response.content

            if 200 <= status < 300:
                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=status,
                    bytes_read=len(raw_bytes),
                    elapsed_s=time.perf_counter() - t_start,
                    text=response.text,
                    attempts=attempt,
                    error=None,
                )

            last_error = f"HTTP {status}"
            if status not in RETRYABLE_STATUS_CODES:
                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=status,
                    bytes_read=len(raw_bytes),
                    elapsed_s=time.perf_counter() - t_start,
                    text="",
                    attempts=attempt,
                    error=last_error,
                )

            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            exp_backoff = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, exp_backoff * jitter_ratio)
            retry_wait = max(exp_backoff + jitter, retry_after or 0.0)

        except httpx.TooManyRedirects:
            return FetchResult(
                url=url,
                final_url=url,
                status_code=None,
                bytes_read=0,
                elapsed_s=time.perf_counter() - t_start,
                attempts=attempt,
                error="TooManyRedirects",
            )
        except (TimeoutError, httpx.TimeoutException):
            last_status = None
            last_error = "TimeoutError"
            exp_backoff = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, exp_backoff * jitter_ratio)
            retry_wait = exp_backoff + jitter
        except httpx.TransportError as exc:
            last_status = None
            last_error = f"TransportError: {exc}"
            exp_backoff = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, exp_backoff * jitter_ratio)
            retry_wait = exp_backoff + jitter

        if attempt <= max_retries and retry_wait is not None:
            logger.warning(
                "Повтор %d/%d для %s через %.3f с (причина: %s)",
                attempt,
                max_retries,
                url,
                retry_wait,
                last_error,
            )
            await asyncio.sleep(retry_wait)

    return FetchResult(
        url=url,
        final_url=url,
        status_code=last_status,
        bytes_read=0,
        elapsed_s=time.perf_counter() - t_start,
        attempts=max_retries + 1,
        error=last_error or "MaxRetriesExceeded",
    )


class RobotsCache:
    """Асинхронний кеш правил robots.txt окремо для кожного хоста."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, client: httpx.AsyncClient, url: str) -> bool:
        parts = urlsplit(url)
        host_key = f"{parts.scheme.lower()}://{(parts.hostname or '').lower()}"
        if parts.port:
            host_key = f"{host_key}:{parts.port}"

        if host_key not in self._parsers:
            lock = self._locks.setdefault(host_key, asyncio.Lock())
            async with lock:
                if host_key not in self._parsers:
                    self._parsers[host_key] = await self._load_robots(client, host_key)

        return self._parsers[host_key].can_fetch(self.user_agent, url)

    async def _load_robots(self, client: httpx.AsyncClient, host_key: str) -> RobotFileParser:
        robots_url = f"{host_key}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            async with asyncio.timeout(self.timeout):
                resp = await client.get(robots_url, follow_redirects=True)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in {401, 403}:
                rp.parse(["User-agent: *", "Disallow: /"])
            else:
                rp.parse([])
        except Exception as exc:
            logger.debug("Не вдалося завантажити %s (%s); дозволяємо обхід.", robots_url, exc)
            rp.parse([])
        return rp


def _write_crawl_log(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def crawl(
    start_urls: list[str] | str,
    *,
    max_pages: int = 50,
    concurrency: int = 5,
    per_host_concurrency: int = 2,
    min_delay_per_host: float = 0.05,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.2,
    same_host_only: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    robots_cache: RobotsCache | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    log_path: Path | None = None,
) -> AsyncIterator[CrawledPage]:
    """Асинхронний генератор обходу сайту на базі TaskGroup, семафорів та черги."""
    if max_pages < 1:
        return

    seeds = [start_urls] if isinstance(start_urls, str) else list(start_urls)
    canonical_seeds: list[str] = []
    allowed_hosts: set[str] = set()

    for raw in seeds:
        canon = canonicalize_url(raw)
        if canon is not None:
            canonical_seeds.append(canon)
            host = urlsplit(canon).hostname
            if host:
                allowed_hosts.add(host.lower())

    if not canonical_seeds:
        return

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    live_stats = stats if stats is not None else CrawlStats()
    live_stats.start_time = time.perf_counter()

    robots = robots_cache if robots_cache is not None else RobotsCache(user_agent=user_agent)
    url_queue: asyncio.Queue[str] = asyncio.Queue()
    results_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    seen: set[str] = set()

    for seed in canonical_seeds:
        if seed not in seen:
            seen.add(seed)
            url_queue.put_nowait(seed)
    live_stats.queue_depth = url_queue.qsize()

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_delay_locks: dict[str, asyncio.Lock] = {}
    host_last_ts: dict[str, float] = {}
    state_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    active_workers = 0

    async def wait_host_slot(host: str) -> None:
        if min_delay_per_host <= 0:
            return
        d_lock = host_delay_locks.setdefault(host, asyncio.Lock())
        async with d_lock:
            now = time.monotonic()
            last = host_last_ts.get(host, 0.0)
            wait_s = min_delay_per_host - (now - last)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            host_last_ts[host] = time.monotonic()

    async def worker(client: httpx.AsyncClient, worker_tasks: list[asyncio.Task[None]]) -> None:
        nonlocal active_workers
        while not stop_event.is_set():
            try:
                url = await url_queue.get()
            except asyncio.CancelledError:
                return

            async with state_lock:
                if stop_event.is_set():
                    url_queue.task_done()
                    live_stats.queue_depth = url_queue.qsize()
                    return
                active_workers += 1
                live_stats.in_flight += 1
                live_stats.queue_depth = url_queue.qsize()

            try:
                host = (urlsplit(url).hostname or "").lower()
                host_sem = host_sems.setdefault(
                    host, asyncio.Semaphore(max(1, per_host_concurrency))
                )

                async with global_sem, host_sem:
                    if stop_event.is_set():
                        continue

                    allowed = await robots.can_fetch(client, url)
                    if not allowed:
                        live_stats.errors += 1
                        _write_crawl_log(log_path, f"DISALLOWED robots.txt | {url}")
                        continue

                    await wait_host_slot(host)
                    if stop_event.is_set():
                        continue

                    res = await fetch(
                        client,
                        url,
                        timeout=timeout,
                        max_retries=max_retries,
                        base_delay=base_delay,
                    )

                if res.error is not None or res.status_code is None:
                    live_stats.errors += 1
                    _write_crawl_log(
                        log_path,
                        f"ERROR status={res.status_code} attempts={res.attempts} "
                        f"time={res.elapsed_s:.3f}s reason={res.error} | {url}",
                    )
                    continue

                title, text, links = extract_page_content(res.text, base_url=res.final_url)
                _write_crawl_log(
                    log_path,
                    f"OK status={res.status_code} bytes={res.bytes_read} "
                    f"attempts={res.attempts} time={res.elapsed_s:.3f}s | {url}",
                )

                should_emit = False
                reached_limit = False
                async with state_lock:
                    if live_stats.pages_fetched < max_pages:
                        live_stats.pages_fetched += 1
                        live_stats.bytes_downloaded += res.bytes_read
                        should_emit = True
                        if live_stats.pages_fetched >= max_pages:
                            reached_limit = True
                            stop_event.set()
                        else:
                            for link in links:
                                if same_host_only:
                                    link_host = (urlsplit(link).hostname or "").lower()
                                    if link_host not in allowed_hosts:
                                        continue
                                if link not in seen:
                                    seen.add(link)
                                    url_queue.put_nowait(link)
                        live_stats.queue_depth = url_queue.qsize()

                if should_emit:
                    await results_queue.put(
                        CrawledPage(
                            url=res.final_url,
                            title=title,
                            text=text,
                            links=tuple(links),
                            status_code=res.status_code,
                            bytes_read=res.bytes_read,
                            elapsed_s=res.elapsed_s,
                        )
                    )

                if reached_limit:
                    for task in worker_tasks:
                        if task is not asyncio.current_task():
                            task.cancel()
                    return

            finally:
                url_queue.task_done()
                async with state_lock:
                    active_workers -= 1
                    live_stats.in_flight = max(0, live_stats.in_flight - 1)
                    live_stats.queue_depth = url_queue.qsize()
                    if active_workers == 0 and url_queue.empty():
                        stop_event.set()
                        for task in worker_tasks:
                            if task is not asyncio.current_task():
                                task.cancel()

    async def produce() -> None:
        headers = {"User-Agent": user_agent}
        try:
            async with httpx.AsyncClient(headers=headers, transport=transport) as client:
                worker_tasks: list[asyncio.Task[None]] = []
                with contextlib.suppress(asyncio.CancelledError):
                    async with asyncio.TaskGroup() as tg:
                        for _ in range(max(1, concurrency)):
                            t = tg.create_task(worker(client, worker_tasks))
                            worker_tasks.append(t)
        finally:
            await results_queue.put(None)

    producer_task = asyncio.create_task(produce())
    try:
        while True:
            page = await results_queue.get()
            if page is None:
                break
            yield page
    finally:
        stop_event.set()
        if not producer_task.done():
            producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task

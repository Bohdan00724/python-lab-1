import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.core import iter_documents
from findex.crawler import (
    CrawledPage,
    CrawlStats,
    RobotsCache,
    canonicalize_url,
    crawl,
    fetch,
)

runner = CliRunner()


@pytest.mark.parametrize(
    ("raw", "base", "expected"),
    [
        (
            "HTTP://Example.COM:80/Docs//Guide/../Intro/#section1",
            None,
            "http://example.com/Docs/Intro",
        ),
        (
            "https://example.com:443/search?b=2&a=1#top",
            None,
            "https://example.com/search?a=1&b=2",
        ),
        (
            "/page/?z=9&a=1",
            "https://docs.example.org/base/",
            "https://docs.example.org/page?a=1&z=9",
        ),
        ("mailto:admin@example.com", "https://example.org", None),
        ("javascript:void(0)", "https://example.org", None),
    ],
)
def test_canonicalize_url(raw: str, base: str | None, expected: str | None) -> None:
    assert canonicalize_url(raw, base_url=base) == expected


def test_fetch_retries_on_429_with_retry_after_and_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Too Many Requests")
        return httpx.Response(200, text="<html><body>Recovered</body></html>")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.test/item",
                timeout=2.0,
                max_retries=3,
                base_delay=0.01,
            )
        assert res.status_code == 200
        assert res.attempts == 3
        assert "Recovered" in res.text

    asyncio.run(run())


def test_fetch_does_not_retry_non_429_4xx() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="Not Found")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.test/missing",
                max_retries=3,
                base_delay=0.01,
            )
        assert res.status_code == 404
        assert res.attempts == 1

    asyncio.run(run())


def test_fetch_retries_on_timeout_up_to_max_retries() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("Simulated timeout")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.test/slow",
                timeout=0.5,
                max_retries=3,
                base_delay=0.005,
            )
        assert res.status_code is None
        assert res.attempts == 4
        assert res.error == "TimeoutError"

    asyncio.run(run())


def test_robots_cache_caches_per_host_and_blocks_disallowed() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="OK")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        cache = RobotsCache()
        async with httpx.AsyncClient(transport=transport) as client:
            assert await cache.can_fetch(client, "https://docs.test/public/page") is True
            assert await cache.can_fetch(client, "https://docs.test/private/secret") is False
            assert await cache.can_fetch(client, "https://docs.test/public/other") is True
        assert robots_hits == 1

    asyncio.run(run())


def test_crawl_respects_max_pages_dedup_robots_and_streams_jsonl(tmp_path: Path) -> None:
    pages_db: dict[str, str] = {
        "/robots.txt": "User-agent: *\nDisallow: /secret\n",
        "/": (
            "<html><head><title>Home</title></head><body>"
            "Welcome to async Python index. "
            "<a href='/p1?b=2&a=1#frag'>P1</a> "
            "<a href='/p1?a=1&b=2'>P1 duplicate</a> "
            "<a href='/secret'>Secret</a> "
            "<a href='/p2'>P2</a>"
            "</body></html>"
        ),
        "/p1?a=1&b=2": (
            "<html><head><title>Page One</title></head><body>"
            "Concurrency with TaskGroup and Semaphore. "
            "<a href='/p3'>P3</a>"
            "</body></html>"
        ),
        "/p2": (
            "<html><head><title>Page Two</title></head><body>"
            "Exponential backoff and jitter in httpx."
            "</body></html>"
        ),
        "/p3": (
            "<html><head><title>Page Three</title></head><body>"
            "Should not be fetched when max_pages=3."
            "</body></html>"
        ),
        "/secret": "<html><body>Top secret content</body></html>",
    }

    fetched_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        full_path = request.url.path
        if request.url.query:
            full_path = f"{full_path}?{request.url.query.decode()}"
        if full_path != "/robots.txt":
            fetched_paths.append(full_path)
        body = pages_db.get(full_path)
        if body is None:
            return httpx.Response(404, text="Not found")
        return httpx.Response(200, text=body)

    jsonl_path = tmp_path / "crawled.jsonl"
    log_path = tmp_path / "crawl.log"
    index_path = tmp_path / "crawled_index.json"

    async def run() -> None:
        stats = CrawlStats()
        transport = httpx.MockTransport(handler)
        collected: list[CrawledPage] = []
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            async for page in crawl(
                "https://crawler.test/",
                max_pages=3,
                concurrency=2,
                per_host_concurrency=2,
                min_delay_per_host=0.0,
                transport=transport,
                stats=stats,
                log_path=log_path,
            ):
                collected.append(page)
                jf.write(
                    json.dumps(
                        {"url": page.url, "title": page.title, "text": page.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        assert len(collected) == 3
        assert stats.pages_fetched == 3
        assert "/secret" not in fetched_paths
        assert fetched_paths.count("/p1?a=1&b=2") == 1

    asyncio.run(run())

    docs = list(iter_documents(jsonl_path))
    assert len(docs) == 3

    res_idx = runner.invoke(app, ["index", str(jsonl_path), "--out", str(index_path)])
    assert res_idx.exit_code == 0

    res_search = runner.invoke(
        app, ["search", "TaskGroup AND Semaphore", "--index", str(index_path)]
    )
    assert res_search.exit_code == 0
    assert "https://crawler.test/p1?a=1&b=2" in res_search.stdout
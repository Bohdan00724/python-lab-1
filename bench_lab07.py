import asyncio
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
from fastapi import Depends, Query

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from findex.index import Index  # noqa: E402
from findex.ranking import ScorerName  # noqa: E402
from findex.web.app import (  # noqa: E402
    _execute_paginated_search,
    create_app,
    get_required_index,
)
from findex.web.schemas import SearchResponseOut  # noqa: E402


def build_bench_app():
    """Фабрика FastAPI з маршрутом /search (def) та /search_async (async def) для замірів."""
    app = create_app()

    @app.get("/search_async", response_model=SearchResponseOut)
    async def search_async(
        index: Annotated[Index, Depends(get_required_index)],
        q: Annotated[str, Query()] = "python",
        k: Annotated[int, Query()] = 10,
        scorer: Annotated[ScorerName, Query()] = "bm25",
        page: Annotated[int, Query()] = 1,
    ) -> SearchResponseOut:
        return _execute_paginated_search(index=index, q=q, k=k, scorer=scorer, page=page)

    return app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, math.ceil(len(sorted_vals) * pct / 100.0) - 1)
    return sorted_vals[max(0, idx)]


async def _wait_healthy(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.15)
    raise RuntimeError(f"Сервер {base_url} не відповів /health за {timeout_s} с")


async def run_load_test(
    base_url: str,
    endpoint: str = "/search",
    total_requests: int = 400,
    concurrency: int = 20,
) -> tuple[float, float, float, float]:
    latencies_ms: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        await client.get(endpoint, params={"q": "python", "k": 10})

        async def one_req(i: int) -> None:
            query = "python" if i % 2 == 0 else "algorithm OR data"
            async with sem:
                t0 = time.perf_counter()
                resp = await client.get(endpoint, params={"q": query, "k": 10})
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    latencies_ms.append(dt_ms)

        t_start = time.perf_counter()
        await asyncio.gather(*(one_req(i) for i in range(total_requests)))
        total_s = max(time.perf_counter() - t_start, 1e-6)

    latencies_ms.sort()
    rps = len(latencies_ms) / total_s
    p50 = _percentile(latencies_ms, 50.0)
    p95 = _percentile(latencies_ms, 95.0)
    p99 = _percentile(latencies_ms, 99.0)
    return rps, p50, p95, p99


def _start_server(port: int, workers: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{ROOT_DIR}"
    env.setdefault("INDEX_PATH", str(ROOT_DIR / "index.json"))
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "bench_lab07:build_bench_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            str(workers),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


async def main() -> None:
    deployed_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
    rows: list[tuple[str, float, float, float, float]] = []

    p1 = _free_port()
    proc1 = _start_server(p1, workers=1)
    try:
        url1 = f"http://127.0.0.1:{p1}"
        await _wait_healthy(url1)

        t_async_20 = time.perf_counter()
        await run_load_test(url1, endpoint="/search_async", total_requests=20, concurrency=20)
        wall_async_20 = (time.perf_counter() - t_async_20) * 1000.0

        t_def_20 = time.perf_counter()
        await run_load_test(url1, endpoint="/search", total_requests=20, concurrency=20)
        wall_def_20 = (time.perf_counter() - t_def_20) * 1000.0

        print(
            f"20 одночасних запитів /search: "
            f"async def = {wall_async_20:.2f} мс | def = {wall_def_20:.2f} мс"
        )

        rps_async, p50_a, p95_a, p99_a = await run_load_test(
            url1, endpoint="/search_async", total_requests=400, concurrency=20
        )
        rows.append(("1 воркер (`async def`)", rps_async, p50_a, p95_a, p99_a))

        rps_def, p50_d, p95_d, p99_d = await run_load_test(
            url1, endpoint="/search", total_requests=400, concurrency=20
        )
        rows.append(("1 воркер (`def`)", rps_def, p50_d, p95_d, p99_d))
    finally:
        proc1.terminate()
        proc1.wait(timeout=5)

    p4 = _free_port()
    proc4 = _start_server(p4, workers=4)
    try:
        url4 = f"http://127.0.0.1:{p4}"
        await _wait_healthy(url4)
        rps_4, p50_4, p95_4, p99_4 = await run_load_test(
            url4, endpoint="/search", total_requests=400, concurrency=20
        )
        rows.append(("4 воркери (`def`)", rps_4, p50_4, p95_4, p99_4))
    finally:
        proc4.terminate()
        proc4.wait(timeout=5)

    if deployed_url:
        await _wait_healthy(deployed_url, timeout_s=30.0)
        rps_dep, p50_dep, p95_dep, p99_dep = await run_load_test(
            deployed_url, endpoint="/search", total_requests=200, concurrency=10
        )
        rows.append((f"Розгорнутий сервіс ({deployed_url})", rps_dep, p50_dep, p95_dep, p99_dep))

    print("\n=== ТАБЛИЦЯ НАВАНТАЖЕННЯ (LAB 07) ===")
    print("| Конфігурація | Запитів/с (RPS) | p50 (мс) | p95 (мс) | p99 (мс) |")
    print("|---|---:|---:|---:|---:|")
    for label, rps, p50, p95, p99 in rows:
        print(f"| {label} | {rps:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} |")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        uvicorn.run(build_bench_app(), host="127.0.0.1", port=8000)
    else:
        asyncio.run(main())

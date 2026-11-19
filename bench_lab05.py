import ctypes
import os
import platform
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import wintypes
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from findex.evaluate import CORPUS_DOCS  # noqa: E402
from findex.index import (  # noqa: E402
    ExecutorType,
    PartialIndex,
    _build_partial_task,
    build_index,
    build_partial,
    merge,
    split_into_chunks,
)


def get_cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    return platform.processor() or "Unknown CPU"


def get_peak_rss_mb() -> float:
    if sys.platform == "win32":

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0


def prepare_full_corpus(corpus_dir: Path, num_files: int = 1200, repeats: int = 45) -> list[Path]:
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    base_texts = list(CORPUS_DOCS.values())
    paths: list[Path] = []
    for i in range(num_files):
        body = "\n".join(base_texts[(i + k) % len(base_texts)] for k in range(repeats))
        file_path = corpus_dir / f"doc_{i:04d}.txt"
        file_path.write_text(f"Document {i}\n{body}\n", encoding="utf-8")
        paths.append(file_path)
    return paths


def measure_once(
    paths: list[Path], workers: int, executor: ExecutorType
) -> tuple[float, float, float, float]:
    """Повертає (wall_s, cpu_s, peak_rss_mb, merge_s)."""
    chunks = split_into_chunks(paths, workers)
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    partials: list[PartialIndex] = []
    if executor == "serial":
        for start_id, chunk_paths in chunks:
            partials.append(build_partial(chunk_paths, start_doc_id=start_id))
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [(chunk_paths, start_id, True) for start_id, chunk_paths in chunks]
            partials = list(pool.map(_build_partial_task, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = [(chunk_paths, start_id, True) for start_id, chunk_paths in chunks]
            partials = list(pool.map(_build_partial_task, tasks))

    t_merge_start = time.perf_counter()
    merged = merge(partials)
    merge_s = time.perf_counter() - t_merge_start

    wall_s = time.perf_counter() - t_wall_start
    cpu_s = time.process_time() - t_cpu_start
    rss_mb = get_peak_rss_mb()

    merged.clear()
    partials.clear()
    return wall_s, cpu_s, rss_mb, merge_s


def save_speedup_plot(
    worker_grid: list[int],
    thread_speedups: list[float],
    proc_speedups: list[float],
    cpu_model: str,
    cpu_cores: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("docs").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(worker_grid, thread_speedups, marker="o", linewidth=2, label="Threads (GIL)")
    plt.plot(worker_grid, proc_speedups, marker="s", linewidth=2, label="Processes")
    plt.plot(
        worker_grid,
        worker_grid,
        linestyle="--",
        color="gray",
        label="Ideal linear (y = x)",
    )
    plt.title(f"findex Parallel Indexing Speedup ({cpu_model}, {cpu_cores} cores)")
    plt.xlabel("Number of workers")
    plt.ylabel("Speedup (x)")
    plt.xticks(worker_grid)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plot_path = Path("docs/speedup_lab05.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    return plot_path


def demo_gil_and_race() -> None:
    gil_fn = getattr(sys, "_is_gil_enabled", None)
    gil_state = gil_fn() if callable(gil_fn) else True
    print(f"\nPython: {sys.version.split()[0]} | sys._is_gil_enabled() = {gil_state}")

    bench_dir = Path("data_gil_check")
    try:
        paths = prepare_full_corpus(bench_dir, num_files=400, repeats=30)
        _ = build_index(paths, workers=1, executor="serial")  # прогрів кешу ОС

        t0 = time.perf_counter()
        _ = build_index(paths, workers=1, executor="serial")
        t_serial = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = build_index(paths, workers=4, executor="threads")
        t_threads = time.perf_counter() - t1
        print(
            f"findex threads (4 workers) vs serial: "
            f"{t_serial:.4f}s -> {t_threads:.4f}s (speedup {t_serial / t_threads:.2f}x)"
        )
    finally:
        shutil.rmtree(bench_dir, ignore_errors=True)

    shared_counter = 0
    iterations = 100_000

    def unsafe_inc() -> None:
        nonlocal shared_counter
        for _ in range(iterations):
            tmp = shared_counter
            if tmp % 500 == 0:
                time.sleep(0)
            shared_counter = tmp + 1

    threads = [threading.Thread(target=unsafe_inc) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Unsafe counter (очікувалось {4 * iterations}): {shared_counter}")

    safe_counter = 0
    lock = threading.Lock()

    def safe_inc() -> None:
        nonlocal safe_counter
        for _ in range(iterations):
            with lock:
                tmp = safe_counter
                if tmp % 500 == 0:
                    time.sleep(0)
                safe_counter = tmp + 1

    threads = [threading.Thread(target=safe_inc) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Safe counter з Lock (очікувалось {4 * iterations}): {safe_counter}")


def main() -> None:
    if "--gil-only" in sys.argv:
        demo_gil_and_race()
        return

    cpu_model = get_cpu_model()
    cpu_cores = os.cpu_count() or 4
    worker_grid = sorted({1, 2, 4, 8, cpu_cores})

    print(f"CPU: {cpu_model}")
    print(f"Logical cores (os.cpu_count()): {cpu_cores}")
    print(f"Worker grid: {worker_grid}")

    bench_dir = Path("data_bench")
    paths = prepare_full_corpus(bench_dir, num_files=1200, repeats=45)

    try:
        # Прогрівочний запуск, щоб вирівняти файловий кеш ОС
        _ = measure_once(paths, 1, "serial")

        serial_runs = [measure_once(paths, 1, "serial") for _ in range(3)]
        base_wall = statistics.median(r[0] for r in serial_runs)
        base_cpu = statistics.median(r[1] for r in serial_runs)
        base_rss = statistics.median(r[2] for r in serial_runs)
        base_merge = statistics.median(r[3] for r in serial_runs)

        rows: list[tuple[str, int, float, float, float, float, float]] = [
            ("serial", 1, base_wall, base_cpu, base_rss, base_merge, 1.0)
        ]

        thread_speedups: list[float] = []
        proc_speedups: list[float] = []

        for exec_type in ("threads", "processes"):
            for w in worker_grid:
                runs = [measure_once(paths, w, exec_type) for _ in range(3)]
                med_wall = statistics.median(r[0] for r in runs)
                med_cpu = statistics.median(r[1] for r in runs)
                med_rss = statistics.median(r[2] for r in runs)
                med_merge = statistics.median(r[3] for r in runs)
                speedup = base_wall / med_wall
                rows.append((exec_type, w, med_wall, med_cpu, med_rss, med_merge, speedup))
                if exec_type == "threads":
                    thread_speedups.append(speedup)
                else:
                    proc_speedups.append(speedup)

        print("\n=== MARKDOWN TABLE (MEDIAN OF 3 RUNS) ===")
        print(
            "| Executor | Workers | Wall time (s) | CPU time (s) | "
            "Peak RSS (MB) | Merge (s) | Speedup |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for ex, w, wall, cpu, rss, mrg, sp in rows:
            print(
                f"| `{ex}` | {w} | {wall:.4f} | {cpu:.4f} | "
                f"{rss:.1f} | {mrg:.4f} | {sp:.2f}x |"
            )

        plot_path = save_speedup_plot(
            worker_grid, thread_speedups, proc_speedups, cpu_model, cpu_cores
        )
        print(f"\nГрафік збережено у {plot_path}")

        demo_gil_and_race()
    finally:
        shutil.rmtree(bench_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

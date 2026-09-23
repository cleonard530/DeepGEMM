#!/usr/bin/env python3
"""Benchmark DeepGEMM under representative Python-thread contention.

Run this on the same TORCH_LIBRARY branch twice: once with DeepJIT's normal
GIL support and once with ``DJ_DISABLE_GIL=1``. Change ``BUILD_LABEL`` between
runs. Comparing those two reports isolates the DeepJIT macro.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import deep_gemm


REPO_ROOT = Path(__file__).resolve().parents[1]

# Edit only the label between the two builds being compared.
BUILD_LABEL = "torch_deepjit_torch_temp"
M, N, K = 128, 4096, 7168
WARMUPS = 20
CALLS = 1_000
TRIALS = 3
PYTHON_BURST_ITERS = 2_000
SCHEDULER_WAIT_SECONDS = 0.001
IO_WAIT_SECONDS = 0.001
OUTPUT_FILE = REPO_ROOT / f"gil_production_results_{BUILD_LABEL}.json"


def make_kernel_call():
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    return lambda: deep_gemm.bf16_gemm_nt(a, b, d)


def run_python_burst(state: int) -> int:
    for value in range(PYTHON_BURST_ITERS):
        state = (state * 1_664_525 + value + 1_013_904_223) & 0xFFFFFFFF
    return state


def worker_loop(profile: str, stop: threading.Event, ready: threading.Event,
                result: list[int]) -> None:
    tasks = 0
    state = 1
    ready.set()
    while not stop.is_set():
        if profile == "io_bound":
            tasks += 1
            stop.wait(IO_WAIT_SECONDS)
        elif profile == "scheduler":
            state = run_python_burst(state)
            tasks += 1
            stop.wait(SCHEDULER_WAIT_SECONDS)
        elif profile == "cpu_bound":
            state = run_python_burst(state)
            tasks += 1
        else:
            raise ValueError(f"unknown worker profile: {profile}")
    result[:] = [tasks, state]


def start_worker(profile: str):
    stop = threading.Event()
    ready = threading.Event()
    result = [0, 0]
    worker = threading.Thread(
        target=worker_loop, args=(profile, stop, ready, result), daemon=True
    )
    worker.start()
    ready.wait()
    return worker, stop, result


def run_foreground(call, profile: str | None) -> tuple[float, int]:
    torch.cuda.synchronize()
    worker = stop = result = None
    if profile is not None:
        worker, stop, result = start_worker(profile)

    start = time.perf_counter()
    for _ in range(CALLS):
        call()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tasks = 0
    if worker is not None and stop is not None and result is not None:
        stop.set()
        worker.join()
        tasks = result[0]
    return elapsed, tasks


def run_worker_alone(profile: str, duration: float) -> int:
    worker, stop, result = start_worker(profile)
    time.sleep(duration)
    stop.set()
    worker.join()
    return result[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    call = make_kernel_call()
    for _ in range(WARMUPS):
        call()
    torch.cuda.synchronize()

    results = []
    for profile in (None, "io_bound", "scheduler", "cpu_bound"):
        profile_name = profile or "no_worker"
        for trial in range(1, TRIALS + 1):
            elapsed, worker_tasks = run_foreground(call, profile)
            standalone_tasks = (
                run_worker_alone(profile, elapsed) if profile is not None else 0
            )
            worker_progress = (
                worker_tasks / standalone_tasks if standalone_tasks else None
            )
            row = {
                "profile": profile_name,
                "trial": trial,
                "foreground_seconds": elapsed,
                "foreground_us_per_call": elapsed / CALLS * 1e6,
                "foreground_calls_per_second": CALLS / elapsed,
                "worker_tasks": worker_tasks,
                "standalone_worker_tasks": standalone_tasks,
                "worker_progress_ratio": worker_progress,
            }
            results.append(row)
            progress = "n/a" if worker_progress is None else f"{worker_progress:.1%}"
            print(
                f"{profile_name:10} trial {trial}: "
                f"{row['foreground_us_per_call']:9.2f} us/call, "
                f"worker progress={progress}"
            )

    summaries = {}
    for profile in ("no_worker", "io_bound", "scheduler", "cpu_bound"):
        rows = [row for row in results if row["profile"] == profile]
        progress_values = [
            row["worker_progress_ratio"] for row in rows
            if row["worker_progress_ratio"] is not None
        ]
        summaries[profile] = {
            "median_foreground_us_per_call": statistics.median(
                row["foreground_us_per_call"] for row in rows
            ),
            "median_worker_progress_ratio": (
                statistics.median(progress_values) if progress_values else None
            ),
        }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_label": BUILD_LABEL,
        "deep_gemm_path": str(Path(deep_gemm.__file__).resolve()),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "shape": {"m": M, "n": N, "k": K},
        "warmups": WARMUPS,
        "calls": CALLS,
        "trials": TRIALS,
        "summaries": summaries,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
